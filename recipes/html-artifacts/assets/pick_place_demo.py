"""Write the synthetic pick-and-place recording that the gallery captures.

The gallery shows real Rerun viewer output of this recording: screenshots and a
sweep video made with the rerun-viewer-validation skill. The data is synthetic;
the viewer pixels are real.

Run:
    pixi exec --spec rerun-sdk==0.38.1 --spec tyro --spec jaxtyping --spec numpy \
        python pick_place_demo.py --out /tmp/pick-place.rrd
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float64, Int64

# Keyframes: time [s], gripper palm x, y, z [m], gripper closed (0 or 1).
KEYS: Float64[np.ndarray, "k 5"] = np.array(
    [
        [0.0, -0.30, 0.0, 0.30, 0.0],
        [1.4, -0.10, 0.0, 0.30, 0.0],
        [2.2, -0.10, 0.0, 0.06, 0.0],
        [2.6, -0.10, 0.0, 0.06, 1.0],
        [3.6, -0.10, 0.0, 0.30, 1.0],
        [5.0, 0.35, 0.0, 0.30, 1.0],
        [5.8, 0.35, 0.0, 0.07, 1.0],
        [6.2, 0.35, 0.0, 0.07, 0.0],
        [7.2, 0.35, 0.0, 0.30, 0.0],
        [8.0, 0.35, 0.0, 0.30, 0.0],
    ]
)
EVENTS: list[tuple[float, str, str]] = [
    (0.0, "INFO", "approach: move above the cube"),
    (2.2, "INFO", "grasp: descend and close"),
    (2.6, "INFO", "contact: cube held"),
    (5.8, "INFO", "place: lower into the bin"),
    (6.2, "INFO", "release: cube in the bin"),
]
CUBE_HALF_M: float = 0.025
PALM_TO_CUBE_M: float = 0.035
FINGER_OPEN_M: float = 0.045
FINGER_CLOSED_M: float = 0.029


@dataclass(slots=True)
class DemoConfig:
    """Options for the demo recording."""

    out: Path = Path("/tmp/pick-place.rrd")
    """Where to write the .rrd file."""
    fps: int = 30
    """Samples per second on the `time` timeline."""


def palm_track(times_s: Float64[np.ndarray, "n"]) -> tuple[Float64[np.ndarray, "n 3"], Float64[np.ndarray, "n"]]:
    """Interpolate palm position (smoothstep per segment) and closed state at each time.

    Args:
        times_s: Sample times in seconds, shape (n,), float64.

    Returns:
        Palm positions in metres, shape (n, 3), float64, and the closed state, shape (n,), float64.
    """
    seg: Int64[np.ndarray, "n"] = np.clip(np.searchsorted(KEYS[:, 0], times_s, side="right") - 1, 0, len(KEYS) - 2)
    t0: Float64[np.ndarray, "n"] = KEYS[seg, 0]
    t1: Float64[np.ndarray, "n"] = KEYS[seg + 1, 0]
    k: Float64[np.ndarray, "n"] = np.clip((times_s - t0) / (t1 - t0), 0.0, 1.0)
    k = k * k * (3.0 - 2.0 * k)
    palm_xyz: Float64[np.ndarray, "n 3"] = KEYS[seg, 1:4] + (KEYS[seg + 1, 1:4] - KEYS[seg, 1:4]) * k[:, None]
    closed: Float64[np.ndarray, "n"] = KEYS[seg, 4]
    return palm_xyz, closed


def main(cfg: DemoConfig) -> None:
    """Log the scene, the time-varying columns, the events, and the blueprint to an .rrd file.

    Args:
        cfg: Output path and sample rate.
    """
    times_s: Float64[np.ndarray, "n"] = np.arange(0, int(KEYS[-1, 0] * cfg.fps) + 1) / cfg.fps
    palm_xyz: Float64[np.ndarray, "n 3"]
    closed: Float64[np.ndarray, "n"]
    palm_xyz, closed = palm_track(times_s)

    # Fingers ease between open and closed over 0.3 s so the opening plot is continuous.
    finger_m: Float64[np.ndarray, "n"] = np.where(closed > 0, FINGER_CLOSED_M, FINGER_OPEN_M)
    kernel: Float64[np.ndarray, "w"] = np.ones(int(0.3 * cfg.fps)) / int(0.3 * cfg.fps)
    finger_m = np.convolve(np.pad(finger_m, (len(kernel) - 1, 0), mode="edge"), kernel, mode="valid")

    # The cube rides with the palm while closed; otherwise it rests where it was released.
    cube_xyz: Float64[np.ndarray, "n 3"] = np.empty_like(palm_xyz)
    rest: Float64[np.ndarray, "3"] = np.array([-0.10, 0.0, CUBE_HALF_M])
    for i in range(len(times_s)):
        if closed[i] > 0:
            rest = palm_xyz[i] - np.array([0.0, 0.0, PALM_TO_CUBE_M])
        cube_xyz[i] = rest

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="Scene",
                eye_controls=rrb.EyeControls3D(position=(0.03, -0.8, 0.46), look_target=(0.03, 0.0, 0.09), eye_up=(0.0, 0.0, 1.0)),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/signals", name="Signals"),
                rrb.TextLogView(origin="/events", name="Events"),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    rr.init("html_artifacts_pick_place_demo")
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(cfg.out), default_blueprint=blueprint)  # before any log call, so every chunk lands in the file
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/table", rr.Boxes3D(centers=[0.0, 0.0, -0.005], half_sizes=[0.55, 0.3, 0.005], colors=[60, 80, 88], fill_mode="solid"), static=True)
    rr.log("world/bin", rr.Boxes3D(centers=[0.35, 0.0, 0.06], half_sizes=[0.08, 0.08, 0.06], colors=[143, 163, 166], labels="bin"), static=True)
    rr.log("world/cube", rr.Boxes3D(half_sizes=[CUBE_HALF_M] * 3, colors=[242, 196, 109], fill_mode="solid"), static=True)
    rr.log("world/gripper", rr.Boxes3D(centers=[0.0, 0.0, 0.0], half_sizes=[0.06, 0.012, 0.006], colors=[140, 196, 230], fill_mode="solid"), static=True)
    for side in ("left", "right"):
        rr.log(f"world/gripper/{side}", rr.Boxes3D(centers=[0.0, 0.0, -0.025], half_sizes=[0.004, 0.012, 0.022], colors=[140, 196, 230], fill_mode="solid"), static=True)
    rr.log("world/plan", rr.LineStrips3D([KEYS[:, 1:4]], colors=[127, 209, 192], radii=0.002), static=True)

    time_col: rr.TimeColumn = rr.TimeColumn("time", duration=times_s)
    zeros: Float64[np.ndarray, "n"] = np.zeros_like(finger_m)
    rr.send_columns("world/gripper", indexes=[time_col], columns=rr.Transform3D.columns(translation=palm_xyz))
    rr.send_columns("world/gripper/left", indexes=[time_col], columns=rr.Transform3D.columns(translation=np.stack([-finger_m, zeros, zeros], axis=1)))
    rr.send_columns("world/gripper/right", indexes=[time_col], columns=rr.Transform3D.columns(translation=np.stack([finger_m, zeros, zeros], axis=1)))
    rr.send_columns("world/cube", indexes=[time_col], columns=rr.Transform3D.columns(translation=cube_xyz))
    rr.send_columns("signals/gripper_opening_mm", indexes=[time_col], columns=rr.Scalars.columns(scalars=2000.0 * finger_m))
    rr.send_columns("signals/cube_height_mm", indexes=[time_col], columns=rr.Scalars.columns(scalars=1000.0 * cube_xyz[:, 2]))
    for t, level, text in EVENTS:
        rr.set_time("time", duration=t)
        rr.log("events", rr.TextLog(text, level=level))

    rr.get_global_data_recording().flush()
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(DemoConfig))
