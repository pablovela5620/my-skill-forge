# API Layer Template

This is the annotated template for a node's API module (`<module>/apis/<name>.py`). The API layer contains the node class (computation + verbose Rerun logging), result logging helper, typed configs/results, and the CLI entry point.

## Structure

```python
"""<Name> node.

<One paragraph describing what this node does, what model/algorithm it wraps,
and what its input/output contract is.>

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class <Name>Config:
    """Configuration for <name> prediction."""

    # Every field gets a per-field docstring (nerfstudio pattern).
    # Use jaxtyping Annotated types where applicable.
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend."""
    verbose: bool = False
    """Emit intermediate Rerun logging during computation when True."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class <Name>Result:
    """Output of <name> prediction."""

    # Every array field uses jaxtyping dtype + shape annotation.
    depth_map: Float32[ndarray, "H W"]
    """Predicted depth map in meters."""
    # Non-array fields are annotated normally.
    # Use slots=True for pure data containers when no inheritance is needed.


# ---------------------------------------------------------------------------
# Node class (computation + verbose intermediate logging)
# ---------------------------------------------------------------------------
class <Name>Node:
    """<Name> node: computation + verbose-gated intermediate Rerun logging.

    The node encapsulates both config and model. In Gradio, it lives as a
    module-level ``_NODE`` singleton (re-created only when model-affecting
    config fields change). In CLI, it is created in ``main()`` and used once.

    The node relies on the **caller** setting the Rerun recording context
    (``with recording:`` in Gradio, global init in CLI). All ``rr.log()``
    calls inside ``__call__`` are gated behind ``config.verbose``.

    Args:
        config: Node configuration.
        parent_log_path: Root Rerun log path (e.g. ``Path("world")``).
    """

    def __init__(self, config: <Name>Config, parent_log_path: Path) -> None:
        self.config = config
        self.parent_log_path = parent_log_path
        # Load model/predictor — done once, reused across calls.
        # For lightweight nodes (no model), this can be a no-op.
        self.predictor = create_<name>_predictor(config)

    def __call__(
        self,
        *,
        rgb: UInt8[ndarray, "H W 3"],
    ) -> <Name>Result:
        """Run <name> prediction.

        Args:
            rgb: Input RGB image.

        Returns:
            <Name>Result with prediction outputs.
        """
        # Call the model
        raw_output = self.predictor(rgb)
        depth_map: Float32[ndarray, "H W"] = raw_output.depth

        # Intermediate logging (gated behind verbose)
        if self.config.verbose:
            rr.log(
                f"{self.parent_log_path}/camera/pinhole/depth",
                rr.DepthImage(depth_map, meter=1),
            )

        return <Name>Result(depth_map=depth_map)


# ---------------------------------------------------------------------------
# Result logging helper (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def log_<name>_result(
    result: <Name>Result,
    parent_log_path: Path,
) -> None:
    """Log <name> prediction results to the active Rerun recording.

    Called by both CLI ``main()`` and the Gradio streaming callback.
    Relies on the caller having set the recording context.

    This pairs with ``create_<name>_blueprint()`` — one builds the layout,
    the other populates it with data.

    Args:
        result: Prediction output from ``<Name>Node``.
        parent_log_path: Root log path (e.g. ``Path("world")``).
    """
    rr.log(
        f"{parent_log_path}/camera/pinhole/depth",
        rr.DepthImage(result.depth_map, meter=1),
        static=True,
    )
    # ... log other result fields (images, point clouds, etc.) ...


# ---------------------------------------------------------------------------
# Blueprint builder (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def create_<name>_blueprint(
    parent_log_path: Path,
    # Add parameters that affect layout (e.g. num_images for multi-view)
) -> rrb.ContainerLike:
    """Create a Rerun blueprint for <name> visualization.

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).

    Returns:
        Rerun blueprint layout.
    """
    return rrb.Horizontal(
        rrb.Spatial3DView(
            origin=f"{parent_log_path}",
            contents=["+ $origin/**"],
        ),
        rrb.Vertical(
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
        ),
        column_shares=[3, 1],
    )


# ---------------------------------------------------------------------------
# CLI config (wraps node config + RerunTyroConfig + input paths)
# ---------------------------------------------------------------------------
@dataclass
class <Name>CLIConfig:
    """CLI configuration for <name> prediction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration (--save, --connect, --spawn)."""
    image_path: Path = Path("data/examples/<name>/example1.jpg")
    """Path to input image (or image_dir for multi-image nodes)."""
    config: <Name>Config = field(default_factory=lambda: <Name>Config(verbose=True))
    """<Name> prediction configuration."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(config: <Name>CLIConfig) -> None:
    """CLI entry point for <name> prediction with Rerun visualization.

    Creates the node, runs the pipeline, and logs results. The node handles
    intermediate logging; ``log_<name>_result`` handles final output logging.
    """
    parent_log_path: Path = Path("world")

    # 1. Load input data
    bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(config.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image {config.image_path}")
    rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_<name>_blueprint(parent_log_path=parent_log_path),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # 3. Log input (one-shot asset — caller's responsibility)
    rr.log(
        f"{parent_log_path}/camera/pinhole/image",
        rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
        static=True,
    )

    # 4. Create node and run (node handles intermediate logging)
    node: <Name>Node = <Name>Node(config=config.config, parent_log_path=parent_log_path)
    result: <Name>Result = node(rgb=rgb)

    # 5. Log final results (shared with Gradio)
    log_<name>_result(result, parent_log_path=parent_log_path)
```

## Key Patterns

### Config Dataclass Conventions

- Use `@dataclass` (not attrs, not pydantic)
- Every field has a per-field docstring on the line below:
  ```python
  keep_top_percent: KeepTopPercent = 30.0
  """Fraction of high-confidence pixels retained after filtering."""
  ```
- Use jaxtyping + beartype `Annotated` for constrained types:
  ```python
  from beartype.vale import Is
  KeepTopPercent = Annotated[int | float, Is[lambda percent: 1 <= percent <= 100]]
  ```
- Use `Literal` for string enums: `Literal["crop", "pad"]`
- Default values match the most common use case
- `device` field uses `Literal["cuda", "cpu"]` with default `"cuda"`
- `verbose` field controls intermediate `rr.log()` calls in the node

### Separation of Concerns

The API layer has **3 reusable pieces** for Rerun integration:

1. **`<Name>Node.__call__()`** — intermediate logging during computation (gated by `config.verbose`)
2. **`log_<name>_result()`** — final output logging (called once after computation)
3. **`create_<name>_blueprint()`** — Rerun viewer layout

The **caller** (CLI `main()` or Gradio callback) is responsible for:
- Setting the recording context (`with recording:` or global init)
- Logging one-shot input assets (video files, input images)
- Sending the blueprint
- Calling `log_<name>_result()` after the node returns

### Multi-Image Nodes

For nodes that take multiple images (like VGGT geometry):
- Input: `list[UInt8[ndarray, "H W 3"]]` (not a batch tensor)
- CLI config has `image_dir: Path` instead of `image_path: Path`
- Use `load_rgb_images()` from shared utilities
- Log per-camera results under `parent_log_path / f"camera_{i}" / "pinhole" / ...`

### Single-Image Nodes

For nodes that process one image at a time (like metric depth, segmentation):
- Input: `UInt8[ndarray, "H W 3"]`
- The Gradio UI or orchestrator loops over images if needed
- Log under `parent_log_path / "camera" / "pinhole" / ...`

## Real Examples

- **Class-based (primary)**: `pysfm/apis/video_to_image.py` — `VideoToImageNode`, `VideoToImageConfig`, `VideoToImageResult`, `create_video_to_image_blueprint()`
- **Multi-image (older, function-based)**: `monopriors/apis/multiview_geometry.py` — `MultiviewGeometryConfig`, `MultiviewGeometryResult`, `run_multiview_geometry()`
- **Single-image (older, function-based)**: `monopriors/apis/metric_depth.py` — `MetricDepthNodeConfig`, `run_metric_depth()`, `create_metric_predictor()`
