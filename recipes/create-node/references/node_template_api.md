# API Layer Template

This is the annotated template for a node's API module (`<module>/apis/<name>.py`). The API layer contains pure computation, typed configs/results, and the CLI entry point with Rerun logging.

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
    """Emit per-camera/per-frame detail logging when True."""


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
# Core pipeline function (NO Rerun, NO file I/O)
# ---------------------------------------------------------------------------
def run_<name>(
    *,
    rgb: UInt8[ndarray, "H W 3"],
    predictor: <Predictor>,
    config: <Name>Config,
) -> <Name>Result:
    """Run <name> prediction.

    This function is pure computation. It does NOT call ``rr.log()`` and does
    NOT read files. The caller (CLI or Gradio) handles visualization and I/O.

    Args:
        rgb: Input RGB image.
        predictor: Pre-initialised predictor (model already loaded).
        config: Prediction configuration.

    Returns:
        <Name>Result with prediction outputs.
    """
    # Call the model
    raw_output = predictor(rgb)
    # Post-process
    ...
    return <Name>Result(depth_map=...)


# ---------------------------------------------------------------------------
# Blueprint builder (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def create_<name>_blueprint(
    parent_log_path: Path,
    # Add parameters that affect layout (e.g. num_images for multi-view)
) -> "rrb.ContainerLike":
    """Create a Rerun blueprint for <name> visualization.

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).

    Returns:
        Rerun blueprint layout.
    """
    import rerun.blueprint as rrb

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
    <name>_config: <Name>Config = field(default_factory=lambda: <Name>Config(verbose=True))
    """<Name> prediction configuration."""


# ---------------------------------------------------------------------------
# CLI entry point (Rerun logging happens HERE, not in run_<name>)
# ---------------------------------------------------------------------------
def main(config: <Name>CLIConfig) -> None:
    """CLI entry point for <name> prediction with Rerun visualization.

    Loads input data, initialises the predictor, runs the pipeline,
    sets up the Rerun blueprint, and logs all results.
    """
    # Lazy imports for heavy deps — keeps import time fast
    import cv2
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    from simplecv.rerun_log_utils import log_pinhole

    parent_log_path: Path = Path("world")

    # 1. Load input data
    bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(config.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image {config.image_path}")
    rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Init predictor
    predictor = create_<name>_predictor(config.<name>_config)

    # 3. Run pipeline (pure computation, no Rerun)
    result: <Name>Result = run_<name>(
        rgb=rgb,
        predictor=predictor,
        config=config.<name>_config,
    )

    # 4. Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_<name>_blueprint(parent_log_path=parent_log_path),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # 5. Log results to Rerun
    rr.log(
        f"{parent_log_path}/camera/pinhole/image",
        rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
        static=True,
    )
    rr.log(
        f"{parent_log_path}/camera/pinhole/depth",
        rr.DepthImage(result.depth_map, meter=1),
        static=True,
    )
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

### Separation of Concerns

- `run_<name>()` is pure: arrays in, result out. No Rerun, no file I/O.
- `main()` handles: file loading, model init, Rerun setup, logging. These are CLI-specific concerns.
- The Gradio UI has its own streaming callback that handles the same Rerun setup differently (binary stream).
- Blueprint builders are separate functions, importable by both CLI and Gradio.

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

- **Multi-image**: `monopriors/apis/multiview_geometry.py` — `MultiviewGeometryConfig`, `MultiviewGeometryResult`, `run_multiview_geometry()`
- **Single-image**: `monopriors/apis/metric_depth.py` — `MetricDepthNodeConfig`, `run_metric_depth()`, `create_metric_predictor()`
