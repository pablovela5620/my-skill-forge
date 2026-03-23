# Rerun Integration Patterns

Rerun is non-negotiable at every layer of a node. This reference covers the two execution modes (CLI and Gradio) and shared conventions.

## Core Principle: Thread-Local Recording

The pipeline function (`run_<name>()`) itself does NOT call `rr.log()`. It returns a result dataclass. The Rerun logging happens in the caller:

- **CLI**: `main()` calls `rr.log()` using the global recording (controlled by `RerunTyroConfig`)
- **Gradio**: the streaming callback calls `rr.log()` inside `with recording:` which scopes all calls to a binary stream

This means the same pipeline function works in both contexts without any Rerun awareness.

## Pattern 1: CLI with RerunTyroConfig

`RerunTyroConfig` from `simplecv.rerun_log_utils` auto-configures Rerun from CLI flags:

```python
from simplecv.rerun_log_utils import RerunTyroConfig

@dataclass
class <Name>CLIConfig:
    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/<name>/example.jpg")
    <name>_config: <Name>Config = field(default_factory=<Name>Config)
```

CLI flags it provides:
- `--rr-config.spawn` — spawn a Rerun viewer process and stream to it
- `--rr-config.connect` — connect to an already-running viewer
- `--rr-config.save <path.rrd>` — save to an RRD file

In `main()`, after running the pipeline, log results using global `rr.log()`:

```python
def main(config: <Name>CLIConfig) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    # ... load data, run pipeline ...

    # Setup Rerun
    blueprint = rrb.Blueprint(create_<name>_blueprint(...), collapse_panels=True)
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Log results
    rr.log("world/camera/pinhole/image", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(), static=True)
    rr.log("world/camera/pinhole/depth", rr.DepthImage(result.depth_map, meter=1), static=True)
```

## Pattern 2: Gradio with Binary Stream

The Gradio streaming callback creates a scoped recording and binary stream:

```python
def <name>_fn(
    recording_id: uuid.UUID,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Generator[tuple[bytes | None, str], None, None]:
    # 1. Create a recording stream scoped to this session
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id="<name>", recording_id=recording_id
    )
    stream: rr.BinaryStream = recording.binary_stream()

    # 2. Run everything inside the recording context
    with recording:
        # Blueprint
        rr.send_blueprint(rrb.Blueprint(..., collapse_panels=True))
        rr.log("world", rr.ViewCoordinates.RFU, static=True)

        # Run pipeline and log results
        result = run_<name>(rgb_list=rgb_list, predictor=_PREDICTOR, config=_CONFIG)

        # Log to Rerun (goes to binary stream, not global recording)
        for i, pred in enumerate(result.predictions):
            rr.log(f"world/camera_{i}/pinhole/depth", rr.DepthImage(...), static=True)

    # 3. Yield accumulated bytes to the Rerun viewer component
    yield stream.read(), "<Name> complete"
```

The `Rerun(streaming=True)` Gradio component receives these bytes and renders them.

### Incremental Streaming

For long-running pipelines, yield intermediate results:

```python
with recording:
    rr.send_blueprint(...)
    for i, image in enumerate(rgb_list):
        result_i = process_single(image, predictor=_PREDICTOR)
        rr.log(f"world/camera_{i}/pinhole/depth", rr.DepthImage(result_i.depth), static=True)
        yield stream.read(), f"Processed {i+1}/{len(rgb_list)}"

yield stream.read(), "Complete"
```

## Blueprint Builders

Every node defines a blueprint builder function. This can live in the API module (if shared) or the UI module (if Gradio-specific).

```python
def create_<name>_blueprint(
    parent_log_path: Path,
    num_images: int = 1,
) -> rrb.ContainerLike:
    """Create Rerun blueprint for <name> visualization."""
    import rerun.blueprint as rrb

    return rrb.Horizontal(
        rrb.Spatial3DView(
            origin=f"{parent_log_path}",
            contents=[
                "+ $origin/**",
                # Exclude raw depth from 3D (noisy)
                f"- {parent_log_path}/camera/pinhole/depth",
            ],
        ),
        rrb.Vertical(
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
        ),
        column_shares=[3, 1],
    )
```

### Multi-Camera Blueprints

For nodes producing per-camera results, group cameras into tabs:

```python
def create_multiview_blueprint(parent_log_path: Path, num_images: int) -> rrb.ContainerLike:
    tabs = []
    for i in range(num_images):
        tab = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera_{i}/pinhole/image"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera_{i}/pinhole/depth"),
        )
        tabs.append(rrb.Vertical(contents=[tab], name=f"Camera {i+1}"))

    view_3d = rrb.Spatial3DView(origin=f"{parent_log_path}", contents=["+ $origin/**"])
    view_2d = rrb.Tabs(contents=tabs)
    return rrb.Horizontal(contents=[view_3d, view_2d], column_shares=[3, 2])
```

## Log Path Conventions

All nodes log under `PARENT_LOG_PATH = Path("world")`:

```
world/                                  ViewCoordinates (RFU or RDF)
world/camera/pinhole/                   PinholeProjection (single-image nodes)
world/camera/pinhole/image              rr.Image (RGB, compressed)
world/camera/pinhole/depth              rr.DepthImage (meter=1)
world/camera/pinhole/confidence         rr.Image (grayscale confidence)
world/camera_{i}/pinhole/               per-camera (multi-image nodes)
world/point_cloud                       rr.Points3D
world/mesh                              rr.Mesh3D
```

### ViewCoordinates

- `rr.ViewCoordinates.RFU` (Right-Forward-Up) — for multi-camera scenes, reconstructions
- `rr.ViewCoordinates.RDF` (Right-Down-Forward) — for single-camera / image-centric views

### Static vs Timeline

- Use `static=True` for most logged data (results don't change over time)
- Use `rr.set_time(TIMELINE, duration=0)` when logging sequential steps in a pipeline

### Compression

Always compress RGB images to reduce stream/file size:
```python
rr.log("path/image", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(), static=True)
```

Depth images and confidence maps are NOT compressed (they're already compact or need precision).

## Rerun Viewer Component (Gradio)

```python
from gradio_rerun import Rerun

rr_viewer = Rerun(
    streaming=True,
    panel_states={
        "time": "collapsed",
        "blueprint": "collapsed",
        "selection": "collapsed",
    },
    height=800,
)
```

- `streaming=True` enables incremental data reception
- `panel_states` collapses side panels for a clean default view
- `height=800` gives adequate vertical space
- Place in the right column with `scale=5`

## Real Examples

- **CLI + Rerun**: `monopriors/apis/multiview_geometry.py` lines 158-235 (`main()`)
- **Gradio + binary stream**: `monopriors/gradio_ui/nodes/multiview_geometry_ui.py` (`multiview_geometry_fn()`)
- **Blueprint builder**: `monopriors/gradio_ui/nodes/multiview_geometry_ui.py` (`create_multiview_blueprint()`)
