# Rerun Integration Patterns

Rerun is non-negotiable at every layer of a node. This reference covers the node class pattern, the two execution modes (CLI and Gradio), and the `@rr.recording_stream_generator_ctx` decorator for safe incremental streaming.

## Core Principle: Node Owns Intermediate Logging

The API layer has **3 reusable pieces** for Rerun integration:

1. **`<Name>Node.__call__()`** — intermediate `rr.log()` calls during computation, gated by `config.verbose`. Runs inside the caller's recording context.
2. **`log_<name>_result()`** — logs the final Result dataclass to Rerun. Called by both CLI and Gradio — eliminates duplicate logging code.
3. **`create_<name>_blueprint()`** — Rerun viewer layout. Shared between CLI and Gradio.

The **caller** (CLI `main()` or Gradio callback) is responsible for:
- Setting the recording context (`with recording:` in Gradio, global init in CLI)
- Logging one-shot input assets (video files, source images)
- Sending the blueprint
- Calling `log_<name>_result()` after the node returns

## Pattern 1: CLI with RerunTyroConfig

`RerunTyroConfig` from `simplecv.rerun_log_utils` auto-configures Rerun from CLI flags:

```python
from simplecv.rerun_log_utils import RerunTyroConfig

@dataclass
class <Name>CLIConfig:
    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/<name>/example.jpg")
    config: <Name>Config = field(default_factory=lambda: <Name>Config(verbose=True))
```

CLI flags it provides:
- `--rr-config.spawn` — spawn a Rerun viewer process and stream to it
- `--rr-config.connect` — connect to an already-running viewer
- `--rr-config.save <path.rrd>` — save to an RRD file

In `main()`, create the node, run it, and log results:

```python
def main(config: <Name>CLIConfig) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    parent_log_path: Path = Path("world")

    # Setup Rerun
    blueprint = rrb.Blueprint(create_<name>_blueprint(parent_log_path), collapse_panels=True)
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # Log input assets (caller's responsibility)
    rr.log(f"{parent_log_path}/camera/pinhole/image", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(), static=True)

    # Create node and run (intermediate logging handled by node)
    node = <Name>Node(config=config.config, parent_log_path=parent_log_path)
    result = node(rgb=rgb)

    # Log final results (shared with Gradio)
    log_<name>_result(result, parent_log_path=parent_log_path)
```

## Pattern 2: Gradio with Binary Stream + `@rr.recording_stream_generator_ctx`

The Gradio streaming callback creates a scoped recording and binary stream, decorated with `@rr.recording_stream_generator_ctx` for safe incremental streaming:

```python
@rr.recording_stream_generator_ctx
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

        # Early yield — viewer shows blueprint while node works
        yield stream.read(), "Running..."

        # Run node (intermediate logging handled by node when verbose)
        result: <Name>Result = _NODE(rgb_list=rgb_list)

        # Log final results (shared with CLI)
        log_<name>_result(result, parent_log_path=PARENT_LOG_PATH)

    # 3. Final yield with all accumulated bytes
    yield stream.read(), "<Name> complete"
```

The `Rerun(streaming=True)` Gradio component receives these bytes and renders them.

## `@rr.recording_stream_generator_ctx` Decorator

### The Problem

Gradio resumes generator callbacks in different async contexts (via `anyio.to_thread.run_sync`). Rerun's `RecordingStream` context manager stores a `ContextVar` token on `__enter__` and validates it on `__exit__`. When you `yield` inside `with recording:`, Gradio resumes the generator in a new context where that token is invalid — `__exit__` raises `ValueError`.

### The Solution

`@rr.recording_stream_generator_ctx` (available since rerun-sdk 0.30.2) wraps generator functions to automatically call `recording.__exit__()` before each `yield` and `recording.__enter__()` when the generator resumes. This makes it safe to yield inside `with recording:`.

```python
@rr.recording_stream_generator_ctx  # <-- required for all Gradio Rerun callbacks
def my_streaming_callback(recording_id, ...) -> Generator[...]:
    recording = rr.RecordingStream(application_id="app", recording_id=recording_id)
    stream = recording.binary_stream()

    with recording:
        rr.send_blueprint(...)
        yield stream.read(), "step 1..."   # safe — decorator handles ContextVar

        # more work, rr.log() calls go to the stream
        yield stream.read(), "step 2..."   # safe
```

### Alternative: `@rr.thread_local_stream`

`@rr.thread_local_stream("app_name")` does the same thing but creates its own `RecordingStream` with a random ID. Use `recording_stream_generator_ctx` when you need to control the `recording_id` (e.g. from Gradio `gr.State`).

## Blueprint Builders

Every node defines a blueprint builder function. This lives in the API module and is imported by both CLI and Gradio.

```python
def create_<name>_blueprint(
    parent_log_path: Path,
    num_images: int = 1,
) -> rrb.ContainerLike:
    """Create Rerun blueprint for <name> visualization."""
    return rrb.Horizontal(
        rrb.Spatial3DView(
            origin=f"{parent_log_path}",
            contents=[
                "+ $origin/**",
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

- **Class-based node (primary)**: `pysfm/apis/video_to_image.py` — `VideoToImageNode`, `create_video_to_image_blueprint()`
- **CLI + Rerun**: `pysfm/apis/video_to_image.py` (`main()`)
- **Gradio + decorator + binary stream**: `pysfm/gradio_ui/nodes/video_to_image_ui.py` (`video_to_image_fn()`)
- **Gradio + binary stream (older)**: `monopriors/gradio_ui/nodes/multiview_geometry_ui.py` (`multiview_geometry_fn()`)
- **Blueprint builder**: `pysfm/apis/video_to_image.py` (`create_video_to_image_blueprint()`)
