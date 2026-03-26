# Gradio UI Template

This is the annotated template for a node's Gradio UI module (`<module>/gradio_ui/nodes/<name>_ui.py`). Node UIs live under `gradio_ui/nodes/`, while composite app UIs live under `gradio_ui/apps/`. The UI layer translates between Gradio widgets and the API layer, manages the node singleton, and streams Rerun data to the viewer.

## Structure

```python
"""Gradio UI for <name> prediction.

Provides an interactive web interface for running <description>.
The left panel holds inputs, a run button, and a config accordion;
the right panel streams results into an embedded Rerun viewer.

The node is loaded once at module import and reused across runs.
Config changes that affect the model trigger lazy re-initialisation.
"""

import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray

from <module>.apis.<name> import (
    <Name>Config,
    <Name>Node,
    <Name>Result,
    create_<name>_blueprint,
    log_<name>_result,
)


# ---------------------------------------------------------------------------
# Example data path (REQUIRED for every node)
# ---------------------------------------------------------------------------
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "examples" / "<name>"
"""Path to bundled example inputs.

Note: ``.parents[3]`` navigates from ``<module>/gradio_ui/nodes/<name>_ui.py``
up to the package root.
"""

# IMPORTANT: register example data with Gradio's static file server
gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

PARENT_LOG_PATH: Final[Path] = Path("world")


# ---------------------------------------------------------------------------
# Module-level state (loaded once at import, reused across runs)
# ---------------------------------------------------------------------------
_CONFIG: <Name>Config = <Name>Config(
    device="cuda",
    verbose=True,
)
"""Module-level config, kept in sync with UI widgets."""

_NODE: <Name>Node = <Name>Node(
    config=_CONFIG,
    parent_log_path=PARENT_LOG_PATH,
)
"""Module-level node singleton. Re-created only when model-affecting
config fields change (see ``_sync_config``)."""


# ---------------------------------------------------------------------------
# _sync_config: widgets → config + node singletons
# ---------------------------------------------------------------------------
def _sync_config(
    # One parameter per Config field that has a widget
    param_a: int | float,
    param_b: str,
    verbose: bool,
) -> None:
    """Sync UI widget values into the module-level config and node singleton.

    Only re-creates the node when a config change requires loading new
    model weights or changing model architecture. Runtime-only fields
    (thresholds, verbosity) are patched in-place.

    Args:
        param_a: Description of param_a.
        param_b: Description of param_b.
        verbose: Whether to log intermediate detail.
    """
    global _CONFIG, _NODE

    # Determine if we need to re-init the node (model-affecting field changed)
    needs_reinit: bool = param_b != _CONFIG.param_b

    _CONFIG = <Name>Config(
        param_a=param_a,
        param_b=param_b,
        device="cuda",
        verbose=verbose,
    )

    if needs_reinit:
        _NODE = <Name>Node(
            config=_CONFIG,
            parent_log_path=PARENT_LOG_PATH,
        )
    else:
        _NODE.config = _CONFIG  # patch runtime-only fields in-place


# ---------------------------------------------------------------------------
# _parse_and_load_<inputs>: Gradio files → domain data
# ---------------------------------------------------------------------------
def _parse_and_load_images(
    img_files: str | list[str],
) -> list[UInt8[ndarray, "H W 3"]]:
    """Parse Gradio file uploads and load them as RGB arrays.

    This is a separate ``.then()`` step so the node never
    touches file paths — it receives pre-loaded arrays from ``gr.State``.

    Args:
        img_files: Single path or list of paths from ``gr.File``.

    Returns:
        Sorted list of RGB images as uint8 numpy arrays.
    """
    from <module>.apis.<shared> import load_rgb_images

    if isinstance(img_files, str):
        img_paths: list[Path] = [Path(img_files)]
    elif isinstance(img_files, list):
        img_paths = [Path(f) for f in img_files]
    else:
        raise gr.Error("Invalid input for images. Please select image files.")

    if not img_paths:
        raise gr.Error("Please select at least one image before running.")

    img_paths.sort()
    rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(img_paths)
    return rgb_list


# ---------------------------------------------------------------------------
# Streaming callback: run node → Rerun binary stream
# ---------------------------------------------------------------------------
@rr.recording_stream_generator_ctx
def <name>_fn(
    recording_id: uuid.UUID,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs <name> prediction.

    Creates a scoped Rerun recording, sends the blueprint, runs the
    node (which handles intermediate logging when verbose), and calls
    ``log_<name>_result()`` for final outputs. Yields binary stream
    bytes so the Rerun viewer updates incrementally.

    The ``@rr.recording_stream_generator_ctx`` decorator suspends and
    restores the Rerun ContextVar around each ``yield``, making it safe
    to yield inside ``with recording:``.

    Args:
        recording_id: Session-scoped recording identifier (fresh UUID per run).
        rgb_list: Pre-loaded RGB images from ``_parse_and_load_images``.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id="<name>", recording_id=recording_id
    )
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        # 1. Send blueprint
        blueprint: rrb.Blueprint = rrb.Blueprint(
            create_<name>_blueprint(parent_log_path=PARENT_LOG_PATH, num_images=len(rgb_list)),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)
        rr.log("world", rr.ViewCoordinates.RFU, static=True)

        # 2. Early yield — viewer shows blueprint while node works
        yield stream.read(), "Running <name>…"

        # 3. Run node (intermediate logging handled by node when verbose)
        result: <Name>Result = _NODE(rgb_list=rgb_list)

        # 4. Log final results (shared with CLI)
        log_<name>_result(result, parent_log_path=PARENT_LOG_PATH)

    yield stream.read(), "<Name> complete"


# ---------------------------------------------------------------------------
# Tab switching helpers
# ---------------------------------------------------------------------------
def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


# ---------------------------------------------------------------------------
# main() → gr.Blocks
# ---------------------------------------------------------------------------
def main() -> gr.Blocks:
    """Build and return the <name> Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (file upload, run
          button, config accordion) and Outputs (status); example sets below.
        - **Right column** (scale=5): Embedded Rerun viewer.

    Returns:
        The assembled ``gr.Blocks`` instance ready for ``.queue().launch()``.
    """
    rr_viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "collapsed",
            "selection": "collapsed",
        },
        height=800,
    )

    with gr.Blocks() as demo:
        # Session state
        recording_id = gr.State(uuid.uuid4())
        rgb_list_state = gr.State([])

        with gr.Row():
            # ---- Left column: controls ----
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        input_imgs = gr.File(
                            label="Input Images",
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg"],
                        )
                        run_btn = gr.Button("Run <Name>")

                        with gr.Accordion("Config", open=False):
                            # One widget per Config field
                            param_a_slider = gr.Slider(
                                label="Param A",
                                minimum=1.0,
                                maximum=100.0,
                                step=1.0,
                                value=_CONFIG.param_a,
                            )
                            param_b_radio = gr.Radio(
                                label="Param B",
                                choices=["option1", "option2"],
                                value=_CONFIG.param_b,
                            )
                            verbose_checkbox = gr.Checkbox(
                                label="Verbose (intermediate detail logging)",
                                value=_CONFIG.verbose,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                # ---- Examples (below tabs, still in left column) ----
                example_set_1: list[str] = sorted(
                    str(p) for p in (EXAMPLE_DATA_DIR / "set_1").glob("*.jpg")
                )
                example_set_2: list[str] = sorted(
                    str(p) for p in (EXAMPLE_DATA_DIR / "set_2").glob("*.jpg")
                )
                gr.Examples(
                    examples=[
                        [example_set_1],
                        [example_set_2],
                    ],
                    inputs=[input_imgs],
                    cache_examples=False,
                )

            # ---- Right column: Rerun viewer ----
            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when examples populate the input
        input_imgs.change(
            fn=_switch_to_inputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        )

        # Click chain: each .then() has ONE job
        run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(  # Fresh recording ID so each run gets its own Rerun session
            fn=lambda: uuid.uuid4(),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        ).then(  # Sync the node singleton with current UI config widgets
            _sync_config,
            inputs=[
                param_a_slider,
                param_b_radio,
                verbose_checkbox,
            ],
        ).then(  # Parse Gradio file uploads into domain data
            _parse_and_load_images,
            inputs=[input_imgs],
            outputs=[rgb_list_state],
        ).then(  # Run node and stream results to the Rerun viewer
            <name>_fn,
            inputs=[recording_id, rgb_list_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
```

## Key Patterns

### Module-Level `_CONFIG` + `_NODE` Singletons

Models are expensive to load (VGGT ~7s, SAM3 ~3s). The node is loaded once at module import:

```python
_CONFIG: <Name>Config = <Name>Config(device="cuda", verbose=True)
_NODE: <Name>Node = <Name>Node(config=_CONFIG, parent_log_path=PARENT_LOG_PATH)
```

`_sync_config()` decides whether to re-create the node or just patch config:

```python
needs_reinit: bool = new_mode != _CONFIG.preprocessing_mode
if needs_reinit:
    _NODE = <Name>Node(config=_CONFIG, parent_log_path=PARENT_LOG_PATH)  # Full reload
else:
    _NODE.config = _CONFIG  # Patch runtime-only fields
```

For lightweight nodes (no model), `_NODE` creation is cheap, but the pattern remains the same for consistency.

### The `@rr.recording_stream_generator_ctx` Decorator

**Required** on all Gradio streaming callbacks that use Rerun. Without it, yielding inside `with recording:` crashes because Gradio resumes generators in different async contexts, invalidating the Rerun ContextVar token.

The decorator suspends the ContextVar before each `yield` and restores it when the generator resumes. This enables true incremental streaming — the viewer updates progressively as computation proceeds.

```python
@rr.recording_stream_generator_ctx  # <-- required
def <name>_fn(recording_id, ...) -> Generator[...]:
    recording = rr.RecordingStream(...)
    stream = recording.binary_stream()
    with recording:
        yield stream.read(), "step 1..."   # safe
        yield stream.read(), "step 2..."   # safe
```

Alternative: `@rr.thread_local_stream("app")` creates its own RecordingStream with a random ID. Use `recording_stream_generator_ctx` when you need to control the `recording_id` (Gradio sessions).

### The Binary Stream Pattern

1. Create a `rr.RecordingStream` scoped to this session's UUID
2. Get a `binary_stream()` from it
3. Run the node inside `with recording:` — all `rr.log()` calls go to this stream
4. `yield stream.read()` sends the accumulated bytes to the `Rerun(streaming=True)` component

### Temp Dir Cleanup

When nodes write to the filesystem (e.g. frame extraction), use `try/finally`:

```python
tmp_dir: Path = Path(tempfile.mkdtemp(prefix="<name>_"))
try:
    with recording:
        result = _NODE(output_dir=tmp_dir, ...)
        # ...
    yield stream.read(), "Done"
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
```

### EXAMPLE_DATA_DIR Convention

```python
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "examples" / "<name>"
```

The `.parents[3]` navigates from `<module>/gradio_ui/nodes/<name>_ui.py` up to the package root. Adjust the index based on your module depth.

### Single-Image vs Multi-Image Inputs

**Multi-image** (like VGGT geometry):
- `gr.File(file_count="multiple")`
- `_parse_and_load_images()` returns `list[UInt8[ndarray, "H W 3"]]`
- `gr.Examples` passes lists of file paths

**Single-image** (like metric depth, segmentation):
- `gr.Image(type="numpy")` or `gr.File(file_count="single")`
- Parse step may be simpler or unnecessary with `gr.Image(type="numpy")`
- Examples are single files

## Real Examples

- **Class-based (primary)**: `pysfm/gradio_ui/nodes/video_to_image_ui.py` — `@rr.recording_stream_generator_ctx`, `_CONFIG` + node creation in callback, `try/finally` cleanup
- **Multi-image (older)**: `monopriors/gradio_ui/nodes/multiview_geometry_ui.py` — `_CONFIG` + `_PREDICTOR` pattern
- **Single-image with config accordion (older)**: `monopriors/gradio_ui/nodes/metric_depth_ui.py`
