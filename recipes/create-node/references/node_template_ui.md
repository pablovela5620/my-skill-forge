# Gradio UI Template

This is the annotated template for a node's Gradio UI module (`<module>/gradio_ui/nodes/<name>_ui.py`). Node UIs live under `gradio_ui/nodes/`, while composite app UIs live under `gradio_ui/apps/`. The UI layer translates between Gradio widgets and the API layer, manages model singletons, and streams Rerun data to the viewer.

## Structure

```python
"""Gradio UI for <name> prediction.

Provides an interactive web interface for running <description>.
The left panel holds inputs, a run button, and a config accordion;
the right panel streams results into an embedded Rerun viewer.

The model is loaded once at module import and reused across runs.
Config changes that affect the model trigger lazy re-initialisation.
"""

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
    <Name>Result,
    run_<name>,
    create_<name>_blueprint,
)
from <module>.models.<model> import <Predictor>


# ---------------------------------------------------------------------------
# Example data path (REQUIRED for every node)
# ---------------------------------------------------------------------------
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "examples" / "<name>"
"""Path to bundled example inputs.

Note: .parents[3] navigates from <module>/gradio_ui/nodes/<name>_ui.py up to the package root.
"""

# IMPORTANT: register example data with Gradio's static file server
gr.set_static_paths([str(EXAMPLE_DATA_DIR)])


# ---------------------------------------------------------------------------
# Module-level state (loaded once at import, reused across runs)
# ---------------------------------------------------------------------------
_CONFIG: <Name>Config = <Name>Config(
    device="cuda",
    verbose=True,
)
"""Module-level config, kept in sync with UI widgets."""

_PREDICTOR: <Predictor> = <Predictor>(
    device=_CONFIG.device,
    # ... other init params from config
)
"""Module-level model singleton. Re-created only when config changes require it."""


# ---------------------------------------------------------------------------
# _sync_config: widgets → config singleton
# ---------------------------------------------------------------------------
def _sync_config(
    # One parameter per Config field that has a widget
    param_a: int | float,
    param_b: str,
    verbose: bool,
) -> None:
    """Sync UI widget values into the module-level config and predictor singleton.

    Only re-creates the predictor when a config change requires loading new
    model weights or changing model architecture. Runtime-only fields
    (thresholds, verbosity) are patched in-place.

    Args:
        param_a: Description of param_a.
        param_b: Description of param_b.
        verbose: Whether to log per-frame detail.
    """
    global _CONFIG, _PREDICTOR

    # Determine if we need to re-init the predictor
    needs_reinit: bool = param_b != _CONFIG.param_b

    _CONFIG = <Name>Config(
        param_a=param_a,
        param_b=param_b,
        device="cuda",
        verbose=verbose,
    )

    if needs_reinit:
        _PREDICTOR = <Predictor>(
            device=_CONFIG.device,
            # ... other init params
        )


# ---------------------------------------------------------------------------
# _parse_and_load_<inputs>: Gradio files → domain data
# ---------------------------------------------------------------------------
def _parse_and_load_images(
    img_files: str | list[str],
) -> list[UInt8[ndarray, "H W 3"]]:
    """Parse Gradio file uploads and load them as RGB arrays.

    This is a separate ``.then()`` step so the pipeline function never
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
# Streaming callback: run pipeline → Rerun binary stream
# ---------------------------------------------------------------------------
def <name>_fn(
    recording_id: uuid.UUID,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs <name> prediction.

    Creates a scoped Rerun recording, runs the pipeline inside it,
    and yields binary stream bytes to the Rerun viewer component.

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
            create_<name>_blueprint(parent_log_path=Path("world"), num_images=len(rgb_list)),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)
        rr.log("world", rr.ViewCoordinates.RFU, static=True)

        # 2. Run pipeline (pure computation)
        result: <Name>Result = run_<name>(
            rgb_list=rgb_list,
            predictor=_PREDICTOR,
            config=_CONFIG,
        )

        # 3. Log results to Rerun
        # ... rr.log() calls for each output
        # All rr.log() calls go to the scoped recording → binary stream

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
                                label="Verbose (per-frame detail logging)",
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
        ).then(  # Sync the predictor singleton with current UI config widgets
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
        ).then(  # Run pipeline and stream results to the Rerun viewer
            <name>_fn,
            inputs=[recording_id, rgb_list_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
```

## Key Patterns

### Module-Level Singletons

Models are expensive to load (VGGT ~7s, SAM3 ~3s). Load once at module import:

```python
_PREDICTOR: VGGTPredictor = VGGTPredictor(device="cuda", preprocessing_mode="pad")
```

Re-create ONLY when a config change requires new model weights. `_sync_config()` checks what changed and decides:

```python
needs_reinit: bool = new_mode != _CONFIG.preprocessing_mode
if needs_reinit:
    _PREDICTOR = VGGTPredictor(...)  # Reload
else:
    _PREDICTOR.some_param = new_value  # Patch in-place
```

### The Binary Stream Pattern

This is how Rerun data flows from the pipeline to the Gradio viewer:

1. Create a `rr.RecordingStream` scoped to this session's UUID
2. Get a `binary_stream()` from it
3. Run the pipeline inside `with recording:` — all `rr.log()` calls go to this stream
4. `yield stream.read()` sends the accumulated bytes to the `Rerun(streaming=True)` component

The pipeline function (`run_<name>()`) doesn't know about streaming. It just returns results. The Rerun logging happens in the streaming callback.

### EXAMPLE_DATA_DIR Convention

```python
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "<name>"
```

The `.parents[3]` navigates from `<module>/gradio_ui/nodes/<name>_ui.py` up to the package root. Adjust the index based on your module depth (the extra level is for the `nodes/` subdirectory).

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

- **Multi-image**: `monopriors/gradio_ui/nodes/multiview_geometry_ui.py`
- **Single-image with Rerun**: `sam3_rerun/gradio_ui/nodes/sam3_rerun_ui.py`
- **Single-image with config accordion**: `monopriors/gradio_ui/nodes/metric_depth_ui.py`
