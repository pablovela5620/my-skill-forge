# Entry Points Template

Entry points are thin wrappers — 3 to 6 lines each. They wire a UI or config to the API layer.

## Gradio Node Launcher (`tools/nodes/<name>_node.py`)

Node launchers live under `tools/nodes/`, while composite app launchers live under `tools/apps/`.

```python
from <module>.gradio_ui.nodes.<name>_ui import main

demo = main()

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
```

Notes:
- `demo = main()` at module level so Gradio can discover the app for `gradio <file>.py`
- `.queue()` enables queueing (required for streaming/generator callbacks)
- `ssr_mode=False` disables server-side rendering (not needed for Rerun streaming apps)
- No other logic — all UI construction lives in the `gradio_ui/nodes/` module

### Variant: Custom launch options

Some apps need specific launch config (e.g. HuggingFace Spaces, custom port):

```python
from <module>.gradio_ui.nodes.<name>_ui import EXAMPLE_DATA_DIR, main

if __name__ == "__main__":
    demo = main()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        allowed_paths=[str(EXAMPLE_DATA_DIR)],
    )
```

## CLI Demo (`tools/demos/<name>.py`)

```python
import tyro

from <module>.apis.<name> import <Name>CLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(<Name>CLIConfig))
```

Notes:
- `tyro.cli()` auto-generates CLI flags from the dataclass fields
- `<Name>CLIConfig` wraps the node config + `RerunTyroConfig` + input paths
- `RerunTyroConfig` gives `--rr-config.save <path.rrd>`, `--rr-config.connect`, `--rr-config.spawn` for free
- No other logic — all pipeline orchestration lives in the `apis` module's `main()`

### Example CLI usage

```bash
# Spawn Rerun viewer and visualize
pixi run -e <package>-dev -- python tools/demos/<name>.py --rr-config.spawn

# Save to RRD file
pixi run -e <package>-dev -- python tools/demos/<name>.py --rr-config.save output.rrd

# Custom input path
pixi run -e <package>-dev -- python tools/demos/<name>.py --image-dir data/examples/<name>/set_1

# Connect to running Rerun viewer
pixi run -e <package>-dev -- python tools/demos/<name>.py --rr-config.connect
```

## Real Examples

- **Gradio node**: `packages/pysfm/tools/nodes/video_to_image_node.py`, `packages/monoprior/tools/nodes/multiview_geometry_app.py`, `packages/monoprior/tools/nodes/metric_depth_app.py`
- **CLI demo**: `packages/pysfm/tools/demos/video_to_image.py`, `packages/monoprior/tools/demos/metric_depth.py`, `packages/monoprior/tools/demos/multiview_depth.py`
- **Gradio composite app** (NOT a node): `packages/monoprior/tools/apps/multiview_calibration_app.py`
- **Custom launch**: `packages/sam3-rerun/tools/nodes/sam3_rerun_app.py`
