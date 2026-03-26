---
name: create-node
description: Create a new single-purpose CV node in the monorepo with API layer (class-based node with verbose Rerun logging, result logging helper, config + result dataclasses), Gradio UI with embedded Rerun viewer (streaming binary stream), CLI entry point (tyro + RerunTyroConfig), and bundled example data. Use when adding a new model, predictor, or pipeline step as a standalone reusable app.
license: BSD-3-Clause
---

# Create Node

A **node** is a single-purpose, reusable CV unit in the monorepo. It wraps one model or algorithm, has a clear typed input/output contract, and integrates Rerun visualization at every layer. Every node works standalone as both a CLI tool and a Gradio app.

## When to Use This Skill

Use when the user wants to:
- Add a new model/predictor as a standalone app
- Create a new pipeline step (e.g. depth estimation, segmentation, alignment)
- Decompose a monolithic pipeline into a reusable node

## Node Anatomy

Every node has exactly three layers, each in its own file:

```
packages/<package>/
  <module>/
    apis/<name>.py                  # Layer 1: API (computation + Rerun logging)
    gradio_ui/
      nodes/<name>_ui.py            # Layer 2: Gradio UI (widgets + streaming)
      apps/<composite>_ui.py        # (Composite apps live here — NOT nodes)
  tools/
    nodes/<name>_node.py             # Layer 3a: Gradio launcher (3 lines)
    apps/<composite>_app.py         # (Composite app launchers — NOT nodes)
    demos/<name>.py                 # Layer 3b: CLI entry point (3 lines)
  data/examples/<name>/             # Required: bundled example inputs
```

**Key distinction**: `nodes/` holds single-purpose, reusable units. `apps/` holds composite UIs that orchestrate multiple nodes (e.g. multiview calibration). This applies to both `gradio_ui/` and `tools/`.

## Step-by-Step: Creating a New Node

### Step 1: Define the API Layer

Read the [API template reference](references/node_template_api.md) for the full annotated template.

Create `<module>/apis/<name>.py` with:

1. **Config dataclass** — all parameters, jaxtyping-annotated, tyro-compatible, docstring-per-field. Include `verbose: bool = False` for intermediate logging control.
2. **Result dataclass** — typed output with jaxtyping array annotations
3. **`<Name>Node` class** — encapsulates config, model, and computation. `__init__(config, parent_log_path)` loads the model once; `__call__(**kwargs)` runs the pipeline and logs intermediate results when `config.verbose` is True. The caller sets the Rerun recording context.
4. **`log_<name>_result()`** — logs the Result dataclass to Rerun. Called by both CLI `main()` and Gradio streaming callback, eliminating duplicate logging code.
5. **`create_<name>_blueprint()`** — returns `rrb.ContainerLike` defining the Rerun layout for this node's outputs
6. **CLI config dataclass** — wraps the node config + `RerunTyroConfig` + input path(s)
7. **`main(config)`** — CLI entry point: loads data, creates node, runs it, calls `log_<name>_result()`, sets up Rerun blueprint

Key rules:
- The node class gates all intermediate `rr.log()` calls behind `config.verbose`. The caller sets the recording context (`with recording:` in Gradio, global init in CLI).
- `log_<name>_result()` handles final output logging — shared between CLI and Gradio. No duplicate `rr.log()` blocks.
- Config uses `@dataclass` with fields annotated using jaxtyping and documented with per-field docstrings (nerfstudio pattern)
- Use top-level imports (`import cv2`, `import rerun as rr`, `import rerun.blueprint as rrb`)

### Step 2: Define the Gradio UI

Read the [UI template reference](references/node_template_ui.md) for the full annotated template.

Create `<module>/gradio_ui/nodes/<name>_ui.py` with:

1. **`EXAMPLE_DATA_DIR: Final[Path]`** — points to bundled example data
2. **Module-level `_CONFIG` + `_NODE` singletons** — `_CONFIG` holds current config, `_NODE` holds the node instance (with model loaded). Both loaded once at import, reused across runs.
3. **`_sync_config()`** — reads widget values into `_CONFIG`, conditionally re-creates `_NODE` only when model-affecting fields change
4. **`_parse_and_load_<inputs>()`** — converts Gradio file paths to domain data (e.g. `list[UInt8[ndarray, "H W 3"]]`)
5. **`<name>_fn()`** — streaming callback decorated with `@rr.recording_stream_generator_ctx`: creates `rr.RecordingStream` + `binary_stream()`, calls `_NODE(...)` and `log_<name>_result()` inside `with recording:`, yields `stream.read()` incrementally
6. **`main() -> gr.Blocks`** — builds the UI layout

Layout pattern (consistent across all nodes):
- **Left column** (scale=1): `gr.Tabs` with Inputs tab (file upload, Run button, Config accordion) and Outputs tab (status text). `gr.Examples` below tabs.
- **Right column** (scale=5): `Rerun(streaming=True)` viewer

Click chain (each `.then()` has one job):
```
run_btn.click(_switch_to_outputs)
  .then(lambda: uuid.uuid4())           # Fresh recording ID
  .then(_sync_config)                    # Sync widgets → config singleton
  .then(_parse_and_load_<inputs>)        # Gradio files → domain data (via gr.State)
  .then(<name>_fn)                       # Run pipeline, stream to Rerun viewer
```

Config accordion: **1:1 mapping** from Config dataclass fields to Gradio widgets. Every config field gets a widget.

### Step 3: Create Entry Points

Read the [entry points template reference](references/node_template_entrypoints.md).

**Gradio node** (`tools/nodes/<name>_node.py`):
```python
from <module>.gradio_ui.nodes.<name>_ui import main

demo = main()

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
```

**CLI demo** (`tools/demos/<name>.py`):
```python
import tyro
from <module>.apis.<name> import <Name>CLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(<Name>CLIConfig))
```

### Step 4: Bundle Example Data

Every node MUST ship with example inputs:
- Store in `data/examples/<name>/` relative to the package root
- At least 2 distinct examples (different scenes, subjects, or conditions)
- If total size < 10MB: commit directly to git
- If larger: create an idempotent pixi download task using `hf download`
- Same examples used in `gr.Examples()` in the Gradio UI
- Reference via `EXAMPLE_DATA_DIR: Final[Path]` in the UI module
- Call `gr.set_static_paths([str(EXAMPLE_DATA_DIR)])` so Gradio can serve them

### Step 5: Wire into pixi.toml

Read the [pixi integration reference](references/pixi_integration.md).

Add to the package's feature section in root `pixi.toml`:
- Task for the Gradio node (e.g. `<env>-<name>-node`) — `python tools/nodes/<name>_node.py`
- Task for hot-reload dev (e.g. `<env>-<name>-node-dev`) — `gradio tools/nodes/<name>_node.py --watch-dirs $PIXI_PROJECT_ROOT/...`
- Task for the CLI demo (e.g. `<env>-<name>-demo`) — `python tools/demos/<name>.py --rr-config.spawn`
- Download task for example data if needed (idempotent, use `depends-on`)

Pyrefly rules for this monorepo:
- Keep all Pyrefly configuration in the root `pyrefly.toml`
- Never add `[tool.pyrefly]` to `packages/<package>/pyproject.toml`
- If you introduce a new package or source root, update the root `pyrefly.toml` (`project-includes`, `search-path`, and `site-package-path`) so the shared `typecheck` task can resolve it

## Rerun Integration

Read the [Rerun patterns reference](references/rerun_patterns.md) for complete details.

Rerun is non-negotiable at every layer:

| Layer | Pattern | How it works |
|---|---|---|
| API (`<Name>Node`) | Verbose-gated logging | `__call__()` does computation + intermediate `rr.log()` when `config.verbose=True`. Relies on caller setting the recording context. |
| API (`log_<name>_result`) | Final result logging | Logs the Result dataclass to Rerun. Called by both CLI and Gradio — no duplicate logging. |
| API (`create_<name>_blueprint`) | Rerun layout | Returns `rrb.ContainerLike`. Shared between CLI and Gradio. |
| CLI (`main`) | Global recording | `RerunTyroConfig` gives `--save`, `--connect`, `--spawn` flags. Creates node, calls it, calls `log_<name>_result()`. |
| Gradio (`<name>_fn`) | Binary stream | `@rr.recording_stream_generator_ctx` + `rr.RecordingStream` + `binary_stream()`. Calls `_NODE(...)` + `log_<name>_result()` inside `with recording:`. Yields `stream.read()` incrementally. |

The `@rr.recording_stream_generator_ctx` decorator is **required** on all Gradio streaming callbacks that use Rerun. It suspends/restores the Rerun ContextVar around each `yield`, making it safe to yield inside `with recording:`. Without it, Gradio's async context switching invalidates the ContextVar token.

## Checklist

Before considering a node complete:

- [ ] API: Config dataclass with jaxtyping fields + per-field docstrings + `verbose: bool`
- [ ] API: Result dataclass with jaxtyping fields
- [ ] API: `<Name>Node` class with `__init__(config, parent_log_path)` + `__call__()` with verbose-gated intermediate logging
- [ ] API: `log_<name>_result()` — logs Result to Rerun (shared between CLI and Gradio)
- [ ] API: `create_<name>_blueprint()` Rerun layout
- [ ] API: CLI config wrapping node config + `RerunTyroConfig` + input path(s)
- [ ] API: `main(config)` CLI entry point using node + `log_<name>_result()`
- [ ] UI: Module-level `_CONFIG` + `_NODE` singletons
- [ ] UI: `_sync_config()` with conditional `_NODE` re-creation
- [ ] UI: `_parse_and_load_<inputs>()` converting Gradio files to domain data
- [ ] UI: `<name>_fn()` streaming callback with `@rr.recording_stream_generator_ctx` + binary stream
- [ ] UI: `main() -> gr.Blocks` with standard layout
- [ ] UI: `EXAMPLE_DATA_DIR` + `gr.set_static_paths` + `gr.Examples`
- [ ] Entry point: 3-line Gradio node launcher in `tools/nodes/<name>_node.py`
- [ ] Entry point: 3-line tyro CLI wrapper in `tools/demos/`
- [ ] Example data: at least 2 examples bundled or downloadable
- [ ] pixi.toml: tasks for node + node-dev + demo (+ download if needed)

## Testing

> **TODO**: A comprehensive testing strategy will be added in a future iteration. Planned tiers:
> - **T0: Import** — smoke test verifying package imports
> - **T1: API unit** — config creation, result shapes, pipeline with example data, verify Rerun output (non-empty `stream.read()`)
> - **T2: CLI integration** — run demo via subprocess with `--rr-config.save <tmp>.rrd`, verify exit code + RRD file exists
> - **T3: App E2E** — Playwright: launch app, upload examples, click Run, verify Rerun viewer renders

## Reference Implementations

These existing nodes demonstrate the pattern. Read them to see the conventions in practice.

> **Note**: Existing code is being migrated to the class-based node pattern. Older references (Multiview Geometry, Metric Depth) still use the function-based `run_<name>()` pattern — follow Video-to-Image for the current conventions.

- **Video-to-Image** (class-based node — **primary reference**):
  - API: `packages/pysfm/pysfm/apis/video_to_image.py` — `VideoToImageNode`, `VideoToImageConfig`, `VideoToImageResult`, `create_video_to_image_blueprint()`
  - UI: `packages/pysfm/pysfm/gradio_ui/nodes/video_to_image_ui.py` — `@rr.recording_stream_generator_ctx`, `_CONFIG` + node singleton, `_sync_config()`, `try/finally` cleanup
  - Node: `packages/pysfm/tools/nodes/video_to_image_node.py`
  - Demo: `packages/pysfm/tools/demos/video_to_image.py`

- **Multiview Geometry** (multi-image node, function-based — older pattern):
  - API: `packages/monoprior/monopriors/apis/multiview_geometry.py`
  - UI: `packages/monoprior/monopriors/gradio_ui/nodes/multiview_geometry_ui.py`
  - Node: `packages/monoprior/tools/nodes/multiview_geometry_app.py`

- **Metric Depth** (single-image node, function-based — older pattern):
  - API: `packages/monoprior/monopriors/apis/metric_depth.py`
  - UI: `packages/monoprior/monopriors/gradio_ui/nodes/metric_depth_ui.py`
  - Node: `packages/monoprior/tools/nodes/metric_depth_app.py`

- **SAM3 Segmentation** (single-image, different package):
  - UI: `packages/sam3-rerun/src/sam3_rerun/gradio_ui/nodes/sam3_rerun_ui.py`
  - Node: `packages/sam3-rerun/tools/nodes/sam3_rerun_app.py`

- **Multiview Calibration** (composite app — NOT a node):
  - UI: `packages/monoprior/monopriors/gradio_ui/apps/multiview_calibration_ui.py`
  - App: `packages/monoprior/tools/apps/multiview_calibration_app.py`

## Architecture Docs

For deeper context on the overall architecture and future plans:
- `packages/monoprior/docs/app_architecture.md` — detailed walkthrough of the layered pattern
- `packages/monoprior/docs/node_decomposition_plan.md` — plan for decomposing monolithic pipelines into nodes
