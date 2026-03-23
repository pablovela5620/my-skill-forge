---
name: create-node
description: Create a new single-purpose CV node in the monorepo with API layer (config + result dataclasses, pipeline function with Rerun logging), Gradio UI with embedded Rerun viewer (streaming binary stream), CLI entry point (tyro + RerunTyroConfig), and bundled example data. Use when adding a new model, predictor, or pipeline step as a standalone reusable app.
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
    nodes/<name>_app.py             # Layer 3a: Gradio launcher (3 lines)
    apps/<composite>_app.py         # (Composite app launchers — NOT nodes)
    demos/<name>.py                 # Layer 3b: CLI entry point (3 lines)
  data/examples/<name>/             # Required: bundled example inputs
```

**Key distinction**: `nodes/` holds single-purpose, reusable units. `apps/` holds composite UIs that orchestrate multiple nodes (e.g. multiview calibration). This applies to both `gradio_ui/` and `tools/`.

## Step-by-Step: Creating a New Node

### Step 1: Define the API Layer

Read the [API template reference](references/node_template_api.md) for the full annotated template.

Create `<module>/apis/<name>.py` with:

1. **Config dataclass** — all parameters, jaxtyping-annotated, tyro-compatible, docstring-per-field
2. **Result dataclass** — typed output with jaxtyping array annotations
3. **`run_<name>()`** — core pipeline function: takes domain data (arrays, NOT file paths), returns Result. Calls `rr.log()` using thread-local recording so the caller controls where data goes.
4. **`create_<name>_blueprint()`** — returns `rrb.ContainerLike` defining the Rerun layout for this node's outputs
5. **CLI config dataclass** — wraps the node config + `RerunTyroConfig` + input path(s)
6. **`main(config)`** — CLI entry point: loads data, inits model, runs pipeline, sets up Rerun blueprint, logs results

Key rules:
- The pipeline function's input is **domain data** (e.g. `UInt8[ndarray, "H W 3"]`), never file paths or Gradio artifacts
- `run_<name>()` does NOT import rerun — Rerun logging happens in `main()` (CLI) and the streaming callback (Gradio). The core function is pure computation.
- Config uses `@dataclass` with fields annotated using jaxtyping and documented with per-field docstrings (nerfstudio pattern)
- `main()` does lazy imports of heavy deps (`rerun`, `open3d`, `cv2`) to keep import time fast

### Step 2: Define the Gradio UI

Read the [UI template reference](references/node_template_ui.md) for the full annotated template.

Create `<module>/gradio_ui/nodes/<name>_ui.py` with:

1. **`EXAMPLE_DATA_DIR: Final[Path]`** — points to bundled example data
2. **Module-level config + model singleton** — loaded once at import, reused across runs
3. **`_sync_config()`** — reads widget values into config, conditionally re-inits model only when necessary
4. **`_parse_and_load_<inputs>()`** — converts Gradio file paths to domain data (e.g. `list[UInt8[ndarray, "H W 3"]]`)
5. **`<name>_fn()`** — streaming callback: creates `rr.RecordingStream` + `binary_stream()`, runs pipeline inside `with recording:`, yields bytes
6. **`create_<name>_blueprint()`** — Rerun blueprint for this node (may import from API if shared with CLI)
7. **`main() -> gr.Blocks`** — builds the UI layout

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

**Gradio app** (`tools/nodes/<name>_app.py`):
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
- Task for the Gradio app (e.g. `<name>-app`)
- Task for the CLI demo (e.g. `<name>-demo`)
- Download task for example data if needed (idempotent)

Pyrefly rules for this monorepo:
- Keep all Pyrefly configuration in the root `pyrefly.toml`
- Never add `[tool.pyrefly]` to `packages/<package>/pyproject.toml`
- If you introduce a new package or source root, update the root `pyrefly.toml` (`project-includes`, `search-path`, and `site-package-path`) so the shared `typecheck` task can resolve it

## Rerun Integration

Read the [Rerun patterns reference](references/rerun_patterns.md) for complete details.

Rerun is non-negotiable at every layer:

| Layer | Pattern | How it works |
|---|---|---|
| API (`run_<name>`) | Pure computation | Returns result dataclass. Does NOT call `rr.log()`. |
| CLI (`main`) | Global recording | `RerunTyroConfig` gives `--save`, `--connect`, `--spawn` flags. Calls `rr.log()` after pipeline returns. |
| Gradio (`<name>_fn`) | Binary stream | `rr.RecordingStream` + `binary_stream()`. Calls `rr.log()` inside `with recording:`. Yields `stream.read()` to `Rerun(streaming=True)`. |

Blueprint builders are shared between CLI and Gradio when the layout is the same. Define in the UI module (or API module) and import where needed.

## Checklist

Before considering a node complete:

- [ ] API: Config dataclass with jaxtyping fields + per-field docstrings
- [ ] API: Result dataclass with jaxtyping fields
- [ ] API: `run_<name>()` pure pipeline function (no Rerun, no file I/O)
- [ ] API: `create_<name>_blueprint()` Rerun layout
- [ ] API: CLI config wrapping node config + `RerunTyroConfig` + input path(s)
- [ ] API: `main(config)` CLI entry point with Rerun logging
- [ ] UI: Module-level model singleton
- [ ] UI: `_sync_config()` with conditional re-init
- [ ] UI: `_parse_and_load_<inputs>()` converting Gradio files to domain data
- [ ] UI: `<name>_fn()` streaming callback with binary stream
- [ ] UI: `main() -> gr.Blocks` with standard layout
- [ ] UI: `EXAMPLE_DATA_DIR` + `gr.set_static_paths` + `gr.Examples`
- [ ] Entry point: 3-line Gradio app launcher in `tools/nodes/`
- [ ] Entry point: 3-line tyro CLI wrapper in `tools/demos/`
- [ ] Example data: at least 2 examples bundled or downloadable
- [ ] pixi.toml: tasks for app + demo (+ download if needed)

## Testing

> **TODO**: A comprehensive testing strategy will be added in a future iteration. Planned tiers:
> - **T0: Import** — smoke test verifying package imports
> - **T1: API unit** — config creation, result shapes, pipeline with example data, verify Rerun output (non-empty `stream.read()`)
> - **T2: CLI integration** — run demo via subprocess with `--rr-config.save <tmp>.rrd`, verify exit code + RRD file exists
> - **T3: App E2E** — Playwright: launch app, upload examples, click Run, verify Rerun viewer renders

## Reference Implementations

These existing nodes demonstrate the pattern. Read them to see the conventions in practice.

> **Note**: Existing code is being migrated from the flat `gradio_ui/` and `tools/apps/` layout to the new `gradio_ui/nodes/` and `tools/nodes/` convention. If a reference file still lives in the old location, follow the pattern but place new files in the correct `nodes/` subdirectory.

- **Multiview Geometry** (multi-image node):
  - API: `packages/monoprior/monopriors/apis/multiview_geometry.py`
  - UI: `packages/monoprior/monopriors/gradio_ui/nodes/multiview_geometry_ui.py`
  - App: `packages/monoprior/tools/nodes/multiview_geometry_app.py`

- **Metric Depth** (single-image node):
  - API: `packages/monoprior/monopriors/apis/metric_depth.py`
  - UI: `packages/monoprior/monopriors/gradio_ui/nodes/metric_depth_ui.py`
  - App: `packages/monoprior/tools/nodes/metric_depth_app.py`

- **SAM3 Segmentation** (single-image, different package):
  - UI: `packages/sam3-rerun/src/sam3_rerun/gradio_ui/nodes/sam3_rerun_ui.py`
  - App: `packages/sam3-rerun/tools/nodes/sam3_rerun_app.py`

- **Multiview Calibration** (composite app — NOT a node):
  - UI: `packages/monoprior/monopriors/gradio_ui/apps/multiview_calibration_ui.py`
  - App: `packages/monoprior/tools/apps/multiview_calibration_app.py`

## Architecture Docs

For deeper context on the overall architecture and future plans:
- `packages/monoprior/docs/app_architecture.md` — detailed walkthrough of the layered pattern
- `packages/monoprior/docs/node_decomposition_plan.md` — plan for decomposing monolithic pipelines into nodes
