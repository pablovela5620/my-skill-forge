# pixi.toml Integration

When adding a new node, wire its tasks into the root `pixi.toml`. All pixi configuration lives in the root, and all `pyrefly` configuration lives in the root `pyrefly.toml` — per-package `pyproject.toml` files only have standard Python packaging metadata plus tool config that is truly package-local (for example `ruff`).

## Adding Tasks

Tasks go in the package's feature section. Every node needs three tasks: a node launcher, a hot-reload dev task, and a CLI demo.

### Gradio Node Task

```toml
[feature.<package>.tasks.<env>-<name>-node]
cmd = "python tools/nodes/<name>_node.py"
depends-on = ["_download-<name>-examples"]
cwd = "packages/<package-dir>"
description = "Launch <Name> Gradio node app"
```

### Hot-Reload Dev Task

```toml
[feature.<package>.tasks.<env>-<name>-node-dev]
cmd = "gradio tools/nodes/<name>_node.py --watch-dirs $PIXI_PROJECT_ROOT/packages/<package-dir>/<module>"
depends-on = ["_download-<name>-examples"]
cwd = "packages/<package-dir>"
description = "<Name> Gradio node with hot reload"
```

Key points about `--watch-dirs`:
- Must use `$PIXI_PROJECT_ROOT` for an **absolute path** — relative paths silently fail to clear cached modules during reload
- The `tools/` directory is auto-watched (parent of the demo file); only the module source dir needs explicit `--watch-dirs`
- Without `--watch-dirs`, Gradio watches CWD which includes `.pixi/`, causing stdlib modules to get cleared from `sys.modules` during reload

### CLI Demo Task

```toml
[feature.<package>.tasks.<env>-<name>-demo]
cmd = "python tools/demos/<name>.py --rr-config.spawn"
depends-on = ["_download-<name>-examples"]
cwd = "packages/<package-dir>"
description = "Run <Name> CLI demo with Rerun viewer"
```

Note: `--rr-config.spawn` spawns a Rerun viewer by default. Users can override with `--rr-config.save output.rrd` etc.

### Download Task (for example data > 10MB)

```toml
[feature.<package>.tasks._download-<name>-examples]
cmd = "test -d data/examples/<name> || hf download <hf-repo-id> --repo-type dataset --local-dir data/examples/<name>"
cwd = "packages/<package-dir>"
description = "Download <Name> example data from HuggingFace"
```

Key points:
- `test -d ...` makes it idempotent — won't re-download if data already exists
- Use `hf download` (NOT `huggingface-cli download`) — conda's huggingface_hub provides `hf`
- `--repo-type dataset` for data repos
- `--local-dir` controls where files land
- Prefix with `_` to mark as private (convention) — other tasks use `depends-on` to call it

### Task Dependencies

All three main tasks should `depends-on` the download task:

```toml
[feature.<package>.tasks.<env>-<name>-node]
cmd = "python tools/nodes/<name>_node.py"
depends-on = ["_download-<name>-examples"]

[feature.<package>.tasks.<env>-<name>-node-dev]
cmd = "gradio tools/nodes/<name>_node.py --watch-dirs $PIXI_PROJECT_ROOT/packages/<package-dir>/<module>"
depends-on = ["_download-<name>-examples"]

[feature.<package>.tasks.<env>-<name>-demo]
cmd = "python tools/demos/<name>.py --rr-config.spawn"
depends-on = ["_download-<name>-examples"]
```

## Environment Naming Convention

Each package has two environments in pixi.toml:

| Environment | Use | Contains |
|---|---|---|
| `<package>` | Production: running demos/apps | Package deps + CUDA + common |
| `<package>-dev` | Development: lint, test, typecheck | Production + ruff, pytest, beartype, pyrefly |

Tasks like `<env>-<name>-node` and `<env>-<name>-demo` work in both environments. Dev tasks (`lint`, `typecheck`, `tests`) only work in `*-dev` environments.

## Pyrefly Configuration

`pyrefly` is monorepo-wide in this repo:

- Do not add `[tool.pyrefly]` to `packages/<package>/pyproject.toml`
- Update the root `pyrefly.toml` when a new package adds a new source root
- Ensure the root config includes:
  - `project-includes` for the package's Python files
  - `search-path` for the package import root
  - `site-package-path` for the package's `*-dev` environment if it has unique third-party deps

The shared Pixi `typecheck` task points `pyrefly` at the root config explicitly so per-package
checks use the monorepo search paths consistently.

## Running Tasks

```bash
# Gradio node (prod env is fine)
pixi run -e <package> <env>-<name>-node

# Gradio node with hot reload (dev env recommended)
pixi run -e <package>-dev <env>-<name>-node-dev

# CLI demo (prod env is fine)
pixi run -e <package> <env>-<name>-demo

# Download example data
pixi run -e <package> _download-<name>-examples
```

## Feature Composition Pattern

Dependencies are organized as composable features in root `pixi.toml`:

- **`common`** — shared deps all envs get (numpy, opencv, rerun-sdk, gradio, jaxtyping, etc.)
- **`cuda`** — CUDA toolkit + PyTorch GPU
- **`dev`** — ruff, pytest, beartype, pyrefly
- **Per-package features** — package-specific deps, editable install, tasks

Each environment composes features:
```toml
[environments]
<package> = ["common", "cuda", "<package>"]
<package>-dev = ["common", "cuda", "<package>", "dev"]
```

## Adding Package-Specific Dependencies

If the new node requires a dependency not in `common`:

```toml
[feature.<package>.pypi-dependencies]
some-new-dep = ">=1.0"
```

Or for conda dependencies:
```toml
[feature.<package>.dependencies]
some-conda-dep = ">=1.0"
```

## Real Examples

Look at existing task definitions in root `pixi.toml`:
- `feature.pysfm.tasks` for pysfm's node/demo tasks (class-based pattern)
- `feature.monoprior.tasks` for monoprior's app/demo tasks
- `feature.sam3-rerun.tasks` for sam3-rerun's tasks
- `feature.pysfm.tasks._download-*` for download task patterns
