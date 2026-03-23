# pixi.toml Integration

When adding a new node, wire its tasks into the root `pixi.toml`. All pixi configuration lives in the root, and all `pyrefly` configuration lives in the root `pyrefly.toml` — per-package `pyproject.toml` files only have standard Python packaging metadata plus tool config that is truly package-local (for example `ruff`).

## Adding Tasks

Tasks go in the package's feature section. Every node needs at minimum an app task and a demo task.

### Gradio Node App Task

```toml
[feature.<package>.tasks.<name>-app]
cmd = "python tools/nodes/<name>_app.py"
cwd = "packages/<package-dir>"
description = "Launch <Name> Gradio node app"
```

### CLI Demo Task

```toml
[feature.<package>.tasks.<name>-demo]
cmd = "python tools/demos/<name>.py --rr-config.spawn"
cwd = "packages/<package-dir>"
description = "Run <Name> CLI demo with Rerun viewer"
```

Note: `--rr-config.spawn` spawns a Rerun viewer by default. Users can override with `--rr-config.save output.rrd` etc.

### Download Task (for example data > 10MB)

```toml
[feature.<package>.tasks.download-<name>-examples]
cmd = "test -d data/examples/<name> || hf download <hf-repo-id> --repo-type dataset --local-dir data/examples/<name>"
cwd = "packages/<package-dir>"
description = "Download <Name> example data from HuggingFace"
```

Key points:
- `test -d ...` makes it idempotent — won't re-download if data already exists
- Use `hf download` (NOT `huggingface-cli download`) — conda's huggingface_hub provides `hf`
- `--repo-type dataset` for data repos
- `--local-dir` controls where files land

### Making demo depend on download

```toml
[feature.<package>.tasks.<name>-demo]
cmd = "python tools/demos/<name>.py --rr-config.spawn"
cwd = "packages/<package-dir>"
depends-on = ["download-<name>-examples"]
description = "Run <Name> CLI demo with Rerun viewer"
```

## Environment Naming Convention

Each package has two environments in pixi.toml:

| Environment | Use | Contains |
|---|---|---|
| `<package>` | Production: running demos/apps | Package deps + CUDA + common |
| `<package>-dev` | Development: lint, test, typecheck | Production + ruff, pytest, beartype, pyrefly |

Tasks like `<name>-app` and `<name>-demo` work in both environments. Dev tasks (`lint`, `typecheck`, `tests`) only work in `*-dev` environments.

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
# Gradio app (prod env is fine)
pixi run -e <package> <name>-app

# CLI demo (prod env is fine)
pixi run -e <package> <name>-demo

# Download example data
pixi run -e <package> download-<name>-examples
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
- `feature.monoprior.tasks` for monoprior's app/demo tasks
- `feature.sam3-rerun.tasks` for sam3-rerun's tasks
- `feature.monoprior.tasks.download-*` for download task patterns
