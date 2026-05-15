# Project Summary

This repository packages coding-agent skills as noarch conda packages and publishes them to the
`https://prefix.dev/my-skill-forge` channel.

Skills are Markdown-based directories for coding agents. A skill must contain `SKILL.md` with YAML
frontmatter, and may include `references/`, `scripts/`, assets, or other files that the skill reads
on demand.

## Package Layout

Package each skill as `agent-skill-<skill>` from `recipes/<skill>/recipe.yaml`.

Recipe conventions:

- Use `noarch: generic`.
- Install files under `$PREFIX/share/agent-skills/<skill>/`.
- Do not use `etc/agent-skills`; `pixi-skills` discovers packaged skills from `share/agent-skills`.
- Include a strict `package_contents` test for `share/agent-skills/<skill>/SKILL.md` and any required extra files.
- Include `agentskills validate $CONDA_PREFIX/share/agent-skills/<skill>` in script tests.
- Prefer `requirements.run_constraints` for tool or library compatibility unless the package truly needs a bundled runtime dependency.

## Adding A Skill

For a local or original skill, follow `examples/example-skill`:

- Do not specify a `source` field.
- Copy files from `$RECIPE_DIR`.
- If the user gives special build/update instructions, place them in `PROMPT.md` next to `recipe.yaml`.
- Do not include `PROMPT.md` in the final package.

For a mirrored upstream skill:

- Inspect the upstream repository and identify the exact skill directory before writing the recipe.
- Use `source.git` with a pinned commit, or `source.url` with `sha256` for release/archive sources.
- Copy only the intended skill subtree into `share/agent-skills/<skill>/`.
- Add `context.upstream_path` for `git-main` mirrors when possible so autobump can avoid commits where the skill path was removed or moved.
- Use a small patch only when needed to adapt upstream content to this repo.

Whenever you add a skill, update `README.md`.

## Autobump And Updates

If a skill fetches content from a remote source, add it to `.github/workflows/autobump.yml`.

Update strategy defaults:

- Use `github-latest-release` when the upstream repository has active, recent releases.
- Use `git-main` otherwise.
- Use `yolo` only for fixed URLs where the sha256 should be refreshed directly.

When updating skill content:

1. Check the latest upstream source in a temporary checkout or with GitHub/API inspection.
2. Compare the current skill against upstream/tool behavior.
3. Apply the minimal viable update: keep accurate content unchanged, and only edit outdated, incorrect, or missing parts.
4. Read any `PROMPT.md` beside the recipe and follow it while updating.
5. Do not blindly bump to an upstream commit that no longer contains the packaged skill path.

The updater can be run as:

```bash
pixi run .github/scripts/update.sh <skill> <strategy>
```

## Build, Test, Publish, Install

Build all new packages:

```bash
pixi run build-new
```

Build a specific package:

```bash
pixi run rattler-build build -r recipes/<skill>
```

Always run:

```bash
pixi run pre-commit-run
```

Do not install `my-skill-forge` packages from local build artifacts or `--path output/...`.
Local artifact installs bypass the review/publish path and can leave the user's global environment ahead of the channel.

For real user-level installation:

1. Push and merge the change.
2. Wait for the `Package` workflow on `main` to upload to prefix.dev.
3. Verify the package is available from `https://prefix.dev/my-skill-forge`.
4. Install from the channel-backed global environment:

```bash
pixi global add --environment agent-skill-forge <agent-skill-package-name>
```

Then link and verify with `pixi-skills`:

```bash
pixi skills manage --backend codex --scope global
pixi skills status --backend codex
test -f ~/.codex/skills/<skill-name>/SKILL.md
```
