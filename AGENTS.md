# my-skill-forge

This repo packages coding-agent skills as `noarch: generic` conda packages named
`agent-skill-<skill>`, built from `recipes/<skill>/recipe.yaml` and published to
`https://prefix.dev/my-skill-forge`. A skill is a directory with a `SKILL.md`
(YAML frontmatter) and any files it reads on demand.

## Recipe conventions

- Install the skill under `$PREFIX/share/agent-skills/<skill>/`. pixi-skills and
  the fleet discover skills there.
- Test with a strict `package_contents` list of every installed file, and with
  `agentskills read-properties $CONDA_PREFIX/share/agent-skills/<skill> > /dev/null`
  (from `skills-ref`). That checks the frontmatter parses and has a name and a
  description, and it accepts agent-specific fields.
- Keep upstream frontmatter as published, including `user-invocable`,
  `user_invocable`, and `disable-model-invocation`.
- State tool or library compatibility with `requirements.run_constraints`. Add a
  run dependency only when the skill cannot work without it.
- The recipe is the catalog: the package name, description, and upstream links
  live in `recipe.yaml` only.

## Adding a skill

A local skill starts from `examples/example-skill`: no `source`, files copied
from `$RECIPE_DIR`. Put special build or update instructions in `PROMPT.md`
beside the recipe; the package leaves it out.

A mirrored upstream skill:

1. Find the exact skill directory upstream.
2. Pin `source.git` to a commit, or `source.url` with `sha256` for a release
   archive.
3. Set `context.upstream_path` to the skill directory and copy only that
   subtree. Patch only what this repo must adapt.
4. Add the skill to the matrix in `.github/workflows/autobump.yml`:
   `github-latest-release` when upstream cuts regular releases, otherwise
   `git-main`.

## Updating skill content

1. Read the latest upstream skill, and any `PROMPT.md` beside the recipe.
2. Change only what is outdated, wrong, or missing.
3. Pin a commit that still contains the skill path;
   `pixi run autobump <skill> git-main` picks one.

## Versions and publishing

Builds use `--skip-existing=all`: a build skips any name, version, and build
string the channel already has, and editing the recipe does not change the
build string.

- A change to installed files needs a new `package.version`, or a new
  `build.number` when the version tracks upstream.
- A test-only edit needs no bump. CI does not rebuild it, so build it locally.

A merge to `main` publishes every new version through the Package workflow.
Autobump runs daily. For each skill it bumps the recipe on one `bump-<skill>`
PR, builds and tests the package, merges, and dispatches Package. When a build
fails, the PR stays open and the Autobump run is red.

A change is done when `pixi run build <skill>` passes the recipe tests for each
changed recipe and `pixi run check` passes.

## Installing

A package reaches machines only through agent-fleet. When Package has uploaded
the version, pin it under `[envs.agent-skill-forge.dependencies]` in an
agent-fleet manifest and converge with the `fleet` skill. Install from the
channel only: the next fleet sync removes a local artifact or a
`pixi global add`, and both skip review.
