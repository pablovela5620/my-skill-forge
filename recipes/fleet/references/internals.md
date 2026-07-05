# Sync Internals & Failure Signatures

Disclosed reference for the [`fleet`](../SKILL.md) skill — read when a
converge fails, onboarding stops, or a machine behaves unexpectedly.

## sync.sh pipeline

1. `git pull --ff-only`, then re-exec itself (so the updated script runs, not
   the stale buffer). Diverged history → hard stop. Offline → converges only
   if the checkout matches the last-synced stamp
   (`~/.pixi/manifests/pixi-global.toml.last-synced`), else refuses unless
   `FLEET_OFFLINE_OK=1` (tar-shipped machines set this: the fresh checkout IS
   the intended state).
2. First run with no overlay: `capture-overlay.sh` snapshots every live env
   not in base into `manifests/machines/<host>.toml` and stops for review.
   Envs whose exposed binaries collide with base are skipped (reported).
   Name-colliding envs carrying DIFFERENT packages than base stop everything
   with a conflict report (exit 2) — resolve before converging.
3. Render base+overlay to a temp file, then safety gates before the atomic
   swap: fewer than 10 envs → refuse (corrupted base); more than 5 deletions
   → refuse unless `FLEET_ALLOW_MASS_DELETE=1`.
4. `pixi global sync` — installs missing, DELETES unlisted.
5. Skill symlinks: forge envs (`agent-skill-*`) plus this repo's `skills/`
   dir, linked into `~/.claude/skills` and `~/.codex/skills`; dangling links
   pruned.

## Failure signatures

- **"Model unavailable" in claude** → almost always a dead OAuth token, not
  model gating. Probe: `claude -p --model claude-fable-5 "ok"`. A 401 means
  re-login (`claude` → `/login` in a TTY); stale entitlement caches in
  `~/.claude.json` clear on next authenticated launch.
- **Stale tool version** → machine hasn't converged: run sync.sh, compare the
  stamp file against `git rev-parse HEAD` in `~/agent-fleet`.
- **App config broken after a version bump** (e.g. yazi themes after a major
  jump) → per-app config is not fleet-managed; fix locally
  (yazi: `ya pkg upgrade --discard`).
- **`gh pr create` fails / git pushes rejected on a machine** → check WHICH
  account gh holds (`gh auth status`) — work vs personal credentials differ
  per machine.
