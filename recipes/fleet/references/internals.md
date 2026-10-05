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
2. First run with no overlay: `capture_overlay.py` snapshots every live env
   not in base into `manifests/machines/<host>.toml` and stops for review.
   NB: "first run" is keyed on the overlay filename matching `hostname -s`
   — if the OS renames the machine (macOS does on network name conflicts),
   sync re-enters onboarding for the "new" host; fix by `git mv`-ing the
   overlay to the new name (the fresh capture should be near-identical).
   Envs whose exposed binaries collide with base are skipped (reported).
   Name-colliding envs stop everything (exit 2) only when the live env has
   EXTRA packages base lacks (deletion risk); base merely adding packages is
   normal converge-forward. Resolve extras by promoting them to base or
   dropping them.
3. Render base+overlay to a temp file, then safety gates before the atomic
   swap: fewer than 10 envs → refuse (corrupted base); more than 5 deletions
   → refuse unless `FLEET_ALLOW_MASS_DELETE=1`.
4. `pixi global sync` — installs missing, DELETES unlisted. The previous
   manifest is snapshotted first and restored if pixi rejects the render.
   NB: pixi treats a per-env solve failure as a WARNING (exit 0) — a broken
   pin can silently leave an env missing; the CI solve gate exists to catch
   this at PR time (`pixi run solve-check`, three-platform matrix).
5. Skill symlinks: installed forge envs (`agent-skill-*`), linked into
   `~/.claude/skills` and `~/.codex/skills`; dangling links
   pruned.
6. Prepare both work profiles, share their skills, apply Codex defaults, and
   add the managed rc line sourcing `config/shell/fleet.sh`.
7. `paseo_apply.py` — fleet paseo config (tailnet-IP listen) + service
   unit; restarts the daemon only on fleet-initiated config change (shadow
   copy `.fleet-config-applied`); no-op without the pixi paseo binary.
8. Ruler reads shared instructions and MCP declarations, then writes both
   agents' personal and work profile files. Sync supplies output paths,
   installed stdio command paths, and the live Paseo endpoint. Ruler performs
   the native merges; unrelated MCPs and settings remain. Invalid agent
   configuration fails convergence without writing a success stamp.

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

- **A pinned tool runs the wrong version** → PATH shadowing: a brew/apt/native
  copy resolves before `~/.pixi/bin`. fleet.sh prepends pixi in sourced
  shells; check `command -v <tool>` in the real interactive shell, and clean
  dormant duplicates (`~/.local/bin`, brew) when convenient.
