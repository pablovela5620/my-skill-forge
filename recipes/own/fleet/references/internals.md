# Sync Internals & Failure Signatures

Disclosed reference for the [`fleet`](../SKILL.md) skill. Read it when a
converge fails, onboarding stops, or a machine behaves unexpectedly. The code
lives in agent-fleet's `fleetcore/sync.py`; `scripts/sync.sh` is the wrapper.

## Sync pipeline

Each stage stops the converge on failure, and the success stamp
(`~/.pixi/manifests/pixi-global.toml.last-synced`) is written only after every
stage passes.

1. **Checkout gates.** `~/agent-fleet` must resolve to the running checkout;
   a second real checkout there is refused (two checkouts risk converging
   stale state). Uncommitted changes are refused unless `FLEET_ALLOW_DIRTY=1`.
2. **Update.** `git fetch`, then a fast-forward merge to the upstream branch
   and a re-exec through `scripts/sync.sh`, so the updated code runs. Diverged
   history is a hard stop. Offline, sync converges only if the checkout matches
   the last-synced stamp, else it refuses unless `FLEET_OFFLINE_OK=1`
   (tar-shipped machines set this: the fresh checkout is the intended state).
3. **Overlay onboarding.** With no overlay for this machine,
   `capture_overlay.py` snapshots every live env not in base into
   `manifests/machines/<host>.toml` and stops for review. `<host>` is the
   stable identity: lowercase short hostname on Linux, `scutil --get
   LocalHostName` on macOS (`FLEET_HOST` overrides). If that identity changes
   (macOS renames on network name conflicts), sync re-enters onboarding for the
   "new" host; `git mv` the overlay to the new name (the fresh capture should be
   near-identical).
   - Envs whose exposed binaries collide with base are skipped and reported.
   - A name-colliding env stops everything (exit 2, no overlay written) only
     when the live env has extra packages base lacks, since converging would
     delete them. Base merely adding packages is normal converge-forward.
     Resolve extras by moving them to the overlay under a new env name, or by
     accepting base's definition.
4. **Render and safety gates.** Base plus overlay render to the live manifest.
   A base with fewer than 10 envs is refused (corrupted base). More than 5
   deletions is refused unless `FLEET_ALLOW_MASS_DELETE=1`.
5. **`pixi global sync`** installs what is missing and deletes what is
   unlisted. The previous manifest is snapshotted (`.prev`) and restored if
   pixi rejects the render. Pixi treats a per-env solve failure as a warning
   (exit 0), so a broken pin can silently leave an env missing; the CI solve
   gate (`pixi run solve-check`, three-platform matrix) catches this at PR
   time.
6. **Blender add-on.** Deploys the add-on matching the installed fleet Blender
   package.
7. **Agent files.** Installed `agent-skill-*` envs are linked into
   `~/.claude/skills` and `~/.codex/skills`; dangling links are pruned and
   non-symlink paths are kept with a warning. `~/.claude-work/skills` and
   `~/.codex-work/skills` link to those directories. `.zshrc` and `.bashrc`
   get the managed line sourcing `config/shell/fleet.sh`.
8. **Codex defaults.** Root model, reasoning effort, and service tier are set
   in `~/.codex` and `~/.codex-work`; named profiles are preserved.
9. **Paseo.** `paseo_apply.py` applies the fleet Paseo config (tailnet-IP
   listen) and service unit. It restarts the daemon only on a fleet-initiated
   config change (shadow copy `.fleet-config-applied`) and does nothing without
   the pixi paseo binary.
10. **Ruler** reads the shared instructions and MCP declarations, then writes
    the personal and work profile files for both agents. Sync supplies output
    paths, installed stdio command paths, and the live Paseo endpoint. Ruler
    performs the native merges; unrelated MCPs and settings remain. Invalid
    agent configuration fails the converge.

## Failure signatures

- **"Model unavailable" in claude** → almost always a dead OAuth token, not
  model gating. Probe: `claude -p --model claude-fable-5 "ok"`. A 401 means
  re-login (`claude` → `/login` in a TTY); stale entitlement caches in
  `~/.claude.json` clear on the next authenticated launch.
- **Stale tool version** → the machine has not converged: run sync.sh, then
  compare the stamp file against `git rev-parse HEAD` in `~/agent-fleet`.
- **App config broken after a version bump** (for example yazi themes after a
  major jump) → per-app config is not fleet-managed; fix it locally
  (yazi: `ya pkg upgrade --discard`).
- **`gh pr create` fails or git pushes are rejected on a machine** → check
  which account gh holds (`gh auth status`); work and personal credentials
  differ per machine.
- **A pinned tool runs the wrong version** → PATH shadowing: a brew, apt, or
  native copy resolves before `~/.pixi/bin`. fleet.sh prepends pixi in sourced
  shells; check `command -v <tool>` in the real interactive shell, and clean
  dormant duplicates (`~/.local/bin`, brew) when convenient.
