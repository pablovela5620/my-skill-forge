---
name: fleet
description: Fleet operations — use when a task involves other machines (discovering, reaching, or reasoning about them), installing/updating/removing any user-level tool, converging a machine, or working with paseo or the agent-fleet repo.
---

# Fleet Operations

The fleet is defined by the `agent-fleet` repo (`~/agent-fleet` on every
managed machine). One exact-pinned pixi-global manifest (base + per-machine
overlay) declares each machine's tools; a machine **converges** to it — the
manifest is the complete state, and converging installs what's missing AND
deletes what's unlisted.

## Discovering machines — query, never hardcode

| Question | Source of truth | Command |
|---|---|---|
| What exists / is online? | Tailscale (self-updating) | `tailscale status` |
| Specs, GPUs, roles? | Rackpeek (curated, can lag) | `ssh pablo-dl-server docker exec rackpeek rpk summary` (also `rpk servers describe <name>`, `rpk systems list`) |
| Fleet-managed? | agent-fleet repo | overlay at `manifests/machines/<lowercased short hostname>.toml`; `~/agent-fleet` checked out |

Stable facts: `pablo-dl-server` is the conductor (always-on, hosts rackpeek).
`spark-9232` is linux-aarch64 — account for it in platform availability.
Macs are osx-arm64.

## Reaching machines

SSH rides the tailnet: any hostname from `tailscale status` is an SSH target
(`ssh pablo-ubuntu`, `ssh spark-9232`; `~/.ssh/config` may hold shorter
aliases). Patterns that matter:

- Non-interactive commands: `ssh -o ConnectTimeout=5 -o BatchMode=yes <host> '…'`
  — BatchMode fails fast instead of hanging on a password prompt.
- PATH gotcha: non-interactive SSH does NOT load the shell rc, so pixi tools
  are missing. Prefix every remote command chain with
  `export PATH="$HOME/.pixi/bin:$PATH"`.
- `pkill -f <pattern>` over SSH can match and kill your own SSH session's
  command line — prefer `pgrep -a` first, then kill exact PIDs.

Moving files — scripted transfers always use SSH; Taildrop is for humans:

- Single file, exact destination: `scp file <host>:path`.
- Directory trees / preserving structure:
  `COPYFILE_DISABLE=1 tar czf - . | ssh <host> 'tar xzf - -C <dest>'`
  (COPYFILE_DISABLE strips macOS xattr noise).
- Phone/iPad ↔ machine: Taildrop (share sheet / `tailscale file cp`) — its
  niche is devices without SSH. No destination control: macOS receives to
  ~/Downloads; headless Linux needs a second step (`tailscale file get`).
  Alpha; requires the tailnet "Send Files" setting.

Tailscale itself is OS-layer (never pixi-managed); the tailnet is also the
auth for Tailscale SSH on some machines (e.g. spark).

## Changing a machine's tools

1. Edit `manifests/base.toml` (fleet-wide) or `manifests/machines/<host>.toml`
   (one machine). Exact `==` pins; base packages must solve on linux-64,
   linux-aarch64, AND osx-arm64. Channels: conda-forge → own channels
   (`prefix.dev/ai-demos`, `prefix.dev/my-skill-forge`) →
   `prefix.dev/github-releases` as gap-filler.
2. `pixi run check`, commit, push.
3. Converge each machine: `~/agent-fleet/scripts/sync.sh`.

Done when every target machine prints `✓ <host> converged to <rev>` at the
pushed revision — not before.

JS CLIs get rattler-build recipes in the ai-demos repo (npm-tarball pattern,
see `recipes/ruler` there) and arrive via the channel.

## Holds

- Range pin (`tmux = "3.3.*"`): the range IS the hold — pin-bump ignores it.
- `# pin-bump: hold` on an `==` pin line: frozen exact pin.

## Updates

Mon 05:00 UTC: ai-demos autobump PRs recipe bumps (auto-merges on green CI,
publishes). Mon 06:00: agent-fleet pin-bump PRs base.toml updates —
human-gated; merging converges the fleet on next sync. Rollback = revert the
pin PR.

## Paseo (orchestration layer)

CLI daemon on headless machines (pixi `paseo` package), desktop app on the
primary laptop. `paseo daemon status|start|stop`. State lives in `~/.paseo`
(machine-local, survives binary swaps); the daemon password is
`PASEO_PASSWORD` in `~/.secrets.env`, never in config files.

## When a converge fails or a machine misbehaves

Read [`references/internals.md`](references/internals.md) — sync.sh's
pipeline and safety gates, capture/overlay onboarding, and known failure
signatures (dead OAuth masquerading as "model unavailable", stale stamps,
app configs broken by version bumps).
