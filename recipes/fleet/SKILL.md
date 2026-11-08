---
name: fleet
description: Fleet operations — use when a task reaches another machine, installs, updates, or removes a user-level tool or skill, converges a machine, places large data on the NAS, or changes Paseo daemon, MCP, or agent-fleet configuration.
---

# Fleet Operations

The fleet is defined by the `agent-fleet` repo (`~/agent-fleet` on every
managed machine). One version-controlled pixi-global manifest (base plus a
per-machine overlay) declares each machine's tools. A machine **converges** to
it: the manifest is the complete state, so converging installs what is missing
and deletes what is unlisted.

## Discovering machines

Query these sources each time; machine lists in documents go stale.

| Question | Source of truth | Command |
|---|---|---|
| What exists / is online? | Tailscale (self-updating) | `tailscale status` |
| Specs, GPUs, roles? | Rackpeek (curated, can lag) | `ssh pablo-dl-server docker exec rackpeek rpk summary` (also `rpk servers describe <name>`, `rpk systems list`) |
| Fleet-managed? | agent-fleet repo | an overlay in `manifests/machines/` and a `~/agent-fleet` checkout; the overlay name is the local identity sync selects, which can differ from the Tailscale name |

For Tailscale-specific connectivity, DNS, routing, access policy, or service
configuration, also read the `tailscale` skill. This skill still owns machine
discovery and installation rules; repository instructions own artifact
publishing routes.

## Reaching machines

SSH rides the tailnet: any hostname from `tailscale status` is an SSH target
(`~/.ssh/config` may hold shorter aliases). On some machines the tailnet is
also the auth for Tailscale SSH. Tailscale itself is OS-layer, not
pixi-managed.

- Non-interactive commands: `ssh -o ConnectTimeout=5 -o BatchMode=yes <host> '…'`.
  BatchMode fails fast instead of hanging on a password prompt.
- Non-interactive SSH does not load the shell rc, so pixi tools are missing
  from PATH. Start every remote command chain with
  `export PATH="$HOME/.pixi/bin:$PATH"`.
- `pkill -f <pattern>` over SSH can match your own SSH session's command line.
  Find the process with `pgrep -a`, then kill exact PIDs.

Scripted transfers use SSH:

- Single file, exact destination: `scp file <host>:path`.
- Directory trees: `COPYFILE_DISABLE=1 tar czf - . | ssh <host> 'tar xzf - -C <dest>'`
  (COPYFILE_DISABLE strips macOS xattr noise).
- Phone or iPad: Taildrop (share sheet / `tailscale file cp`) is for devices
  without SSH. It has no destination control: macOS receives to `~/Downloads`;
  headless Linux needs `tailscale file get`. Alpha; requires the tailnet
  "Send Files" setting.

## NAS storage

Active project data works on the SSD tier, `/mnt/nas/ssd-8tb`. The HDD shares
hold the stable copy, and Pablo decides what moves there. The SSD is one disk
with no backup, so report any costly result that exists only on it.

Read `~/agent-fleet/docs/nas-best-practices.md` before you place large data,
copy to or from the NAS, remove NAS data, or serve a catalog from it. It is the
source of truth for layout, capacity limits, transfers, and catalogs.

## Changing a machine's tools

1. Edit `manifests/base.toml` (fleet-wide) or `manifests/machines/<host>.toml`
   (one machine). Use exact `==` pins. Two forms are holds that pin-bump
   skips; keep them as written:
   - a range pin (`tmux = "3.3.*"`): the range is the hold;
   - `# pin-bump: hold` on an `==` pin line: a frozen exact pin.

   Base packages must solve on linux-64, linux-aarch64, and osx-arm64.
   Channel order: conda-forge, then the own channels (`prefix.dev/ai-demos`,
   `prefix.dev/my-skill-forge`), then `prefix.dev/github-releases` as
   gap-filler. JS CLIs get rattler-build recipes in the ai-demos repo
   (npm-tarball pattern, see `recipes/ruler` there) and arrive via the channel.
2. `pixi run check`, commit, push.
3. Converge each machine: `~/agent-fleet/scripts/sync.sh`.

**Done when** every target machine prints `✓ <host> converged to <rev>` at the
pushed revision.

## Adding a skill fleet-wide

Skills are `agent-skill-<name>` conda packages from
`https://prefix.dev/my-skill-forge`.

1. Recipe in the my-skill-forge repo (vendored SKILL.md, `noarch: generic`,
   `agentskills read-properties` in tests; copy an existing recipe). Merging
   publishes to the channel.
2. Pin it: add `agent-skill-<name> = "==<version>"` under
   `[envs.agent-skill-forge.dependencies]` in agent-fleet's
   `manifests/base.toml`, push.
3. Converge machines. Sync links each installed skill into `~/.claude/skills`
   and `~/.codex/skills`; the work profiles share those directories.

The pin is the membership act: publishing alone reaches no machine, and the
weekly pin-bump only bumps skills already pinned. **Done when** the skill
directory appears under `~/.claude/skills/` on every converged machine.

## Updates

Monday 05:00 UTC: ai-demos autobump opens recipe-bump PRs, which auto-merge on
green CI and publish. Monday 06:00 UTC: agent-fleet pin-bump opens
`base.toml` PRs, which a human merges; machines pick them up on the next
converge. Rollback is a revert of the pin PR.

## Agent configuration and Paseo

MCP servers and the Paseo daemon are fleet-managed: edit agent-fleet, push,
converge. Before you add or change an MCP server, or touch the Paseo daemon or
its config, read [`references/agent-config.md`](references/agent-config.md).

## When a converge fails or a machine misbehaves

Read [`references/internals.md`](references/internals.md): the sync pipeline
and its safety gates, overlay onboarding, and known failure signatures (dead
OAuth that looks like "model unavailable", stale stamps, app configs broken by
version bumps, PATH shadowing).
