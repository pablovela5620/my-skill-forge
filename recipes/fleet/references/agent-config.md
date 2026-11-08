# MCP servers and the Paseo daemon

Disclosed reference for the [`fleet`](../SKILL.md) skill. Read it before you
add or change an MCP server, or touch the Paseo daemon or its config. Both
follow the fleet loop: edit agent-fleet, push, converge.

## MCP servers

Declare each fleet MCP server once, in a `[mcp_servers.<name>]` block in
agent-fleet's `.ruler/ruler.toml`. Sync runs Ruler for the personal and work
Claude and Codex profiles; Ruler owns instruction generation and the native MCP
configuration merges. The source is linked at `~/.config/ruler` (or
`$XDG_CONFIG_HOME/ruler`).

- Change a fleet server only in `ruler.toml`. `claude mcp add` or
  `codex mcp add` for a fleet server is drift that reverts on the next
  converge. Servers added by hand under other names survive Ruler's merge.
- Sync resolves stdio commands from the installed fleet tools and skips
  optional tools absent from that machine. Package pins own compatibility.
- Paseo's existing package supplies its CLI and HTTP MCP. Sync reads the
  daemon's listen address from `~/.paseo/config.json`; agents launched by
  Paseo receive its caller-scoped MCP connection automatically.
- Ruler preserves unrelated configuration values; it normalizes Codex TOML
  formatting and drops comments on update. Fleet sync links skill packages,
  so Ruler skill copying stays disabled.
- New sessions pick changes up automatically; restart running sessions.

## Paseo daemon

Every machine runs the CLI daemon from the pixi `paseo` pin under a service
unit, which owns the daemon:

- Linux: `systemctl --user status|restart paseo` (linger required for boot
  start).
- macOS: `launchctl print|kickstart -k gui/$UID/sh.paseo.daemon` (starts at
  login).

Restart through the unit; a hand-run `paseo daemon start` fights it. Check
health with `paseo daemon status`.

Config is one file in git, `agent-fleet/config/paseo/config.json`: edit, push,
converge. `scripts/paseo_apply.py` renders the machine's tailnet IP and
restarts the daemon only on a fleet-initiated change, tracked by the
`.fleet-config-applied` shadow copy. UI edits are drift that reverts on the
next fleet change. State lives in `~/.paseo` (machine-local, survives binary
swaps).

The daemon has no password: auth is the daemon keypair plus relay pairing. One
web UI serves the fleet, `https://pablo-dl-server.ilish-ruler.ts.net:8767`
(PWA); pair a daemon into it with `paseo daemon pair`, then Add connection.
