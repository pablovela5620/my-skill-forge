# Codex delegation from Claude Code

Read this file only after routing work to Codex.

## Boundary

Agent-initiated work uses the Codex CLI. The Claude/Codex plugin remains
limited to ambient hooks and commands invoked by the user.

Use one self-contained prompt. Inherit the configured model and speed tier,
and set the highest supported single-agent CLI effort:

```bash
codex exec -c 'model_reasoning_effort="xhigh"' ...
```

Use `--sandbox read-only` for investigation and `--sandbox workspace-write`
for edits. Start hardware probes in the sandbox; if device isolation blocks
the probe, request approval for only that command — do not broaden the
sandbox.

## Durable execution — the only mode

Never run `codex exec` directly through a tool call: a run the harness
backgrounds at a tool timeout wedges silently and never finishes. Every Codex
launch is durable:

1. Use a teardown-proof tmux server. An agent-spawned tmux server dies with
   the agent session's cgroup, killing every session on it — boot the server
   via `ssh <this-host> 'tmux new-session -d …'` or `systemd-run --user`.
   Use absolute binary paths in ssh-launched scripts (`~/.pixi/bin` is not
   on a non-interactive PATH).
2. Start Codex in a uniquely named tmux session. Redirect its log to a file
   and capture the final response with `--output-last-message`.
3. Pick the wake-up: a background watcher on the output file for work
   collected this session; `paseo heartbeat create` when the work must
   outlive the session.
4. Verify the tmux session, first log output, report path, and wake-up
   before promising completion. On heartbeat wake-up, re-check the status
   sentinel and re-boot the work if it has gone stale.

Track the exact tmux session. A live session remains in progress; a valid
final report is complete; a missing session without a valid report is failed.
Required permission or information is blocked and must name the user's next
action.

Remove the heartbeat after a terminal outcome. Resume the recorded Codex
session when recovery is possible. Launch the primary job before optional
research.

**Complete when:** Codex survival and the wake-up path are both verified, and
the final report has been read and assessed.
