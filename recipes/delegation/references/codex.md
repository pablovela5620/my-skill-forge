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

## Foreground

Foreground is the default. Set an explicit tool timeout and collect the result
before yielding. Use `--sandbox read-only` for investigation and
`--sandbox workspace-write` for edits.

**Complete when:** Codex has returned and its output has been assessed.

## Durable

Use durable execution when work must outlive the current Claude turn:

1. Start Codex in a uniquely named tmux session.
2. Capture its log and final response with `--output-last-message`.
3. Register a finite wake-up with `paseo heartbeat create`.
4. Verify the tmux session, first log output, report path, and heartbeat before
   promising completion notification.

Track the exact tmux session. A live session remains in progress; a valid
final report is complete; a missing session without a valid report is failed.
Required permission or information is blocked and must name the user's next
action.

Remove the heartbeat after a terminal outcome. Resume the recorded Codex
session when recovery is possible. Launch the primary job before optional
research.

**Complete when:** Codex survival and Claude wake-up are both verified.
