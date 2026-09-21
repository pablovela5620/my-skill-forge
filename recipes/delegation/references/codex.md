# Codex delegation from Claude Code

Read this file only after routing work to Codex.

## Boundary

Agent-initiated work uses the Codex CLI. The Claude/Codex plugin remains
limited to ambient hooks and commands invoked by the user.

## Preflight: identity and model

Check `printenv CLAUDE_CONFIG_DIR` and `claude auth status` in the parent
session's environment. Use the profile directory to select Codex; use the
reported account to catch mismatches:

| Parent profile | Codex profile home |
|---|---|
| Personal Claude: unset or the normal `.claude` directory | `$HOME/.codex` |
| Claude Work: `.claude-work` | `$HOME/.codex-work` |

Resolve custom paths, missing login, or account mismatches before dispatch.
Keep accounts separate: never substitute the other login or copy credentials.
Set `delegation_codex_home` to the selected absolute path on the execution host.
Pass it explicitly through tmux, SSH, and resumes; non-interactive shells may
not have the `codexw` alias.

Default to GPT-6 Astra (`gpt-6-astra`) with low effort; honor explicit user
overrides and retain the configured speed tier:

```bash
env CODEX_HOME="$delegation_codex_home" codex login status
env CODEX_HOME="$delegation_codex_home" codex exec -m gpt-6-astra -c 'model_reasoning_effort="low"' ...
```

`codex login status` must pass, but does not prove the account email.
Include the selected profile and repo identity rules in the worker's prompt.
GitHub is separate: before private-repo access, pushes, or PR changes, check
`gh auth status` and select the account required by the repo's `AGENTS.md`.

**Complete when:** the parent profile is verified, the matching Codex login
passes, and the launch uses the selected home, model, and effort.

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

## Desktop tasks

Use the same Astra/low default for computer use. Include the target host and
application, ask the worker to read `cua-driver`, and require fresh visual or
application-state evidence. Confirm that its execution host has the driver,
skill, graphical session, and OS grants. Keep one active computer-use worker
per desktop session; isolated desktops may run in parallel. An unavailable
model or desktop is a reported blocker, not permission to switch accounts.
