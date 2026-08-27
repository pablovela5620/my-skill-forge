---
name: delegation
description: "Closed-loop delegation from Claude Code: use when choosing between the current Claude session, Opus workers, and Codex."
---

# Delegation

Every delegation is closed-loop: route, dispatch, collect.

## Scope

This skill governs Claude Code. When the host runtime is Codex, stop here and
follow its active delegation instructions and native tools.

## 1. Route

Keep small or tightly coupled work in the current Claude session. Delegate work
that benefits from separation, specialization, or parallel execution.

Use only Codex, Opus, or Fable:

| Work | Worker |
|---|---|
| Implementation, testing, migrations, research, or codebase sweeps | Codex |
| Ambiguous architecture, user-facing judgment, or synthesis | Fable or Opus |
| Independent review | A different runtime from the author |

Codex uses the configured model and speed tier with
`model_reasoning_effort="xhigh"`. Ultra is outside this single-agent policy.
Opus and Fable use their default effort. Launch Opus with `model: opus` and
Fable with `model: inherit`.

Capability and taste outrank cost. Escalate when the first worker misses the
bar.

**Complete when:** every delegated stage has an explicit worker and reason.

## 2. Dispatch

- Launch Claude workers through Agent or Workflow calls.
- Before launching Codex, read
  [`references/codex.md`](references/codex.md) completely and follow its
  durable execution contract — never a bare `codex exec` tool call.
- Give parallel editing workers separate worktrees.
- Launch the primary requested work before optional supporting research.

**Complete when:** the worker has returned a result, or its durable execution
and wake-up path are verified.

## 3. Collect

Read and assess every delegated result. Resume tracked durable work instead of
silently starting a duplicate. Surface the result, failure, or exact blocker
to the user.

**Complete when:** every delegated stage has reported a terminal outcome.
