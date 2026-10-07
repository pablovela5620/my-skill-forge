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
| Native app or desktop computer use | Codex, GPT-6 Astra at low effort |
| Implementation, testing, migrations, research, or codebase sweeps | Codex |
| Ambiguous architecture, user-facing judgment, or synthesis | Fable or Opus |
| Independent review | A different runtime from the author |

Codex launch defaults and profile checks live in `references/codex.md`.
Opus and Fable use their default effort. Launch Opus with `model: opus` and
Fable with `model: inherit`.

For computer use, Astra is the fleet preference based on the user's observed
results. Have the worker read the installed `cua-driver` skill before acting.
Run it on the host with the target desktop, verify desktop readiness, and give
each desktop session one active computer-use worker. Explicit user model
choices take precedence.

Capability and taste outrank cost. Escalate when the first worker misses the
bar.

**Complete when:** every delegated stage has an explicit worker and reason.

## 2. Dispatch

- Launch Opus and Fable workers through Agent or Workflow calls. Do not run
  Anthropic models through Paseo.
- Launch Codex workers through Paseo in `auto-review` mode: pass
  `--mode auto-review` (MCP: `settings.modeId: "auto-review"`). Codex's own
  reviewer then decides its sandbox requests. Plain `auto` waits for a human on
  each request, and you cannot approve them. Never switch a running worker's
  mode; the auto-mode classifier blocks that as creating an unsafe agent. Right
  after launch, check `currentModeId` and `pendingPermissions`.
- Never launch a worker in `full-access`. If the reviewer refuses a step the
  task needs (a commit in a git worktree, a write outside the workspace and
  `/tmp`, the GPU), the worker leaves its changes uncommitted and reports the
  step; you do it.
- Before launching Codex, read
  [`references/codex.md`](references/codex.md) completely and follow its
  Paseo dispatch and account selection. Use the MCP when available or the
  installed CLI; Paseo owns the worker's lifetime.
- Give parallel editing workers separate worktrees.
- Launch the primary requested work before optional supporting research.

**Complete when:** the worker has returned a result, or its Paseo agent ID
and collection path are recorded.

## 3. Collect

Read and assess every delegated result. Resume tracked durable work instead of
silently starting a duplicate. Surface the result, failure, or exact blocker
to the user.

**Complete when:** every delegated stage has reported a terminal outcome.
