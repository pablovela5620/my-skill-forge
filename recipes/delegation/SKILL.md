---
name: delegation
description: "Delegation from Claude Code: use before spawning a subagent, Workflow, or Codex worker, or when choosing which model runs a task."
---

# Delegation

Every delegation is closed-loop: route, dispatch, collect.

This skill governs Claude Code. When the host runtime is Codex, follow Codex's
own delegation instructions and native tools instead.

## 1. Route

Keep small or tightly coupled work in the current Claude session. Delegate work
that benefits from separation, specialization, or parallel execution. Workers
are Codex, Opus, or Fable:

| Work | Worker |
|---|---|
| Native app or desktop computer use | Codex, GPT-6.1 Sol at low effort |
| Implementation, testing, migrations, research, or codebase sweeps | Codex |
| Ambiguous architecture, user-facing judgment, or synthesis | Fable or Opus |
| Independent review | A different runtime from the author |

Launch Opus with `model: opus` and Fable with `model: inherit`, at their
default effort. Opus 5.5 defaults to `medium`, which matches or beats Opus 5 at
`high`; reserve `xhigh` and `max` for work where a quality gain has been
measured. Codex launch defaults live in
[`references/codex.md`](references/codex.md).

Explicit user model and effort choices take precedence. Capability and taste
outrank cost: when the first worker misses the bar, escalate to a stronger one.

**Complete when:** every delegated stage has an explicit worker and reason.

## 2. Dispatch

- Launch Opus and Fable workers through Agent or Workflow calls. Anthropic
  models run there, not through Paseo.
- Launch Codex workers through Paseo in `auto-review` mode; never
  `full-access`. Before launching, read
  [`references/codex.md`](references/codex.md) completely and follow its
  account selection, mode rules, and dispatch. Paseo owns the worker's
  lifetime.
- Give parallel editing workers separate worktrees.
- Launch the primary requested work before optional supporting research.
- Brief every unattended worker with its completion condition up front, and
  have it keep the task's parts in a checklist it updates (a to-do tool or a
  file).
- When a worker leads its own team (a Workflow lead, or a worker that spawns
  subagents) and you can estimate the duration, give it a time budget and
  end each message you send back with the elapsed time against it, in seconds
  (`elapsed 340s / 1200s`). Set the budget somewhat above the time you want
  spent. The budget is advisory, so keep your own timeout. Without an
  estimate, tell it that time matters and an earlier correct result is better.

**Complete when:** the worker has returned a result, or its Paseo agent ID
and collection path are recorded.

## 3. Collect

Read and assess every delegated result against its completion condition. A
text-only end of turn is a report, not proof that the task is done:

- If checklist items are still open and the worker names no blocker, send one
  short message that names the open items and asks it to continue or say what
  blocks it. Stop after two or three automatic continuations on the same task
  and report it as stuck.
- If the worker started something that is still running (a background command,
  a subagent), wait for its output before you judge the task.

Resume tracked durable work instead of silently starting a duplicate. Surface
the result, failure, or exact blocker to the user.

**Complete when:** every delegated stage has a terminal outcome: done against
its completion condition, failed, or blocked with the blocker named.
