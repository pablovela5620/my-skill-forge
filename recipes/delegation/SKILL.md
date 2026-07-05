---
name: delegation
description: Model routing for delegated work — use when spawning subagents or workflows, choosing a model for a task, or handing work to the Codex runtime. (Claude Code side; other agents rarely delegate.)
---

# Delegation — Model Routing

Rankings (higher = better). Cost reflects actual spend, not list price.

| model    | cost | intelligence | taste |
|----------|------|--------------|-------|
| gpt-5.5  | 9    | 8            | 5     |
| sonnet-5 | 6    | 5            | 7     |
| opus-4.8 | 4    | 8            | 8     |
| fable-5  | 2    | 9            | 9     |

## Routing rules

- Bulk/mechanical work (clear-spec implementation, data analysis,
  migrations): gpt-5.5 via the Codex runtime — very cheap, token efficient.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8; optionally gpt-5.5
  as an extra independent perspective.
- Never use Haiku.
- When axes conflict for anything that ships: intelligence > taste > cost.

## Defaults, not limits

If a cheaper model's output misses the bar, redo with a smarter model
without asking. Judge the output, not the price tag. Escalating costs less
than shipping mediocre work.

## Mechanics

- Claude models: the `model` parameter on Agent/Workflow calls.
- gpt-5.5: the Codex plugin runtime (`/codex:*` commands, `codex:codex-rescue`
  agent type). State the per-stage model routing when launching multi-stage
  work so it can be reviewed.
