---
name: delegation
description: Model routing for delegated work — use when spawning subagents or workflows, choosing a model for a task, or handing work to the Codex runtime. (Claude Code side; other agents rarely delegate.)
---

# Delegation — Model Routing

Rankings (higher = better). Cost reflects actual spend (OpenAI limits are
very generous), not list price. Intelligence is how hard a problem you can
hand the model unsupervised. Taste covers UI/UX, code quality, API design,
and copy.

| model    | cost | intelligence | taste |
|----------|------|--------------|-------|
| gpt-5.5  | 9    | 8            | 5     |
| sonnet-5 | 6    | 5            | 7     |
| opus-4.8 | 4    | 8            | 8     |
| fable-5  | 2    | 9            | 9     |

## Routing rules

- Bulk/mechanical work (clear-spec implementation, data analysis, migrations,
  convention mining, codebase sweeps): gpt-5.5 via the Codex runtime — very
  cheap and token efficient.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8; optionally gpt-5.5
  as an extra independent perspective.
- Never use Haiku.
- When axes conflict for anything that ships: intelligence > taste > cost.
- Claude stays for synthesis, judgment, and taste stages — the orchestrator,
  not the workhorse.

## Defaults, not limits

These are defaults with standing permission to override: if a cheaper
model's output misses the bar, rerun or redo with a smarter model WITHOUT
asking. Judge the output, not the price tag. Escalating costs less than
shipping mediocre work.

## The habitual failure mode

The recurring violation is not picking the wrong model deliberately — it is
running bulk work on Claude *by default* because Claude is the runtime you
are already in. Spawning a Claude subagent for a codebase sweep, or running
every Workflow stage on the session model, both feel natural and both
misroute. Before ANY delegation (Agent call, Workflow stage, background
task), classify first: is this bulk/mechanical? Then it goes to gpt-5.5,
and staying on a Claude model requires a reason, not a habit. When launching
multi-stage work, state the per-stage model routing up front so it can be
reviewed before tokens burn.

## Mechanics

gpt-5.5 runs natively via the `openai/codex-plugin-cc` plugin, adopting
user-level config from `~/.codex/config.toml`. Use the plugin's tools —
never hand-rolled bash wrappers around the codex CLI:

- `/codex:review` — non-destructive read-only code quality assessment;
  supports `--base <ref>` for branch analysis.
- `/codex:adversarial-review` — skeptical design review pressure-testing
  tradeoffs, auth, and reliability; append focus text to steer it.
- `/codex:rescue` — subcontract active debugging, multi-file refactoring,
  or implementation loops when a second pass is required.
- `/codex:status` / `/codex:result` / `/codex:cancel` — check, fetch, or
  abort asynchronous jobs when using `--background` on heavy tasks.

Inside Workflows and subagents: delegate via the plugin's slash commands or
its exposed `codex-cli-runtime` skills directly — bulk stages route to
Codex; Claude subagents take only the synthesis/judgment stages.

Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent/Workflow
`model` parameter.

Closed-loop QA: keep the review gate on via
`/codex:setup --enable-review-gate` — a stop hook challenges Claude's output
with Codex before it reaches the main session unvetted.
