---
name: delegation
description: Model routing for delegated work — use when spawning subagents or workflows, choosing a model for a task, or handing work to the Codex runtime. (Claude Code side; other agents rarely delegate.)
---

# Delegation — Model Routing

## General preferences

- If asked to do too much work at once, stop and state that clearly.
- If computer use is helpful for completing or verifying work, shell out to
  gpt-5.5 with Codex for it.

## Picking the right models for workflows and subagents

Rankings, higher = better. Cost reflects what I actually pay (OpenAI has
really generous limits), not list price. Intelligence is how hard a problem
you can hand the model unsupervised. Taste covers UI/UX, code quality, API
design, and copy.

| model    | cost | intelligence | taste |
|----------|------|--------------|-------|
| gpt-5.5  | 9    | 8            | 5     |
| sonnet-5 | 6    | 5            | 7     |
| opus-4.8 | 4    | 8            | 8     |
| fable-5  | 2    | 9            | 9     |

How to apply:

- These are defaults, not limits. You have standing permission to override
  them: if a cheaper model's output doesn't meet the bar, rerun or redo the
  work with a smarter model without asking. Judge the output, not the price
  tag. Escalating costs less than shipping mediocre work.
- Don't let cost prevent you from using the right model for the job.
  Instead, take advantage of cheaper options to get more information and
  try things before moving the work to a more expensive option.
- Cost is a tie-breaker only; when axes conflict for anything that ships,
  intelligence > taste > cost.
- Bulk/mechanical work (clear-spec implementation, data analysis,
  migrations, convention mining, codebase sweeps): gpt-5.5 — it's very
  cheap and token efficient.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8, optionally gpt-5.5
  as an extra independent perspective.
- Never use Haiku.
- Claude stays for synthesis, judgment, and taste stages — the orchestrator,
  not the workhorse.

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

gpt-5.5 is handled natively via the `openai/codex-plugin-cc` plugin inside
Claude Code, automatically adopting your user-level configuration from
`~/.codex/config.toml`. Avoid writing custom bash scripts; instead, utilize
the plugin's built-in tools and skills:

- `/codex:review` — run non-destructive, read-only code quality
  assessments. Supports `--base <ref>` for branch analysis.
- `/codex:adversarial-review` — perform a skeptical design review to
  pressure-test tradeoffs, auth, and reliability. Append custom focus text
  at the end of the command to steer the focus.
- `/codex:rescue` — subcontract active debugging, multi-file refactoring,
  or implementation loops to Codex when a second pass is required.
- `/codex:status` / `/codex:result` / `/codex:cancel` — use these to check,
  fetch, or abort asynchronous jobs when using the `--background` flag on
  heavy tasks.

Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent/Workflow
`model` parameter.

Sandbox & network (verified empirically 2026-07-08): Codex network access
is a CONFIG SETTING, not a fixed limit. Modes: `read-only` (no network),
`workspace-write` (network off by default; on when the machine's
`~/.codex/config.toml` sets `[sandbox_workspace_write] network_access =
true`, or per-run via `-c sandbox_workspace_write.network_access=true`),
`danger-full-access` (no boundary). With network on, both web fetches and
outbound SSH work from inside the sandbox — proven with live runs. So:

- Before rerouting a network-needing stage to a Claude agent, check the
  machine's codex config — on network-enabled machines Codex handles ssh/
  web stages fine. The banner line of every run states the effective mode.
- The flip side: a network-enabled sandbox makes delegated prompts an
  injection surface. Never feed untrusted content (web pages, third-party
  code) into a Codex run that can also reach the tailnet.
- Fleet state (2026-07-08): M5 + dl-server set `network_access = true`;
  spark / mini / ubuntu run the network-off default — one config line to
  change, per machine, if a stage there needs egress.

Using gpt-5.5 inside workflows and subagents:

- Subagents and automated workflows should call the plugin's native slash
  commands or its exposed `codex-cli-runtime` skills to delegate tasks
  directly, omitting the need for raw terminal wrappers.
- For closed-loop quality assurance, keep the review gate turned on via
  `/codex:setup --enable-review-gate`. This ensures a stop hook
  automatically challenges Claude's outputs using Codex before finalizing,
  preventing broken code or weak design assumptions from reaching the main
  session unvetted.
