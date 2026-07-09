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

## Codex mechanics

gpt-5.5 runs through the Codex CLI: `codex exec` for work, `codex review` for
reviews. The claude-code plugin stays out of the agent's model of the world —
its stop-time review gate runs as an ambient hook, and its slash commands are
for the human to type. Route everything agent-initiated through the CLI:

- One self-contained prompt per run: `codex exec -s read-only "..."` for
  investigation/analysis, `codex exec --sandbox workspace-write "..."` for
  edits. Long-running goals: `nohup codex exec ... > /tmp/<job>.log &`,
  observed via that log and the rollout jsonl under
  `~/.codex/sessions/<y>/<m>/<d>/`.
- One goal = one session. Parallel sessions sharing a state file (a
  PROGRESS.md, a scratch dir) stomp each other's checkpoints — kill the old
  session (`pgrep -f "codex exec"`) before relaunching with a steering note.
- Codex runs can exceed Bash's 10-minute timeout: pass an explicit timeout,
  or run in the background and poll for the report file.

Using gpt-5.5 inside workflows and subagents (the model parameter only takes
Claude models, so use a wrapper):

- Spawn a thin Claude wrapper agent (`model: 'sonnet', effort: 'low'`) whose
  prompt instructs it to write a self-contained codex prompt, run
  `codex exec` via Bash, and return the report (use `schema` on the wrapper
  for structured output back).
- Always label these agents with a `gpt-5.5:` prefix, e.g.
  `{label: 'gpt-5.5:review-auth'}` — the workflow UI shows the wrapper's
  Claude model, so the label is the only indication the real worker is
  gpt-5.5.
- Parallel gpt-5.5 implementation agents must use `isolation: 'worktree'` so
  codex edits don't collide in the shared checkout.
- Workflow token budgets only count Claude tokens; codex work is free and
  invisible to `budget.spent()`.

Sandbox (workspace-write) — verified empirically 2026-07-09:

- Network is a config setting — check it, don't remember it:
  `grep -A1 sandbox_workspace_write ~/.codex/config.toml` on the machine you
  are on, or enable per-run with
  `-c sandbox_workspace_write.network_access=true`. The banner line of every
  run states the effective mode.
- GPU adapters, binding local sockets, and killing processes are NOT
  grantable in workspace-write; outbound localhost is allowed. For goals
  that need GPU, eval sweeps, or process control, front-load the access
  decision: ask the user for `--sandbox danger-full-access` at launch time.
  A relay — the sandboxed agent writing a script for the orchestrator to
  execute unsandboxed — is permission laundering; the permission classifier
  blocks it. Alternative when full access isn't warranted: a server split,
  where the orchestrator owns the long-lived GPU process unsandboxed and
  sandboxed codex connects to it over localhost.
- A network-enabled sandbox makes delegated prompts an injection surface.
  Never feed untrusted content (web pages, third-party code) into a codex
  run that can also reach the tailnet.
