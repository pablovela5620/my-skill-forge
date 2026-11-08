# Codex delegation from Claude Code

Paseo runs durable Codex workers. The MCP and CLI control the same daemon; the
worker survives the calling command. Read this file after routing to Codex.

## Account

Check `printenv CLAUDE_CONFIG_DIR` and `claude auth status` in the parent
session. Match that account on the execution host:

| Parent account | Paseo provider | Codex home |
|---|---|---|
| Personal Claude | `codex` | `$HOME/.codex` |
| Claude Work | `codex-work` | `$HOME/.codex-work` |

Fleet config gives each provider its own account directory. If the matching
login is unavailable, report it; never substitute the other provider or copy
credentials. Resolve custom profile paths or an account mismatch before
dispatch. Use `paseo provider diagnostic <provider> --json` on the selected
daemon to verify the executable and provider readiness. On that host, verify
the matching `CODEX_HOME` with `codex login status`; it does not prove the
account email.

Provider login and GitHub login are separate: include the repo's GitHub
identity rules in the prompt.

## Launch settings

Use the fleet defaults: `gpt-6.1-sol`, low thinking, and `auto-review` mode.
Inherit the service tier (speed) from the selected Codex home, where fleet
config sets it; a shared launch profile's speed setting does not replace it.
If the selected tier is unsupported or omitted, report it rather than claiming
it is active. Honor explicit user model, effort, and speed choices.

Pass `auto-review` at launch for every task, reviews and investigation
included; a read-only task says so in its prompt (`--mode auto-review`; MCP:
`settings.modeId: "auto-review"`). `auto-review` keeps the workspace-write
sandbox and routes eligible requests to Codex's auto-reviewer: a commit in a
git worktree, whose git dir is outside the workspace; a write outside the
workspace and `/tmp`; the GPU. Plain `auto` waits for a human on each request,
and the parent session cannot approve them. Right after launch, check
`currentModeId` and `pendingPermissions`.

Set the mode only at launch: the auto-mode classifier blocks switching a
running worker's mode as creating an unsafe agent. Never use `full-access`. If
a hardware probe needs broader access, request approval for only that command.
If the reviewer refuses a step, the worker leaves its changes uncommitted and
reports it; the parent does that step.

## Dispatch

Use the `paseo` MCP when its tools are available. Call `list_profiles` to read
any configured launch bundles; retain the matching personal/work provider.
Use `list_providers`, `list_models`, or `inspect_provider` when settings need
verification. Create or select the execution workspace, then call
`create_agent` with its workspace ID, provider/model, initial prompt, and
`settings.thinkingOptionId` / `settings.modeId` (`auto-review`). A profile is
launch settings, not a `profile` argument. If it defines features, copy them to
`settings.features`.

Use the installed CLI when MCP tools are unavailable:

```bash
paseo run --background --provider codex --model gpt-6.1-sol --thinking low --mode auto-review --cwd /absolute/repo "<task and acceptance criteria>" --json
paseo wait <agent-id> --timeout 60 --json
paseo inspect <agent-id> --json
paseo logs <agent-id> --tail 40 --json
paseo send <agent-id> "<follow-up>"
```

Choose `codex-work` for the work account. An agent-scoped `paseo run` runs in
the caller's workspace and ignores `--cwd`, so a Codex worker can write only
there and in `/tmp`. Give a worker that writes elsewhere its own workspace
(`--new-workspace`), or name a `/tmp` output path. For another machine, use
`paseo --host <target> ...` and a directory on that host; discover targets with
the fleet skill. `paseo run --help` lists the workspace options.

## Collect

Record the daemon target, agent ID, provider, and workspace. In Paseo-hosted
sessions, keep `notifyOnFinish` enabled and use the completion notification.
Outside Paseo, wait in bounded calls, inspect status, and read the final
activity with `get_agent_activity` or CLI logs. Idle alone does not prove the
task succeeded: assess the result and acceptance criteria.

Resume the recorded agent with `send_agent_prompt` or `paseo send` instead
of launching a duplicate. Report a permission request or failure with the next
required action. Use a heartbeat only when the user asks for follow-up that
outlives this conversation, and remove it when that work ends.

## Desktop tasks

Use the same Sol/low defaults and the selected Codex home's speed for computer
use. Name the target host and application in the prompt, and require fresh
visual or application-state evidence. Run the worker on the host with the
target desktop, and confirm that the host has a graphical session and the OS
grants before it acts. Keep one active computer-use worker per desktop session;
isolated desktops may run in parallel.
