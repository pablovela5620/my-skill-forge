Mirror `tools/skills/svelte-code-writer` from sveltejs/ai-tools.

Keep `use-fleet-mcp.patch` as a focused adaptation of the upstream workflow:

- Use the connected, fleet-managed Svelte MCP tools instead of downloading or
  running `@sveltejs/mcp` through npx. Agent-fleet owns the server configuration.
- Pass source text to the remote autofixer; it cannot read local file paths.
- Preserve documentation lookup, project-version selection, and the validation
  loop. Take tool argument names from the current server schema.
- Do not require the upstream plugin's `svelte-file-editor` subagent, which is
  not installed by these skill packages. Follow the caller's delegation rules.
- Keep build, type, and browser checks distinct from static autofixer analysis.

When updating, inspect the upstream changes and refresh the patch if needed.
Package only the patched skill subtree. Do not install this file or the patch
as skill content, and do not add another copy of the workflow to AGENTS.md.
