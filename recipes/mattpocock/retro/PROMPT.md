Mirror `skills/engineering/retro` from mattpocock/skills.

Keep `use-agent-traces.patch` as a focused adaptation of the upstream steps:

- The agent-traces Rerun catalog is the primary evidence source, read through
  `scripts/retro_pull.py` (copied from this folder) and the
  `rerun-catalog-queries` skill. Point to the agent-traces README for the
  layout; do not copy it.
- The live transcript fills what the catalog lacks: the current session alone,
  or sessions newer than the last catalog refresh.
- Keep upstream's categories, the presentation step and the Reference section.

When updating, inspect the upstream changes and refresh the patch if needed.
Do not install this file or the patch as skill content.
