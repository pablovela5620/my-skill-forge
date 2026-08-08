# Update contract

Keep `pixi-only.patch` minimal while syncing upstream. Preserve one co-located
Pixi-only setup section and adapt conflicting examples to the same process:

1. Inspect `pixi.toml` and its declared conda channels, normally conda-forge.
2. Use `pixi run` for workspace tools, `pixi exec` for one-off tools, and
   `pixi add` for recurring tools.
3. Use `pixi add --pypi` only when the declared conda channels lack a package.
4. Apply this process to linked chapters.

The update is complete when the patch applies cleanly, the installed-skill
tests in `recipe.yaml` pass, and every upstream `SKILL.md` change is preserved
or intentionally adapted by this policy.
