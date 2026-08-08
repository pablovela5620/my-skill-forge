# my-skill-forge

[![my-skill-forge](https://img.shields.io/badge/prefix.dev%2Fmy--skill--forge-F7CC49?style=flat-square)](https://prefix.dev/channels/my-skill-forge)
[![my-skill-forge](https://img.shields.io/badge/pablovela5620%2Fmy--skill--forge-181717?style=flat-square&logo=github)](https://github.com/pablovela5620/my-skill-forge)

A collection of agent skills packaged as conda packages and published to the [my-skill-forge](https://prefix.dev/channels/my-skill-forge) channel on prefix.dev.

Agent skills are markdown files that give AI coding agents specialized knowledge about libraries, tools, and domains.
They are managed by [pixi-skills](https://github.com/pavelzw/pixi-skills) and can be installed into any pixi project.

For more background on why distributing agent skills through package managers makes sense, check out the blog post [Managing Agent Skills with Your Package Manager](https://pavel.pink/blog/pixi-skills).

## Skills

The [Prefix channel](https://prefix.dev/channels/my-skill-forge) is the
published package catalog. Source recipes and their pinned upstreams live in
[`recipes/`](recipes/). Recipe metadata is canonical; this README does not copy
the catalog.

## Usage

### Managing skills with pixi-skills

The recommended way to use agent skills is through [pixi-skills](https://github.com/pavelzw/pixi-skills).
Install it with:

```bash
pixi exec pixi-skills manage
```

This will interactively guide you through adding skills to your project.

### Manual setup

Add the `my-skill-forge` channel and the desired skill packages to your `pixi.toml`:

```toml
[workspace]
channels = ["conda-forge", "https://prefix.dev/my-skill-forge"]
platforms = ["linux-64", "osx-arm64", "win-64"]

[dependencies]
rattler-build = "*"

[feature.dev.dependencies]
agent-skill-rattler-build = "*"
```
