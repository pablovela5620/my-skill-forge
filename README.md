# my-skill-forge

[![my-skill-forge](https://img.shields.io/badge/prefix.dev%2Fmy--skill--forge-F7CC49?style=flat-square)](https://prefix.dev/channels/my-skill-forge)
[![my-skill-forge](https://img.shields.io/badge/pablovela5620%2Fmy--skill--forge-181717?style=flat-square&logo=github)](https://github.com/pablovela5620/my-skill-forge)

A collection of agent skills packaged as conda packages and published to the [my-skill-forge](https://prefix.dev/channels/my-skill-forge) channel on prefix.dev.

Agent skills are markdown files that give AI coding agents specialized knowledge about libraries, tools, and domains.
They are managed by [pixi-skills](https://github.com/pavelzw/pixi-skills) and can be installed into any pixi project.

For more background on why distributing agent skills through package managers makes sense, check out the blog post [Managing Agent Skills with Your Package Manager](https://pavel.pink/blog/pixi-skills).

## Available skills

| Skill | Package | Description |
|-------|---------|-------------|
| [chrome-cdp](https://github.com/pasky/chrome-cdp-skill) | `agent-skill-chrome-cdp` | Chrome DevTools Protocol skill for live local browser sessions |
| [conda-forge](https://conda-forge.org) | `agent-skill-conda-forge` | conda-forge packaging operations |
| [create-node](https://github.com/pablovela5620/my-skill-forge) | `agent-skill-create-node` | Create single-purpose CV nodes with API, Gradio UI, Rerun viewer, and CLI |
| [daggr](https://github.com/gradio-app/daggr/tree/main/.agents/skills/daggr) | `agent-skill-daggr` | Build visual DAG-based AI pipelines with Gradio Spaces, Hugging Face models, and Python functions |
| [gh-cli](https://github.com/github/awesome-copilot/tree/main/skills/gh-cli) | `agent-skill-gh-cli` | GitHub CLI workflows and command reference |
| [grill-me](https://skills.sh/mattpocock/skills/grill-me) | `agent-skill-grill-me` | Stress-test plans and designs with a focused planning interview |
| [grill-with-docs](https://skills.sh/mattpocock/skills/grill-with-docs) | `agent-skill-grill-with-docs` | Stress-test plans against existing domain documentation and capture glossary or ADR decisions |
| [hf-cli](https://github.com/huggingface/skills/tree/main/skills/hf-cli) | `agent-skill-hf-cli` | Hugging Face Hub CLI workflows and command reference |
| [html-artifacts](https://github.com/pablovela5620/my-skill-forge) | `agent-skill-html-artifacts` | Readable single-file HTML artifacts for plans, reports, reviews, diagrams, decks, and editors |
| [karpathy-guidelines](https://github.com/multica-ai/andrej-karpathy-skills/tree/main/skills/karpathy-guidelines) | `agent-skill-karpathy-guidelines` | Behavioral guidelines to reduce common LLM coding mistakes |
| [playwright-cli](https://github.com/microsoft/playwright-cli/tree/main/skills/playwright-cli) | `agent-skill-playwright-cli` | Browser automation and Playwright test workflows |
| [port-model](https://github.com/facebook/pyrefly/tree/main/.claude/skills/port-model) | `agent-skill-port-model` | Port PyTorch models to pyrefly tensor shape types |
| [rattler-build](https://rattler.build) | `agent-skill-rattler-build` | Build conda packages with rattler-build |
| [rerun-viewer-validation](https://github.com/pablovela5620/my-skill-forge) | `agent-skill-rerun-viewer-validation` | Validate Rerun 0.33 viewer output with native headless screenshots and size-limited WebViewer reports |
| [tdd](https://github.com/mattpocock/skills/tree/main/skills/engineering/tdd) | `agent-skill-tdd` | Test-driven development with red-green-refactor workflows |

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
