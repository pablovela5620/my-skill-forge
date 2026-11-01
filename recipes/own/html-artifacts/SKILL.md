---
name: html-artifacts
description: Create self-contained single-file HTML artifacts when an explanation, plan, report, review, comparison, diagram, deck, prototype, or lightweight editor would be more readable, useful, or reusable in a browser than in markdown. Use for making dense information scannable and beautiful with semantic HTML, inline CSS, inline JavaScript, responsive layout, typography, charts, tables, annotations, tabs, collapsibles, timelines, diagrams, and copy/export affordances. Especially use when the user wants an artifact they are likely to open, read, share, or iterate on directly as a .html file.
license: BSD-3-Clause
---

# Single-File HTML Artifacts

Build one browser-openable `.html` file when HTML will communicate the work better than markdown. The goal is not a tiny website or a decorative report. The goal is a readable, useful artifact the user will actually open: clear structure, strong hierarchy, enough interactivity to reduce cognitive load, and enough visual craft to make the work inviting.

This skill is inspired by Impeccable's frontend design discipline and by the "unreasonable effectiveness of HTML" pattern: use the browser as a rich document medium for plans, reviews, explainers, reports, diagrams, decks, and small editing tools. Borrow Impeccable's taste, structure, and anti-slop standards, not its command system: keep the artifact agent-agnostic, inline, and directly openable.

## When to Use HTML

Prefer a single HTML artifact when the content has any of these shapes:

- Multiple options to compare side by side.
- A process, timeline, architecture, dependency graph, or decision tree.
- A code review, PR summary, or implementation plan with many references.
- A research explainer with glossary terms, examples, tabs, or progressive disclosure.
- A report where charts, status bands, or summaries would help scanning.
- A small interactive tool or editor where the user should manipulate inputs and export the result.
- A deck-like story that benefits from keyboard navigation.

Do not use HTML just to wrap a normal answer in decoration. If the artifact would still read like a wall of prose, restructure it first. Transform the material into a browser-native reading object: summary bands, comparison tables, timelines, diagrams, filters, tabs, collapsibles, annotations, and copy/export affordances.

## Output Contract

Unless the user asks otherwise:

- Produce exactly one `.html` file.
- Keep it self-contained: inline CSS and inline JavaScript by default.
- Make it open directly from the filesystem, with no build step and no server.
- Use semantic landmarks, headings, labels, buttons, and tables.
- Make it responsive from phone width to desktop.
- Do not depend on external libraries, fonts, or images unless the user explicitly allows remote assets.
- Use inline SVG, CSS, and small data structures for diagrams and charts.
- Include export/copy controls for editor-like artifacts.
- Keep text real and specific. No placeholder lorem ipsum, fake links, or dead controls.
- Add `<link rel="icon" href="data:,">` to the `<head>`, so the browser does not request a missing favicon.
- Exception: a page that embeds a Rerun recording loads the shared viewer and `embed.js` from the NAS. Read [references/nas-publishing.md](references/nas-publishing.md) before you publish to the NAS or embed an `.rrd`.

## Workflow

1. Decide the artifact type.
   - If the structure is not obvious, read [references/patterns.md](references/patterns.md).
   - Pick the pattern that makes the user's next decision or reading path easiest.

2. Shape the information before styling.
   - Identify the primary question the artifact answers.
   - Put the answer, status, or decision at the top.
   - Group details by how the user will scan them, not by how the source material arrived.
   - Convert long prose into tables, timelines, callouts, diagrams, tabs, or collapsibles only when those structures reduce effort.

3. Apply the house style and the design quality bar.
   - The house style is Rerun's: read [references/DESIGN.md](references/DESIGN.md) (a [DESIGN.md](https://github.com/google-labs-code/design.md) file: tokens plus rules) and start from [assets/rerun.css](assets/rerun.css). Draw diagrams with [assets/rerun_figure.py](assets/rerun_figure.py). Where DESIGN.md and design-quality.md disagree, DESIGN.md wins.
   - Use a different style only when the user asks for one or the artifact belongs to another brand.
   - Read [references/design-quality.md](references/design-quality.md) before substantial visual work.
   - Write the scene sentence before theme, palette, density, or motion: the user opens this artifact on what device/context, with what urgency, to decide what.
   - Use restrained but intentional styling. Avoid generic AI-looking decoration.
   - Prefer a dark, tinted reading surface when the scene allows it. Switch to light when the user asks for it, the source system requires it, or the artifact is primarily for printing, bright-room projection, or daytime public sharing.

4. Build the file.
   - Use CSS custom properties for color, spacing, type, and radii.
   - Use readable measures for prose, usually 65-75 characters.
   - Use stable layout primitives: grid, flex, sticky sidebars, tabs, details/summary, tables, and inline SVG.
   - Add small JavaScript only when it materially improves use: search, filters, tabs, copy buttons, slide navigation, simple editors.

5. Verify.
   - Open or screenshot the file when tooling is available.
   - Check at 1400 px (desktop), 1133x744 and 744x1133 (iPad mini) and 390 px (phone). Keep the layout fluid from 700 to 1200 px; make touch targets at least 44x44 px.
   - Open every embedded Rerun viewer in GPU Chrome at those sizes and look at the pixels (see [references/nas-publishing.md](references/nas-publishing.md)).
   - Check keyboard navigation for controls.
   - Check there is no horizontal overflow, clipped text, broken asset, console error, or unreadable contrast.

## Design Defaults

- Dark by default, in the Rerun house style: pure black page, neutral greys, hairline borders, colour only on data. Outside the house style, dark means tinted near-black surfaces, clear contrast, visible borders, and restrained accents.
- Include print-friendly light styling for print-heavy plans, reports, and review artifacts.
- Use one accent color plus tinted neutrals unless the content genuinely needs a broader palette.
- Use system fonts: the system sans for text, JetBrains Mono (falling back to the system mono) for labels, buttons and code, as DESIGN.md sets out. Load no web fonts.
- Use motion sparingly: 150-250ms, ease-out, state feedback only. Respect `prefers-reduced-motion`.
- Avoid hover-only meaning. Every control must work with keyboard and touch.

## Hard Bans

- No gradient text.
- No glassmorphism as decoration.
- No dark mode with glowing neon accents.
- No nested card stacks.
- No endless identical card grids.
- No everything-centered layouts except for a true deck, poster, or presentation moment.
- No pure black or pure white as the full visual system, except the Rerun house style's black page with neutral greys. Elsewhere, tint neutrals subtly.
- No gray text on saturated colored backgrounds.
- No oversized hero section unless the artifact is explicitly a deck, poster, or landing-style narrative.
- No rounded-square icon tile above every heading.
- No modal-first interaction for content that should be browsed, compared, or scanned.
- No all-primary button rows.
- No decorative fake charts, sparklines, or metrics.
- No monospace "technical" styling unless code, paths, commands, logs, or structured data are actually being shown, or it is the house style's uppercase mono labels and buttons.
- No redundant intro paragraphs that restate the heading.
- No visible instructional text explaining that the page is an HTML artifact or how the design works.
- No lorem ipsum, fake links, placeholder content, or controls that do nothing.

## File Naming

Use a descriptive, stable filename in the workspace, for example:

- `implementation-plan.html`
- `code-review-map.html`
- `incident-timeline.html`
- `research-explainer.html`
- `option-comparison.html`

If the user names a path, use it. Otherwise choose a filename that matches the artifact's job.
