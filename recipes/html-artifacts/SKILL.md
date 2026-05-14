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

## Workflow

1. Decide the artifact type.
   - If the structure is not obvious, read [references/patterns.md](references/patterns.md).
   - Pick the pattern that makes the user's next decision or reading path easiest.

2. Shape the information before styling.
   - Identify the primary question the artifact answers.
   - Put the answer, status, or decision at the top.
   - Group details by how the user will scan them, not by how the source material arrived.
   - Convert long prose into tables, timelines, callouts, diagrams, tabs, or collapsibles only when those structures reduce effort.

3. Apply the design quality bar.
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
   - Check mobile and desktop widths.
   - Check keyboard navigation for controls.
   - Check there is no horizontal overflow, clipped text, broken asset, console error, or unreadable contrast.

## Design Defaults

- Prefer a dark theme when the artifact's reading context does not require light. Dark does not mean black: use tinted near-black surfaces, clear contrast, visible borders, and restrained accents.
- Include print-friendly light styling for print-heavy plans, reports, and review artifacts.
- Use one accent color plus tinted neutrals unless the content genuinely needs a broader palette.
- Prefer system fonts for utility and report artifacts. Use a distinctive type pairing only when the artifact is closer to editorial, deck, or brand work.
- Use motion sparingly: 150-250ms, ease-out, state feedback only. Respect `prefers-reduced-motion`.
- Avoid hover-only meaning. Every control must work with keyboard and touch.

## Hard Bans

- No gradient text.
- No glassmorphism as decoration.
- No dark mode with glowing neon accents.
- No nested card stacks.
- No endless identical card grids.
- No everything-centered layouts except for a true deck, poster, or presentation moment.
- No pure black or pure white as the full visual system. Tint neutrals subtly.
- No gray text on saturated colored backgrounds.
- No oversized hero section unless the artifact is explicitly a deck, poster, or landing-style narrative.
- No rounded-square icon tile above every heading.
- No modal-first interaction for content that should be browsed, compared, or scanned.
- No all-primary button rows.
- No decorative fake charts, sparklines, or metrics.
- No monospace "technical" styling unless code, paths, commands, logs, or structured data are actually being shown.
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
