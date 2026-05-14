---
name: html-artifacts
description: Create self-contained single-file HTML artifacts when an explanation, plan, report, review, comparison, diagram, deck, prototype, or lightweight editor would be more readable, useful, or reusable in a browser than in markdown. Use for making dense information scannable and beautiful with semantic HTML, inline CSS, inline JavaScript, responsive layout, typography, charts, tables, annotations, tabs, collapsibles, timelines, diagrams, and copy/export affordances. Especially use when the user wants an artifact they are likely to open, read, share, or iterate on directly as a .html file.
license: BSD-3-Clause
---

# Single-File HTML Artifacts

Build one browser-openable `.html` file when HTML will communicate the work better than markdown. The goal is not a tiny website or a decorative report. The goal is a readable, useful artifact the user will actually open: clear structure, strong hierarchy, enough interactivity to reduce cognitive load, and enough visual craft to make the work inviting.

This skill is inspired by Impeccable's frontend design discipline and by the "unreasonable effectiveness of HTML" pattern: use the browser as a rich document medium for plans, reviews, explainers, reports, diagrams, decks, and small editing tools.

## When to Use HTML

Prefer a single HTML artifact when the content has any of these shapes:

- Multiple options to compare side by side.
- A process, timeline, architecture, dependency graph, or decision tree.
- A code review, PR summary, or implementation plan with many references.
- A research explainer with glossary terms, examples, tabs, or progressive disclosure.
- A report where charts, status bands, or summaries would help scanning.
- A small interactive tool or editor where the user should manipulate inputs and export the result.
- A deck-like story that benefits from keyboard navigation.

Do not use HTML just to wrap a normal answer in decoration. If the artifact would still read like a wall of prose, restructure it first.

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
   - Use restrained but intentional styling. Avoid generic AI-looking decoration.
   - Choose a physical reading context before theme and palette: where the user opens this, under what light, with what urgency.

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

- Use a light theme unless the artifact's reading context argues otherwise.
- Use one accent color plus tinted neutrals unless the content genuinely needs a broader palette.
- Prefer system fonts for utility and report artifacts. Use a distinctive type pairing only when the artifact is closer to editorial, deck, or brand work.
- Use motion sparingly: 150-250ms, ease-out, state feedback only. Respect `prefers-reduced-motion`.
- Avoid hover-only meaning. Every control must work with keyboard and touch.

## Hard Bans

- No gradient text.
- No glassmorphism as decoration.
- No nested card stacks.
- No endless identical card grids.
- No pure black or pure white as the full visual system. Tint neutrals subtly.
- No gray text on saturated colored backgrounds.
- No oversized hero section unless the artifact is explicitly a deck, poster, or landing-style narrative.
- No visible instructional text explaining that the page is an HTML artifact or how the design works.
- No lorem ipsum, placeholder metrics, decorative fake charts, or controls that do nothing.

## File Naming

Use a descriptive, stable filename in the workspace, for example:

- `implementation-plan.html`
- `code-review-map.html`
- `incident-timeline.html`
- `research-explainer.html`
- `option-comparison.html`

If the user names a path, use it. Otherwise choose a filename that matches the artifact's job.
