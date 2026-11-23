---
version: alpha
name: Rerun
description: House style for HTML artifacts, from rerun.io (measured 2026-11-01), the Rerun docs illustrations, the Rerun FigJam diagram kit, and the 2026 brand. Dark by default.
colors:
  primary: "oklch(0.97 0 0)"
  bg: "#000000"
  panel: "oklch(0.145 0 0)"
  panel-2: "oklch(0.205 0 0)"
  figure: "#0e0f10"
  fg: "oklch(0.97 0 0)"
  subtle: "oklch(0.922 0 0)"
  text: "oklch(0.708 0 0)"
  dim: "oklch(0.6 0 0)"
  border: "#64646440"
  border-strong: "oklch(0.371 0 0)"
  on-fg: "#000000"
  row: "#22b5e6"
  time: "#de7df3"
  comp: "#6cefb4"
  api: "#3fe0a0"
  good: "#6cefb4"
  warn: "#f6da75"
  bad: "#ff6b5e"
  brand-blue: "#186ADD"
  brand-red: "#D72F21"
  brand-yellow: "#F6DA75"
  brand-green: "#86D9A6"
  code-keyword: "oklch(0.827 0.119 306)"
  code-string: "oklch(0.845 0.143 165)"
  code-function: "oklch(0.808 0.114 19.6)"
  code-comment: "oklch(0.6 0 0)"
  light-bg: "#ffffff"
  light-fg: "oklch(0.145 0 0)"
  light-text: "oklch(0.439 0 0)"
  light-border: "oklch(0.87 0 0)"
  kit-board: "#0a0c10"
  kit-card: "#14171d"
  kit-card-2: "#1d2129"
  kit-stroke: "#34446b"
  kit-arrow: "#6b93ff"
  kit-index: "#f28ff0"
  kit-data: "#30dc9a"
  kit-note: "#4ed37a"
  kit-pill: "#2a3bff"
  kit-caption: "#9aa0a8"
typography:
  display:
    fontFamily: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif
    fontSize: 3.5rem
    fontWeight: 450
    lineHeight: 1
    letterSpacing: -0.01em
  heading:
    fontFamily: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif
    fontSize: 1.75rem
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: -0.01em
  body:
    fontFamily: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif
    fontSize: 15px
    fontWeight: 450
    lineHeight: 1.5
  caption:
    fontFamily: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif
    fontSize: 0.85em
    fontWeight: 500
    lineHeight: 1.5
  label-caps:
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: 13px
    fontWeight: 500
    lineHeight: 1.33
    letterSpacing: 0.0125em
  feature-title:
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: 17px
    fontWeight: 600
    lineHeight: 1.33
    letterSpacing: 0.0125em
  diagram-title:
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: 30px
    fontWeight: 500
    lineHeight: 1.25
  diagram-label:
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.4
  code:
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: 0.86em
    fontWeight: 400
    lineHeight: 1.6
rounded:
  control: 2px
  panel: 8px
  figure: 14px
  pill: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
  measure: 70ch
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-fg}"
    typography: "{typography.label-caps}"
    rounded: "{rounded.control}"
    padding: 10px 14px
  button-secondary:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.fg}"
    typography: "{typography.label-caps}"
    rounded: "{rounded.control}"
    padding: 10px 14px
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.panel}"
    padding: 16px
  figure:
    backgroundColor: "{colors.figure}"
    textColor: "{colors.fg}"
    rounded: "{rounded.figure}"
    padding: 24px
  pill:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.fg}"
    typography: "{typography.caption}"
    rounded: "{rounded.pill}"
    padding: 4px 12px
  code-block:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.fg}"
    typography: "{typography.code}"
    rounded: "{rounded.panel}"
    padding: 14px
  body-text:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.text}"
    typography: "{typography.body}"
  caption-text:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.dim}"
    typography: "{typography.caption}"
  table-cell:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.subtle}"
    typography: "{typography.body}"
  bar-track:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.fg}"
    rounded: "{rounded.control}"
  divider:
    backgroundColor: "{colors.border}"
    textColor: "{colors.fg}"
    height: 1px
  input-outline:
    backgroundColor: "{colors.border-strong}"
    textColor: "{colors.fg}"
    height: 1px
  chip-row:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.row}"
    typography: "{typography.code}"
  chip-time:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.time}"
    typography: "{typography.code}"
  chip-component:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.comp}"
    typography: "{typography.code}"
  diagram-api-name:
    backgroundColor: "{colors.figure}"
    textColor: "{colors.api}"
    typography: "{typography.code}"
  wip-badge:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.warn}"
    typography: "{typography.label-caps}"
    rounded: "{rounded.pill}"
    padding: 6px 12px
  status-good:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.good}"
    typography: "{typography.label-caps}"
  status-warn:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.warn}"
    typography: "{typography.label-caps}"
  status-bad:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.bad}"
    typography: "{typography.label-caps}"
  brand-fill-blue:
    backgroundColor: "{colors.brand-blue}"
  brand-fill-red:
    backgroundColor: "{colors.brand-red}"
  brand-rule-start:
    backgroundColor: "{colors.brand-yellow}"
    height: 2px
  brand-rule-end:
    backgroundColor: "{colors.brand-green}"
    height: 2px
  code-keyword:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.code-keyword}"
    typography: "{typography.code}"
  code-string:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.code-string}"
    typography: "{typography.code}"
  code-function:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.code-function}"
    typography: "{typography.code}"
  code-comment:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.code-comment}"
    typography: "{typography.code}"
  light-page:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-text}"
    typography: "{typography.body}"
  light-heading:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-fg}"
    typography: "{typography.heading}"
  light-divider:
    backgroundColor: "{colors.light-border}"
    textColor: "{colors.light-fg}"
    height: 1px
  kit-board:
    backgroundColor: "{colors.kit-board}"
    textColor: "{colors.fg}"
    typography: "{typography.diagram-title}"
    rounded: "{rounded.panel}"
  kit-card:
    backgroundColor: "{colors.kit-card}"
    textColor: "{colors.fg}"
    typography: "{typography.diagram-label}"
    rounded: "{rounded.panel}"
  kit-card-nested:
    backgroundColor: "{colors.kit-card-2}"
    textColor: "{colors.fg}"
    typography: "{typography.diagram-label}"
    rounded: "{rounded.panel}"
  kit-stroke:
    backgroundColor: "{colors.kit-stroke}"
    textColor: "{colors.fg}"
    height: 2px
  kit-arrow:
    backgroundColor: "{colors.kit-board}"
    textColor: "{colors.kit-arrow}"
  kit-table-index:
    backgroundColor: "{colors.kit-card-2}"
    textColor: "{colors.kit-index}"
  kit-table-data:
    backgroundColor: "{colors.kit-card-2}"
    textColor: "{colors.kit-data}"
  kit-note:
    backgroundColor: "{colors.kit-board}"
    textColor: "{colors.kit-note}"
    typography: "{typography.diagram-label}"
  kit-caption:
    backgroundColor: "{colors.kit-card}"
    textColor: "{colors.kit-caption}"
    typography: "{typography.caption}"
  kit-file-pill:
    backgroundColor: "{colors.kit-pill}"
    textColor: "{colors.fg}"
    typography: "{typography.code}"
    rounded: "{rounded.pill}"
---

## Overview

Rerun's website is a pure black page with neutral greys, light system-sans headings with tight
tracking, and small uppercase JetBrains Mono labels. Borders are hairlines; buttons are square and
uppercase. The interface stays monochrome. Colour belongs to the data (chunk columns, chart series,
API names in diagrams) and to rare brand moments, the way rerun.io keeps colour for its rainbow
product renders.

Sources: rerun.io CSS custom properties and computed styles (2026-11-01); the Rerun docs chunk
illustration and the D2 diagram on the chunk processing page; brand colours and the heading font
(Basis Grotesque Pro, licensed for the website only) from the 2026 rebrand. There is no written diagram
guide. The FigJam "Diagram starter kit" (read 2026-11-01) draws light slide boards; this system uses its shapes and
colours on dark boards, because artifacts are dark by default.

## Colors

- **primary:** the key colour is the `fg` white, as on rerun.io's primary buttons. Rerun's interface is monochrome; the brand blue is not the primary colour.
- **bg (#000000):** the page. Pure black, as on rerun.io; this overrides the "tinted near-black" default.
- **panel / panel-2:** neutral-950 and neutral-900 for code, cards and bar tracks. Greys have zero chroma.
- **fg:** headings and key values. **text:** body copy (the rerun.io body colour). **dim:** captions, axes; a step lighter than neutral-500 so small captions pass WCAG AA (4.5:1) on `bg` and `panel`.
- **border:** 1 px hairlines. **border-strong:** outlines of secondary buttons and inputs.
- **row / time / comp:** the docs' chunk picture: row ids blue, time magenta, component data green.
  Reuse them as the first three data series in charts.
- **api:** green for API names and identifiers inside diagrams, as in the D2 diagrams.
- **good / warn / bad:** status, always paired with a word or icon.
- **brand-blue, brand-red, brand-yellow → brand-green gradient:** at most one brand moment per artifact
  (a rule under the title, one highlighted figure). Never a page background, never gradient text.
  Blue and red are about 4:1 on black: use them as fills and rules with no text on them.
- **kit-*:** the FigJam diagram kit translated to dark (see Diagrams). Read from screenshots, not exact Figma values.
- **light-*:** the light theme, used only when asked for: white page, neutral-950 text, neutral-600 body.
- **Dark is the default.** Never switch to light on your own, not even for print.

## Typography

Body is the system sans at 15 px, weight 450, in `text` grey. One display heading per page, light
(450) with −0.01 em tracking. Section headings are sans 500. Labels, eyebrows, buttons and feature
titles are JetBrains Mono UPPERCASE with slight positive tracking: this is the strongest Rerun cue.
Code, paths and API names are JetBrains Mono. Do not load web fonts: JetBrains Mono shows when
installed and the stack falls back to the system mono.

## Layout

Readable measure (about 70 characters) for prose; figures, flows and wide tables may span the full content width (up to about 1,560 px on a large screen) while the prose beside them keeps the measure. Type sizes may scale fluidly with `clamp()` between the phone and desktop values. Feature blocks sit in 3–4 column grids without card
borders. Generous vertical space between sections; hairline rules separate them. Left-aligned text,
except a centred title on a deck or poster page.

## Elevation & Depth

Flat. No shadows, no glass, no glow. Depth comes from panel greys (`panel`, `panel-2`) and hairlines.

## Shapes

- Buttons and inputs: 2 px corners (rerun.io uses 1.875 px).
- Panels, cards, code: 8 px. Diagram figures: 14 px, as in the docs pictures.
- Pills and tags: fully rounded, 1 px outline.

## Components

- **Buttons:** primary = `fg` background, black mono uppercase text; secondary = outline in
  `border-strong`. Never a row of primaries.
- **Feature block:** small sans label in `text`, then a mono uppercase title in `fg`, then body in `text`.
- **Stat tile:** big sans value in `fg`, mono uppercase label below in `dim`.
- **Table:** 1 px row rules, no zebra; mono uppercase header in `dim`; numbers right-aligned, tabular.
- **Callout:** `panel` with a 2 px left rule in one data or status colour.
- **Code block:** `panel`, 8 px, 1 px border, rerun.io token colours.
- **Work-in-progress badge** (`.rr-wip`, `data-wip`): a dashed `warn` pill at the very top of the page, in the normal
  flow (never fixed, so it covers nothing), visible without scrolling at every size: the Tabler `progress` glyph, the
  words "Work in progress", "updated <time>" (UTC, or the reader's zone when a script localises it), and optionally one
  line on what is being worked on now. Static glyph; animate only under `prefers-reduced-motion: no-preference`.
  Remove it in the final publish.

## Diagrams

Two diagram themes, both dark, both in `assets/rerun_figure.py`. Use the **docs figure** to explain a mechanism inside a page; use
the **kit board** for a slide-like overview of a system or pipeline.

### Docs figure

- One idea per `figure` panel (`#0e0f10`, 14 px corners).
- Rerun data is drawn with the **chunk glyph**: a header bar, then columns for row ids (`row`), time
  (`time`) and component data (`comp`). Single rows are small glyphs with one bar per column.
- D2 shapes: parallelogram = reader or terminal call, cylinder = store, rectangle = process,
  dashed `api` rectangle = a lazy plan.
- Node labels mono UPPERCASE in `fg`; API names in `api` mono, on edges or inside nodes.
- Explanations are plain sans text on dotted leader lines, never long text inside boxes.
- Arrows: 1.4 px grey lines with small solid heads. Lane titles mono uppercase in `dim`; the
  recommended lane's title in `api`.
- Measured numbers go in a bar panel of the same style, titled `MEASURED · <what>`, in data colours.
  Only measured values, and say what was measured.
- Render and look: no label may touch a shape or another label.

### Kit board

From the Rerun FigJam diagram starter kit, translated to dark with the kit's own dark rule: step from darkest to
lightest with minimal steps, and alternate two shades for nesting.

- Board `kit-board` with 8 px corners; optional `diagram-title` top left, mono, normal case.
- Cards `kit-card`, nested cards `kit-card-2`, all with 2 px `kit-stroke` outlines and 8 px corners. A dashed
  outline marks an optional group.
- Labels mono in normal case (`diagram-label`), never uppercase. Group names above glyphs are small grey sans
  (`kit-caption`).
- Arrows 2 px `kit-arrow` with open line heads. Relationship lines are dashed with no heads.
- Table glyph: header of blue dashes, then per row a pink index dot (`kit-index`) and green data dashes (`kit-data`).
- File glyph: a blank page with a rounded corner and a `kit-pill` extension pill (`.mcap`, `.rrd`).
- Green mono notes (`kit-note`) above a step say why the step matters.
- Flat fills only: no gradients, shadows or glow.

## Do's and Don'ts

- Do keep the page `#000` and the greys neutral; do put colour only on data, API names, status, or one
  brand moment.
- Do use mono uppercase for labels and buttons, light sans for headings.
- Do draw Rerun concepts with the chunk glyph and D2 shapes, or as a kit board, and check the render.
- Do default to dark; translate light sources (the FigJam kit, slides) to dark unless light is asked for.
- Don't tint the neutrals warm or cool, add shadows, load web fonts, or use gradient text.
- Don't switch to the light theme, a white board or a light print style unless asked.
- Don't spread brand colours across cards; don't use the data colours for decoration.
