---
version: alpha
name: Rerun
description: House style for HTML artifacts, from rerun.io (measured 2026-11-01), the Rerun docs illustrations, and the 2026 brand.
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
guide; a FigJam diagram starter kit exists. If a Figma frame from it is available, match it.

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
- **light-*:** the light theme for print or on request: white page, neutral-950 text, neutral-600 body.

## Typography

Body is the system sans at 15 px, weight 450, in `text` grey. One display heading per page, light
(450) with −0.01 em tracking. Section headings are sans 500. Labels, eyebrows, buttons and feature
titles are JetBrains Mono UPPERCASE with slight positive tracking: this is the strongest Rerun cue.
Code, paths and API names are JetBrains Mono. Do not load web fonts: JetBrains Mono shows when
installed and the stack falls back to the system mono.

## Layout

Readable measure (about 70 characters) for prose. Feature blocks sit in 3–4 column grids without card
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

## Diagrams

Follow the Rerun docs illustrations; `assets/rerun_figure.py` has every primitive.

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

## Do's and Don'ts

- Do keep the page `#000` and the greys neutral; do put colour only on data, API names, status, or one
  brand moment.
- Do use mono uppercase for labels and buttons, light sans for headings.
- Do draw Rerun concepts with the chunk glyph and D2 shapes, and check the render.
- Don't tint the neutrals warm or cool, add shadows, load web fonts, or use gradient text.
- Don't spread brand colours across cards; don't use the data colours for decoration.
