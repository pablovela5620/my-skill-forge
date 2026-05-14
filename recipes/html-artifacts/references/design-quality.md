# Design Quality Reference

Use this when making the HTML artifact visually strong enough that the user wants to read it.

## Scene Before Style

Before choosing theme, palette, and density, write a concrete scene sentence:

> The user opens this artifact on [device/context], while [mood/urgency], to decide [primary decision].

Let that sentence choose the visual system. A calm weekly status read on Monday morning wants a different surface than an incident report opened during an outage.

When the scene does not force a light surface, prefer a dark one. Use light for print, mandated source-system parity, bright-room projection, or contexts where sharing and legibility clearly benefit from paper-like contrast.

## Artifact Role

Choose the role before drawing the interface:

- Utility artifact: dense, scannable, restrained, system fonts, low ornament. Use for plans, reviews, reports, status pages, and operational tools.
- Narrative artifact: stronger hierarchy, richer pacing, and more memorable composition. Use for decks, explainers, and persuasive comparisons.
- Tool artifact: controls, state, validation, copy/export, and keyboard/touch affordances. Use for calculators, editors, filters, and generators.

The role decides density and expression. A utility artifact should not turn into a landing page, and a narrative artifact should not hide its argument behind decorative controls.

## Hierarchy

- The page should answer "what am I looking at?" in the first viewport.
- Put the conclusion, current state, or main decision above the supporting detail.
- Use no more than 3 primary type levels in a small artifact.
- Make scan paths obvious: summary strip, section headers, numbered steps, or a persistent table of contents.
- Keep prose columns readable at 65-75 characters. Tables and code can be wider.

## Dark-First Theme

- Start with a dark, tinted base unless the scene says otherwise. Good dark artifacts feel calm and legible, not theatrical.
- Avoid pure black page backgrounds and pure white text. Use near-black surfaces with a subtle hue, softer foreground text, and a brighter text color only for headings or critical values.
- Define a real surface ladder: page, section, elevated panel, border, hover, active, and selected. Do not rely on shadows alone for separation.
- Use one accent color for navigation, focus, and key state. Keep saturation limited so the accent remains meaningful.
- Use `oklch()` when practical. It makes dark palettes easier to tune without dead grayscale.
- Keep data states semantic: success, warning, danger, neutral, and active should have both color and text labels or icons.
- For reports and plans that may be printed, include `@media print` styles that switch to a light paper surface, dark text, and visible table borders.

## Color

- Pick a named color strategy:
  - Restrained: dark tinted neutrals and one accent. Default for reports, plans, reviews, tools.
  - Committed: one strong color carries a major surface. Use for decks, editorial explainers, memorable decision artifacts.
  - Full palette: 3-4 roles with clear semantics. Use for charts, category-heavy reports, systems maps.
- Use CSS `oklch()` when practical.
- Tint neutrals toward the accent hue. Avoid dead grayscale.
- Reserve saturated color for state, emphasis, or navigation. Do not spray accents across every card.
- Never rely on color alone. Pair color with labels, icons, shape, or position.

## Typography

- Product/report artifacts can use system fonts. They should feel native and easy to scan.
- Editorial/deck artifacts may use a stronger heading font if local or remote fonts are allowed.
- Avoid negative letter spacing.
- Avoid viewport-based font scaling. Use fixed rem steps with responsive layout, not fluid text.
- Use `text-wrap: balance` for short headings and `text-wrap: pretty` for prose when supported.
- Body text should usually sit between 15px and 18px.

## Layout

- Prefer spatial structure over boxed repetition.
- Use cards only for real repeated units, not as the default page-building material.
- Do not put cards inside cards.
- Vary spacing by section importance. Identical padding everywhere feels mechanical.
- Use sticky navigation or a compact table of contents for long artifacts.
- Give tables enough structure: sticky headers where useful, zebra rows only when they help tracking, aligned numbers, and clear captions.

## Interaction

Add interaction only when it changes how the artifact is used:

- Tabs for alternate views of the same object.
- Details/summary for optional depth.
- Search or filters for long lists.
- Copy buttons for commands, snippets, or generated output.
- Keyboard navigation for decks.
- Export buttons for editor-like artifacts.

Prefer native HTML controls and patterns: labeled inputs, buttons, tables, `details`/`summary`, and `dialog` only when a modal is truly needed. Every control needs hover, focus-visible, active, and disabled states when applicable. Touch targets should be at least 44px on mobile.

## Visual Anti-Slop Checks

Before finishing, remove or redesign these:

- Purple-blue gradients used as a default identity.
- Bokeh blobs, decorative orbs, and ambient blobs.
- Rounded-square icon tiles above every heading.
- Side-stripe borders on callouts or cards.
- Hero metric blocks with one huge number plus tiny labels.
- Repeated icon-heading-body cards that all have the same weight.
- Overly soft shadows used to fake depth.
- Decorative dashboards with fake data.
- Dark mode that depends on glow, neon edges, or high-saturation bloom.
- All-centered layouts when the artifact needs comparison, scanning, or repeated use.
- All-primary button rows where every action appears equally important.
- Modal-first browsing for complex content.
- Monospace styling used to imply technical depth when the content is ordinary prose.
- Redundant intros that restate the section heading.
- UI text rasterized into images or SVG when semantic text would work.

## Accessibility and Robustness

- Use landmarks: `header`, `main`, `nav`, `section`, `article`, `aside`, `footer`.
- Keep heading levels logical.
- Give controls accessible names.
- Preserve focus outlines.
- Respect `prefers-reduced-motion`.
- Check contrast in both normal and emphasized states.
- Test long text, short text, empty sections, and narrow widths.
- Check browser zoom from 125% to 200% for dense artifacts.
- Avoid horizontal scrolling except in intentionally scrollable tables or code blocks.

## Finish Line

The artifact is done when a reader can:

1. Understand the point in under 10 seconds.
2. Find supporting detail without reading linearly.
3. Trust the visual hierarchy.
4. Use any interactive control without instructions.
5. Open the file locally without broken assets or setup.
6. Print the file acceptably when the artifact is a plan, report, review, or reference.
7. Confirm there are no dead controls, fake links, placeholder data, external dependencies, clipped labels, or missing assets.
