# Design Quality

Use this for theme, colour, type, motion, and accessibility decisions. The
visual-first rules and bans in `SKILL.md` come first.

## Contents

- Scene before style
- Surface and colour
- Typography
- Layout
- Motion
- Interaction
- Accessibility
- Finish line

## Scene before style

Write one sentence before you choose a theme or density:

> The user opens this on [device and place], while [mood or urgency], to
> decide [the main decision].

A Monday status read and an outage report want different pages. Let the
sentence choose.

## Surface and colour

- Start dark and tinted: near-black with a hue, not pure black, and soft
  off-white text, not pure white. Switch to light for print, projection, or
  when the user asks.
- Define a surface ladder: page, alternate band, panel, raised panel, border,
  strong border. Do not rely on shadows for separation.
- One colour per meaning, the same on every scene. Default set: blue = reuse
  or existing, amber = change or warning, green = new or success, rose = risk
  or failure, teal = navigation and focus.
- Never use colour alone. Pair it with a label, an icon, or a position.
- `oklch()` and `color-mix()` help tune tints; keep contrast at WCAG AA or better.

## Typography

- Use the system font stack. It renders offline on every machine.
- Use 3 type levels or fewer per page: scene heading, sub-line, body/caption.
- Use big numerals (about 2 rem) for counters and totals; they are visuals.
- Do not scale font size with the viewport. Use fixed `rem` steps and change
  the layout instead. Use `text-wrap: balance` on headings.
- Body text is 15 to 17 px; captions about 14 px; never below 12 px.

## Layout

- One idea per full-width scene. Content column about 1120 px wide,
  left-aligned. Centre only a true deck slide or poster.
- Put the visual high in the scene: heading, one sub-line, then the visual.
- Use cards only for real repeated units (flip cards, counters). No cards
  inside cards. No card grid as page filler.
- Vary space by importance; identical padding everywhere looks mechanical.
- Stack in reading order on phones. Long rows turn into columns.

## Motion

- Animate the mechanism: tokens that move along a pipeline, a playhead that
  moves along tracks, steps that light in order, bars that fill. Motion that
  only decorates the page is banned.
- Prefer CSS keyframes. Use JS (`requestAnimationFrame`) only for state that
  CSS cannot drive, such as a scrubbable timeline.
- One strong moment per scene. Loops are slow and calm: 2 to 3 s per cycle.
  State feedback takes 150 to 250 ms. A card flip takes about 0.5 s.
- Pause animation off screen. Under `prefers-reduced-motion`, show the final or
  key state and keep manual controls.

## Interaction

- Each control changes what the reader sees: presets, toggles, play/pause,
  scrub, flip, replay.
- Use real `<button>` elements with `aria-pressed` for state and an
  accessible name. Touch targets are at least 44 px.
- No hover-only meaning. Every control works with keyboard and touch.
- No modal-first browsing.

## Accessibility

- Landmarks (`main`, `section`, `nav`), logical heading levels, visible focus.
- Inline SVG diagrams get `role="img"` and an `aria-label` that states the point.
- Screenshots get `alt` text that says what the image proves.
- Live counters and captions use `aria-live="polite"`.
- Check contrast for text on coloured segments; use dark text on light fills.

## Finish line

The page is done when a reader can:

1. Get the point in 10 seconds from the first viewport.
2. Follow every scene from its visual and caption alone.
3. Use every control without instructions.
4. Open the file offline on a phone with nothing broken or clipped.
5. Trust every number: each one is measured, computed, or labeled as an example.
