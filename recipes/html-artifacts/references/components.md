# Visual Components

Every recipe is a marked block in `assets/gallery.html`. Each block holds its
own `<style>`, markup, and `<script>`. All blocks pass the checker at 390 px and
1440 px with their controls exercised (`assets/gallery-steps.js`).

## Contents

- Copy a recipe
- Shared base
- rerun-hero
- pipeline
- toggle
- presets
- flip-cards
- stepper
- metro
- capacity
- evidence
- Write a new component

## Copy a recipe

```bash
G=<skill-dir>/assets/gallery.html
sed -n '/<!-- recipe:base-head -->/,/<!-- \/recipe:base-head -->/p' "$G"   # into <head>
sed -n '/<!-- recipe:base-body -->/,/<!-- \/recipe:base-body -->/p' "$G"   # first thing in <body>
sed -n '/<!-- recipe:flip-cards -->/,/<!-- \/recipe:flip-cards -->/p' "$G" # into <main>, once per recipe
```

Keep the order: `base-head`, then `base-body` (it defines `art`), then the
recipe blocks. Replace the example content and keep the structure. Delete the
gallery-only `<code class="recipe-tag">` lines. To see a recipe run, open the
gallery in a browser.

## Shared base

- **Tokens.** Surfaces `--bg --bg2 --panel --panel2 --line --line2`, text
  `--ink --ink2 --muted`, and one colour per meaning: `--reuse` blue,
  `--adapt` amber, `--new` green, `--risk` rose, `--accent` for navigation and
  focus. Keep each meaning the same on every scene.
- **Scene shell.** `<section class="scene" id="s-x" data-nav="Label"><div
  class="scene-in">`. The `id` is required: the checker takes one screenshot
  per `section[id]`. `data-nav` adds a nav dot.
- **`art` helpers.** `art.reduced()` is true under `prefers-reduced-motion`.
  `art.onMotionChange(fn)` runs when that setting changes. `art.onLive(el,
  fn)` calls `fn(true|false)` when `el` enters or leaves the screen.
- **Motion control.** CSS animations pause when their scene is off screen
  (`.scene:not(.is-live) *`). A global reduced-motion rule removes CSS
  animations and transitions. JS-driven motion must check `art.reduced()`.
- **No global smooth scroll.** Nav dots call `scrollIntoView` with
  `behavior` set from `art.reduced()`. `html { scroll-behavior: smooth }`
  breaks scripted scrolling in the checker.
- **Icons.** One inline sprite of `<symbol>`s; use
  `<svg class="ic"><use href="#i-name"/></svg>`. New symbols use the same
  24×24 line style: stroke 1.8, round caps and joins, no fill.

## rerun-hero

**Use for** the end product: a run, a recording, a robot task, any data over
time. It plays a real sweep video of the Rerun viewer. Chapter buttons seek to
the recording's own events, and a live caption names the current event.

**Make it** from a real `.rrd`, or from a synthetic one when no data exists
yet (`assets/pick_place_demo.py` is a template). Sweep and capture with
`rerun-viewer-validation`; see `rerun.md`.

**Edit** the `<video>` source, `data-span` (recording seconds the sweep
covers), and `data-events` (copy `[seconds, text]` from the recording's event
log, never invent them). Keep the badge true: "Real viewer · synthetic data" or
"Real viewer · real data".

**Rules.** The video plays only while visible and never starts by itself under
reduced motion; chapters still seek. Crop the sweep to the view that tells the
story when phones matter: full viewer chrome is too small to read at 390 px.

## pipeline

**Use for** work or data that moves through 3 to 6 stages. Token colour shows
what kind of thing flows; add a legend.

**Rules.** Stage name of 1 or 2 words plus a sub-label of 3 words or fewer. The
row turns vertical under 640 px. When nothing passes between steps, use
`stepper` instead.

## toggle

**Use for** before and after, today and proposed, or hidden mechanics made
visible. The counters read the DOM, so they stay true when content changes.

**Rules.** Two views, three at most. Name the buttons after the states, not
"A" and "B". Give the views a similar height so the page does not jump.

## presets

**Use for** behaviour under conditions: load, failure, slow consumer, stall.
Two or three preset buttons, each one a story, with big coloured counters.

**Rules.** Write the model as a pure, deterministic function `run(params)` that
matches the real design. Counters come from the model, never from typed
numbers. Check the model against the real design before animating it: a
smooth model once showed a capture burst that the real sampler cannot produce.
A raw slider is an extra, never the only control.

## flip-cards

**Use for** decisions, options with a cost, glossary terms. Front: icon and
answer (6 words or fewer). Back: the tradeoff (12 words or fewer).

**Rules.** Each card is a `<button aria-pressed>`. Keep "Flip all". Reduced
motion swaps faces without rotation. Keep the `fc-` class prefix: a generic
`.back` class once flipped arrow labels in another component.

## stepper

**Use for** an ordered process. Steps light up in turn when the scene appears;
"Replay" runs it again. Leaving mid-run cancels the run. Reduced motion,
also when switched on later, lights all steps at once. Step text:
a 3-word name and a 4-word note.

## metro

**Use for** a roadmap, milestones, or rollout status. `class="done"` fills a
stop and the segment after it; `class="now"` pulses; other stops stay hollow.
The line turns vertical on phones.

## capacity

**Use for** a budget, quota, disk size, or time box. Set `data-cap` and the
`PARTS` rows (`[label, value, "--colour"]`, same unit as the cap). Bars fill
when the scene appears. Show the total as a big number above the bar, and
say what the total counts.

## evidence

**Use for** proof. A real screenshot, cropped to the part that matters, with
one box, one short label, and the rest dimmed. Box position uses percentages.
Add `data-label="end"` when the box is near the right edge.

Capture web UIs with playwright (`locator.screenshot()`). Capture Rerun at an
exact time with the viewer MCP (`rerun.md`). Crop, scale, compress, and inline:

```bash
ffmpeg -i shot.png -vf "crop=W:H:X:Y,scale=960:-2" -q:v 6 shot.jpg   # aim under 150 KB
base64 -w0 shot.jpg   # src="data:image/jpeg;base64,..."
```

Set the `width` and `height` attributes to the image size to stop layout shift.

## Write a new component

- Give the root `data-recipe="name"`; the script initialises every root.
- Prefix every class (`name-`). Never use generic classes such as `.back`.
- Use `art.reduced()` and `art.onLive()`; pause loops off screen.
- Give grid and flex children `min-width: 0`; give inline code
  `overflow-wrap: anywhere`. Touch targets are at least 44 px.
- Use buttons with `aria-pressed` or `aria-label`; never hover-only meaning.
- Add the block to `gallery.html`, add clicks to `gallery-steps.js`, and run
  the checker before you use it.
