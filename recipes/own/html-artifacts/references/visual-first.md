# Visual-first pages: figures, Rerun pictures, drawn charts, icons

Read this for any explainer, guide, report or deep dive that teaches how something works. Pablo's bar (2026-11-22):
"visual guides are way, way better ... no weird giant blocks of text", and use the Rerun viewer "as often as it can when
it makes sense to"; when it does not (sizes, byte layouts, counts), "make a good judgment call". A page that passed on
accurate text but looked like a wall of prose was sent back: accuracy is not enough, look at the pixels.

## Page rules

- **Figure first.** Every section opens with its picture (a flow, a Rerun capture, a drawn figure); the text follows.
- **Visible prose budget.** No visible paragraph longer than about three lines; a package or topic page under about 1,000
  visible words (the model page had 23 figures for 808 words). Measure it: count words outside `<details>`, `<svg>`,
  `<table>`, `<pre>` and captions, and strip nested `<svg>` from the innermost out (icons inside a flow SVG otherwise leak
  their labels into the count).
- **Fold, do not delete.** Keep the first sentence or clause visible; put the rest, PR history, file lists, flags, formulas,
  long tables, paper summaries and notes in `<details>` with a 44 px tall `<summary>`. An inline "More ›" toggle at the end
  of a sentence reads better than a bare boxed "More": a block that shows only the word "More" looks empty.
- **Captions are one line.** Keep the first sentence (at least ~8 words, so "Left: frame 345." keeps its "Right: ..." half);
  fold the rest and the method line (where the picture came from) under the same toggle.
- **Do not restate a figure in prose.** Drop the sentence that repeats the grid or chart above it.
- **Consistency across pages:** same figure sizes, one-line captions, the same section order and card styles.

## Rerun capture or drawn figure

- **Rerun capture** whenever the thing *is* data Rerun shows well: video frames, 2D overlays, masks, depth, 3D points,
  meshes, cameras and poses, time series (with axes, ticks, legend), recordings and catalog views. Shoot from existing
  `.rrd` files and catalogs first; prefer them to new GPU runs.
- **Drawn figure (inline SVG or HTML)** when Rerun is the wrong tool: sizes and byte layouts, counts and timings,
  before/after comparisons, data-structure layouts, role or registry grids, schematics, pipelines of concepts.
- **Never** matplotlib, stock images, decorative pictures, or a figure that restates the text without data.

## Drawn charts

- Only real, measured numbers that the page cites a source for. Never invent or round away a value: show it as measured
  (18.98 s, not 19.0 s; 114.5 s, not 114 s; integers with thousands separators).
- Write the exact value on each bar. A before/after (or baseline/ours) chart always carries a one-line key (grey = before,
  colour = after) and names both sides ("CUDA" / "Mojo", "before #582" / "after").
- Labels must never be cut. In a small slot put the label above its bar instead of beside it.
- A big-number tile only when the number itself is the point (a threshold, a resolution, a payload size); when a
  measured comparison exists, draw the comparison instead.
- Tables of comparable numbers (timings, speeds, sizes, scores) become bar charts with the exact values; keep tables for
  reference lists (flags, files, configs) and fold them.
- Phones: an SVG drawn 960 units wide shrinks its 15 px text to ~6 px at 390 px. Ship a second, large-type version
  (480 units wide, ≥ 22 px type, height growing with the rows) and switch with a media query; labels stay ≥ 11 px.

## Flow DAGs of real pictures

- Every node shows a picture of the data at that point: a Rerun thumbnail or a drawn picture of its real numbers or shape
  (a tensor-shape ladder, a size bar). A DAG of text boxes is not a visual.
- Until a node has its picture, draw it compact (an icon tile, the label, one line with a number), mark it "NO PICTURE",
  and keep a list of such nodes so the backlog is visible; a node that can never have one carries a written reason
  ("the recording never logged this stage", "no device was attached").
- Every flow must render stacked on a phone, never only behind a full-screen button.

## Rerun thumbnails and stills

- **One view per thumbnail**, shot on its own: a window twice the slot size, the UI zoomed 2x, legends and chrome off.
  Never a whole viewer layout with several panels, panel titles or the timeline at node size; never a crop of a
  full-page screenshot. Section figures may keep a full layout.
- Look at every shot yourself. An "is it empty" pixel test is not enough: a dark frame can pass, and a capture can come
  from the wrong viewer.
- Time cursor: check the recording's timeline range first. Some recordings stamp `video_time` in absolute time
  (1654423 s), so "545 s into the session" is outside it and the view comes out empty.
- 3D auto-framing can zoom onto one huge point (a trail drawn with a big radius); restrict the contents or set eye controls.
- Parallel captures: give each headless viewer its own port (an exclusive lock per port for the viewer's lifetime) and
  a recording id unique to the capture; refuse a viewer that holds any other recording. Otherwise two workers can drive
  each other's viewer and save the wrong picture.
- GPU: take a shared lock for viewer shots, hold it briefly, and never let a long-lived viewer or app hold it.
- Headless Chromium cannot decode the H.264/AV1 embed videos: black video tiles in a headless screenshot are expected;
  check videos in GPU Chrome.

## Icons

- Icons are labels only: flow-node markers (file, video, catalog, GPU, model), file and data chips, requirement chips,
  section and navigation markers, status (always next to a word). They never replace a data figure; no icon where it
  adds nothing.
- One icon set: Tabler Icons outline (the set rerun.io's landing ships), inline SVG, stroke 1.5, `currentColor`, a 20 px
  glyph in a 38 px dark rounded tile (16-18 px inline). Copy the glyphs you use into the artifact; do not load the set.
- One vocabulary: one concept, one glyph, on every page (video → `movie`, catalog → `database`, GPU → `gpu`, mask → `mask`,
  time → `clock`). Pick names by looking at the real glyphs.
- Assign icons from an explicit key, or from a keyword match only on input/output nodes; matching on every node gives
  wrong icons ("no calibration" → calibration).
