# Visual-first pages: figures, Rerun pictures, drawn charts, icons

Read this for any explainer, guide, report or deep dive that teaches how something works. Pablo's bar (2026-11-22):
"visual guides are way, way better ... no weird giant blocks of text", and use the Rerun viewer "as often as it can when
it makes sense to"; when it does not (sizes, byte layouts, counts), "make a good judgment call". A page that passed on
accurate text but looked like a wall of prose was sent back: accuracy is not enough, look at the pixels.

## Page rules

- **Storyboard before building:** one row per section, "heading | lead picture | evidence | caption" (SKILL.md step 1);
  a row without a picture is cut or moves to folded detail. Name the one question the page answers.
- **Concept rows show a pair:** the thing itself (a Rerun capture) beside its representation (a drawn figure of its
  parameters, sizes or structure), both visible.
- **End product first.** The first section shows the result as a real capture (the output, the run, the rendered page).
- **Figure first.** Every section opens with its picture (a flow, a Rerun capture, a drawn figure); the text follows.
- **Results rows show all the evidence** they hold, visible, as a gallery or grid: every scene, view, host, comparison
  and failure. One sample per section with the rest folded reads as less work and less proof (Pablo, 2026-11-23, on a
  rebuild that showed 10 of the original report's 48 pictures).
- **Visible prose budget.** No visible paragraph longer than about three lines; a package or topic page under about 1,000
  visible words (the model page had 23 figures for 808 words). Measure it: count words outside `<details>`, `<svg>`,
  `<table>`, `<pre>` and `<figcaption>`, and strip nested `<svg>` from the innermost out (icons inside a flow SVG
  otherwise leak their labels into the count). A caption that tells how to read its picture belongs to the picture: it
  does not count against the budget and is not cut to meet it.
- **Fold method and provenance, not meaning.** Fold run IDs, flags, PR history, file lists, long reference tables and
  where a picture came from into `<details>` with a 44 px tall `<summary>`. Never fold the explanation of what a picture
  shows or what a claim means, and never fold evidence pictures. A summary names what it opens ("Paper Table 3, every
  scene"); a block that shows only the word "More" looks empty.
- **Captions explain in view.** The visible caption says what the picture shows and how to read it ("Left: frame 345.
  Right: ...", "Each cell is its d = 0 edge, coloured by c₀ + u·g_u + v·g_v"), in as few lines as that takes. Only the
  method line (run ID, flags, the tool that made the picture) may fold.
- **Text budget:** section heading about 6 words; a label inside a picture 1-4 words; prose paragraphs about three
  lines. Short sentences, active voice, common words.
- **Do not restate a figure in prose.** Drop the sentence that repeats the grid or chart above it.
- **Links go to the most specific target** (the subfolder, the file and line, the section), not a repo or site root.
- **Every deep page is reachable** from its hub or parent page; a page only the author can find does not exist for readers.
- **Consistency across pages:** same figure sizes, caption style, section order and card styles.

## Rerun capture or drawn figure

- **Rerun capture** whenever the thing *is* data Rerun shows well: video frames, 2D overlays, masks, depth, 3D points,
  meshes, cameras and poses, time series (with axes, ticks, legend), recordings and catalog views. Shoot from existing
  `.rrd` files and catalogs first; prefer them to new GPU runs.
- **A process over time gets a video of the whole timeline**, not one still: a sweep of the recording (the embed video
  of [nas-publishing.md](nas-publishing.md)), with stills only for the moments the text points at.
- **Drawn figure (inline SVG or HTML)** when Rerun is the wrong tool: sizes and byte layouts, counts and timings,
  before/after comparisons, data-structure layouts, role or registry grids, schematics, pipelines of concepts.
- **Never** matplotlib, stock images, decorative pictures, or a figure that restates the text without data.
- **Never draw an imitation of the Rerun viewer** in SVG or HTML: show real viewer pixels from a real `.rrd`.
- **No data yet (a proposal):** log a small synthetic recording with the real Rerun SDK (the planned entity paths,
  timelines and views, filled with made-up values) and capture it like real data; label it "synthetic data" in the
  caption and never present it as a measurement.
- **Truthful motion.** Animate the mechanism (data moves, time passes, state changes), not the page: no entrance fades on
  every section, no hover effects on every card, no parallax. Prefer 2-3 presets to a raw slider when the presets tell
  the story.
- **Colour has one meaning per page** (for example blue = reuse, amber = change, green = new or success, red = risk),
  always paired with a word or icon.

## Drawn charts

- Draw them with `assets/rerun_figure.py`: `chart_html({"bars": ...})` or `chart_html({"pairs": ...})` returns the wide SVG
  and its phone version (the `.rr-chart` rules in `rerun.css` switch them by the chart's own width). Pass a value as a
  string (`"0.50"`) to keep it as measured. Give a wide chart a slot of about 700 px or more.
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
- One icon set: Tabler Icons outline **3.50.0**, pinned because it is the set rerun.io's landing page ships. It is bundled
  with this skill: `assets/tabler-outline/<name>.svg` (5,203 glyphs, MIT licence beside them;
  the package build adds the folder, so read it from the installed skill, not the recipe source). Copy the glyphs a page
  uses into it as inline SVG with `stroke-width="1.5"` and `stroke="currentColor"`; never load the set from the network.
  A 20 px glyph in a 38 px dark rounded tile, or 16-18 px inline.
- Find a glyph: `grep -i <concept> assets/tabler-outline/INDEX.tsv` (name, then its categories), then look at the glyph
  before you use it.
- One vocabulary for every artifact: [icon-vocabulary.md](icon-vocabulary.md) (video → `movie`, catalog → `database`,
  GPU → `gpu`, ...). A page may add concepts; it must not give an existing concept another glyph.
- Assign icons from an explicit key, or from a keyword match only on input/output nodes; matching on every node gives
  wrong icons ("no calibration" → calibration).
