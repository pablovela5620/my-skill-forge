---
name: html-artifacts
description: HTML artifacts, visual-first pages in the Rerun house style. Use when a result reads better in a browser than in markdown (explainer or visual guide, report or PR review, plan or comparison, deck, small tool), or when a page goes to the NAS or embeds a Rerun recording.
license: BSD-3-Clause
compatibility: The checker (scripts/check-artifact.mjs) needs node and playwright-cli on PATH (both in the fleet manifest).
---

# HTML artifacts

A page the user opens, refreshes while you work, and decides from. Three leading words carry this skill:
**visual-first** (every section opens with its picture), the **house style** (Rerun's, in
[references/DESIGN.md](references/DESIGN.md)), and **publish early** (the user sees the page grow at one URL).

## Contract

- **Files.** One `.html` with inline CSS and JavaScript. A page built on many real pictures, or a set of linked pages
  (a hub and its topic pages), keeps its pictures in a `media/` folder beside the HTML and is generated from data by a
  small build script. A page may reuse another published page's media by same-origin path, read-only, to avoid a
  large copy; name the source page in a comment, since the link breaks if that page moves.
- **Self-contained.** The page ships everything it shows: system fonts, inline SVG, local media. The one outside load is
  the shared Rerun viewer and `embed.js` on the NAS, for an embedded recording.
- **Rerun recordings.** For a recording on the page, follow the `rerun-viewer-validation` skill: it decides still,
  video or viewer, the size budget per connection, slicing, versions and the embed.
- **Real.** Every number, link, control and picture is real and specific. A picture comes from a Rerun capture, a
  measured run, or a figure drawn from real numbers.
- **Private data.** A page about a private or restricted dataset shows counts, metadata, and public or labelled
  synthetic samples. Hard guardrail: no private frame, thumbnail, crop or video goes into a page or a screenshot, and
  you do not open those frames yourself; age-restricted items are never rendered.
- **Style.** The house style, dark, from [references/DESIGN.md](references/DESIGN.md) and
  [assets/rerun.css](assets/rerun.css). Use another style only when the user asks for it or the page belongs to
  another brand.
- **"Presentation"** means this kind of page. Build slides only when the user asks for slides.

## Steps

### 1. Storyboard

Write the **scene sentence** first: the user opens this page on [device], while [urgency], to decide [what]. Then pick
the role: a *utility* page (plan, review, report, status) is dense and scannable; a *narrative* page (explainer, deck,
persuasive comparison) paces a story; a *tool* page (editor, filter, calculator) is controls, state and export. Pick
the pattern in [references/patterns.md](references/patterns.md).

Write the storyboard, one row per section: heading | lead picture | evidence | how-to-read caption. Row one holds the
answer, the result or the decision. Two kinds of row need more than one picture:

- A **concept** row ("what X is") shows the thing itself (a capture) beside its representation (a drawn figure of its
  parameters, sizes or structure), so the reader sees both what it looks like and what it is made of.
- A **results** row shows the evidence in full: every comparison, view, scene and failure you have, as a gallery or a
  grid, not one sample. Depth of evidence is what lets the reader trust the claim.

For an explainer, guide, report or deep dive, follow [references/visual-first.md](references/visual-first.md).

**Done when** every row has its lead picture and all the evidence you hold for it (or is folded into detail or cut),
every concept row shows the thing and its representation, and row one answers the scene sentence.

### 2. Publish the skeleton

Build the title, the section headings and the first picture or placeholders, with the work-in-progress badge
(`.rr-wip` in DESIGN.md). Publish it as [references/nas-publishing.md](references/nas-publishing.md) describes, and send
the user the URL. The URL stays the same from now on.

**Done when** the live URL shows the badge and the user has the URL.

### 3. Build section by section

For each storyboard row, build the section in the house style. Republish after each section or small batch, and at
least every 15 minutes while you work, so a refresh always shows progress; run the live-content check
(nas-publishing.md) each time. Update the badge's "updated" time and its "now" line.

- Figures and charts: [assets/rerun_figure.py](assets/rerun_figure.py) (docs figures, kit boards, `chart_html()`).
- Each caption's visible line says how to read the picture: what the colours, axes, panels or formula mean. Fold
  only the method and provenance (where the picture came from, flags, run IDs).
- Icons label words, from the bundled Tabler set in `<skill-dir>/assets/tabler-outline/` (grep `INDEX.tsv`), with the
  shared [references/icon-vocabulary.md](references/icon-vocabulary.md).
- Structure comes from space and hairlines; a card holds a real repeated unit, one level deep.
- Content is browsed in place: tabs for alternate views, `<details>` for depth, search or filters for long lists,
  copy buttons for commands, export for editors, keyboard for decks. A dialog is for a choice that must interrupt.
- Text on a coloured fill is `fg` or black at full contrast.
- Page text is about the subject. Labels, numbers and captions stay live text (HTML or SVG `<text>`), so they scale,
  wrap and can be searched.
- Every control works with keyboard and touch and shows its focus. Landmarks, logical headings and accessible names
  hold the structure.

**Done when** every storyboard row is live and the page is within the visual-first prose budget.

### 4. Verify

Run `node <skill-dir>/scripts/check-artifact.mjs <published URL>`, then read every screenshot it saves: a PASS proves
only what the checker measures, and a page can pass and still be a wall of text. Open each embedded Rerun viewer in GPU
Chrome at the same sizes (nas-publishing.md). For a dense page, also try long and empty content and 125-200 % zoom.
Details and pitfalls: [references/verification.md](references/verification.md).

**Done when** the checker passes at all four sizes, you have read every screenshot, and every section leads with its
picture.

### 5. Final publish

Remove the badge, republish, and run `check-artifact.mjs --final <published URL>` and the live-content check. Send the
user the URL; for a visual change, the cache-busting form (`<url>?v=<short-hash>`).

**Done when** `--final` passes on the live URL and the served file's sha256 matches your build.
