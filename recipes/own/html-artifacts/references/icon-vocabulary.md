# Icon vocabulary (Tabler Icons outline 3.50.0)

The default concept-to-glyph map for every artifact: one concept, one glyph, on every page. A page may add concepts it
needs (add them here when they recur); it must not give an existing concept another glyph. Every name below exists in
`assets/tabler-outline/INDEX.tsv` (checked against 3.50.0).

| Group | concept → glyph |
|---|---|
| Data | `video` → `movie`, `camera` → `camera`, `image` → `photo`, `depth` → `ruler-measure`, `points` → `grain`, `mesh` → `polygon`, `body` → `body-scan`, `keypoints` → `vector`, `hand` → `hand-stop`, `mask` → `mask`, `detection` → `scan`, `track` → `focus-2`, `trajectory` → `route`, `map` → `map-2`, `imu` → `activity`, `calibration` → `perspective`, `fisheye` → `rotate-360`, `layer` → `stack-2`, `recording` → `file-3d`, `file` → `file`, `folder` → `folder`, `table` → `table`, `text` → `file-text` |
| Sources | `catalog` → `database`, `download` → `cloud-download`, `web` → `world`, `device` → `device-mobile`, `disk` → `server`, `bundled` → `package` |
| Compute and code | `gpu` → `gpu`, `cpu` → `cpu`, `model` → `brain`, `speed` → `bolt`, `rust` → `brand-rust`, `python` → `brand-python`, `library` → `package`, `recipe` → `book`, `code` → `code`, `github` → `brand-github`, `terminal` → `terminal-2`, `viewer` → `device-desktop` |
| Measures | `time` → `clock`, `accuracy` → `target`, `metrics` → `chart-bar` |
| Sections | `flow` → `hierarchy-2`, `walkthrough` → `player-play`, `deep-dive` → `zoom-in`, `models` → `box-multiple`, `input` → `database-import`, `stages` → `list-numbers`, `evidence` → `photo-scan`, `paper` → `school`, `change` → `adjustments`, `notes` → `notes`, `related` → `affiliate` |
| Status | `ok` → `circle-check`, `partial` → `circle-half-2`, `blocked` → `circle-x`, `todo` → `circle-dashed`, `wip` → `progress`, `pending` → `clock` |
| Process stages | `collect` → `camera`, `prepare` → `filter`, `train` → `barbell`, `evaluate` → `target`, `deploy` → `rocket`, `foundations` → `building` |

## Brand marks

Tabler 3.50.0 has `brand-github`, `brand-python` and `brand-rust` (grep `INDEX.tsv` for `^brand-`); it has no Hugging Face
or NVIDIA glyph. For those, use the brand's own monochrome mark from Simple Icons (CC0; `huggingface`, `nvidia`), filled
with `currentColor` at the same size as the Tabler glyphs, and say where it came from in the page source. The Rerun logo
comes from the house style ([DESIGN.md](DESIGN.md)), never from an icon set.

## Use

- Find a glyph: `grep -i <concept> assets/tabler-outline/INDEX.tsv` (name, then its Tabler categories), then open the
  SVG and look at it before you use it; names can mislead.
- Copy the glyph's drawing elements into the page as inline SVG: `viewBox="0 0 24 24"`, `fill="none"`,
  `stroke="currentColor"`, `stroke-width="1.5"` (the files ship with 2), round caps and joins. Drop the invisible
  `<path stroke="none" d="M0 0h24v24H0z"/>` frame.
- Icons are labels, always next to a word; never a picture of data (see [visual-first.md](visual-first.md)).
