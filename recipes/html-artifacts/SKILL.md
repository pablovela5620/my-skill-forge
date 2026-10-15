---
name: html-artifacts
description: Builds visual-first, single-file HTML artifacts (explainers, plans, proposals, reviews, reports, comparisons, roadmaps, guides) from animated diagrams, interactive simulations, real screenshots, and real Rerun captures, with very little prose. Use when the user asks for an HTML page, artifact, visual guide, explainer, or plan to open in a browser; when an answer has a process, pipeline, timeline, architecture, before/after change, set of options, or data over time that a picture or interaction shows faster than text; or when a result goes to the fleet HTML artifact host.
license: BSD-3-Clause
compatibility: The checker needs node and playwright-cli on PATH. Image crops use ffmpeg.
---

# Visual-First HTML Artifacts

Build one self-contained `.html` file that explains by showing. The reader is a
visual learner. A page of paragraphs and tables fails, even when every fact is
right. Give each idea a picture that moves or responds. Text only labels it.

## Defaults

- **Visual first.** Every scene has a diagram, animation, simulation, or real
  image. A text-only scene moves to the reference page or goes away.
- **End product first.** Scene one shows the result as a real capture.
- **Rerun first for data over time.** For robots, sensors, recordings, and
  pipelines, show real Rerun viewer output, captured and proven with the
  `rerun-viewer-validation` skill. With no data yet, log a small synthetic
  recording with the Rerun SDK, label it synthetic, and capture that. Never
  draw an imitation of the viewer. Read [references/rerun.md](references/rerun.md).
- **Truthful motion.** Animate the mechanism (data moves, time passes, state
  changes), not the page. A smooth model must not show behaviour that the real
  system does not have.
- **One file.** Inline CSS, JS, SVG, and images. No CDN, network fonts, or
  build step. It opens from disk and works on a phone. Allowed exceptions: a
  linked `<name>-reference.html` for dense detail; a sidecar video over about
  2 MB in the same folder; a live Rerun embed, only when the user agrees,
  because it needs the network.
- **Dark, tinted surface** unless the scene needs light (print, projection).

## Text budget

| Element | Limit |
|---|---|
| Scene heading | 6 words |
| Sub-line under the heading | 1 sentence, about 15 words |
| Caption, card face | 12 words |
| Paragraph | none; at most 2 short sentences when there is no other way |
| Prose in one scene (heading, sub-line, captions) | about 40 words |
| Label inside a visual (stage, track, chip) | 1 to 4 words |

Write in ASD-STE100 style: short sentences, active voice, common words, no
filler. Put dense detail (sources, line numbers, tables, logs, long lists) on a
separate linked page (`<name>-reference.html`), not in the guide.

## Pick the visual

| Content shape | Recipe |
|---|---|
| The result, a run | `rerun-hero`: real Rerun sweep video with event chapters |
| A UI or page result | `evidence` screenshot |
| Data moves through stages | `pipeline` with moving tokens |
| Ordered steps | `stepper` that lights each step in turn |
| Before and after, hidden and shown | `toggle` |
| Behaviour under different conditions | `presets` simulation with big counters |
| Decisions and their costs | `flip-cards` |
| Roadmap, milestones | `metro` line |
| Budget, capacity, quota | `capacity` bar |
| Proof that something works | `evidence`: cropped real screenshot with a box |
| Data over time | real Rerun capture ([references/rerun.md](references/rerun.md)) |

Every recipe is a marked block in [assets/gallery.html](assets/gallery.html),
tested at 390 px and 1440 px. [references/components.md](references/components.md)
tells when to use each one and how to copy it.

## Workflow

Copy this checklist and tick it off:

```
- [ ] 1 Question: the one question the page answers; the end product to show first
- [ ] 2 Storyboard: one row per scene, "heading | visual | caption"; no row without a visual
- [ ] 3 Evidence: real screenshots, Rerun captures, measured numbers
- [ ] 4 Build: gallery base blocks + recipe blocks; prefix each component's classes
- [ ] 5 Check: run the checker, read every screenshot, fix, run again until PASS
- [ ] 6 Critique: test against the bans; remove one thing that does not earn its place
- [ ] 7 Publish: fleet host, verify the served file, give the full URL
```

## Check

```bash
node <skill-dir>/scripts/check-artifact.mjs page.html --steps steps.js
```

The checker opens a fresh playwright-cli session at 390 px and 1440 px and
saves one screenshot per `section[id]` and per steps-file `shot()`. It fails on
console errors, horizontal overflow or broken media after any screenshot,
looping CSS animations under reduced motion, and a throwing steps file. A PASS
does not prove the page looks right:

1. Read every screenshot. Look for clipped labels, empty visuals, and overlaps.
   Read every Rerun frame you embed too: a moved playhead is not proof.
2. Click every control in a steps file and photograph the result.
3. Fix the cause, then run again.

Steps-file API, pitfalls, and publishing: [references/verification.md](references/verification.md).

## Bans

Name the pattern to avoid. "Avoid a generic look" only swaps one default for another.

- Walls of text: paragraphs over 2 sentences, text-only scenes, a table as the
  main content.
- Decorative motion: a fade-and-slide entrance on every section, hover effects
  on every card, parallax, spinners.
- Raw sliders when 2 or 3 presets tell the story.
- A drawn imitation of the Rerun viewer, and invented data shown as real. Log a
  real recording, or a synthetic one labeled as synthetic, and capture it.
- Default styles: "01/02/03" labels on content that is not a sequence; ALL-CAPS
  tracked eyebrow labels; monospace for labels that are not code; one accent
  word in a headline; cream or off-white pages; purple gradients; gradient
  text; glassmorphism; neon glow; nested cards; identical card grids as filler.
- Colour without meaning. Give each colour one meaning on every scene (for
  example blue = reuse, amber = change, green = new or success, rose = risk)
  and pair it with a label or icon.
- External assets, dead controls, placeholder text, and text that explains how
  the page works.

## References

- [references/components.md](references/components.md): recipe catalog, copy command, rules per recipe.
- [references/rerun.md](references/rerun.md): real Rerun screenshots, sweep videos, synthetic recordings, embeds.
- [references/verification.md](references/verification.md): checker, steps files, pitfalls, publishing.
- [references/design-quality.md](references/design-quality.md): theme, colour, type, accessibility.
- [references/patterns.md](references/patterns.md): scene lists for plans, reviews, reports, decks.
