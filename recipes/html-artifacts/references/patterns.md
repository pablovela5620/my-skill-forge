# Artifact Patterns

Each pattern is a storyboard: one row per scene, "heading | visual | caption".
Keep only rows with a visual. Dense detail goes on the reference page.

## Contents

- Choose a pattern
- Proposal or implementation plan
- Option comparison
- Code review or PR map
- Research explainer
- Report or status update
- Incident timeline
- Deck
- Lightweight tool
- Reference page

## Choose a pattern

| The reader must | Pattern |
|---|---|
| approve a plan | Proposal or implementation plan |
| choose | Option comparison |
| review a change | Code review or PR map |
| learn how something works | Research explainer |
| catch up | Report or status update |
| rebuild what happened | Incident timeline |
| present | Deck |
| change inputs and export | Lightweight tool |

## Proposal or implementation plan

1. The result: real capture, or `hero-mock` labeled as an illustration.
2. How it works: `pipeline` with tokens for what flows.
3. What changes: `toggle`, today and proposed.
4. Behaviour under stress: `presets` with counters.
5. Proof so far: `evidence` screenshots of tests that ran.
6. Roadmap: `metro`.
7. Decisions: `flip-cards` with the tradeoff on the back.

Reference page: sources, file and line map, risk table, commands.

## Option comparison

1. The recommended option, shown working.
2. Each option in one `toggle` view with the same visual.
3. Behaviour of each option: `presets`, one preset per option.
4. Cost, size, or time: `capacity` bars side by side.
5. Tradeoffs: `flip-cards`.

## Code review or PR map

1. Behaviour before and after: `toggle` or two `evidence` shots.
2. Changed path: `pipeline`, changed stages amber, new stages green.
3. Findings: `flip-cards` ordered by severity (rose for high).
4. Test results: `evidence` crops of real output.

Reference page: file list with reasons, line references, full diffs.

## Research explainer

1. The concept as an interactive model (`hero-mock` or `presets`).
2. The mechanism: `stepper` or `pipeline`.
3. Parameters: `presets`, one story per preset.
4. Real examples: `evidence` or a real Rerun capture.
5. Terms: `flip-cards` (term on the front, meaning on the back).

## Report or status update

1. Status: `metro` with the current stop.
2. Budgets and quotas: `capacity` bars.
3. What shipped: `evidence` screenshots.
4. Trends: an inline SVG chart of real data only.
5. Decisions needed: `flip-cards`.

## Incident timeline

1. The timeline: `hero-mock` layout with real timestamps, one track per
   service, ticks for events. Label it real, not an illustration.
2. Detection, mitigation, resolution: `stepper`.
3. Evidence: cropped log or dashboard screenshots.
4. Follow-ups: `metro`.

## Deck

One scene per slide, the same recipes, arrow-key navigation, and a progress
bar. Keep speaker notes on the reference page.

## Lightweight tool

Controls, a live preview, and copy or export of the result. Offer presets
before raw inputs. Keep local state in `localStorage` only when it helps.

## Reference page

Name it `<name>-reference.html`. Use the same base blocks and plain sections.
Tables, long lists, sources, and line numbers belong here. Link to it from the
guide's last scene and from any scene that needs detail.
