# Artifact Patterns

Choose the smallest pattern that makes the material easier to understand than markdown.

## Option Comparison

Use for architecture choices, implementation approaches, vendor/tool selection, design directions.

Structure:
- Top recommendation strip with chosen option and why.
- Side-by-side columns for options.
- Shared criteria rows: complexity, risk, time, dependencies, reversibility.
- A risk table below the comparison.
- "When to choose this" bullets per option.

Useful additions:
- Weighted score sliders only if the user needs to tune priorities.
- Inline code snippets for implementation choices.
- Sticky criteria column on desktop.

## Implementation Plan

Use for multi-step engineering work.

Structure:
- Outcome statement.
- Milestone timeline.
- Dependency map or data-flow diagram.
- Work packages with owner/scope/test notes.
- Risk register.
- Verification checklist.

Useful additions:
- Expandable details for each milestone.
- Copyable command blocks.
- Status chips for `ready`, `blocked`, `risky`, `optional`.

## Code Review or PR Map

Use for reviews, pull request summaries, and unfamiliar diffs.

Structure:
- Review focus at the top.
- File-by-file map with why each file changed.
- Annotated diff snippets or behavior before/after.
- Severity-ranked findings.
- Test and rollout checklist.

Useful additions:
- Jump links by severity.
- Toggle between "reviewer summary" and "implementation detail".
- Module diagram for call flow.

## Research Explainer

Use for learning a concept, summarizing a library, or explaining how a feature works.

Structure:
- TL;DR.
- Concept map or request/path diagram.
- Progressive sections from "what" to "why" to "how".
- Tabbed examples.
- Glossary sidebar.
- Common mistakes.

Useful additions:
- Hover-linked terms.
- Small interactive simulation when it is genuinely clarifying.
- Copyable examples.

## Report or Status Update

Use for weekly status, project health, incident postmortems, research summaries.

Structure:
- Status headline.
- Key changes since last update.
- Timeline or trend chart.
- What shipped, slipped, and needs attention.
- Decisions needed.
- Follow-up list.

Useful additions:
- Inline SVG charts.
- Severity bands.
- Print-friendly CSS.

## Incident Timeline

Use for postmortems, outages, debugging narratives.

Structure:
- Impact summary.
- Timeline with timestamps.
- Detection, mitigation, resolution.
- Root cause tree.
- Follow-ups with owners and due dates.

Useful additions:
- Log excerpts in collapsibles.
- Highlighted uncertainty or evidence gaps.
- "What changed" comparison.

## Deck

Use when the artifact will be presented or walked through live.

Structure:
- One `<section>` per slide.
- Keyboard navigation with left/right arrows.
- Progress indicator.
- Speaker notes hidden in `<details>` or a notes panel.

Useful additions:
- Print stylesheet with one slide per page.
- Overview mode.

## Diagram Sheet

Use for architecture maps, process flows, mental models, figure sets.

Structure:
- Figure gallery.
- Each figure has title, caption, and takeaway.
- Inline SVG with semantic text where possible.
- Copy/export affordance if the diagram is intended to be reused.

Useful additions:
- Click-to-highlight paths.
- Toggle labels or layers.
- Legend with consistent color semantics.

## Lightweight Editor

Use when the user needs to sort, classify, tune, or transform information.

Structure:
- Input area or editable controls.
- Live preview/result.
- Validation or warnings.
- Export/copy button that produces markdown, JSON, diff, or a command.

Useful additions:
- Local-only state in `localStorage` when useful.
- Reset button.
- Import textarea for round-tripping.

## Pattern Selection Heuristics

- If the user must choose: option comparison.
- If the user must execute: implementation plan.
- If the user must review: code review map.
- If the user must learn: research explainer.
- If the user must report: status/report.
- If the user must reconstruct events: incident timeline.
- If the user must present: deck.
- If the user must understand shape: diagram sheet.
- If the user must manipulate inputs: lightweight editor.
