# Live progress pages

A page you keep current while a long task runs, so the user can refresh it and see what is happening. You update data,
never HTML: pages written as running prose turned into walls of text (one agent-written status page measured 4,614
visible words, 13 images and no diagram). The bundled renderer always draws the same shape from a typed state file.

## When to start one

- A task that runs overnight or for more than about an hour, or one you delegate to workers.
- Start it with the plan: the goal, the target metric and the stages exist before the first result does.
- One stable URL per task: `/<project>/progress/<task>.html` on the NAS ([nas-publishing.md](nas-publishing.md)).

## The state file

`state.json`, loaded through `@serde` dataclasses in `assets/progress/progress_state.py` (strict, unknown fields fail):

| Field | Holds | Limit |
|---|---|---|
| `title`, `goal`, `owner` | the task, what done means, who works on it | 80, 140, 60 characters |
| `targets[]` | a metric with `start` and `goal` (FPS 11 → 60), `higher_is_better` | name 40 |
| `stages[]` | the plan in order, each `todo`, `running`, `done`, `blocked` or `failed`, an optional one-line `note` | title 32, note 60 |
| `samples[]` | measured values of a target over time, optionally from a stage | numbers only |
| `evidence[]` | a Rerun shot or video, an image or a link per stage, with a caption that says how to read it | caption 140 |
| `needs[]` | what only the user can give: a decision, an approval, access | 120 |
| `events[]` | the timeline, one line each: `info`, `result`, `problem` or `decision` | 120 |

`replay` (one line) is set only when the page replays a recorded run; the page shows it as a banner, and the times may
then be step numbers (`#0`, `#1`, ...) instead of clock times. A state never mixes the two, and `updated` is always the
real time of the last write.

Every string is one line of at most two sentences; a line break, a longer string or a third sentence fails the load.
There is no free-text field: detail goes into evidence (a picture, a log link), not into the page.

## Update it

Use the commands; they validate before they write, so a rejected update exits 1 with the reason and leaves the file as
it was. Each takes `--t` (a UTC time like `2026-11-23T09:00Z`, or a step number in a replay; default now):

```bash
# Run from the installed skill: the Tabler glyphs ship with the package. From the recipe tree, set
# HTML_ARTIFACTS_TABLER to an installed tabler-outline folder, or the status glyphs are left out.
P="pixi exec -s python=3.13 -s pyserde -s orjson -s tomli-w -s tyro python <skill-dir>/assets/progress/progress_page.py"
$P stage    --state state.json --stage track --status done --note "14.5 ticks/s"
$P sample   --state state.json --target tps --value 26.5 --stage fit
$P event    --state state.json --text "Fit at 15 Hz: 26.5 ticks/s." --kind result --stage fit
$P evidence --state state.json --stage fit --kind image --src media/fit.jpg --caption "Fit overlay, frame 246."
$P need     --state state.json --text "Approve the 15 Hz body rate."     # no --text clears the list
$P finish   --state state.json                                           # the next render has no badge
$P render   --state state.json --out index.html
```

## The page

Always, top to bottom: the work-in-progress badge (with the running stages as its "now" line) and the "needs you" list;
the stage diagram coloured by status (with the status word and glyph); one chart per target against its goal and start;
the newest evidence per stage; the newest 12 events, older ones folded. A finished state renders without the badge.

## How often

- Update at each stage change and at each meaningful result (a new measurement, a picture that shows progress or a
  failure, a decision, a new ask), and at least every 15 minutes while the task runs, so the badge's time is never
  older than that. A quiet interval still gets an update: the current stage and what is running now.
- After each update: render, publish, and run the live-content check ([nas-publishing.md](nas-publishing.md)).
- Raise a "needs you" item as soon as you are blocked on the user, and clear it when answered.
- When the task ends: `finish`, render, publish, then `check-artifact.mjs --final <url>`.
