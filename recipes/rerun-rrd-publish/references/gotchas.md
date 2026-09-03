# Gotchas (rerun 0.37, observed 2026-09-03)

Each line: what you see, why, what to do.

- `rerun rrd route` panics on several inputs when a dataforge base recording is among them. Route one file at a time; export segments from the catalog instead of merging layer files.
- A merged file shows the wrong layout. Every dataforge base .rrd embeds its own blueprint and its activation wins. Export from the catalog (step 1); `join_trim.py` rejects a recording that embeds a blueprint.
- Frusta and transforms vanish from a clip. `rrd split` keeps annotation contexts but drops the other static chunks. `join_trim.py` writes the static slice back with the split part's recording id (`<prefix>0`).
- Only one keyframe is found. Raw dataforge exports mark `is_keyframe` true for frame 0 only. `join_trim.py` runs `rrd optimize --fix-keyframe` first and reads keyframes from that copy; `--fix-keyframe` is also what lets optimize rebatch video.
- Blank video at the end of a clip. Seen once in the web viewer when the cut fell inside a GoP and playback parked on the last frame. Cut on a keyframe (`join_trim.py` does) and loop.
- The viewer parks on the last frame. Blueprint `TimePanel(loop_mode="All")`. An explicit `TimePanel` opts out of `collapse_panels`, so set its `state`; keep `auto_layout`/`auto_views` False on the fragment or the merged layout changes.
- `rerun --save <catalog url>` writes an empty file. Catalog streams carry manifest messages a file cannot hold; use `segment_store(...).write_rrd(...)`.
- `Failed to fetch` in app.rerun.io from a localhost or Tailscale URL. Chrome blocks the https app from private addresses; test the uploaded file.
- Spinner in a view for 20-40 s in headless chromium. Software AV1 decode from the GoP keyframe; wait before judging.
- "Rerun does not yet support native AV1 decoding on Linux ARM64" (rerun-io/rerun#7755). Check video in a browser, or with a viewer build that has the fix.
- `playwright-cli console` returns only messages since the previous `console` call, plus the summary line `Total messages: N (Errors: E, Warnings: W)`.
- Keypoint-connection warnings (`could not be resolved in entity`). The annotation context links keypoints absent from that frame; log sparse keypoint sets without connections in the producer.
- `segment_store(...).summary()` prints every chunk (13 MB of text for a 1 GB segment).
