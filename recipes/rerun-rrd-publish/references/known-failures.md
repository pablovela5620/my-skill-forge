# Known failures (rerun 0.37, 2026-09)

Each entry: symptom, cause, what the scripts do about it.

1. **`rerun rrd route` panics** (`route.rs` chunk-id assertion) when given several inputs that include a dataforge
   base recording. Single-input routes work. The scripts only ever route one file at a time, and stage 1 avoids the
   layer files altogether by exporting the joined segment from the catalog.

2. **Wrong layout after a merge.** Every dataforge base `.rrd` embeds its own blueprint with an activation command;
   after `rrd merge` its activation comes last and the viewer applies that layout instead of yours. `rrd filter`
   cannot drop a store. Fix: export from the catalog (stage 1), which yields a recording without the embedded
   blueprint. Stage 2 rejects a recording that embeds one.

3. **Frusta and transforms vanish after `rrd split`.** Split keeps annotation contexts but drops the other static
   chunks (Pinhole, Transform3D). Stage 2 writes the static slice of the full recording with
   `RrdReader(...).stream().filter(is_static=True).write_rrd(...)` and merges it back, with the same recording id as
   the split part (`<prefix>0`, split appends the part index).
   Pick that part by its recording id, never by file name: split names its output `<stem>_<data-start>__<end>.rrd`,
   so a glob on `*_0ns__*.rrd` finds nothing as soon as the recording does not start at 0 (a part covering
   10.759-16.139 s came out as `fixed_10_692348928s__16_138764163s.rrd`, its name taken from the earliest chunk of
   any component, not from the cut). Stage 2 also deletes the other parts, which are dead weight in the work dir.

4. **Blank video at the end of a clip.** Observed once, on 2026-09-03 against app.rerun.io 0.37.0, and not
   re-tested since: the frames of a truncated final GoP stayed blank once playback parked on the last frame (still
   blank after 30 s), while mid-clip frames rendered. The mitigation stands whatever the cause is, because it costs
   nothing: cut exactly at a keyframe. Split puts the frame at the cut time in the second part, part 0 ends on a
   whole GoP (161 frames, up to 5.346 s, for the cut at 5.379588054 s here), and split has no cutoff to revise. Any
   other time makes split log `revising cutoff time to match video keyframe` for every video: later in the GoP it
   still leaves part 0 the first frame of the next GoP (162 frames for a 5.4 s cut), earlier it revises back to the
   GoP start. Stage 2 snaps the cut forward, so the clip runs a little longer than requested, and it snaps with
   `>=`: a request that lands exactly on a keyframe cuts there instead of carrying one more GoP. The length is
   measured from the timeline's first video stamp, so it means the same thing for a recording that starts at 0 and
   for one that starts at 10.759 s.

5. **Viewer parks on the last frame.** Default playback stops at the end, and a 5 s clip reaches it before the
   viewer has finished decoding. `TimePanel(loop_mode="All")` in the blueprint keeps it playing. An explicit
   `TimePanel` opts out of `Blueprint(collapse_panels=True)`, so its `state` must be set explicitly. A
   `Blueprint(TimePanel(...))` fragment also writes `/viewport` `auto_layout`/`auto_views`; keep both `False` or the
   merged layout changes.

6. **`is_keyframe` is wrong in a dataforge export.** It marks frame 0 `true` and every later frame `false`, where
   0.37 wants sparse `true` markers only. Reading keyframes from the raw export therefore finds exactly one, and
   `rrd split` cannot rebatch it: every part carries each video sample row twice (320 rows for 160 frames, about
   twice the bytes). `rerun rrd optimize --fix-keyframe` rewrites the markers. Stage 2 runs it before reading the
   keyframes, splits that fixed copy, and keeps `--fix-keyframe` on the final optimize as well.

7. **`rerun --save <catalog-url>` writes an empty file** ("Received a RrdManifest which can't be stored in a file").
   Use `segment_store(...).write_rrd(...)` from Python; both `application_id` and `recording_id` are required kwargs.

8. **`Failed to fetch` from app.rerun.io against a localhost or Tailscale server.** Chrome's private network access
   policy blocks the https app from fetching loopback or private addresses, CORS headers or not. Test through a
   scratch upload to the public dataset instead (stage 3 does this and deletes it afterwards).

9. **"Rerun does not yet support native AV1 decoding on Linux ARM64"** in the native viewer (rerun-io/rerun#7755).
   Overlays render, videos show a decode error. The browser decodes fine, and a viewer built with the AV1 fix works;
   only the stock native viewer on aarch64 is affected.

10. **Seeks look stalled in headless chromium.** Software AV1 decode of 1080p from the GoP keyframe takes many
    seconds; the three-dot spinner in a view means "decoding", not "broken". Judge a frame after 20-40 s.

11. **`segment_store(...).summary()` prints every chunk.** On a 1 GB segment that is 13 MB of text; never log it.

12. **Keypoint-connection warnings** (`could not be resolved in entity ...`) come from an annotation context whose
    skeleton links reference keypoint ids that are absent in a frame. Sparse per-frame keypoint sets should log the
    context without connections; fix it in the producer, not in the export.

13. **`playwright-cli` drops a `.playwright-cli/` session log in the working directory.** Driven by hand from a repo
    checkout, its console log and page dump land in the next commit. Stage 3 runs the browser check in a subshell
    that first cds into `--out-dir`.

14. **`playwright-cli console` reports errors two ways.** It prints a summary line
    `Total messages: 185 (Errors: 0, Warnings: 0)` and then the messages it returns, each tagged
    `[ERROR] ... @ <url>:<line>`. A gate that greps only one of them, under `set -euo pipefail`, ends the script
    silently when that shape is missing, and a stage-3 failure then looks like a clean exit with the scratch file
    still uploaded. Stage 3 reads the summary first, counts `[ERROR]` lines when there is none, says which it used,
    and puts `|| true` on every grep. It dumps the console at the `warning` level, which returns errors too. Do not
    expect message bodies in `console.txt`: `playwright-cli` returns only what arrived since the previous console
    call, and the `open` at the start of the check consumes the first batch, so against app.rerun.io the file holds
    the summary line and little else. Every message is in the `.playwright-cli/console-*.log` file in `--out-dir`.
