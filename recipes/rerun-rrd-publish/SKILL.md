---
name: rerun-rrd-publish
description: "Turn a Rerun catalog segment, or any recording .rrd plus blueprint .rbl, into one standalone .rrd that app.rerun.io opens by URL, optionally trimmed to a short clip that ends on a video keyframe boundary and loops, then publish it to a Hugging Face dataset and prove the link renders. Use when asked to export, trim, share, or upload a Rerun recording or .rrd, make a web-viewer example, or publish to an rrd-examples dataset."
compatibility: Stages 1-2 run their scripts with a Python that has rerun-sdk[catalog] >= 0.37 (the project env's python; the rerun CLI is invoked as `python -m rerun`). Stage 3 needs the `hf` CLI logged in as the dataset owner and `playwright-cli` with chromium. Pixel checks of AV1 video on Linux ARM64 need a browser or a viewer build with AV1 enabled.
---

# Rerun rrd publish

Three stages, each a separate script with plain files as the interface. Run them in order for the
common case, or one alone: stage 2 works on any `.rrd` + `.rbl`, stage 3 on any `.rrd`.

| stage | input | output | script |
|---|---|---|---|
| 1 fetch | catalog URL, dataset, segment | `segment.rrd` (all layers, one store) + `blueprint.rbl` (same application id) | `scripts/fetch_segment.py` |
| 2 join + trim | recording `.rrd`, blueprint `.rbl`, optional clip length | one verified `.rrd`, one recording + one blueprint | `scripts/join_trim.py` |
| 3 publish | `.rrd`, dataset repo, target name, viewer version | live viewer link + evidence screenshots | `scripts/publish_hf.sh` |

Acceptance is pixel evidence in the target viewer, never a successful upload. Read `references/known-failures.md`
before debugging anything: every entry there cost real time and none is visible from CLI help.

## Stage 1: fetch a segment from the catalog

```bash
<env>/bin/python scripts/fetch_segment.py --catalog-url rerun+http://127.0.0.1:9988 \
    --dataset <name> --segment <segment-id> --out-dir /tmp/rrd-publish/<name> [--app-id <id>]
```

- Exports the segment with `segment_store(...).write_rrd(...)`. The catalog joins the layers server-side, so the
  result is one recording store. Never assemble the on-disk layer files instead (see known failures 1 and 2).
- Takes the dataset's default blueprint from its `file://` storage URL and routes it to the export's application id.
  Refuses to run without a default blueprint: a file without one opens with an auto layout. Pass `--blueprint` when
  the catalog is on another host.
- `rerun --save <catalog-url>` cannot do this (it rejects the catalog's manifest messages), and the segment must be
  exported untrimmed: stage 2 reads keyframes and static chunks from the full recording.

## Stage 2: join, and trim to a clip

```bash
<env>/bin/python scripts/join_trim.py --recording segment.rrd --blueprint blueprint.rbl \
    --out <name>+rbl.rrd [--clip-seconds 5] [--timeline video_time] [--no-loop] [--keep-work]
```

What it does, in order:

1. Asserts one recording store, one blueprint store, same application id. A recording that embeds a blueprint is
   rejected: that is a dataforge base file, go through stage 1.
2. With `--clip-seconds`, rebuilds the keyframe markers first (`rrd optimize --fix-keyframe`, about 10 s per GB),
   reads the `VideoStream:is_keyframe` rows of that copy, and cuts exactly at the first keyframe after the requested
   length. Split puts the frame at the cut time in the second part, so the clip ends on a whole GoP. It reports the
   snapped length; expect it to be longer than asked (a 5 s request on 161-frame GoPs at 29.93 fps became 5.380 s).
3. `rerun rrd split` at that time on the fixed copy, keeping the first part. Split drops static chunks (pinholes,
   transforms) while keeping annotation contexts, so the static slice of the recording is written out and merged back.
4. Clips get `TimePanel(loop_mode=All)` merged into the blueprint, collapsed by default, so the viewer keeps playing
   instead of parking on the final frame. `--no-loop` opts out; `--loop` adds it to an untrimmed export.
5. `rerun rrd merge` of the parts and the blueprint, `rerun rrd optimize --fix-keyframe`, `rerun rrd verify`, then
   the store assertions again on the output. `--fix-keyframe` is required for dataforge-written video.

Naming convention for published examples: `<name>+rbl.rrd`, the `+rbl` marking that the blueprint is inside.

## Stage 3: publish and prove it

```bash
scripts/publish_hf.sh <name>+rbl.rrd <user>/<dataset> <name>+rbl.rrd --viewer-version <X.Y.Z> [--dry-run]
```

1. Uploads to `scratch/<timestamp>-<name>` first and checks the raw URL returns 200.
2. Opens the scratch link in headless chromium through `app.rerun.io/version/<X.Y.Z>/`, screenshots at 20 s and
   35 s, dumps the console. Stops if the console has errors. A localhost server is not an option here: Chrome blocks
   the https app from fetching localhost or private-network addresses.
3. Uploads under the final name, deletes the scratch file, prints the dataset tree and the viewer link.

Then do the acceptance test yourself: Read both screenshots. Pass only if the blueprint layout is applied (not an
auto layout), video pixels are present in the 2D views, and when looping the time cursor differs between the two
frames. Headless chromium decodes AV1 in software: a spinner at 20 s is normal, blank at 35 s is a failure. For a
finer check (seek to a time, read panels) use the `rerun-viewer-validation` skill's MCP branch on a native viewer;
on Linux ARM64 that needs a viewer build with AV1 enabled.

Pick `--viewer-version` at or above the SDK that wrote the data; Rerun has no forward compatibility.

## Evidence and hand-back

Report the viewer link, file size, requested vs actual clip length, and the two screenshot paths. Say that a
replaced file stays in the dataset's git history until the history is squashed, and do not squash it yourself.

## What this skill does not do

It never registers layers or blueprints in a catalog, never restarts or kills a catalog server, and never rewrites
dataset history. Those are separate decisions for the user.
