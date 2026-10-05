---
name: rerun-rrd-publish
description: Publish a Rerun recording as one standalone .rrd that app.rerun.io opens by URL. Use when asked to export a catalog segment to a file, trim an .rrd to a short clip, or upload an .rrd example to Hugging Face.
compatibility: Scripts run with a Python that has rerun-sdk >= 0.37 with the catalog extra (the project env's python); they call the rerun CLI as `python -m rerun`. Step 3 uses the `hf` CLI and `playwright-cli`.
---

# Rerun rrd publish

Three steps; each hands the next one a file. Flags: `--help` on each script.

## 1. Export the segment

```bash
<env>/bin/python scripts/fetch_segment.py --catalog-url rerun+http://<host>:<port> --dataset <name> --segment <id> --out-dir <dir>
```

Done when it prints the layer list and `<dir>` holds `segment.rrd` and `blueprint.rbl`. The catalog joins the
layers server-side, so the export is one recording store; the blueprint is the dataset's default, routed to the
same application id. Skip this step when you already have a recording and its blueprint.

## 2. Join, and clip

```bash
<env>/bin/python scripts/join_trim.py --recording segment.rrd --blueprint blueprint.rbl --out <name>+rbl.rrd [--clip-seconds 5]
```

Done when it prints `recordings=1 blueprints=1`. A clip ends on the first video keyframe at or after the requested
length, so a 5 s request becomes about 5.4 s, and it loops. Name published files `<name>+rbl.rrd`: the `+rbl`
says the blueprint is inside.

## 3. Upload and look

```bash
hf upload <user>/<dataset> <name>+rbl.rrd <name>+rbl.rrd --repo-type dataset
```

Link: `https://app.rerun.io/version/<rerun version>/index.html?url=<encoded>`, where `<encoded>` is the
percent-encoded `https://huggingface.co/datasets/<user>/<dataset>/resolve/main/<name>+rbl.rrd`, so the file name
ends in `%2Brbl.rrd`. Use the viewer version that wrote the data or newer.

```bash
playwright-cli open "<link>" && playwright-cli resize 1920 1080 && sleep 30 && playwright-cli screenshot --filename check.png && playwright-cli console
```

Done when the screenshot shows the blueprint's layout with video pixels in the 2D views and the console reports
`Errors: 0`. Headless chromium decodes AV1 in software, so a spinner at 30 s means wait, and blank video at 60 s
means fail. To fix a file, upload it again under the same name; the link stays.

Something failed or looks wrong: read `references/gotchas.md`.
