---
name: rerun-viewer-validation
description: Rerun pixels and Rerun media in pages and public shares. Use when a recording, blueprint or Rerun rendering must be proven with screenshots, when a recording or catalog segment becomes stills, a video, an embed in a page or a public app.rerun.io link (medium, size budget, slicing, versions), or when a web viewer or gradio surface needs browser checks.
compatibility: Needs the rerun binary (0.42+; its MCP server is `rerun mcp` from 0.44, `rerun viewer-mcp` on 0.42) and ffmpeg (libx264, libsvtav1) on PATH, and a GPU for the headless viewer. The slice check and the catalog export need the rerun SDK; browser checks need playwright-cli or Chrome.
---

# Rerun viewer validation

**Pixels** prove what rendered; logs, a clean exit and `rrd stats` prove only what was sent. Work **versions first**,
capture in **batches** from one headless viewer, and keep every page recording inside its **budget**.

## Steps

### 1. Versions first

Read three versions before you judge anything:

- **Writer** of the recording: `python scripts/rrd_check.py version run.rrd` (RRD header bytes 4-7; works on `.rbl`
  too), or "Source RRD version" in the viewer's Selection panel.
- **Viewer**: `<env>/bin/rerun --version` for the native viewer; `data-version` and `/rerun-viewer/versions.json` for
  a page.
- **Catalog server**, when one serves the data: its `rerun --version`.

The viewer must be the same or newer than the writer. A newer viewer opens older recordings as they are: the 0.42.0
web viewer opened 0.37.0 and 0.38.1 recordings with their blueprints (it migrates the data on load). An older viewer
cannot open a newer recording. On a skew:

- Newer writer: use a viewer of at least the writer's version (the project env's binary; for pages, install that web
  viewer with `pixi run python -m scripts.rerun_viewer_install <version>` in `~/agent-fleet`).
- Messages such as "newer than the local Rerun version", a Sorbet version mismatch, or `Operation is not implemented`
  from a server mean a skew: switch binaries before you judge the recording.
- `rerun rrd migrate f.rrd` rewrites a file to the newest format in place (it keeps `f.backup.rrd`) and replaces the
  writer version, so run it only when a tool needs the new format. What changed between releases:
  [migration guides](https://rerun.io/docs/reference/migration) and the changesets
  (`https://rerun.io/docs/changelog/changeset-0-<minor>`).

**Done when** your notes name the writer, viewer and server versions, and viewer ≥ writer.

### 2. Start one viewer on the GPU

`python scripts/headless_viewer.py start --rerun-bin <env>/bin/rerun <file>…` (or let the capture scripts start it).
It claims a free port with a lock file (never 9876), and stops the viewer by process group. Start it with
`RUST_LOG=info,re_renderer=debug` once and read the `Picked adapter:` line: a CPU adapter (llvmpipe, lavapipe,
SwiftShader) makes timings and video-decode evidence invalid. A GPU probe that fails in the agent sandbox says nothing
about the host: ask to rerun it outside the sandbox. Take AV1 video evidence on an x86_64 GPU host (pablo-dl-server),
not the Spark.

**Done when** the adapter line names a GPU, and nothing else answers on the viewer's port.

### 3. Capture in batches

| Need | Tool |
|---|---|
| Stills at chosen times | `scripts/stills.py --rrd run.rrd [--rbl run.rbl] --timeline t --at 12s --at 14s --out-dir shots/` |
| A video or a sweep | `scripts/rrd_to_video.py --rrd run.rrd --out sweep.mp4` (report) or `--embed <dir>/run` (page set) |
| UI state: clicks, panels, selection | the viewer MCP: [references/mcp.md](references/mcp.md) |
| Proof inside a Python script | `rerun.experimental.ViewerClient`: `spawn(headless=True, executable_path=…)`, `open`, `set_time`, `save_screenshot` |

A page set for one 4 s window, with its poster:

```bash
python scripts/rrd_to_video.py --rrd run.rrd --rbl run.rbl --rerun-bin <env>/bin/rerun \
  --timeline video_time --start 10s --end 14s --poster-at 12s --collapse-panels --embed <dir>/run
```

`stills.py` and `rrd_to_video.py` drive one viewer through one MCP session (`rerun mcp`, or `viewer-mcp` on 0.42): 10 stills take under 1 s after
load, where one MCP screenshot per agent turn took a median 10 s in agent traces. `stills.py` refuses a viewer that
holds recordings other than yours.

`rrd_to_video.py` plays a temporal window at real speed up to `--max-seconds` (default 15), so a 4 s window is
60 frames. `--seconds` forces slow or fast motion. The screenshot waits for the frame, so `--settle-ms` stays 0 unless
the stale-frame check fails. That check compares the views without the time panel; when it fails on a scene that
really does not change, pass `--allow-static '<why>'`. The script prints `TIMING` with set_time, settle and screenshot
seconds.

**Done when** you have opened and looked at every still and 2-3 frames of every video (`--keep-frames`), and each
shows the data, not an empty or loading view.

### 4. Put a recording in a page or a public share

Follow [references/pages.md](references/pages.md): the source (a file, or a catalog segment exported with
`scripts/catalog_export.py`), the medium (still, video or viewer), the size budget for a 10 Mbps reader, the slice
(`rerun rrd split` + `rerun rrd optimize` + `scripts/rrd_check.py slice`), the size gate (`scripts/rrd_check.py
gate`), versions in pages, the embed, and the public share (Hugging Face and an app.rerun.io link). Browser checks of
the result: [references/web.md](references/web.md).

**Done when** `rrd_check.py gate` passes (or prints an `--allow-large` reason), the slice check passes, and the
embed passes the browser checks at the four check sizes; for a public share, Pablo has the checked file, the
`hf upload` command and the link.

### 5. Record the evidence

Write `notes.md` next to the pictures (`/tmp/rerun-viewer-validation/<timestamp>/`): versions, adapter, commands,
recording path and size, timeline and range, `TIMING` lines, and pass or fail per check.

**Done when** another agent can rerun every capture from `notes.md` alone.

## Rules

- **Own what you start.** Stop viewers by process group (`headless_viewer.py stop <PID>`); confirm by port
  (`ss -ltnp`) that nothing you started still listens. Stop processes by PID; shared catalog servers are also named
  `rerun`.
- **One viewer, one capture.** A port that already answers belongs to someone else; claim a new one.
- **Settle with a real condition.** `wait_for` takes a filter observed in `widget_tree`, or `min_steps`, with a
  `timeout_secs` that fits the scene. For a plain pause, use `min_steps`.
- **Look at the picture.** A moved playhead or `ok: true` proves nothing about decoded video; nonblank, changing
  pixels do.
- **Panels.** For clean frames, collapse panels in the blueprint (`rrb.Blueprint(..., collapse_panels=True)` or
  per-panel `state="collapsed"`) or pass `--collapse-panels` to `rrd_to_video.py`.
- **Blank or wrong pictures:** inspect the data first (`rerun rrd verify|stats|print`, with the project env's binary).
