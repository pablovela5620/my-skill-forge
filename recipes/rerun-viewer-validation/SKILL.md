---
name: rerun-viewer-validation
description: Prove what the Rerun viewer (≥0.34) rendered — pixel evidence over logs. Use when a .rrd, blueprint, or Rerun rendering must be visually verified, when a timeline sweep or video of a recording is wanted, when an .rrd must be embedded in an HTML page, or when a gradio/WebViewer surface needs browser validation.
---

# Rerun Viewer Validation (0.34)

Prove what rendered. Logs, metadata, and `rrd stats` say what was *sent*; only pixels say what the viewer *shows*.

## Decision tree — pick by what you're validating

1. **Static render proof** (does the .rrd load? does the blueprint lay out? do views render?) → **scripted ViewerClient**. No MCP needed, deterministic, CI-friendly.
2. **Time or UI state** (scrub to frame N, verify view X at time T, click/select entities, read panels) → **viewer MCP**.
3. **Video / timeline sweep** (watch an algorithm run) → **`scripts/rrd_to_video.py`** (this skill's helper). Never loop MCP screenshots for video.
4. **Web** (embed an .rrd in an HTML page; validate a gradio-rerun app or WebViewer embed) → read **`references/web.md`** — the iframe embed recipe (CORS, Chrome Local Network Access, tailscale) and the playwright validation recipes live there. The MCP structurally cannot reach a WASM viewer in a browser — no gRPC server to dial.

Version rule for every branch: the viewer that validates must be ≥ the SDK that wrote the data. Rerun has no forward compat — a 0.34-written `.rrd` will not load in a 0.33 viewer (including the 0.33 WASM viewer inside `gradio-rerun==0.33.0` apps).

## Headless vs headed

Default **headless** for both `ViewerClient.spawn(headless=True)` and `rerun --headless`:

- Renders real frames offscreen (1920×1080 default) given a GPU or a software rasterizer (lavapipe). In a bare container with no Vulkan adapter it panics with "No graphics adapter was found" — then fall back to the browser branch.
- No OS window → immune to the occluded/minimized-window failure (MCP `screenshot` times out if a *headed* window can't render, notably on macOS).
- Works over SSH/CI/no-`DISPLAY`. A headed spawn without `DISPLAY` wedges silently (channel fills, `rr.log` blocks forever).

Go **headed only when a human co-views**: the user wants to watch you scrub, or wants the viewer left open afterwards. All tools work identically against either — headed vs headless is user preference, not capability.

Lifecycle gotchas (both modes):
- The MCP **never spawns a viewer**. Always: spawn viewer → `connect` → work.
- `ViewerClient.spawn` resolves `rerun` from PATH — stale global installs win. **Always pass `executable_path=` pointing at the project env's ≥0.34 binary.**
- `detach_process` defaults: headless → attached (dies with your script / `close()`); headed → detached (survives; only explicit `close()` kills it). Clean up detached viewers when done.

## MCP: getting the tools

The server is `rerun viewer-mcp` (stdio); it dials a running viewer's gRPC `ViewerControlService`. In order of preference:

1. `mcp__rerun__*` tools already in your surface → use them.
2. No tools, no restart possible → drive the server over stdio yourself: newline-delimited JSON-RPC (`initialize` → `notifications/initialized` → `tools/call`); reuse `McpStdioClient` from `scripts/rrd_to_video.py`.
3. Register for future sessions: `claude mcp add rerun -- <env>/bin/rerun viewer-mcp` (or `codex mcp add …`). Older binaries lack the subcommand.
4. Delegating to a *different* agent CLI (e.g. `claude -p --mcp-config …` from a non-Claude harness) crosses a provider boundary — confirm with the user first.

## MCP: driving the viewer

17 tools: `connect`, `disconnect`, `viewer_state`, `set_time`, `open_url` (rerun-specific) + `query_tree`, `get_node`, `screenshot`, `click`, `drag`, `hover`, `scroll`, `press_key`, `type_text`, `resize`, `wait_for`, `batch` (egui UI, accessibility-tree based). Work observe → act → verify.

- `connect` takes `endpoint: "http://127.0.0.1:<port>"` — plain http, **not** the SDK's `rerun+http://…/proxy` URL.
- `open_url` loads recordings: absolute file path (no `file://` prefix), `rerun://` dataset URI, or https URL.
- `viewer_state` first, always: recordings + per-timeline `{timeline, type, min, max}` + current time. Choose the timeline from this data, never by assumption.
- `set_time`: `time` is a sequence index for `sequence` timelines, **nanoseconds** for duration/timestamp timelines. `play: true` to run from there; default stays paused.
- `screenshot` **always returns the PNG inline into context**; `save_path` writes to disk *in addition*. Budget ≤ ~10 MCP screenshots per validation — seeing evidence frames is the point; sweeping is the helper's job.
- Prefer locators (`id` from `query_tree`, `role`/`label_contains`) over raw `pos`; everything is in logical points (screenshot pixels at `pixels_per_point: 1.0` align 1:1 with click coordinates).
- `batch` chains act+observe (e.g. `set_time` → `screenshot`) in one round trip.

## ViewerClient: scripted static proof

```python
import rerun as rr
from rerun.experimental import ViewerClient

with ViewerClient.spawn(
    headless=True, port=9877, hide_welcome_screen=True,
    executable_path="<env>/bin/rerun",  # NEVER rely on PATH
) as viewer:
    rr.init("rrd_check", default_enabled=True, strict=True)
    rr.connect_grpc(url=viewer.url)
    rr.log_file_from_path("recording.rrd")  # preserves saved blueprint/layout
    rr.get_global_data_recording().flush(timeout_sec=30.0)
    import time; time.sleep(3.0)  # let ingestion + first render settle
    viewer.save_screenshot("native-full.png")
```

Prefer `save .rrd → reload → screenshot`: it validates serialization, blueprint, and viewer loading in one pass. `ViewerClient` has **no time-cursor setter** — the playhead lives in the MCP (`set_time`) only.

**Per-view capture works only for views the viewer is currently rendering.** The safe pattern is authoring the blueprint in-process: `view = rrb.Spatial3DView(…); rr.send_blueprint(view); viewer.save_screenshot(p, view_id=view.id)` — returns in milliseconds. The trap: a `view_id` the viewer can't resolve to a rendered view (an unknown uuid, or a saved-blueprint view right after replaying an `.rrd`) gets **no reply and the blocking call hangs forever, with no diagnostic on 0.34.0**. So always run `view_id` calls in a killable child process with a timeout, and for replayed recordings prefer cropping the full screenshot (view rectangles are deterministic for a fixed viewport). To enumerate a recording's saved views (their `/view/<uuid>` ids are the same namespace as `view.id`, but resolve only while rendered):

```python
import rerun.experimental as rrx
r = rrx.RrdReader("recording.rrd")
for chunk in r.stream(store=r.blueprints()[0]).to_chunks():
    if str(chunk.entity_path).startswith("/view/"):
        print(chunk.entity_path, chunk.to_record_batch())
```

## Video: timeline sweep to mp4

```bash
python scripts/rrd_to_video.py --rrd recording.rrd --out sweep.mp4 \
  --rerun-bin <env>/bin/rerun [--timeline frame] [--frames 150] [--fps 15] [--collapse-panels]
```

Spawns a headless viewer, drives `rerun viewer-mcp` over stdio (`set_time` → `screenshot save_path` per frame — zero agent context), ffmpeg-encodes. 120 frames at 1080p ≈ 10 s: the per-frame cost is the settle wait plus a ~32 ms screenshot RPC, so `--settle-ms` is the speed/fidelity dial. Auto-picks the first non-`log_time` timeline; handles sequence and temporal timelines (`--frames` samples evenly across the range); stdlib-only — needs just `ffmpeg` on PATH and a ≥0.34 rerun binary. The default `--settle-ms 30` is enough for decoded video frames; raise to 100–400 when overlay-heavy views (detections, segmentation) must fully stabilize per frame — a mostly-duplicate sweep fails loudly with that advice (`--allow-static` overrides for genuinely static scenes). Verify 2–3 sampled frames visually (Read start/middle/end PNGs with `--keep-frames`) before trusting the mp4.

## Panel visibility

Collapse the blueprint/selection/time panels whenever the frame should be all content — videos, embeds, clean screenshots:

- **Live viewer, any recording**: the top bar has one labeled toggle per panel; MCP `click` with `label_contains` = `"Blueprint panel toggle"`, `"Time panel toggle"`, `"Selection panel toggle"`. A fresh viewer starts with panels expanded, so one click each collapses; confirm via `query_tree` (the `_streams_tree` / `_selection_panel` panes disappear). The video helper does this for you: `--collapse-panels`.
- **Recordings you author — and therefore embeds**, since panel state rides the saved blueprint: `rrb.Blueprint(<views>, collapse_panels=True)`, or per-panel `rrb.BlueprintPanel(state="collapsed")` / `rrb.SelectionPanel(…)` / `rrb.TimePanel(…)` with `"collapsed" | "hidden" | "expanded"`. An `.rrd` re-saved this way opens chrome-free everywhere, including the WASM viewer iframe.

## Evidence & checks

- Reports under `/tmp/rerun-viewer-validation/<timestamp>/`: screenshots, `notes.md` recording Rerun version, command, `.rrd` path/size, chosen timeline + range, wait times, renderer string, pass/fail.
- Blank or wrong visuals → inspect data before blaming blueprints: `rerun rrd verify|stats|print <file>` (use the ≥0.34 binary).
- Keep viewport fixed; wait after load and after each time change. For encoded video streams, a moved playhead proves nothing about decode — only nonblank, changing pixels do.
- Remote viewing (optional): `tailscale serve --https <port> --bg <report-dir>`, pick an unused port, `curl -k -I` the URL to confirm reachability. Path mode is fine for plain HTML + screenshots; a report that *embeds* an .rrd needs the CORS proxy setup in `references/web.md`.

## Docs

- Viewer MCP: https://rerun.io/docs/reference/viewer/mcp
- Python `ViewerClient`: https://ref.rerun.io/docs/python/main/experimental/
- Timelines: https://rerun.io/docs/concepts/logging-and-ingestion/timelines
