---
name: rerun-viewer-validation
description: Validate Rerun 0.34 viewer output — prove what rendered instead of trusting logs. Native headless screenshots via ViewerClient, timeline scrubbing and UI interaction via the rerun viewer MCP, timeline-sweep videos via a shipped helper, and playwright for gradio/WebViewer browser deliverables. Use when Rerun rendering, a .rrd, a blueprint, or a viewer-embedding must be visually verified.
---

# Rerun Viewer Validation (0.34)

Prove what rendered. Logs, metadata, and `rrd stats` say what was *sent*; only pixels say what the viewer *shows*.

## Decision tree — pick by what you're validating

1. **Static render proof** (does the .rrd load? does the blueprint lay out? do views render?) → **scripted ViewerClient**. No MCP needed, deterministic, CI-friendly.
2. **Time or UI state** (scrub to frame N, verify view X at time T, click/select entities, read panels) → **viewer MCP**.
3. **Video / timeline sweep** (watch an algorithm run) → **`scripts/rrd_to_video.py`** (this skill's helper). Never loop MCP screenshots for video — every frame returns inline into context.
4. **Browser deliverable** (gradio app with gradio-rerun, embedded WebViewer HTML report) → **playwright / chrome-devtools**. The MCP structurally cannot reach a WASM viewer in a browser — no gRPC server to dial. To *build* such a deliverable, see "Embedding an .rrd in an HTML report".

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

The server is `rerun viewer-mcp` (stdio); it dials a running viewer's gRPC `ViewerControlService`. If `mcp__rerun__*` tools are already in your surface, use them. If not, **don't wait for a restart — delegate to a cold subagent**:

```bash
cat > /tmp/rerun-mcp.json <<EOF
{"mcpServers": {"rerun": {"command": "<env>/bin/rerun", "args": ["viewer-mcp"]}}}
EOF
claude -p --strict-mcp-config --mcp-config /tmp/rerun-mcp.json \
  --allowedTools "mcp__rerun__*" --model sonnet \
  "Connect to the viewer at http://127.0.0.1:<port>, then <task>. Report findings."
```

For a persistent registration instead: `claude mcp add rerun -- <env>/bin/rerun viewer-mcp` (takes effect next session). The binary must come from an env whose `rerun --version` is ≥ 0.34 — never bare `rerun` from PATH.

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
    viewer.save_screenshot("native-3d.png", view_id=view_3d_id)  # per-view
```

Prefer `save .rrd → reload → screenshot`: it validates serialization, blueprint, and viewer loading in one pass. `ViewerClient` has **no time-cursor setter** — the playhead lives in the MCP (`set_time`) only. To find saved blueprint view IDs:

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
  --rerun-bin <env>/bin/rerun [--timeline frame] [--frames 150] [--fps 15]
```

Spawns a headless viewer, drives `rerun viewer-mcp` over stdio (`set_time` → `screenshot save_path` per frame — zero agent context), ffmpeg-encodes. ~150 frames ≈ 2 min. Auto-picks the first non-`log_time` timeline; needs `ffmpeg` on PATH and rerun-sdk importable. Verify 2–3 sampled frames visually (Read start/middle/end PNGs with `--keep-frames`) before trusting the mp4.

## Embedding an .rrd in an HTML report (no JS)

Preferred embed: an `<iframe>` of the hosted viewer — never self-host `@rerun-io/web-viewer` for a report.

```html
<iframe src="https://app.rerun.io/version/<ver>/index.html?url=<rrd-url>&theme=dark"
        style="width:100%;aspect-ratio:16/9" allow="fullscreen"></iframe>
```

Three requirements, all verified:
1. **CORS**: the WASM viewer fetches `<rrd-url>` cross-origin, so the `.rrd` response must send `Access-Control-Allow-Origin: *`. `tailscale serve` path mode sets no headers — run a small CORS+Range file server on localhost and use proxy mode: `tailscale serve --https=<port> http://127.0.0.1:<local>` (headers pass through).
2. **Chrome Local Network Access**: Chrome 138+ gates fetches from a public origin (app.rerun.io) to private address space (Tailscale 100.64/10, LAN IPs) behind a user permission. Symptom: the rrd request sits `pending` forever in headless/CI, and real browsers show a one-time "allow local network" prompt. **Zero-prompt dodge**: also proxy the viewer itself — `tailscale serve --https=<port2> https://app.rerun.io` — and iframe `https://<node>.ts.net:<port2>/version/<ver>/index.html?url=…`; both origins are then private, so no gate. Firefox/Safari don't gate.
3. **Version + size**: pin `/version/<ver>/` ≥ the SDK that wrote the `.rrd`; keep files under ~1.5 GiB (WASM allocation fails near 2 GiB).

`?url=` also accepts `rerun://` dataset URIs and `rerun+http://…/proxy` live-SDK endpoints. The recording never leaves your network: the viewer assets are static; the `.rrd` fetch happens in the reader's browser.

## Browser branch: gradio apps & WebViewer validation

Only for validating the web deliverable itself. Use playwright / chrome-devtools against the running gradio app or embedded-viewer report. Surviving recipes:

- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- **WebGL**: reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`). Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` (+ real `DISPLAY`/`XAUTHORITY`) only if it still falls back.
- **Size limit**: embed WebViewer only under ~1.5 GiB `.rrd`; WASM allocation fails near 2 GiB (`RuntimeError: unreachable`). Larger files → native screenshots only.
- WebViewer JS `set_current_time(recordingId, timeline, value)` encodes like MCP `set_time` (sequence index / ns / epoch-ns); after seeking, wait for decode and confirm nonblank pixels.

## Evidence & checks

- Reports under `/tmp/rerun-viewer-validation/<timestamp>/`: screenshots, `notes.md` recording Rerun version, command, `.rrd` path/size, chosen timeline + range, wait times, renderer string, pass/fail.
- Blank or wrong visuals → inspect data before blaming blueprints: `rerun rrd verify|stats|print <file>` (use the ≥0.34 binary).
- Keep viewport fixed; wait after load and after each time change. For encoded video streams, a moved playhead proves nothing about decode — only nonblank, changing pixels do.
- Remote viewing (optional): `tailscale serve --https <port> --bg <report-dir>`, pick an unused port, `curl -k -I` the URL and the `.rrd` to confirm reachability.

## Docs

- Viewer MCP: https://rerun.io/docs/reference/viewer/mcp
- Python `ViewerClient`: https://ref.rerun.io/docs/python/main/experimental/
- Timelines: https://rerun.io/docs/concepts/logging-and-ingestion/timelines
- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
