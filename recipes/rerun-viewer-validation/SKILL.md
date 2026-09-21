---
name: rerun-viewer-validation
description: Prove what the Rerun viewer rendered — pixel evidence over logs. Use when a .rrd, blueprint, or Rerun rendering must be visually verified, when a timeline sweep or video of a recording is wanted, when an .rrd must be embedded in an HTML page, or when a gradio/WebViewer surface needs browser validation.
compatibility: Requires the rerun binary (with the viewer-mcp subcommand) and ffmpeg on PATH, plus a GPU or software rasterizer for headless rendering. The web branch additionally needs a browser automation tool (playwright or chrome-devtools) and network access.
---

# Rerun Viewer Validation

Prove what rendered. Logs, metadata, and `rrd stats` say what was *sent*; only pixels say what the viewer *shows*.

## Decision tree — pick by what you're validating

1. **Static render proof** (does the .rrd load? does the blueprint lay out? do views render?) → **scripted ViewerClient**. No MCP needed, deterministic, CI-friendly.
2. **Time or UI state** (scrub to frame N, verify view X at time T, click/select entities, read panels) → **viewer MCP**.
3. **Video / timeline sweep** (watch an algorithm run) → **`scripts/rrd_to_video.py`** (this skill's helper). Never loop MCP screenshots for video.
4. **Web** (embed an .rrd in an HTML page; validate a gradio-rerun app or WebViewer embed) → read **`references/web.md`** — the iframe embed recipe (CORS, Chrome Local Network Access, tailscale) and the playwright validation recipes live there. The MCP structurally cannot reach a WASM viewer in a browser — no gRPC server to dial.

**Video vs embed** (branches 3 vs 4, when both fit): the embedded rrd is the richer artifact — fully inspectable, orbitable, scrubbable — so prefer it when the recording is browser-sized. The WASM viewer holds the whole recording in memory, so check size first (`ls -lh`, `rerun rrd stats`) and gate embeds at a few hundred MB (hard ceiling ~1.5 GiB — see `references/web.md`). Choose video when the recording is huge, when the audience only needs to *watch* (Slack, PR description), or when the data needs a visualizer the web viewer doesn't have (custom visualizers). Best of both, often: a trimmed/downsampled preview rrd for the embed plus a full-fidelity video.

Version rule for every branch: the viewer that validates must be ≥ the SDK that wrote the data — Rerun has no forward compat, so an `.rrd` written by a newer SDK will not load in an older viewer (including the WASM viewer inside a gradio-rerun app pinned to an older release). Check the writer and actual viewer versions; this skill's minimum version constraint does not guarantee recording or tool-schema compatibility.

## Headless vs headed

Default **headless** for both `ViewerClient.spawn(headless=True)` and `rerun --headless`:

- Renders real frames offscreen (1920×1080 default) given a GPU or a software rasterizer (lavapipe). In a bare container with no Vulkan adapter it panics with "No graphics adapter was found" — then fall back to the browser branch.
- No OS window → immune to the occluded/minimized-window failure (MCP `screenshot` times out if a *headed* window can't render, notably on macOS).
- Works over SSH/CI/no-`DISPLAY`. A headed spawn without `DISPLAY` wedges silently (channel fills, `rr.log` blocks forever).

Go **headed only when a human co-views**: the user wants to watch you scrub, or wants the viewer left open afterwards. All tools work identically against either — headed vs headless is user preference, not capability.

Lifecycle gotchas (both modes):
- The MCP **never spawns a viewer**. Always: spawn viewer → `connect` → work.
- `ViewerClient.spawn` resolves `rerun` from PATH — stale global installs win. **Always pass `executable_path=` pointing at the project env's rerun binary.**
- `detach_process` defaults: headless → attached (dies with your script / `close()`); headed → detached (survives; only explicit `close()` kills it). Clean up detached viewers when done.

### Linux aarch64: temporary AV1 fix for 0.38.1

Stock Rerun 0.38.1 on Linux aarch64 reports "Rerun does not yet support native AV1 decoding on Linux ARM64" for AV1 video ([upstream issue #7755](https://github.com/rerun-io/rerun/issues/7755)). This error alone does not mean the recording is broken: text logs, phases, and overlays still load, so read those before judging the recording.

The `ai-demos` channel packages tested, prebuilt `rerun-sdk` 0.38.1 wheels with the patched native viewer. Pixi installation requires no compilation. Its build string contains `av1arm64`. Select it only on Linux aarch64:

```toml
[target.linux-aarch64.dependencies]
rerun-sdk = { version = "==0.38.1", build = "av1arm64_*", channel = "https://prefix.dev/ai-demos" }
```

Projects that pin `rerun-sdk` through PyPI instead can point the dependency at the wheel behind that package (the `rerun-sdk-0.38.1-av1arm64-wheels` release in the ai-demos repo, `{ url = "…whl" }`) rather than mixing a conda and a PyPI install. Check the installed build with `pixi list` and use that environment's viewer explicitly. `rerun --version` must include `Video features: av1`, but that flag alone is not decode proof: capture nonblank, changing video frames as described below.

This is a stopgap. On a Rerun version bump, check #7755 and the release notes. Once an upstream release supports native AV1 on Linux aarch64, use that release, remove the patched package pin and recipe, and remove this subsection. Do not apply the patch to a version that already includes the fix.

## MCP: getting the tools

The server is `rerun viewer-mcp` (stdio); it dials a running viewer's gRPC `ViewerControlService`. In order of preference:

1. `mcp__rerun__*` tools already in your surface → use them.
2. No tools, no restart possible → drive the server over stdio yourself: newline-delimited JSON-RPC (`initialize` → `notifications/initialized` → `tools/call`); reuse `McpStdioClient` from `scripts/rrd_to_video.py`.
3. Register for future sessions: `claude mcp add rerun -- <env>/bin/rerun viewer-mcp` (or `codex mcp add …`). (The `viewer-mcp` subcommand exists since 0.34.)
4. Delegating to a *different* agent CLI (e.g. `claude -p --mcp-config …` from a non-Claude harness) crosses a provider boundary — confirm with the user first.

## MCP: driving the viewer

Use this order. Read **[MCP schemas and recovery](references/mcp.md)** before the first call; use `tools/list` from the selected binary to resolve tool names and arguments.

1. **Connect:** spawn the viewer, then disconnect → connect to its plain HTTP control endpoint. Reuse one connection per MCP process; reset it after a viewer restart or endpoint change.
2. **Loaded:** open the recording, then poll viewer state until it lists the intended recording and a nonempty timeline range. Bound the polling; an accepted open request is not a loaded recording.
3. **Observe:** choose the recording, timeline, range, and time units from viewer state. Read `query_tree` for current UI locators.
4. **Act:** make timeline/control calls separately. `batch` accepts only egui UI tools, for example click → `wait_for` → screenshot. For UI actions, prefer a fresh widget ID; verify the action result and resulting state.
5. **Settle:** call `wait_for` with nonzero `min_steps` before every screenshot, including retries. A presence filter must come from accessible tree text; check nonempty `matched`, not just `ok: true`.
6. **Prove:** inspect the returned or saved image. Keep viewport fixed and capture only the evidence needed (usually ≤10 MCP images). A moved playhead or successful RPC does not prove video decoding.

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

**Compatibility gate:** the bundled helper uses the legacy `connect` / `viewer_state` / `set_time` schema. Check `tools/list` first; the newer `rerun_*` schema is not supported by this helper. Use a compatible viewer that can also read the recording, or report the helper limitation and use the appropriate validation branch. Do not downgrade below the writer version.

Spawns a headless viewer, drives `rerun viewer-mcp` over stdio (`set_time` → `screenshot save_path` per frame — zero agent context), ffmpeg-encodes. Capture cost depends on scene and renderer; screenshot calls can take seconds. Measure a short sample before planning a sweep. Auto-picks the first non-`log_time` timeline; handles sequence and temporal timelines (`--frames` samples evenly across the range); stdlib-only — needs just `ffmpeg` on PATH and the project env's rerun binary. The default `--settle-ms 30` is a starting point; raise it when decoded video or overlays have not stabilized — a mostly-duplicate sweep fails loudly with that advice (`--allow-static` overrides for genuinely static scenes). Verify 2–3 sampled frames visually (Read start/middle/end PNGs with `--keep-frames`) before trusting the mp4.

## Panel visibility

Collapse the blueprint/selection/time panels whenever the frame should be all content — videos, embeds, clean screenshots:

- **Live viewer, any recording**: inspect `query_tree` for each panel's current state and toggle. Click the fresh ID only when the panel is expanded, then confirm the intended pane disappeared. Saved blueprints can override initial panel state. The video helper does this for you: `--collapse-panels`.
- **Recordings you author — and therefore embeds**, since panel state rides the saved blueprint: `rrb.Blueprint(<views>, collapse_panels=True)`, or per-panel `rrb.BlueprintPanel(state="collapsed")` / `rrb.SelectionPanel(…)` / `rrb.TimePanel(…)` with `"collapsed" | "hidden" | "expanded"`. An `.rrd` re-saved this way opens chrome-free everywhere, including the WASM viewer iframe.

## Evidence & checks

- Reports under `/tmp/rerun-viewer-validation/<timestamp>/`: screenshots, `notes.md` recording Rerun version, command, `.rrd` path/size, chosen timeline + range, wait times, renderer string, pass/fail.
- Blank or wrong visuals → inspect data before blaming blueprints: `rerun rrd verify|stats|print <file>` (use the project env's binary).
- Keep viewport fixed; wait after load and after each time change. For encoded video streams, a moved playhead proves nothing about decode — only nonblank, changing pixels do.
- Remote viewing (optional): `tailscale serve --https <port> --bg <report-dir>`, pick an unused port, `curl -k -I` the URL to confirm reachability. Path mode is fine for plain HTML + screenshots; a report that *embeds* an .rrd needs the CORS proxy setup in `references/web.md`.

## Docs

- Viewer MCP: https://rerun.io/docs/reference/viewer/mcp
- Python `ViewerClient`: https://ref.rerun.io/docs/python/main/experimental/
- Timelines: https://rerun.io/docs/concepts/logging-and-ingestion/timelines
