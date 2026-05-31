---
name: rerun-viewer-validation
description: Validate Rerun 0.33 Viewer output with native headless screenshots by default, saved .rrd reloads, blueprints, view-specific screenshots, exact timeline screenshots, self-hosted WebViewer HTML reports, hardware WebGL checks, theme parity, optional Tailscale Serve links, and data-path checks. Use when Codex must prove what rendered in Rerun instead of relying on logs or metadata.
---

# Rerun Viewer Validation

Use Rerun SDK/CLI 0.33. Default to native headless screenshots; use WebViewer only when it adds browser/timeline evidence.

## Requirements

| Capability | Level | Check |
| --- | --- | --- |
| Rerun 0.33 | Required | Verify `import rerun as rr; print(rr.__version__)` and `rerun --version`. |
| Native headless | Default | Use `ViewerClient.spawn(headless=True)`, not a real desktop window. |
| WebViewer | Optional | Use for exact timeline/playhead screenshots or embedded reports only when the `.rrd` is under 1.5 GiB. |
| Tailscale Serve | Optional | If `tailscale` exists and remote viewing is useful, serve the report directory and give the URL. |

## Workflow

1. Prefer `save .rrd -> reload .rrd -> screenshot`; this validates serialization, blueprints/layout, and viewer loading.
2. Capture native headless first: full viewer plus important 2D/3D view IDs.
3. Inspect timelines before choosing frames. Prefer domain timelines such as `video_time`, `frame`, or sensor time.
4. Use WebViewer JS for exact begin/middle/end playhead screenshots only for small `.rrd` files.
5. Put disposable reports under `/tmp/rerun-viewer-validation/<timestamp>/`; include `index.html`, screenshots, summaries, and `notes.md`.
6. If visuals are blank or wrong, inspect data paths before changing blueprints: `rerun rrd verify|stats|print <file>`.

## Native Headless

```python
import rerun as rr
from rerun.experimental import ViewerClient

with ViewerClient.spawn(headless=True, port=9877, hide_welcome_screen=True) as viewer:
    rr.init("rrd_check", default_enabled=True, strict=True)
    rr.connect_grpc(url=viewer.url)
    rr.log_file_from_path("recording.rrd")  # preserves saved layout better than send_chunks replay
    rr.get_global_data_recording().flush(timeout_sec=30.0)
    viewer.save_screenshot("native-full.png")
    viewer.save_screenshot("native-3d.png", view_id=view_3d_id)
```

Native `ViewerClient` in 0.33 does not expose a playhead setter. It proves the current/default viewer state; use WebViewer JS when exact timeline positions matter.

To find saved blueprint view IDs:

```python
import rerun.experimental as rrx

r = rrx.RrdReader("recording.rrd")
for chunk in r.stream(store=r.blueprints()[0]).to_chunks():
    if str(chunk.entity_path).startswith("/view/"):
        print(chunk.entity_path, chunk.to_record_batch())
```

## Timelines

Choose the timeline from the data, not by assumption. For saved `.rrd` files, read index columns, types, and min/max ranges. For live gRPC, use the producer's `rr.set_time(..., sequence=|duration=|timestamp=)` contract.

For WebViewer exact-time screenshots:

```js
viewer.set_active_timeline(recordingId, "video_time")
viewer.set_current_time(recordingId, "video_time", encodedValue)
```

Encode values by timeline type: sequence number for sequence timelines, nanoseconds for duration/timedelta timelines, and Unix epoch nanoseconds for timestamp timelines. After setting time, wait for loading/video decode, then confirm `get_current_time(...)`, renderer errors, no visible loading overlay, and nonblank pixels.

## WebViewer

Use a same-origin report:

```text
report/
  index.html
  recording.rrd        # symlink is fine if the server follows it
  viewer.html
  screenshots/
  web-viewer/          # official @rerun-io/web-viewer JS/WASM pinned to Rerun 0.33
```

Self-host `@rerun-io/web-viewer` assets next to the `.rrd`; make `viewer.html` import `WebViewer` and call `viewer.start("./recording.rrd", element, options)`. Embed `viewer.html` with an iframe only when the `.rrd` is under 1.5 GiB. For larger files, skip WebViewer and report native screenshots; browser/WASM loading can fail near 2 GiB, and a 5.1 GiB EPFL RRD hit `RuntimeError: unreachable` from allocation failure.

Default WebViewer to native visual parity. Native screenshots usually render dark; in 0.33, `theme: "dark"` alone may not override the browser's light `prefers-color-scheme`. For Playwright, set `colorScheme: "dark"`. For embedded reports, pass `theme=dark` and force the media query before `viewer.start`:

```js
const theme = new URLSearchParams(location.search).get("theme") || "dark"
const realMatchMedia = window.matchMedia.bind(window)
window.matchMedia = (query) =>
  String(query).includes("prefers-color-scheme: dark")
    ? { matches: theme === "dark", media: query, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent() { return false } }
    : realMatchMedia(query)
```

For WebGL validation, reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`, `Software`). On Linux/NVIDIA browser automation, use hardware GPU flags and, only if needed, the real X11 session:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  node capture_webviewer_timeline.mjs --ozone=x11
```

Use Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` only if Chrome falls back to software.

## Tailscale Reports

If remote viewing is needed and Tailscale is available:

```bash
command -v tailscale
tailscale serve status
tailscale serve --https 10465 --bg /tmp/rerun-viewer-validation/<report>
```

Choose a new unused port if `serve status` shows a listener. Return the `https://<node>.<tailnet>.ts.net:<port>/` URL. Verify the report and `.rrd` are reachable when possible:

```bash
curl -k -I https://<node>.<tailnet>.ts.net:<port>/
curl -k -I https://<node>.<tailnet>.ts.net:<port>/recording.rrd
```

## Checks

- Record Rerun version, command, `.rrd` path/size, chosen timeline/range, wait time, renderer string, screenshots, and pass/fail in `notes.md`.
- Keep viewport/window size fixed and wait after initial load and after each timeline change.
- For encoded video, `set_current_time` only moves the playhead; it does not prove decode/render. Use screenshots and visible pixels as evidence.
- For catalog/table flows, capture catalog, table, and selected recording/segment separately.

## Docs

- Rerun web embedding: https://rerun.io/docs/howto/integrations/embed-web
- Rerun timelines: https://rerun.io/docs/concepts/logging-and-ingestion/timelines
- Rerun video: https://rerun.io/docs/concepts/logging-and-ingestion/video
- WebViewer API: https://ref.rerun.io/docs/js/0.33.0/web-viewer/classes/WebViewer.html
- Python `ViewerClient`: https://ref.rerun.io/docs/python/main/common/experimental/
- Chrome headless GPU testing: https://developer.chrome.com/blog/supercharge-web-ai-testing
- Tailscale Serve: https://tailscale.com/docs/features/tailscale-serve
