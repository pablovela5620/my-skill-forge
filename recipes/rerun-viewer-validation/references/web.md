# Web branch: embedding .rrd files and validating browser surfaces

## Embedding an .rrd in an HTML page

### On Pablo's NAS (tailnet pages, `:8768`)

Use the shared viewer and the embed script; never vendor another viewer copy.

- `/rerun-viewer/<ver>/` is the app build from `app.rerun.io/version/<ver>/`;
  `/rerun-viewer/<ver>/npm/` is `@rerun-io/web-viewer` `<ver>`. Installed: 0.42.0.
  Add a version with `cd ~/agent-fleet && pixi run python -m scripts.rerun_viewer_install <ver>`.
- Mark each embed with one element; `/rerun-viewer/embed.js` shows the video
  (`data-video` stem: `.av1.mp4` first, `.h264.mp4` fallback) with the poster,
  "Open live viewer (NN MB)" (lazy JS viewer, one at a time), a
  "Video | Live viewer" toggle, "Open full screen" and "Copy rerun command".
  Every embed has a video and a poster; the live viewer is opt-in:

```html
<link rel="stylesheet" href="/rerun-viewer/embed.css">
<script type="module" src="/rerun-viewer/embed.js"></script>
<figure class="rr-embed" data-rrd="run.rrd" data-rbl="run.rbl" data-video="run"
        data-poster="run.poster.jpg" data-version="0.42.0" data-time="frame@805">
  <a href="/rerun-viewer/0.42.0/embed.html?url=<abs rrd url>&url=<abs rbl url>&seek=frame@805"><img src="run.poster.jpg" alt="..."></a>
  <figcaption>...</figcaption>
</figure>
```

- Make the video and poster with
  `cd ~/agent-fleet && pixi run python -m scripts.rerun_embed_video <rrd> --rbl <rbl> --out <dir>/run --timeline <tl> --start <t0> --end <t1> --poster-at <t>`.
  It drives `rerun --headless` through `viewer-mcp` (set time, screenshot)
  and encodes AV1 (SVT-AV1 CRF 38, screen content, keyframe every 2 s) and
  H.264. Two traps it handles: the native viewer registers a local `.rrd`
  with its Viewer catalog, which gives the recording a random application id
  so the `.rbl` does not apply (it serves local files over 127.0.0.1 HTTP
  instead), and an `.rbl` passed with the `.rrd` on the command line applies
  only some of the time (it opens the `.rbl` after the recording has loaded
  and waits until the views change).
- Same origin, so no CORS: the page, the viewer and the `.rrd` (published
  next to the page) are all on `https://pablos-4800gt.ilish-ruler.ts.net:8768`.
  That server is Tailscale Serve path mode: no CORS headers, and
  `<dir>/index.html?query` redirects to the site root, so link to `<dir>/?query`.
- `data-version` >= the SDK that wrote the recording. Keep the `.rrd` under
  ~300 MB and log video as `VideoStream`.
- Reference page and source: `https://pablos-4800gt.ilish-ruler.ts.net:8768/rerun-viewer/demo.html`,
  `~/agent-fleet/config/html-artifacts/` (README lists every attribute).

### What the 0.42 viewer takes in the URL

| Feature | Status |
|---|---|
| Several `url=` (`.rrd` + `.rbl`) | Works (`query.getAll("url")`; the JS `start` takes an array). |
| `.rbl` as a URL | Works: applies to the recording with the same application id. |
| `theme`, `persist`, `renderer`, `hide_welcome_screen` | Work. |
| `#when=` / `#time_selection=` on an `.rrd` URL | Not applied: only catalog dataset URIs (`rerun+http://`) carry fragments. Set the time with the JS API: `viewer.set_time({ timeline, sequence \| duration_nanos \| timestamp_nanos, play_state: "paused" })`, after `get_state().loading` is empty (a blueprint that loads later resets the cursor). |

0.42 JS API names are snake_case (`get_state`, `set_time`, `set_location`,
`open`, `close`, `stop`); newer docs show camelCase. Pass
`check_for_updates_on_startup: false` to skip the GitHub version check (it
returns 403 on the tailnet). The viewer sets inline styles on the parent
element it gets; give it an inner element that fills your box.

### `.rbl` "Blueprint failed validation"

`rrb.Blueprint(...).save(app_id, path)` (0.42.0) writes valid blueprints. A
blueprint rebuilt from chunks read back with `RrdReader`/`LazyChunkStream`
and saved again can carry nullable list items, and the 0.42 web viewer then
drops it and shows the automatic layout. Console:

```text
WARN re_viewer::app::logic: Failed to make blueprint default: Blueprint failed validation
ScalarAxis:range of /view/<uuid>/ScalarAxis: found FixedSizeList(2 x Float64), expected FixedSizeList(2 x non-null Float64)
```

(also `VisualBounds2D:range`, `LineGrid3D:plane`). The native viewer
accepted the same blueprints over `send_blueprint`. Prefer composing
blueprints with `rrb` and `save()`. If you must re-save read-back chunks,
rebuild every fixed-size-list item field as non-null before `send_chunks`
and `save_blueprint` (example:
`https://pablos-4800gt.ilish-ruler.ts.net:8768/monorepo-guide/rrd/slam-rs/repro/viewers_rbl_fix.py`).
Check the console of every embed: a blueprint that does not apply shows the
automatic layout, and the warning names the component.

### Other origins

A public page or another host needs the files to be fetchable cross-origin:

- Public: viewer from `https://rerun.io/viewer/version/<ver>/` (or
  `app.rerun.io`) and the npm build from jsDelivr; host the `.rrd` where it
  sends `Access-Control-Allow-Origin` (for example `static.rerun.io`).
- NAS files from another origin (not deployed; choose when needed):
  `rerun server -d <dir> --cors-allow-origin <origin>` (OSS catalog server;
  segment URLs take `#when=` and `#time_selection=`), `rerun --serve-web` /
  `--serve-grpc --cors-allow-origin <origin>` for live streams, or a static
  server with CORS headers behind Tailscale Serve proxy mode.
- Chrome Local Network Access: Chrome 138+ asks for permission before a public
  origin (app.rerun.io) fetches from private addresses (Tailscale 100.64/10,
  LAN). In headless runs the request stays `pending`. Serving the viewer from a
  tailnet origin avoids the prompt.
- Size: WASM allocation fails near 2 GiB (`RuntimeError: unreachable`).

For a chrome-free embed (no blueprint/selection/time panels), bake the panel state into the recording before embedding — see "Panel visibility" in SKILL.md (`collapse_panels=True` on the saved blueprint).

## Check sizes and GPU Chrome

Check every page with an embedded viewer at 1400 px (desktop), 1133x744 and
744x1133 (iPad mini) and 390 px (phone): no horizontal scroll, readable panel
titles, the viewer fits its box (full-screen apps: no page scroll), touch
targets of 44 px or more. Keep layouts fluid from 700 to 1200 px.

On the RTX 5090 host, start headed Chrome on a private Xvfb display with
`/tmp/simplerecon-pages/latency/gpu_chrome.sh <profile> 1400x1000 <port>` (it
prints `PID`, `XVFB_PID`, `CDP`; stop both by PID). Prove the GPU with
`gpu_proof.sh <cdp> <out_prefix> <page_url>` in the same directory, then drive
the page with `playwright-cli attach --cdp=<cdp>` or `playwright-core`
`connectOverCDP`. Never use display `:1` (Pablo's desktop) or port 9876
(Rerun's default). In the sandbox, Chrome cannot reach the GPU: run it outside.

Pass for each embed: the video plays and shows data; the network log (CDP
`Network` events) has no `.rrd`/`.rbl` GET, `npm/` script or wasm before the
click (HEAD requests for the size are expected) and shows which video source
loaded (AV1 in Chrome); click "Open live viewer", poll `get_state()` until
`loading` is empty, screenshot the stage and look at it (data visible, not an
empty view), and read the console (only egui "viewport command" warnings are
expected). Once more with low data: `Network.emulateNetworkConditions`
(3G-like) plus `saveData` true; pass = no autoplay, no video bytes, one
confirm before the live viewer. Bound each `get_state()` call with a timeout; a call on a viewer
that is stopping can hang.

## Validating gradio apps & WebViewer embeds

**Playwright to prove, chrome-devtools to diagnose.** Playwright is the default for validation evidence: headless by default with a pinned viewport, so screenshots are deterministic and no browser window pops up on the user's desktop. chrome-devtools defaults to a *headed* browser at a small window — screenshots pick up scrollbars and cramped layouts unless you resize or pass headless flags — but it's the right tool when something misbehaves: inspecting network requests (gradio queue SSE bodies, rrd fetch status codes) and the WASM viewer's console debug stream is its home turf. Whichever you use, set the viewport explicitly (e.g. 1920×1080).

- **Slow loads look like failures**: a hosted-rrd gradio app can spend minutes in silent WASM ingest *after* the fetch completes. Before declaring a load failure, confirm the rrd request succeeded (network tab: 200 on the rrd URL; console: `open_url`), then wait — only a still-empty viewer after that is a real failure.
- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- **WebGL**: reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`). Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` (+ real `DISPLAY`/`XAUTHORITY`) only if it still falls back.
- WebViewer JS `set_current_time(recordingId, timeline, value)` encodes like MCP `set_time` (sequence index / ns / epoch-ns); after seeking, wait for decode and confirm nonblank pixels.

## Docs

- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
