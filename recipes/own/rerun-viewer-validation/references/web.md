# Browser checks: embeds, web viewers and gradio apps

What goes into a page (medium, budget, slice, embed markup) is in [pages.md](pages.md). This file proves the result in a
browser.

## Check sizes and GPU Chrome

Check every page with an embedded viewer at 1400 px (desktop), 1133x744 and
744x1133 (iPad mini) and 390 px (phone): no horizontal scroll, readable panel
titles, the viewer fits its box (full-screen apps: no page scroll), touch
targets of 44 px or more. Keep layouts fluid from 700 to 1200 px.

On a host with an NVIDIA GPU, start headed Chrome on a private Xvfb display
with this skill's `scripts/gpu_chrome.sh <profile> 1400x1000 <port>` (it
prints `PID`, `XVFB_PID`, `CDP`; stop both by PID). Prove the GPU before you
trust a screenshot: `chrome://gpu` must show WebGPU and WebGL "Hardware
accelerated" and a Dawn "Vulkan backend - NVIDIA" adapter, and a page's
`navigator.gpu.requestAdapter()` must not be a fallback adapter. Then drive
the page with `playwright-cli attach --cdp=<cdp>` or `playwright-core`
`connectOverCDP`. Never use display `:1` (Pablo's desktop) or port 9876
(Rerun's default). Give your Chrome its own CDP port and check that the port is
yours before you attach: another session's Chrome on the same port takes your
screenshots. In the sandbox, Chrome cannot reach the GPU: run it outside.
Headless Chromium (Playwright's default) shows embed videos as black: check
videos in GPU Chrome.

For the Safari engine (iPad, iPhone: every iOS browser is WebKit), run
Playwright WebKit on a tailnet Mac (`pablos-mac-mini` has it under
`~/.pixi/envs/playwright-cli`) against the published URL. It has no
`navigator.connection`, like iOS.

Pass for each embed: the video plays and shows data; the network log (CDP
`Network` events) has no `.rrd`/`.rbl` GET, `npm/` script or wasm before the
click (1-byte Range requests for the sizes are expected) and shows which video source
loaded (AV1 in Chrome); click "Open live viewer", poll `get_state()` until
`loading` is empty, screenshot the stage and look at it (data visible, not an
empty view), and read the console (only egui "viewport command" warnings are
expected). Once more with low data: `Network.emulateNetworkConditions`
(3G-like) plus `saveData` true; pass = no autoplay, no video bytes, one
confirm before the live viewer. In normal mode, videos below the fold load
no bytes until they scroll into view (`preload="none"`). Bound each `get_state()` call with a timeout; a call on a viewer
that is stopping can hang.

Expected console noise, nothing else: egui `Unhandled egui viewport command`
warnings (Decorations, Transparent, RequestUserAttention); traces also report
`EyeControls3D … fallback provider already registered` and `Transparent window
was requested` on some 0.42 starts. Every page carries
`<link rel="icon" href="data:,">`, so there is no favicon 404. Any other
warning or error fails the check.

Never open pages through `file://` (Playwright blocks it, and the viewer
needs HTTP). Serve the folder with `python -m http.server --bind 127.0.0.1
<free port>`, or check the published NAS URL. Use the global `playwright-cli`
or `connectOverCDP` to GPU Chrome; do not add Playwright to a project env
just to validate.

## The 0.42 JS API

`@rerun-io/web-viewer` 0.42 names are snake_case (`start`, `open`, `close`, `get_state`, `set_time`, `set_location`,
`stop`); newer docs show camelCase. Pass `check_for_updates_on_startup: false` (the GitHub check returns 403 on the
tailnet). The viewer sets inline styles on the parent element it gets: give it an inner element that fills your box.
Set the time with `viewer.set_time({ timeline, sequence | duration_nanos | timestamp_nanos, play_state: "paused" })`
after `get_state().loading` is empty: a blueprint with a time panel sets the cursor when it loads. Bound every
`get_state()` call with a timeout; a call on a viewer that is stopping can hang. WASM allocation fails near 2 GiB
(`RuntimeError: unreachable`).

## `.rbl` "Blueprint failed validation"

Both the 0.42 web and native viewers drop a blueprint whose fixed-size-list
items are nullable and show the automatic layout. `rrb.Blueprint(...).save()`
writes non-null items, and reading chunks back with `RrdReader` and saving
them again keeps them non-null; a pipeline that rebuilds the Arrow schema
can make them nullable. Web console:

```text
Blueprint has 3 component(s) with unexpected encodings:
  ScalarAxis:range of /view/<uuid>/ScalarAxis: found FixedSizeList(2 x Float64), expected FixedSizeList(2 x non-null Float64)
Failed to make blueprint default: Blueprint failed validation
```

(also `VisualBounds2D:range`, `LineGrid3D:plane`). Compose blueprints with
`rrb` and `save()`; if you rebuild chunks, keep fixed-size-list item fields
non-null. Check the console of every embed: a blueprint that does not apply
shows the automatic layout.


## Validating gradio apps & WebViewer embeds

**Playwright to prove, chrome-devtools to diagnose.** Playwright is the default for validation evidence: headless by default with a pinned viewport, so screenshots are deterministic and no browser window pops up on the user's desktop. chrome-devtools defaults to a *headed* browser at a small window — screenshots pick up scrollbars and cramped layouts unless you resize or pass headless flags — but it's the right tool when something misbehaves: inspecting network requests (gradio queue SSE bodies, rrd fetch status codes) and the WASM viewer's console debug stream is its home turf. Whichever you use, set the viewport explicitly (e.g. 1920×1080).

- **Slow loads look like failures**: a hosted-rrd gradio app can spend minutes in silent WASM ingest *after* the fetch completes. Before declaring a load failure, confirm the rrd request succeeded (network tab: 200 on the rrd URL; console: `open_url`), then wait — only a still-empty viewer after that is a real failure.
- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- **WebGL**: reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`). Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` (+ real `DISPLAY`/`XAUTHORITY`) only if it still falls back.

## Docs

- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
