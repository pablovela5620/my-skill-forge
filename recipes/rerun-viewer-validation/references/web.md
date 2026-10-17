# Web branch: embedding recordings and validating browser surfaces

## Pick the embed

| The recording lives in | Embed | What the browser loads |
|---|---|---|
| A catalog (`rerun server`) | **Catalog segment embed** | Only the *slice* near the time cursor. Opens at an exact moment (`#when`). |
| A single `.rrd` file | **File embed** | The whole file, then plays from the start. |

Measured on one 288 MB hour of game video (0.38.1, Chrome on an RTX 5090, 2× GTX 1080 Ti and an M5 Pro): the catalog embed moved 8–11 MB and used 45–130 MB of viewer memory; the file embed moved 288 MB and used 640–720 MB. Neither path is unbounded: the viewer works within a ~2.5 GB memory budget and skips any single chunk larger than it.

Both embeds iframe the hosted viewer pinned to `/version/<ver>/`, with `<ver>` ≥ the SDK that wrote the data. Trust boundary: the viewer is third-party code that reads the recording in the reader's browser, and `app.rerun.io` also loads an analytics script. For sensitive data, serve a copy of the official static files (`index.html`, `re_viewer.js`, `re_viewer_bg.wasm` from `https://app.rerun.io/version/<ver>/`) from an origin you control: that pins the bytes and, on a private origin, removes the Local Network Access prompt. The copy is ~43 MB unless your host compresses `.wasm`, and the viewer still calls Rerun's telemetry and GitHub.

## Catalog segment embed

1. **Serve the catalog with CORS for the viewer origin.** Use the CLI server; it takes the allowed origins (the Python `rr.server.Server` has no CORS parameter):

   ```bash
   rerun server --host 127.0.0.1 --port 51234 \
     --cors-allow-origin https://app.rerun.io --cors-allow-origin https://<page-origin>
   ```

   The default allowlist is `*://localhost`, `*://127.0.0.1` and `*://rerun.io`, matched on the exact host, so `app.rerun.io` must be added. Register segments with `CatalogClient` (`dataset.register([...])`). The server is in-memory: whatever registers must register again after a server restart, and **each restart gives the dataset a new id**, so deep links made before a restart stop resolving.

2. **Give the catalog HTTPS.** An https page reaches only `rerun+https://`. On a tailnet: `tailscale serve --https=<port> --bg http://127.0.0.1:<port>` (needs Tailscale operator rights; gRPC-web streaming through it was tested on 0.38.1). Elsewhere, any TLS reverse proxy that streams responses unbuffered.
   *Done when* a preflight answers 2xx with the viewer origin echoed:
   `curl -si -X OPTIONS https://<host>:<port>/ -H 'Origin: https://app.rerun.io' -H 'Access-Control-Request-Method: POST' | grep -i '^access-control-allow-origin'`

3. **Build the segment URL** from a `CatalogClient` connected to the **public** URL; `segment_url()` keeps the client's origin:

   ```python
   from datetime import timedelta
   from rerun.catalog import CatalogClient

   dataset = CatalogClient("rerun+https://<host>:<port>").get_dataset("<dataset>")
   segment_ids = dataset.segment_ids()  # dataset.segment_table() has per-segment properties
   url = dataset.segment_url(segment_id, "time", timedelta(seconds=t)).replace("\u2009", "")
   # rerun://<host>:<port>/dataset/<id>?segment_id=<seg>#when=time@+<t>  (rerun:// is the HTTPS alias)
   ```

   The `.replace` is required: the 0.38.1 SDK groups fractional digits with a thin space (`+2.525 964s`), and the viewer's `#when` parser rejects that form and silently opens at +0 s. Fragment keys are `when`, `time_selection` and `selection`, joined with `&`; the viewer drops any other key silently.

4. **Embed**, percent-encoding the *whole* segment URL (`#` → `%23`, `+` → `%2B`):

   ```html
   <iframe allow="local-network-access; fullscreen" loading="lazy" style="width:100%;aspect-ratio:16/9;border:0"
     src="https://app.rerun.io/version/<ver>/index.html?hide_welcome_screen&url=<encodeURIComponent(segment_url)>"></iframe>
   ```

   To jump to another moment, set the iframe `src` to the same URL with another `#when`; the viewer reloads there. A new-tab link to the same viewer URL makes `app.rerun.io` the top-level page, so the reader grants Local Network Access once more, for that origin. The page reads the query keys `url` (repeatable), `renderer`, `video_decoder`, `hide_welcome_screen`, `theme` and `persist`. Panels have no query key: pass a saved `.rbl` as an extra `url=` *before* the segment, made with `rrb.Blueprint(<views>, collapse_panels=True).save("<dataset id>", "layout.rbl")` (the application id must be the dataset id; serve the file from a host with HTTPS and CORS for the viewer origin). Without one, the dataset's registered default blueprint applies. For time evidence keep the time panel collapsed rather than hidden, so the readout stays in the screenshot.
   *Done when* a browser screenshot shows the viewer's time readout at the `#when` time and the expected frame (see "Validating").

## File embed

```html
<iframe src="https://app.rerun.io/version/<ver>/index.html?url=<rrd-url>&theme=dark"
        style="width:100%;aspect-ratio:16/9;border:0" allow="local-network-access; fullscreen"></iframe>
```

- **CORS**: the viewer fetches `<rrd-url>` cross-origin, so the `.rrd` response must send `Access-Control-Allow-Origin`. Scope it to the viewer origin: echo the request's `Origin` only when it matches the viewer and send `Vary: Origin`; a wildcard `*` lets any site a tailnet browser visits read the recording. A preflight may answer any 2xx. `tailscale serve` path mode sets no headers; serve the file from a small CORS + Range server on localhost behind `tailscale serve --https=<port> http://127.0.0.1:<local>`.
- **Size**: the browser downloads the whole file first. Keep it to a few hundred MB; WASM allocation fails near 2 GiB (`RuntimeError: unreachable`). Larger recordings → register them in a catalog and use the catalog segment embed.
- An `http(s)` `.rrd` URL takes no `#when`; it always opens at the start.

## Chrome Local Network Access

Chrome 142+ asks permission when a public origin (`app.rerun.io`) fetches private address space (Tailscale 100.64/10, LAN addresses). The prompt names the **top-level page** ("<page-origin> wants to access other devices on your local network"); one Allow per site and browser. When it is denied (the headless default), the console shows `…Permission was denied for this request to access the 'local' address space`; a catalog embed shows "Failed to load entries" and a file embed leaves its request pending. The iframe delegates the permission with `allow="local-network-access"` (Chrome 145 adds `local-network`; the old token stays an alias). A viewer served from a private origin fetches private-to-private and needs no prompt. An https page cannot rely on plain `http://` to a tailnet host (mixed content), so give the data endpoint HTTPS.

## Viewer catalog setting

Settings → Viewer catalog → "Load files via Viewer catalog" (on by default) routes **local** `.rrd`/`.rbl` opens through an in-process catalog, so they too load lazily; on the web it first copies the file into browser storage (OPFS). Remote catalog URLs and `http(s)` `.rrd` URLs ignore it, and no URL or JS option sets it.

## Validating gradio apps & WebViewer embeds

**Playwright to prove, chrome-devtools to diagnose.** Playwright is the default for validation evidence: headless by default with a pinned viewport, so screenshots are deterministic and no browser window pops up on the user's desktop. chrome-devtools defaults to a *headed* browser at a small window — screenshots pick up scrollbars and cramped layouts unless you resize or pass headless flags — but it's the right tool when something misbehaves: inspecting network requests (gradio queue SSE bodies, rrd fetch status codes) and the WASM viewer's console debug stream is its home turf. Whichever you use, set the viewport explicitly (e.g. 1920×1080).

- **Prove the moment**: for any `#when` or time jump, read the viewer's time readout in the screenshot and compare it with the requested time, then compare the frame. A viewer that loaded at the wrong time looks healthy in every log.
- **Local Network Access**: grant it for each top-level page you open, `context.grantPermissions(['local-network-access'], { origin: '<page-origin>' })` (and `https://app.rerun.io` for a viewer opened directly). Run once without the grant to capture the denied state.
- **Ready and versions**: the viewer draws on a canvas, so page text and DOM say nothing about it. Poll screenshots (bounded, e.g. every 3 s up to 90 s) until the Game view is nonblank and the time readout is visible, then wait for two identical captures. Take the running viewer version from its console ("you are running <ver>"), not from the URL alone, and the writer version from the catalog or `rerun rrd print`.
- **GPU**: record the renderer string (WebGL debug info) in `notes.md`. Headless Linux Chrome offers WebGPU only through SwiftShader (the viewer shows a "Software rasterizer" banner); for GPU evidence add `&renderer=webgl` and launch with `--use-gl=angle --use-angle=gl-egl`. On hybrid-GPU hosts set `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, otherwise Chrome picks the integrated GPU. Elsewhere `--enable-gpu --disable-software-rasterizer`, plus `--ozone-platform=x11` (and a real `DISPLAY`/`XAUTHORITY`) only if it still falls back.
- **Count bytes**: the viewer cancels chunk streams once it has the data, so `net::ERR_ABORTED` on `FetchChunks` is normal and finished-request totals undercount. Sum CDP `Network.dataReceived` (`dataLength` for payload, `encodedDataLength` for wire bytes). The viewer iframe is a separate CDP target: attach to it on `framenavigated` (on `frameattached` it has no session yet), keep one session per frame (each `src` change navigates again), and split catalog traffic (the catalog host) from viewer assets (`app.rerun.io`, ~14 MB compressed).
- **Measure memory**: the JS heap stays near 5 MB; the viewer's memory is WASM linear memory. Read it with an init script that wraps `WebAssembly.instantiate`/`instantiateStreaming`, keeps the exported `WebAssembly.Memory`, and reports `memory.buffer.byteLength` (a capacity, not live allocation).
- **Slow loads look like failures**: a hosted-rrd gradio app can spend minutes in silent WASM ingest *after* the fetch completes. Before declaring a load failure, confirm the rrd request succeeded (network tab: 200 on the rrd URL; console: `open_url`), then wait — only a still-empty viewer after that is a real failure.
- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- WebViewer JS `set_current_time(recordingId, timeline, value)` encodes like MCP `set_time` (sequence index / ns / epoch-ns); after seeking, wait for decode and confirm nonblank pixels.

## Docs

- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
- Catalog server and `CatalogClient`: https://ref.rerun.io/docs/python/main/catalog/
- Chrome Local Network Access: https://developer.chrome.com/blog/local-network-access
