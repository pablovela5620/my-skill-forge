# Web branch: embedding .rrd files and validating browser surfaces

## Embedding an .rrd in an HTML report (no JS)

Preferred embed: an `<iframe>` of the hosted viewer — don't self-host `@rerun-io/web-viewer` just for a report. Trust boundary: the hosted viewer is third-party code that reads the recording bytes in the reader's browser; for genuinely sensitive recordings, keep viewer assets on an origin you control (the tailscale proxy in requirement 2 does this) or skip web embedding and report native screenshots.

```html
<iframe src="https://app.rerun.io/version/<ver>/index.html?url=<rrd-url>&theme=dark"
        style="width:100%;aspect-ratio:16/9" allow="fullscreen"></iframe>
```

Three requirements:

1. **CORS**: the WASM viewer fetches `<rrd-url>` cross-origin, so the `.rrd` response must send `Access-Control-Allow-Origin`. **Scope it to the viewer origin** — echo the request's `Origin` back only when it matches the viewer (`https://app.rerun.io` or your proxy origin) and send `Vary: Origin`; a wildcard `*` lets any website a tailnet browser visits read the recording. `tailscale serve` path mode sets no headers — run a small CORS+Range file server on localhost and use proxy mode: `tailscale serve --https=<port> http://127.0.0.1:<local>` (headers pass through).
2. **Chrome Local Network Access**: Chrome 138+ gates fetches from a public origin (app.rerun.io) to private address space (Tailscale 100.64/10, LAN IPs) behind a user permission. Symptom: the rrd request sits `pending` forever in headless/CI, and real browsers show a one-time "allow local network" prompt. **Zero-prompt dodge**: also proxy the viewer itself — `tailscale serve --https=<port2> https://app.rerun.io` — and iframe `https://<node>.ts.net:<port2>/version/<ver>/index.html?url=…`; both origins are then private, so no gate. Firefox/Safari don't gate.
3. **Version + size**: pin `/version/<ver>/` ≥ the SDK that wrote the `.rrd`; keep files under ~1.5 GiB (WASM allocation fails near 2 GiB, `RuntimeError: unreachable`). Larger files → native screenshots only.

`?url=` also accepts `rerun://` dataset URIs and `rerun+http://…/proxy` live-SDK endpoints. The recording never leaves your network: the viewer assets are static; the `.rrd` fetch happens in the reader's browser.

## Validating gradio apps & WebViewer embeds

Use playwright / chrome-devtools against the running gradio app or embedded-viewer report:

- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- **WebGL**: reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`). Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` (+ real `DISPLAY`/`XAUTHORITY`) only if it still falls back.
- WebViewer JS `set_current_time(recordingId, timeline, value)` encodes like MCP `set_time` (sequence index / ns / epoch-ns); after seeking, wait for decode and confirm nonblank pixels.

## Docs

- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
