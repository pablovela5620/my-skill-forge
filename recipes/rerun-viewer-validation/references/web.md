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

For a chrome-free embed (no blueprint/selection/time panels), bake the panel state into the recording before embedding — see "Panel visibility" in SKILL.md (`collapse_panels=True` on the saved blueprint).

## Validating gradio apps & WebViewer embeds

**Playwright to prove, chrome-devtools to diagnose.** Playwright is the default for validation evidence: headless by default with a pinned viewport, so screenshots are deterministic and no browser window pops up on the user's desktop. chrome-devtools defaults to a *headed* browser at a small window — screenshots pick up scrollbars and cramped layouts unless you resize or pass headless flags — but it's the right tool when something misbehaves: inspecting network requests (gradio queue SSE bodies, rrd fetch status codes) and the WASM viewer's console debug stream is its home turf. Whichever you use, set the viewport explicitly (e.g. 1920×1080).

- **Slow loads look like failures**: a hosted-rrd gradio app can spend minutes in silent WASM ingest *after* the fetch completes. Before declaring a load failure, confirm the rrd request succeeded (network tab: 200 on the rrd URL; console: `open_url`), then wait — only a still-empty viewer after that is a real failure.
- **Dark parity**: set playwright `colorScheme: "dark"`; for embedded reports force the `prefers-color-scheme` media query before `viewer.start` (`theme: "dark"` alone may not override).
- **WebGL**: reject software renderers (`SwiftShader`, `llvmpipe`, `lavapipe`). Chrome flags `--enable-gpu --disable-software-rasterizer`; add `--ozone-platform=x11` (+ real `DISPLAY`/`XAUTHORITY`) only if it still falls back.
- WebViewer JS `set_current_time(recordingId, timeline, value)` encodes like MCP `set_time` (sequence index / ns / epoch-ns); after seeking, wait for decode and confirm nonblank pixels.

## Docs

- Web embedding: https://rerun.io/docs/howto/integrations/embed-web
