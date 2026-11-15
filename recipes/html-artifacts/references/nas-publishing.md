# Publishing on Pablo's NAS, and embedding Rerun recordings

## Where pages go

- Write to `/tmp/fleet-artifacts/<project>/<name>.html`, publish with SSH/SCP to
  `pablos-4800gt:/volume1/personal/html-artifacts/<project>/`, and give the URL
  `https://pablos-4800gt.ilish-ruler.ts.net:8768/<project>/<name>.html`.
- `:8768` is Tailscale Serve in path mode on the NAS
  (`ssh pablos-4800gt tailscale serve status`). It serves static files with
  Range requests and correct MIME types (`.wasm` is `application/wasm`).
- It sends no CORS headers. Pages, the shared viewer and `.rrd` files on this
  origin load each other with no CORS. Other origins (`rerun.io/viewer`, a
  Gradio app, a preview on another host) cannot fetch NAS files.
- Quirks: directory URLs need a trailing slash, and `<dir>/index.html?query`
  redirects to the site root. Link to `<dir>/?query`.
- The source for the viewer tree, embed.js and the install script is in
  `~/agent-fleet` (`config/html-artifacts/`, `scripts/rerun_viewer_install.py`).

## Check sizes

Check every page, app and embedded viewer at these CSS sizes:

| Device | Size |
|---|---|
| Desktop | 1400 px wide |
| iPad mini landscape | 1133x744 |
| iPad mini portrait | 744x1133 |
| Phone | 390 px wide |

At each size: no horizontal scroll, readable text and panel titles, the Rerun
viewer fits its box (full-screen apps: no page scroll), touch targets of at
least 44x44 px. Keep the layout fluid from 700 to 1200 px; the iPad mini 8
viewport can differ a little from 744/1133, so do not tune to one number.

## Rerun recordings in a page

The self-contained rule has one exception: a Rerun viewer. Use the shared,
versioned viewer on the NAS and the embed script; never copy a viewer into a
project directory.

```html
<link rel="stylesheet" href="/rerun-viewer/embed.css">
<script type="module" src="/rerun-viewer/embed.js"></script>

<figure class="rr-embed" data-rrd="run.rrd" data-rbl="run.rbl" data-poster="run.png"
        data-version="0.42.0" data-time="frame@805">
  <a href="/rerun-viewer/0.42.0/?url=<abs rrd url>&url=<abs rbl url>&theme=dark"><img src="run.png" alt="What the recording shows"></a>
  <figcaption>One line on what to look at.</figcaption>
</figure>
```

- embed.js shows the poster with "Open in viewer" (starts the
  `@rerun-io/web-viewer` JS viewer on demand; one viewer runs at a time),
  "Open full screen" (the app build with `?url=`) and "Copy rerun command"
  (`rerun <rrd url>` for the native viewer). The `<a>` is the no-JS fallback.
- `data-version`: pin it, and keep it at or above the SDK version that wrote
  the recording (Rerun has no forward compatibility). Installed: 0.42.0. A
  missing version is one command: `cd ~/agent-fleet && pixi run python -m scripts.rerun_viewer_install <version>`.
- `data-time="<timeline>@<value>"`: integer in the timeline's unit, or a
  duration in seconds (`video_time@26.8s`). The `#when=` fragment does not
  work on plain `.rrd` URLs.
- `data-rbl`: a blueprint saved from the viewer or written with `rrb`.
- Publish the `.rrd` next to the page and keep it under ~300 MB; the browser
  holds the whole recording in memory. Log video as `VideoStream`, not as
  image frames. Make the poster from a viewer screenshot of the same view.
- Reference page: `https://pablos-4800gt.ilish-ruler.ts.net:8768/rerun-viewer/demo.html`.
- Public pages (off the tailnet): set
  `data-app="https://rerun.io/viewer/version/{v}/"` and
  `data-npm="https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@{v}/index.js"`,
  and host the `.rrd` where it sends CORS headers (for example `static.rerun.io`).

## When another origin must open NAS recordings

Not deployed; choose one when the need is real:

- `rerun server -d <dir> --cors-allow-origin <origin>`: the OSS catalog
  server. Segment URLs accept `#when=` and `#time_selection=` fragments.
- `rerun --serve-web` (or `--serve-grpc --cors-allow-origin <origin>`) for
  live streams.
- A static server that sends `Access-Control-Allow-Origin` behind Tailscale
  Serve proxy mode. Path mode cannot add headers.

## Verify embedded viewers in GPU Chrome

Logs and a clean exit do not show what rendered. Open each viewer in GPU
Chrome at the check sizes and look at the screenshot (the
`rerun-viewer-validation` skill, `references/web.md`, has the recipe). Pass
means the viewer shows data, `get_state()` reports no sources still loading,
and the console has no errors.
