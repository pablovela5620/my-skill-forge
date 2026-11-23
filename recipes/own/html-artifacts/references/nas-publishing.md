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
- `/volume1/...` exists only on the NAS: check a folder with
  `ssh pablos-4800gt ls <dir>`, and each published file with
  `curl -sI <url>` (expect 200). From the Spark, use the full name
  `pablos-4800gt.ilish-ruler.ts.net` (the short name can resolve to a LAN
  address). Files written through the NFS mounts can arrive with odd modes
  (even 000): `chmod 644` files and `chmod 755` folders after you write them.
- Put `<link rel="icon" href="data:,">` in every page, so the console check
  has no favicon 404.
- Publish a folder (a page with its `media/`, or a set of pages) as one stream:
  `chmod -R a+rX <dir> && tar -C <dir> -cf - . | ssh pablos-4800gt 'mkdir -p /volume1/personal/html-artifacts/<project> && tar -C /volume1/personal/html-artifacts/<project> -xf -'`.
  rsync fails on this NAS. When several agents publish the same folder, serialise the build and publish with one
  `flock` lock file.
- Before you publish, check the target with `ssh pablos-4800gt test -e <path>`: do not overwrite another artifact unless
  asked; on a collision add a short ID to the name. After publishing, prove the served file is yours:
  `curl -fsS <url> | sha256sum` against `sha256sum <local file>`, then run the checker on the URL.
- Publish iteratively: the first publish happens as soon as a skeleton exists (with the work-in-progress badge), then
  after every section or meaningful change, with the live-content check below after each one; the final publish
  removes the badge. Republishing your own page while it is in progress is expected; overwriting a finished page or
  another agent's page still needs the user's OK.
- Check the content the user will see, not only a 200: grep the live page for a string only your change contains.
  Browsers and Tailscale Serve can show a cached copy; when the change is visual, send a cache-busting link
  (`https://…/<page>.html?v=<short-hash>`) so the user does not judge the old version.
- Link a folder as `<dir>/` (or `./`), never `<dir>/index.html`: the server answers `index.html` with a 301 to the
  server root.
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

The self-contained rule has one exception: a Rerun recording. Use the shared,
versioned viewer on the NAS and the embed script; never copy a viewer into a
project directory. Every embed has a video and a poster; the live viewer is
opt-in.

```html
<link rel="stylesheet" href="/rerun-viewer/embed.css">
<script type="module" src="/rerun-viewer/embed.js"></script>

<figure class="rr-embed" data-rrd="run.rrd" data-rbl="run.rbl" data-video="run"
        data-poster="run.poster.jpg" data-version="0.42.0" data-time="frame@805">
  <a href="/rerun-viewer/0.42.0/embed.html?url=<abs rrd url>&url=<abs rbl url>&seek=frame@805"><img src="run.poster.jpg" alt="What the recording shows"></a>
  <figcaption>One line on what to look at.</figcaption>
</figure>
```

- `data-video` is a stem: `run.av1.mp4` (AV1, tried first) and
  `run.h264.mp4` (fallback for older Apple devices). The browser downloads
  only the one it plays. Videos use `preload="none"` (only the poster loads),
  are muted, loop, and play only while on screen; if autoplay is blocked
  (iOS Low Power Mode) they show their controls. Make both files and the poster from the same `.rrd` + `.rbl` with
  the `rerun-viewer-validation` skill's `scripts/rrd_to_video.py --embed <dir>/run`
  (headless viewer, needs the GPU; keep the AV1 loop under 2 MB).
- "Open live viewer (NN MB)" shows the `.rrd` + `.rbl` size from HEAD
  requests and starts the `@rerun-io/web-viewer` JS viewer only on a click;
  one live viewer runs at a time; a "Video | Live viewer" toggle switches
  back. "Open full screen" opens `embed.html` with the start time when
  `data-time` is set, else the app page; "Copy rerun command" copies
  `rerun <rrd url> [<rbl url>]` for the native viewer. The `<a>` is the
  no-JS fallback. Full attribute reference: `~/agent-fleet/config/html-artifacts/README.md`.
- Low-data mode (`navigator.connection.saveData`, or 2g/3g): no autoplay
  (tap to play), and the first live viewer asks before it loads.
- `data-version`: pin it, and keep it at or above the SDK version that wrote
  the recording (Rerun has no forward compatibility). Installed versions and
  the default are in `/rerun-viewer/versions.json`. A missing version is one
  command: `cd ~/agent-fleet && pixi run python -m scripts.rerun_viewer_install <version>`.
- `data-time="<timeline>@<value>"`: integer in the timeline's unit (exact),
  or a duration in seconds (`video_time@26.8s`). The `#when=` fragment does
  not work on plain `.rrd` URLs.
- Publish the `.rrd` next to the page and keep it under ~300 MB; the browser
  holds the whole recording in memory. Log video as `VideoStream`, not as
  image frames.
- One page can carry many embeds (videos are lazy, one live viewer at a time).
- Reference page: `https://pablos-4800gt.ilish-ruler.ts.net:8768/rerun-viewer/demo.html`.
- Public pages (off the tailnet): set
  `data-app="https://rerun.io/viewer/version/{v}/"` and
  `data-npm="https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@{v}/index.js"`,
  and host the files where they send CORS headers (for example `static.rerun.io`).

## When another origin must open NAS recordings

Not deployed; choose one when the need is real:

- `rerun server -d <dir> --cors-allow-origin <origin>`: the OSS catalog
  server. Segment URLs accept `#when=` and `#time_selection=` fragments.
- `rerun --serve-web` (or `--serve-grpc --cors-allow-origin <origin>`) for
  live streams.
- A static server that sends `Access-Control-Allow-Origin` behind Tailscale
  Serve proxy mode. Path mode cannot add headers.

## Verify embedded viewers in GPU Chrome

Logs and a clean exit do not show what rendered. In GPU Chrome, at the check
sizes: the video plays and shows data; the network log has no `.rrd`, `.rbl`
GET or viewer wasm before "Open live viewer"; after the click the viewer
shows data, `get_state()` reports no sources still loading, and the console
has no errors. Once more with Save-Data emulated (DevTools network
conditions): no video bytes and nothing heavy load before a tap. The
`rerun-viewer-validation` skill (`references/web.md`) has the recipe.
