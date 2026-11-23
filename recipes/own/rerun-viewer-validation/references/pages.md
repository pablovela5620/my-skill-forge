# Rerun recordings in pages

The single source for Rerun media in HTML pages: the medium, the budget, the slice, versions, and the embed. The
`html-artifacts` skill owns the page itself and points here. Design and measurements:
`https://artifacts.ilish-ruler.ts.net/rerun-embed/plan/`.

## Medium: still, video or viewer

| The figure must show | Medium | Viewer button |
|---|---|---|
| One state: a layout, a result, a failure | Still (poster JPG or WebP, ≤ 150 KB) | Yes, when the `.rrd` is published |
| Change over time: motion, a sweep, convergence | Video: AV1 + H.264 loop (≤ 2 MB) with a poster | Always |
| Data to explore: 3D, values, many entities | Video or still, with the viewer one tap away | Always; the slice by default |
| A private or restricted dataset | Counts and public or synthetic samples only | No |

A video always comes with the viewer: same poster, same recording. The page never starts a viewer by itself.

## Budget: size for 10 Mbps

Pablo reads pages on 1 Gbps at home and on about 10 Mbps outside (iPad, phone). iOS Safari and Chrome on iOS have no
`navigator.connection` and no Save-Data, so a page cannot detect a slow link: size every default for 10 Mbps.

| Item | Budget | At 10 Mbps | Loads |
|---|---|---|---|
| Poster | 150 KB | 0.1 s | with the page |
| Video loop (AV1; H.264 fallback) | 2 MB | 1.6 s | on screen (`preload="none"`) |
| Viewer wasm (first viewer per visit) | today 70 MB, no compression (56.7 s measured) | | on tap |
| "Open viewer": the slice | 25 MB | 20 s | on tap |
| "Full recording" | 300 MB | 4 min | on tap; its size on the button |

Gate every file before you publish:

```sh
python scripts/rrd_check.py gate --poster run.poster.jpg --video run.av1.mp4 run.h264.mp4 \
    --slice run.slice.rrd --full run.rrd        # exit 1 when a file is over budget
```

Over budget: cut a shorter window, drop big entities the figure does not need
(`rerun rrd filter --drop-entity <path>`; static meshes do not shrink with the window), or lower the video length or
CRF. When the reader needs the bytes, pass `--allow-large '<why>'` and say so in the caption.

## Slice: a window of the recording

`rerun rrd split` cuts on one timeline and writes `[-inf,t0)`, `[t0,t1)` and `[t1,+inf)`. Proven on a 151 MiB
recording with two AV1 VideoStreams (keyframe every 1 s), a 42 MB static mesh and an embedded blueprint:

```sh
mkdir -p parts    # split fails when the folder does not exist
rerun rrd split --timeline video_time -t 10000000000 -t 14000000000 -o parts run.rrd   # integers in the timeline's unit
rerun rrd optimize parts/run_10s__14s.rrd -o run.slice.rrd       # a raw part shows an empty viewport: always optimize
python scripts/rrd_check.py slice --full run.rrd --slice run.slice.rrd
```

- Times are integers in the timeline's unit (nanoseconds for a duration timeline); `10s` is refused.
- Kept: the blueprint store, all three parts; video cuts move back to a keyframe (`/cam/dino_pca` 10 s → 9.067 s);
  the latest transforms and pinholes from before the cut.
- Lost: one static `Pinhole` in all parts. When `rrd_check.py slice` reports missing static entities, put them back
  first in the stream, then optimize:
  `python scripts/rrd_check.py restore --full run.rrd --slice parts/run_10s__14s.rrd --out static.rrd`,
  `rerun rrd merge static.rrd parts/run_10s__14s.rrd -o merged.rrd`, `rerun rrd optimize merged.rrd -o run.slice.rrd`.
- A static `Mesh3D` is cut into one chunk per component; the viewer then shows "Failed to load mesh … (2 vs 0)" once
  and still draws the mesh.
- The split adds `-1` to the recording id; a separate `.rbl` still applies (it matches by application id).
- Pixel check: one still of the slice and of the full recording at the same time with `scripts/stills.py`; the
  views must match.

## Versions in pages

- `data-version` names a folder under `/rerun-viewer/`; it must be ≥ the writer version
  (`rrd_check.py version`). Installed versions and the default are in `/rerun-viewer/versions.json`.
- A missing viewer folder kills the page's viewer buttons: 8 pages lost theirs when `0.38.1/` went. The 0.42.0
  viewer opened those 0.37/0.38 recordings as they are, so pointing a page at a newer installed viewer is the fix; a
  migrate is not needed.
- A page can read the writer version with one `Range: bytes=0-11` request: `RRF2`, then major, minor, patch, meta.
- Custom viewer builds (`0.42.0-gsplat*`) live next to the official ones and are named in `data-version`.

## The embed

The shared viewer and `embed.js` on the NAS; never a viewer copy in a project folder. Markup, attributes and
behaviour: `~/agent-fleet/config/html-artifacts/README.md`. In short:

```html
<link rel="stylesheet" href="/rerun-viewer/embed.css">
<script type="module" src="/rerun-viewer/embed.js"></script>
<figure class="rr-embed" data-rrd="run.slice.rrd" data-rbl="run.rbl" data-video="run"
        data-poster="run.poster.jpg" data-version="0.42.0" data-time="video_time@12s">
  <a href="/rerun-viewer/0.42.0/embed.html?url=<abs rrd url>&url=<abs rbl url>&seek=video_time@12000000000"><img src="run.poster.jpg" alt="…"></a>
  <figcaption>One line on what to look at.</figcaption>
</figure>
```

Make the video set and the poster from the same `.rrd` and `.rbl`, on the GPU host:

```sh
python scripts/rrd_to_video.py --rrd run.rrd --rbl run.rbl --embed site/run --rerun-bin <env>/bin/rerun \
    --timeline video_time --start 10s --end 14s --poster-at 12s
```

Output: `run.av1.mp4` (SVT-AV1, screen content), `run.h264.mp4` (960 px) and `run.poster.jpg`. A 4 s window gives
60 frames at real speed: 0.6 MB AV1, 0.3 MB H.264, 0.14 MB poster on the metrictrail recording.

## Hosting facts

- Pages, the viewer and `.rrd` files share one origin, the artifact host `https://artifacts.ilish-ruler.ts.net/`
  (Caddy on Conductor, published by Docktail; agent-fleet `config/docker/html-artifacts/`): no CORS needed, and
  everything works on the tailnet with no internet.
- It serves each wasm's precompressed brotli file (`brotli --quality=11`, Rerun's method: 70 MB npm wasm → 10.9 MB,
  101 MB gsplat → 15.4 MB) with `Content-Encoding: br`. `scripts/rerun_viewer_install.py` in agent-fleet writes the
  `.br` files and lists custom builds in `versions.json`.
- The old origin `https://pablos-4800gt.ilish-ruler.ts.net:8768/` (Tailscale Serve path mode) still serves old links,
  with no brotli and no CORS.
- `embed.js` uses the npm build (`/rerun-viewer/<v>/npm/`), which sends no analytics. The app build
  (`/rerun-viewer/<v>/`, the "Open full screen" target) posts analytics to `tel.rerun.io`, checks GitHub for updates
  and loads Google Fonts.
- Public pages: jsDelivr refuses the 0.42 npm wasm (403, over 50 MB); unpkg serves it (gzip, 17.7 MB).
- Another origin that must open NAS files (`rerun.io/viewer`, a Gradio app): `rerun server -d <dir>
  --cors-allow-origin <origin>` (catalog; segment URLs take `#when=`), `rerun --serve-web` or `--serve-grpc
  --cors-allow-origin <origin>` for live streams, or the artifact host, which already allows `https://*.ilish-ruler.ts.net` origins. A public page fetching tailnet addresses meets Chrome's Local Network Access prompt; in headless runs the
  request stays pending.
- Catalog URLs (`rerun+http://`) in a page need `rerun server --cors-allow-origin <page origin>`, TLS through
  `tailscale serve`, and a long keep-alive; dataset ids change on every server restart, so put no catalog URL in a
  permanent page.

### What the 0.42 viewer URL takes

| Feature | Status |
|---|---|
| Several `url=` (`.rrd` and `.rbl`) | Works; the JS `open` takes an array. |
| `.rbl` as a URL | Applies to the recording with the same application id. |
| `theme`, `persist`, `renderer`, `hide_welcome_screen` | Work. |
| `#when=` / `#time_selection=` on an `.rrd` URL | Not applied; only catalog dataset URIs take fragments. Use `data-time` (JS `set_time`) or `embed.html?…&seek=`. |
