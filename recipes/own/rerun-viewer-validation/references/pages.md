# Rerun recordings in pages and public shares

The single source for Rerun media outside the viewer: the source, the medium, the budget, the slice, versions, the
embed and the public share. The `html-artifacts` skill owns the page itself and points here. Design and measurements:
`https://artifacts.ilish-ruler.ts.net/rerun-embed/plan/`.

| Destination | Host | Opens with | Follow |
|---|---|---|---|
| Private page | NAS, served by the artifact host | `embed.js` (`.rrd` and `.rbl` side by side) | Source, Medium, Budget, Slice, The embed |
| Public link | A Hugging Face dataset | `app.rerun.io/version/<v>/index.html?url=` (one `+rbl.rrd`) | Source, Budget, Slice, Public share |
| Both | Both | Both | Export and slice once; the page takes the slice and `.rbl`, the link the joined file |

## Source: a file or a catalog segment

| Source | First step | Blueprint |
|---|---|---|
| A local `.rrd`, with or without an `.rbl` | None | The `.rbl` applies by application id. Other ids: `rerun rrd route --application-id <rrd app id> in.rbl -o run.rbl` |
| A segment of a catalog dataset (`rerun server`, layers) | `scripts/catalog_export.py` | The dataset's default blueprint, routed to the export's application id |

Export a catalog segment to files first: a page or a link holds files, and dataset ids change on every server
restart.

```sh
python scripts/catalog_export.py --catalog rerun+http://127.0.0.1:51234 --dataset <name> --segment <id> --out work/run
```

- The server joins the layers (`base`, `gt`, …) into one recording store. A dataforge base layer file also embeds its
  own blueprint store (application id `dataforge`); the export leaves it out. `rerun rrd route` or `merge` of the layer
  files keeps that store, and `merge` keeps one recording per application id.
- The export carries the exporting SDK's version in its header (0.42.0 from 0.37.0 layer files): check the export
  with `rrd_check.py version`.
- `rerun --save <catalog url>` writes no recording: the stream carries manifests a file cannot hold (`Received a
  RrdManifest which can't be stored in a file`), so the file holds the blueprint only.
- The default blueprint is read from its `file://` storage URL: run the export on the catalog's host, or pass
  `--blueprint`.

**Done when** the script prints the layer list, `run.rrd` holds one recording store and `run.rbl` one blueprint store
with the same application id.

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
- Lost: static archetype components such as `Pinhole` and `Transform3D`, in every part; static custom components on
  the same entities stay, so the entity paths survive. On the msd-odyssey export (0.42.0) the raw part lost all 18
  camera, IMU and magnetometer extrinsics and intrinsics: the 3D views showed error icons and no camera frusta. When
  `rrd_check.py slice` reports missing static components, put them back first in the stream, then optimize:
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

## Public share: Hugging Face and an app.rerun.io link

A public share is one `<name>+rbl.rrd` (the `+rbl` says the blueprint is inside) in a public Hugging Face dataset,
opened by a link to the hosted viewer. Hugging Face answers with `Access-Control-Allow-Origin: *`, so app.rerun.io can
fetch the file.

- **Public data only.** Pablo approves each upload: prepare the file and the exact `hf upload` command, then ask.
  Private and restricted datasets (every `bhuman-*` dataset, agent traces) stay private, slices included.
- **Public URLs only in the link.** The app build sends the page URL, `?url=` included, to `tel.rerun.io`. A tailnet
  URL also fails there (no CORS for app.rerun.io, Chrome's Local Network Access prompt), so check the file on the
  artifact host's own viewer before the upload.

1. **Slice and gate** as for a page (Slice; the joined file has the 25 MB slice budget). A whole recording under
   25 MB needs no slice.
2. **Join** the blueprint and, for a clip, a time panel that loops and starts collapsed. The fragment's newer rows
   replace the blueprint's time panel; `auto_layout=False, auto_views=False` keep its layout.

   ```sh
   rerun rrd print run.rbl | grep -m1 -A3 "store_id: StoreId("   # Blueprint, "<app id>", "<blueprint id>"
   python -c "import rerun.blueprint as rrb; rrb.Blueprint(rrb.TimePanel(loop_mode='All', state='collapsed'), auto_layout=False, auto_views=False).save('<app id>', 'loop-src.rbl')"
   rerun rrd route --recording-id <blueprint id> loop-src.rbl -o loop.rbl
   rerun rrd merge run.slice.rrd run.rbl loop.rbl -o <name>+rbl.rrd
   rerun rrd verify <name>+rbl.rrd
   python scripts/rrd_check.py version <name>+rbl.rrd && python scripts/rrd_check.py gate --slice <name>+rbl.rrd
   ```

3. **Check before the upload.** A still of the joined file with no `.rbl` (`scripts/stills.py --rrd <name>+rbl.rrd`),
   then the file on the artifact host, in GPU Chrome ([web.md](web.md)): copy it to a NAS folder and open
   `https://artifacts.ilish-ruler.ts.net/rerun-viewer/<v>/index.html?url=<percent-encoded artifact URL>`.
4. **Ask, upload, link.** The command for Pablo:

   ```sh
   hf upload <user>/<dataset> <name>+rbl.rrd <name>+rbl.rrd --repo-type dataset
   ```

   The link is `https://app.rerun.io/version/<v>/index.html?url=<encoded>`. `<encoded>` is the percent-encoded
   `https://huggingface.co/datasets/<user>/<dataset>/resolve/main/<name>+rbl.rrd` (`+` becomes `%2B`); `<v>` is the
   writer version or newer. To fix a file, upload it again under the same name; the link stays.

**Done when** the still and the artifact-host viewer show the blueprint's layout with data and video pixels, the
cursor wraps at the clip's end, the console has only the expected noise ([web.md](web.md)), the gate passes, and
Pablo has the file, the command and the link. After his approval and the upload, the app.rerun.io link passes the same
browser check.

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
