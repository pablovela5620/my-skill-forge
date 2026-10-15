# Rerun in HTML Artifacts

Rerun is the main visual medium. Show what the Rerun viewer shows: real viewer
pixels from a real `.rrd`. Never draw an imitation of the viewer in SVG or
HTML. The `rerun-viewer-validation` skill owns capture and proof (headless
viewer, viewer MCP, sweep videos, embeds); read it before you capture. This file
covers what to put in the page.

## Contents

- When Rerun comes first
- Pick the evidence
- No data yet: a synthetic recording
- Exact-time screenshots
- Sweep video
- Live embed
- Size budget

## When Rerun comes first

Use Rerun output for anything with data over time or space: robots, sensors,
cameras, recordings, model outputs, timelines of events, and any pipeline that
writes or could write an `.rrd`. A small Rerun recording often explains a
mechanism better than a diagram, so consider one even for plans.

## Pick the evidence

| You have | Show | Recipe |
|---|---|---|
| A recording, and motion matters | Sweep video with event chapters | `rerun-hero` |
| A recording and one key moment | Screenshot at an exact time, boxed | `evidence` |
| No data yet (a proposal) | A synthetic recording, captured the same way | `rerun-hero`, badge "synthetic data" |
| A small `.rrd` on a host that sends CORS headers | Live embed | iframe, below |

Combine them: a sweep video in the hero, then 2 or 3 exact-time screenshots
that prove specific moments.

## No data yet: a synthetic recording

Log the planned design with the real Rerun SDK: the planned entity paths,
timelines, views, and timings, filled with synthetic values. Then capture it
like real data. Pixels and layout are real; only the values are made up. Say so
in the badge ("Real viewer · synthetic data") and in the caption.

`assets/pick_place_demo.py` is a template: static scene geometry, transform
and scalar columns with `rr.send_columns`, a `TextLog` of events, and a saved
blueprint with collapsed panels. Run it with pixi, no install needed:

```bash
pixi exec --spec rerun-sdk==0.38.1 --spec tyro --spec jaxtyping --spec numpy \
  python <skill-dir>/assets/pick_place_demo.py --out /tmp/demo.rrd
```

Call `rr.save(path, default_blueprint=...)` before the first log call. Pin the
same Rerun version as the viewer that captures it. For real ingestion work,
follow `rerun-data-model` and `rerun-chunk-processing` instead.

## Exact-time screenshots

Use the viewer MCP from `rerun-viewer-validation` (new `rerun_*` schema on
0.38.1):

1. Start `rerun --headless --port <free-port> --window-size 1280x720 rec.rrd`
   from the project's environment. You own it: stop it when done and check
   that no listener remains on the port.
2. `rerun_disconnect`, then `rerun_connect {"endpoint": "http://127.0.0.1:<port>"}`.
3. `rerun_get_viewer_state`: wait until the recording and its timeline range
   appear.
4. `rerun_set_time_cursor {"timeline": {"name": "time"}, "time": {"time": <ns>}, "play": false}`.
5. `wait_for {"min_steps": 5}`, then `screenshot {"save_path": "/abs/path.png"}`.
6. Look at the image. Then crop, box, and inline it with the `evidence`
   recipe. Put the measured fact in the caption, for example "contact at
   2.600 s".

## Sweep video

Use `rerun-viewer-validation`'s `scripts/rrd_to_video.py`. From that skill's
0.38.1 build 2 it speaks both MCP schemas. An older build fails on 0.38.1 with
"no app connected — call `rerun_connect` first".

```bash
python <rerun-viewer-validation>/scripts/rrd_to_video.py --rrd rec.rrd --out sweep.mp4 \
  --rerun-bin <env>/bin/rerun --frames 120 --fps 15 --size 1280x720 --settle-ms 60 --keep-frames
ffmpeg -i sweep.mp4 -c:v libx264 -pix_fmt yuv420p -profile:v baseline \
  -movflags +faststart -crf 26 -an web.mp4
base64 -w0 web.mp4   # src="data:video/mp4;base64,..."
```

Read the first, middle, and last kept frames before you embed the video. With
`--frames N --fps F`, the video lasts N/F seconds; set `data-span` in
`rerun-hero` to the recording seconds the sweep covers. Crop to the view that
tells the story when phones matter (`ffmpeg -vf crop=W:H:X:Y`).

The `rerun-hero` recipe holds the tested playback code. For a bare video, this
snippet (needs the gallery `base-body` block for `art`) plays only while its
scene is on screen, never starts by itself under reduced motion, and stops
when the reader turns reduced motion on:

```html
<video class="rr-video" src="data:video/mp4;base64,..." width="1280" height="720"
       style="width:100%;height:auto" muted loop playsinline controls preload="auto"></video>
<script>
document.querySelectorAll(".rr-video").forEach(v => {
  let live = false;
  const update = () => (live && !art.reduced() ? v.play().catch(() => {}) : v.pause());
  art.onLive(v.closest(".scene"), on => { live = on; update(); });
  art.onMotionChange(update);
});
</script>
```

Inline the video when it is under about 2 MB. A larger video goes next to the
HTML file as a sidecar (`src="sweep.mp4"`). The checker serves the page's
folder, so it loads and checks sidecars too. Publish both files to the same
folder and check both URLs.

## Live embed

The Rerun docs give this embed:

```html
<iframe src="https://app.rerun.io/version/<ver>/index.html?url=<rrd-url>"
        style="width:100%;aspect-ratio:16/9;border:0" allow="fullscreen"></iframe>
```

This snippet loaded Rerun's hosted example
(`<ver>` = `0.38.1`, `<rrd-url>` = `https://app.rerun.io/version/0.38.1/examples/dna.rrd`)
in headless Chrome. It needs all of these. Details are in
`rerun-viewer-validation` `references/web.md`:

- The `.rrd` host sends `Access-Control-Allow-Origin` for the viewer origin.
- Chrome may block a public viewer origin from fetching a private (Tailscale)
  address. Proxy the viewer too, or use another branch.
- `<ver>` is the same as or newer than the SDK that wrote the `.rrd`.
- The file is a few hundred MB or less.

The fleet artifact host sent no CORS headers when checked. Test it before you
plan an embed:

```bash
curl -sI -H 'Origin: https://app.rerun.io' <url-of-any-file-on-the-host> | grep -i access-control
```

No output means no embed from that host. Use screenshots and a sweep video,
and ask the user before you set up a CORS host. An embed also needs the
network, so the page is no longer one offline file.

## Size budget

Base64 makes a file about one third larger. Keep each inline image under
about 150 KB (JPEG `-q:v 5` to `7`, 960 px wide) and the whole page under
about 3 MB, so it loads fast on a phone over Tailscale. A 120-frame,
1280 × 720 sweep of a simple scene encodes to about 100 KB. Larger media go in
sidecar files.
