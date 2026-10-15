# Rerun in HTML Artifacts

Show what the Rerun viewer shows. Do not redraw real data by hand. The
`rerun-viewer-validation` skill owns capture mechanics (headless viewer, MCP,
sweep videos, embeds). This file covers what to put in the page.

## Contents

- When Rerun comes first
- Pick the evidence
- Real screenshots
- Sweep video
- Live embed
- Labeled mock
- Size budget

## When Rerun comes first

Use Rerun output for data over time: robots, sensors, cameras, recordings,
model outputs, and any pipeline that writes or could write an `.rrd`. When the
data exists, real viewer output comes before any drawing.

## Pick the evidence

| You have | Show | Why |
|---|---|---|
| An `.rrd` and a question about one moment | Real screenshots at chosen times | Pixel truth; small |
| An `.rrd` and motion matters | Sweep video (H.264 MP4) | Plays inline from the same file |
| A small `.rrd` on a host that sends CORS headers | Live embed | The reader can scrub and orbit |
| No data yet (a proposal) | `hero-mock`, labeled | Shows the target, honestly |

Combine them: a sweep video in the hero, then 2 or 3 cropped screenshots that
prove specific moments.

## Real screenshots

1. Capture with the `rerun-viewer-validation` skill: a headless viewer and
   `save_screenshot` for a static view, or the viewer MCP to set the time
   cursor first. Use the project environment's `rerun` binary. Collapse the
   panels for a clean frame.
2. Read the screenshot before you use it. A moved playhead does not prove that
   video decoded.
3. Crop, compress, and inline it with the `evidence` recipe
   (`components.md`). Box the detail that proves the point.
4. Put a measured fact in the caption, for example "At 1.000 s Rerun shows
   frame 15." Keep the Rerun version and the time in the reference page.

## Sweep video

Make the sweep with `rerun-viewer-validation` (`scripts/rrd_to_video.py`), then
re-encode for browsers:

```bash
ffmpeg -i sweep.mp4 -vf scale=960:-2 -c:v libx264 -pix_fmt yuv420p \
  -profile:v baseline -movflags +faststart -crf 30 -an web.mp4
base64 -w0 web.mp4   # src="data:video/mp4;base64,..."
```

Tested snippet (needs the gallery `base-body` block for `art`). It plays only
while its scene is on screen. Under reduced motion it never starts by itself,
and it stops when the reader turns reduced motion on:

```html
<video class="rr-video" src="data:video/mp4;base64,..." width="960" height="540"
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

## Labeled mock

Use `hero-mock` only when no real data exists yet. Keep the badge
"Illustration, not a real run". Use the planned entity paths and track names.
Use timings from the real design. Replace the mock with a real capture when
data exists.

## Size budget

Base64 makes a file about one third larger. Keep each inline image under
about 150 KB (JPEG `-q:v 5` to `7`, 960 px wide) and the whole page under
about 3 MB, so it loads fast on a phone over Tailscale. Larger media go in
sidecar files.
