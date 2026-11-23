# Publishing on Pablo's NAS

## Where pages go

- Write to `/tmp/fleet-artifacts/<project>/<name>.html`, publish with SSH/SCP to
  `pablos-4800gt:/volume1/personal/html-artifacts/<project>/`, and give the URL
  `https://artifacts.ilish-ruler.ts.net/<project>/<name>.html`.
- The artifact host is Caddy on Conductor (agent-fleet `config/docker/html-artifacts/`, Docktail name `artifacts`).
  It serves the NAS share read-only with Range requests, correct MIME types, precompressed brotli (a `.br` file next
  to the original) and CORS for `https://*.ilish-ruler.ts.net` origins. A folder URL redirects to its trailing-slash
  form, and `<dir>/index.html` works.
- Old links on `https://pablos-4800gt.ilish-ruler.ts.net:8768/` (Tailscale Serve on the NAS) still work, with no
  brotli and no CORS. Give new links on the artifact host.
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
- Link a folder as `<dir>/` (or `./`): the old `:8768` host answers `<dir>/index.html` with a 301 to its root.
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

Follow the `rerun-viewer-validation` skill: `references/pages.md` for the medium (still, video or viewer), the size
budget for a 10 Mbps reader, slicing, versions, the `embed.js` markup and other origins; `references/web.md` for the
browser checks of an embedded viewer.
