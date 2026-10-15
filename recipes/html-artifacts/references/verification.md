# Verification and Publishing

A page is done when a browser proves it, not when the file is written.

## Contents

- Run the checker
- Steps files
- Read the screenshots
- Pitfalls
- Publish

## Run the checker

```bash
node <skill-dir>/scripts/check-artifact.mjs page.html \
  [--out DIR] [--widths 390,1440] [--scenes "section[id]"] [--wait 900] [--steps steps.js]
```

For each width the checker:

1. Opens a fresh playwright-cli session. A reused session can scroll back to an
   element that it clicked before.
2. Serves the page's folder at `http://artifact.local/` with `page.route`.
   playwright-cli blocks `file://`; this way relative sidecar files load, and
   no HTTP server runs.
3. Turns off smooth scrolling and scrolls to each scene with
   `behavior: "instant"`. It waits `--wait` ms and saves `<width>-<id>.png`.
4. Runs the steps file, if you give one.
5. Measures after every screenshot: horizontal overflow (with the widest
   elements) and broken images or videos.
6. Turns on reduced motion, revisits each scene, and counts infinite
   animations that still run there.
7. Lists requests that leave the page's folder (allowed only for an embed the
   user agreed to).

It exits 1 on console errors, overflow, broken media, reduced-motion loops, or
a throwing steps file. It cannot see JS-driven motion (`requestAnimationFrame`
loops, video): assert those states in the steps file. Screenshots go to
`/tmp/artifact-check/<page-name>/` by default.

## Steps files

A steps file is the body of an async function. `page`, `go(sceneId,
offsetPx)`, and `shot(name)` are in scope. Click each control, then photograph
the result. A thrown error fails the check, so assert states too:

```js
await go("s-flip");
await page.click('[data-recipe="flip-cards"] .fc-all');
await page.waitForTimeout(700);
await shot("flip-all");
const pressed = await page.$$eval(".fc-card", cs => cs.every(c => c.getAttribute("aria-pressed") === "true"));
if (!pressed) throw new Error("Flip all did not flip every card");
```

Set a range input with `$eval(sel, el => { el.value = 400; el.dispatchEvent(new
Event("input")); })`. See `assets/gallery-steps.js` for a full example.

## Read the screenshots

Open every PNG and look at it; a PASS proves only what the checker measures.
For each scene ask:

- Does the first viewport show a visual, not only text?
- Is any label clipped, overlapped, or cut by the frame edge?
- Do animated parts show a mid-motion state that makes sense?
- Does the phone layout stack in reading order, with nothing too small?
- Is the text inside the budget (`SKILL.md`)?

Compare before and after a fix side by side when the change is visual.

## Pitfalls

- **`file://` is blocked** in playwright-cli. The checker serves the folder
  through `page.route`. In `run-code`, `require` and the `URL` global are not
  available; build the script outside and parse URLs with string methods.
- **Sandboxes.** playwright-cli writes its daemon files under
  `~/.cache/ms-playwright`. In a sandbox with a read-only home it fails with
  `EROFS`. Run the checker outside the sandbox, or ask for that write access.
- **Smooth scrolling** (`html { scroll-behavior: smooth }`) breaks scripted
  scrolling. Do not set it globally.
- **Overflow.** Give grid and flex children `min-width: 0`. Give inline code
  and paths `overflow-wrap: anywhere`. Intended scroll containers
  (`overflow-x: auto`) do not count.
- **Class collisions.** Prefix every component's classes. A generic `.back`
  for flip cards once mirrored arrow labels elsewhere.
- **Animation loops.** Keep one `requestAnimationFrame` loop per component.
  Clamp the time step to 0 or more: the first frame can report a time earlier
  than `performance.now()`, and a negative time broke a caption lookup.
- **Truthful models.** Check every simulated number against the real design.
  A smooth sampler once showed a capture burst that the real design cannot make.
- **Reduced motion.** CSS animations stop by the base rule; JS loops must check
  `art.reduced()` and listen with `art.onMotionChange()`.
- **Off-screen work.** Pause loops and videos when their scene leaves the
  screen (`art.onLive`).

## Publish

Follow the fleet HTML-artifact rule in the global instructions. It names the
host, the folder, and the URL pattern. Then:

1. Write the file under `/tmp/fleet-artifacts/<project>/<descriptive-name>.html`.
   Never put it in a repo.
2. Use the host's full MagicDNS name for `ssh` and `scp`. A short name can
   resolve to a public address and time out.
3. Do not overwrite: check first with `ssh <host> test -e <path>`. On a
   collision, add a short ID to the name.
4. Copy the file (and any sidecar media) with `scp`.
5. Prove the served file is the same:

   ```bash
   curl -fsS "$URL" | sha256sum; sha256sum page.html
   ```

6. Give the user the full `https://` URL. Do not start a local HTTP server or
   Tailscale Serve.
