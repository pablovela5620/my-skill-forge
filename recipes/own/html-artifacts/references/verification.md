# Verification

A page is done when a browser proves it and you have looked at the pixels, not when the file is written.

## Run the checker

```bash
node <skill-dir>/scripts/check-artifact.mjs page.html|https://… [--out DIR] \
  [--sizes 1400x1000,1133x744,744x1133,390x844] [--scenes "section[id]"] [--wait 900] [--steps steps.js]
```

`node` and `playwright-cli` come from the fleet manifest. For each size (default: the four fleet check sizes) the checker:

1. Opens a fresh playwright-cli session (a reused one can scroll back to an element it clicked before).
2. Loads the page. A local file is served from its folder at `http://artifact.local/` through `page.route`
   (playwright-cli blocks `file://`), so relative `media/` files load and no HTTP server runs; absolute paths outside the
   folder 404. Check a published page by its `https://` URL instead: then absolute paths such as the shared Rerun viewer
   (`/rerun-viewer/embed.js`) resolve as they do for readers.
3. Turns off smooth scrolling, jumps to each scene (`section[id]` by default), waits, saves `<W>x<H>-<id>.png`.
4. Runs the steps file, if any.
5. Measures after every screenshot: horizontal overflow (with the widest elements), broken images or videos, visible
   buttons, summaries, inputs and selects under 44 x 44 px, and links or controls covered by a fixed, sticky or
   absolute layer of another element (a close button, a toolbar, a viewer overlay).
6. Under reduced motion, revisits each scene and counts infinite animations still running.
7. Lists requests that leave the page's origin (allowed only for an agreed embed).

It exits 1 on console errors (a missing favicon counts: add `<link rel="icon" href="data:,">`), overflow, broken media,
small touch targets, covered controls, reduced-motion loops or a throwing steps file. Screenshots go to `/tmp/artifact-check/<page>/`.
It cannot see JS-driven motion (`requestAnimationFrame` loops, video): assert those states in a steps file.

## Steps files

The body of an async function with `page`, `go(sceneId, offsetPx)` and `shot(name)` in scope. Click each control,
photograph the result, assert the state; a thrown error fails the check:

```js
await go("s-compare");
await page.click("#s-compare .toggle");
await page.waitForTimeout(500);
await shot("compare-after");
if (!(await page.$eval("#s-compare .toggle", b => b.getAttribute("aria-pressed") === "true")))
  throw new Error("the toggle did not switch");
```

Set a range input with `$eval(sel, el => { el.value = 400; el.dispatchEvent(new Event("input")); })`.

## Read the screenshots

A PASS proves only what the checker measures. Open every PNG and ask:

- Does each section open with a picture, not a block of text? Is the visible prose inside the budget
  ([visual-first.md](visual-first.md))?
- Is any label clipped, overlapped or cut by the frame edge? Does a flow stack on the phone?
- Do animated parts show a mid-motion state that makes sense?
- Does the phone layout stack in reading order, with nothing under 11 px?
- Are embedded Rerun videos black? In headless Chromium they are (no H.264/AV1 decoding): check them in GPU Chrome.

Compare before and after a visual fix side by side.

## Pitfalls

- **Lazy content.** Images and late layout change the page after `load`; scroll slowly (about 400 px every 250 ms) or
  visit each section before you trust a width or an "unloaded image" count.
- **Sandboxes.** playwright-cli writes daemon files under `~/.cache/ms-playwright`; with a read-only home it fails with
  `EROFS`. Run the checker outside the sandbox or ask for that write access.
- **`run-code` scripts** have no `require`; parse URLs with string methods, and fill every placeholder (`replaceAll`).
- **Smooth scrolling** (`html { scroll-behavior: smooth }`) breaks scripted scrolling: do not set it globally.
- **Overflow.** Give grid and flex children `min-width: 0`; give inline code and long paths `overflow-wrap: anywhere`.
  Intended scroll containers (`overflow-x: auto`) do not count.
- **Class collisions.** Prefix each component's classes; a generic class name once restyled unrelated parts.
- **Animation loops.** One `requestAnimationFrame` loop per component; clamp the time step to 0 or more; pause loops and
  videos while their section is off screen; JS loops must honour `prefers-reduced-motion`.
- **Truthful models.** Check every simulated number against the real system; a smooth model once showed behaviour the
  real design cannot produce.
