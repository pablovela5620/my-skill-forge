#!/usr/bin/env node
// Check a single-file HTML artifact in a real browser through playwright-cli.
//
// For each width it opens a fresh session and serves the page's folder at
// http://artifact.local/ with page.route (playwright-cli blocks file://, and no
// HTTP server is started), so relative sidecar files resolve. It saves one
// screenshot per scene (`section[id]` by default) and measures after every
// screenshot: horizontal overflow and broken images or videos. Then it walks
// every scene again under prefers-reduced-motion and counts looping
// animations that still run. It also lists requests that leave the page's
// folder. JS-driven motion is invisible to that count: assert it in a steps file.
//
// Usage:
//   node check-artifact.mjs page.html [--out DIR] [--widths 390,1440]
//        [--scenes "section[id]"] [--wait 900] [--steps steps.js]
//
// --steps names a file that holds the body of an async function. It runs after
// the scene shots with `page`, `shot(name)`, and `go(id, offsetPx)` in scope.
// Click toggles, presets, and play buttons there, assert the state, and call
// shot(): every shot() is measured too. A thrown error fails the check.
//
// Exit code 0 when every width has no console errors, no overflow, no broken
// media, no reduced-motion loops, and no step failure; else 1.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";

// Phone and desktop sizes that the skill requires; other widths get 900 px.
const HEIGHTS = { 390: 844, 768: 1024, 1440: 900 };

function parseArgs(argv) {
  const opts = { out: null, widths: [390, 1440], scenes: "section[id]", wait: 900, steps: null };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") opts.out = argv[++i];
    else if (a === "--widths") opts.widths = argv[++i].split(",").map(Number);
    else if (a === "--scenes") opts.scenes = argv[++i];
    else if (a === "--wait") opts.wait = Number(argv[++i]);
    else if (a === "--steps") opts.steps = argv[++i];
    else rest.push(a);
  }
  if (rest.length !== 1 || opts.widths.some(w => !(w > 0)) || !(opts.wait >= 0)) {
    console.error("usage: node check-artifact.mjs page.html [--out DIR] [--widths 390,1440] [--scenes SEL] [--wait MS] [--steps FILE]");
    process.exit(2);
  }
  opts.page = resolve(rest[0]);
  if (!existsSync(opts.page)) { console.error(`no such file: ${opts.page}`); process.exit(2); }
  opts.out = resolve(opts.out ?? `/tmp/artifact-check/${basename(opts.page, ".html")}`);
  return opts;
}

// Runs inside playwright-cli (no require, no URL global). Placeholders become JSON literals.
const RUN_CODE = `async page => {
  const W = __WIDTH__, OUT = __OUT__, WAIT = __WAIT__, DIR = __DIR__, ORIGIN = "http://artifact.local";
  const errors = [], external = new Set(), overflow = [], broken = new Set(), shots = [];
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", e => errors.push(String(e.message)));
  page.on("request", r => { const u = r.url(); if (!/^(data:|blob:|about:)/.test(u) && !u.startsWith(ORIGIN + "/")) external.add(u.slice(0, 120)); });
  await page.route(ORIGIN + "/**", async route => {
    const path = decodeURIComponent(route.request().url().slice(ORIGIN.length).split(/[?#]/)[0]);
    if (path.split("/").includes("..")) return route.fulfill({ status: 403, body: "outside the page folder" });
    try { await route.fulfill({ path: DIR + path }); } catch { await route.fulfill({ status: 404, body: "not found" }); }
  });
  await page.setViewportSize({ width: W, height: __HEIGHT__ });
  await page.goto(ORIGIN + "/" + encodeURIComponent(__FILE__), { waitUntil: "load" });
  await page.waitForTimeout(600);
  await page.evaluate(() => { document.documentElement.style.scrollBehavior = "auto"; });
  const measure = async label => {
    // Wait up to 3 s for images that are still loading, then record what is wide or broken.
    await page.evaluate(() => Promise.race([new Promise(r => setTimeout(r, 3000)),
      Promise.all([...document.images].filter(i => !i.complete).map(i => new Promise(r => { i.onload = i.onerror = r; })))]));
    const m = await page.evaluate(() => {
      const cw = document.documentElement.clientWidth;
      const clipped = el => {
        for (let p = el.parentElement; p; p = p.parentElement) {
          if (["auto", "scroll", "hidden", "clip"].includes(getComputedStyle(p).overflowX)) return true;
        }
        return false;
      };
      const name = e => e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") +
        (typeof e.className === "string" && e.className.trim() ? "." + e.className.trim().split(/\\s+/).join(".") : "");
      const culprits = [...document.querySelectorAll("body *")]
        .filter(e => e.getBoundingClientRect().right > cw + 1 && getComputedStyle(e).position !== "fixed" && !clipped(e))
        .slice(0, 6).map(e => name(e) + " right=" + Math.round(e.getBoundingClientRect().right));
      const broken = [...document.images].filter(i => i.complete && i.naturalWidth === 0).map(i => "img " + (i.getAttribute("src") || "").slice(0, 60))
        .concat([...document.querySelectorAll("video")].filter(v => v.error || v.networkState === 3).map(v => "video " + (v.currentSrc || v.getAttribute("src") || "").slice(0, 60)));
      return { sw: document.documentElement.scrollWidth, cw, culprits, broken };
    });
    if (m.sw > m.cw) overflow.push(label + ": " + m.sw + " > " + m.cw + (m.culprits.length ? " (" + m.culprits.join(", ") + ")" : ""));
    m.broken.forEach(b => broken.add(b));
  };
  const shot = async name => {
    const p = OUT + "/" + W + "-" + name + ".png";
    await page.screenshot({ path: p });
    shots.push(p);
    await measure(name);
  };
  const go = async (id, off = 0) => {
    await page.evaluate(([i, o]) => {
      const el = document.getElementById(i);
      if (!el) throw new Error("no element with id " + i);
      window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY + o, behavior: "instant" });
    }, [id, off]);
    await page.waitForTimeout(WAIT);
  };
  const ids = await page.evaluate(sel => [...document.querySelectorAll(sel)].map(e => e.id).filter(Boolean), __SCENES__);
  if (!ids.length) await shot("page");
  for (const id of ids) { await go(id); await shot(id); }
  __STEPS__
  // Reduced motion: revisit each scene while it is on screen and count infinite animations still running in it.
  await page.emulateMedia({ reducedMotion: "reduce" });
  const motion = [];
  for (const id of ids) {
    await page.evaluate(i => window.scrollTo({ top: document.getElementById(i).getBoundingClientRect().top + window.scrollY, behavior: "instant" }), id);
    await page.waitForTimeout(300);
    const n = await page.evaluate(i => document.getAnimations().filter(a => {
      const t = a.effect && a.effect.target;
      return a.playState === "running" && t && document.getElementById(i).contains(t) && a.effect.getComputedTiming().iterations === Infinity;
    }).length, id);
    if (n) motion.push(id + ": " + n);
  }
  return JSON.stringify({ width: W, errors, overflow, broken: [...broken], motion, external: [...external], shots });
}`;

function runWidth(opts, steps, width) {
  const session = `artifact-check-${process.pid}-${width}`;
  const code = RUN_CODE
    .replace("__WIDTH__", String(width))
    .replace("__HEIGHT__", String(HEIGHTS[width] ?? 900))
    .replace("__OUT__", JSON.stringify(opts.out))
    .replace("__WAIT__", String(opts.wait))
    .replace("__DIR__", JSON.stringify(dirname(opts.page)))
    .replace("__FILE__", JSON.stringify(basename(opts.page)))
    .replace("__SCENES__", JSON.stringify(opts.scenes))
    .replace("__STEPS__", () => steps);
  const file = `${opts.out}/run-${width}.js`;
  writeFileSync(file, code);
  // stdio is piped so the CLI's update banner does not reach the report.
  const cli = (...args) => execFileSync("playwright-cli", [`-s=${session}`, ...args],
    { cwd: opts.out, encoding: "utf8", maxBuffer: 64 << 20, stdio: ["ignore", "pipe", "pipe"] });
  try {
    cli("open", "about:blank");
    let stdout;
    try {
      stdout = cli("run-code", `--filename=${file}`);
    } catch (e) {
      // A throwing page script or steps file exits non-zero; report its message, not a Node stack.
      stdout = String(e.stdout ?? "") + String(e.stderr ?? "");
    }
    const line = stdout.split("\n").find((l, i, all) => i > 0 && all[i - 1].startsWith("### Result"));
    if (!line) {
      const at = stdout.indexOf("### Error");
      const err = (at >= 0 ? stdout.slice(at) : stdout).split("\n").filter(l => l.trim() && !/^[╔║╚]/.test(l)).slice(0, 6).join("\n");
      return { failed: err };
    }
    return JSON.parse(JSON.parse(line));
  } finally {
    try { cli("close"); } catch { /* session already gone */ }
  }
}

const opts = parseArgs(process.argv.slice(2));
mkdirSync(opts.out, { recursive: true });
const steps = opts.steps ? readFileSync(resolve(opts.steps), "utf8") : "";
let ok = true;
for (const width of opts.widths) {
  const r = runWidth(opts, steps, width);
  if (r.failed) {
    ok = false;
    console.log(`\n== ${width}px  the page script or steps file threw:\n${r.failed}`);
    continue;
  }
  ok &&= !r.errors.length && !r.overflow.length && !r.broken.length && !r.motion.length;
  console.log(`\n== ${width}px  errors ${r.errors.length} · overflow ${r.overflow.length} · broken media ${r.broken.length}` +
    ` · reduced-motion loops ${r.motion.length} · external requests ${r.external.length}`);
  for (const e of r.errors) console.log(`   error: ${e}`);
  for (const o of r.overflow) console.log(`   overflow at ${o}`);
  for (const b of r.broken) console.log(`   broken: ${b}`);
  for (const m of r.motion) console.log(`   loops under reduced motion in ${m}`);
  for (const x of r.external.slice(0, 5)) console.log(`   external (allowed only for an agreed embed): ${x}`);
  console.log(`   ${r.shots.length} screenshots in ${opts.out}`);
}
console.log(ok ? "\nPASS" : "\nFAIL");
process.exit(ok ? 0 : 1);
