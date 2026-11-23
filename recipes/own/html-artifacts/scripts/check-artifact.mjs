#!/usr/bin/env node
// Check an HTML artifact in a real browser through playwright-cli, at the fleet check sizes.
//
// For each size it opens a fresh session and serves the page's folder at
// http://artifact.local/ with page.route (playwright-cli blocks file://, and no
// HTTP server is started), so relative sidecar files resolve. It saves one
// screenshot per scene (`section[id]` by default) and measures after every
// screenshot: horizontal overflow and broken images or videos. Then it walks
// every scene again under prefers-reduced-motion and counts looping
// animations that still run. It also lists requests that leave the page's
// folder. JS-driven motion is invisible to that count: assert it in a steps file.
//
// Usage:
//   node check-artifact.mjs page.html|URL [--out DIR] [--sizes 1400x1000,1133x744,744x1133,390x844]
//        [--scenes "section[id]"] [--wait 900] [--steps steps.js]
//
// A published page (an https:// URL) is checked as served, so absolute paths such as the shared Rerun viewer
// (/rerun-viewer/embed.js) resolve; a local file is served from its folder and absolute paths outside it 404.
// It also fails on visible buttons, summaries, inputs and selects smaller than 44 x 44 px (touch targets).
//
// --steps names a file that holds the body of an async function. It runs after
// the scene shots with `page`, `shot(name)`, and `go(id, offsetPx)` in scope.
// Click toggles, presets, and play buttons there, assert the state, and call
// shot(): every shot() is measured too. A thrown error fails the check.
//
// Exit code 0 when every size has no console errors, no overflow, no broken
// media, no small touch target, no reduced-motion loops, and no step failure; else 1.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";

// The fleet check sizes: desktop, iPad mini landscape and portrait, phone.
const SIZES = [[1400, 1000], [1133, 744], [744, 1133], [390, 844]];

function parseArgs(argv) {
  const opts = { out: null, sizes: SIZES, scenes: "section[id]", wait: 900, steps: null };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") opts.out = argv[++i];
    else if (a === "--sizes") opts.sizes = argv[++i].split(",").map(s => s.split("x").map(Number));
    else if (a === "--scenes") opts.scenes = argv[++i];
    else if (a === "--wait") opts.wait = Number(argv[++i]);
    else if (a === "--steps") opts.steps = argv[++i];
    else rest.push(a);
  }
  if (rest.length !== 1 || opts.sizes.some(([w, h]) => !(w > 0 && h > 0)) || !(opts.wait >= 0)) {
    console.error("usage: node check-artifact.mjs page.html|URL [--out DIR] [--sizes 1400x1000,1133x744,744x1133,390x844] [--scenes SEL] [--wait MS] [--steps FILE]");
    process.exit(2);
  }
  opts.url = /^https?:\/\//.test(rest[0]) ? rest[0] : null;
  opts.page = opts.url ? null : resolve(rest[0]);
  if (opts.page && !existsSync(opts.page)) { console.error(`no such file: ${opts.page}`); process.exit(2); }
  const stem = opts.url ? (opts.url.replace(/[?#].*$/, "").replace(/\/$/, "/index").split("/").pop().replace(/\.html$/, "") || "page") : basename(opts.page, ".html");
  opts.out = resolve(opts.out ?? `/tmp/artifact-check/${stem}`);
  return opts;
}

// Runs inside playwright-cli (no require, no URL global). Placeholders become JSON literals.
const RUN_CODE = `async page => {
  const W = __WIDTH__, H = __HEIGHT__, OUT = __OUT__, WAIT = __WAIT__, DIR = __DIR__, URL_ = __URL__;
  const ORIGIN = URL_ ? URL_.split("/").slice(0, 3).join("/") : "http://artifact.local";
  const errors = [], external = new Set(), overflow = [], broken = new Set(), smalls = new Set(), shots = [];
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", e => errors.push(String(e.message)));
  page.on("response", r => { if (r.status() >= 400) errors.push("HTTP " + r.status() + " " + r.url().slice(0, 140)); });
  page.on("request", r => { const u = r.url(); if (!/^(data:|blob:|about:)/.test(u) && !u.startsWith(ORIGIN + "/")) external.add(u.slice(0, 120)); });
  if (!URL_) await page.route(ORIGIN + "/**", async route => {
    const path = decodeURIComponent(route.request().url().slice(ORIGIN.length).split(/[?#]/)[0]);
    if (path.split("/").includes("..")) return route.fulfill({ status: 403, body: "outside the page folder" });
    try { await route.fulfill({ path: DIR + path }); } catch { await route.fulfill({ status: 404, body: "not found" }); }
  });
  await page.setViewportSize({ width: W, height: __HEIGHT__ });
  await page.goto(URL_ || ORIGIN + "/" + encodeURIComponent(__FILE__), { waitUntil: "load" });
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
      const small = [...document.querySelectorAll("button, summary, input, select, [role=button]")].filter(e => {
        const r = e.getBoundingClientRect(), s = getComputedStyle(e);
        return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && r.bottom > 0 && r.top < innerHeight && (r.width < 44 || r.height < 44);
      }).slice(0, 4).map(e => name(e) + " " + Math.round(e.getBoundingClientRect().width) + "x" + Math.round(e.getBoundingClientRect().height));
      return { sw: document.documentElement.scrollWidth, cw, culprits, broken, small };
    });
    m.small.forEach(s => smalls.add(s));
    if (m.sw > m.cw) overflow.push(label + ": " + m.sw + " > " + m.cw + (m.culprits.length ? " (" + m.culprits.join(", ") + ")" : ""));
    m.broken.forEach(b => broken.add(b));
  };
  const shot = async name => {
    const p = OUT + "/" + W + "x" + H + "-" + name + ".png";
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
  return JSON.stringify({ width: W, errors, overflow, broken: [...broken], small: [...smalls], motion, external: [...external], shots });
}`;

function runSize(opts, steps, [width, height]) {
  const session = `artifact-check-${process.pid}-${width}x${height}`;
  const code = RUN_CODE
    .replaceAll("__WIDTH__", String(width))
    .replaceAll("__HEIGHT__", String(height))
    .replace("__URL__", JSON.stringify(opts.url))
    .replace("__OUT__", JSON.stringify(opts.out))
    .replace("__WAIT__", String(opts.wait))
    .replace("__DIR__", JSON.stringify(opts.page ? dirname(opts.page) : ""))
    .replace("__FILE__", JSON.stringify(opts.page ? basename(opts.page) : ""))
    .replace("__SCENES__", JSON.stringify(opts.scenes))
    .replace("__STEPS__", () => steps);
  const file = `${opts.out}/run-${width}x${height}.js`;
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
for (const size of opts.sizes) {
  const tag = `${size[0]}x${size[1]}`;
  const r = runSize(opts, steps, size);
  if (r.failed) {
    ok = false;
    console.log(`\n== ${tag}  the page script or steps file threw:\n${r.failed}`);
    continue;
  }
  ok &&= !r.errors.length && !r.overflow.length && !r.broken.length && !r.small.length && !r.motion.length;
  console.log(`\n== ${tag}  errors ${r.errors.length} · overflow ${r.overflow.length} · broken media ${r.broken.length}` +
    ` · small touch targets ${r.small.length} · reduced-motion loops ${r.motion.length} · external requests ${r.external.length}`);
  for (const e of r.errors) console.log(`   error: ${e}`);
  for (const o of r.overflow) console.log(`   overflow at ${o}`);
  for (const b of r.broken) console.log(`   broken: ${b}`);
  for (const s of r.small) console.log(`   touch target under 44 px: ${s}`);
  for (const m of r.motion) console.log(`   loops under reduced motion in ${m}`);
  for (const x of r.external.slice(0, 5)) console.log(`   external (allowed only for an agreed embed): ${x}`);
  console.log(`   ${r.shots.length} screenshots in ${opts.out}`);
}
console.log(ok ? "\nPASS" : "\nFAIL");
process.exit(ok ? 0 : 1);
