// Steps for check-artifact.mjs --steps: exercise every gallery control, assert, then photograph.
// In scope: page, go(sceneId, offsetPx), shot(name). A thrown error fails the check.
const text = sel => page.$eval(sel, el => el.textContent.trim());
const expect = (what, got, want) => { if (got !== want) throw new Error(`${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); };

await go("s-hero");
const hero = '[data-recipe="rerun-hero"] video';
const playing = await page.$eval(hero, v => ({ ready: v.readyState, paused: v.paused }));
if (playing.ready < 2 || playing.paused) throw new Error("hero video should play on screen: " + JSON.stringify(playing));
await page.$eval(hero, v => v.pause());
await page.click('[data-recipe="rerun-hero"] .rrh-chapters button:nth-child(3)');
await page.waitForTimeout(500);
const caption = await text('[data-recipe="rerun-hero"] .rrh-now');
if (!caption.startsWith("2.6 s") || !caption.includes("contact")) throw new Error("contact chapter: " + caption);
await shot("hero-contact");
await page.$eval(hero, v => v.play());
await page.emulateMedia({ reducedMotion: "reduce" });
await page.waitForTimeout(300);
expect("hero paused under reduced motion", await page.$eval(hero, v => v.paused), true);
await page.emulateMedia({ reducedMotion: "no-preference" });

await go("s-toggle");
await page.click('[data-recipe="toggle"] button[data-view="visual"]');
expect("toggle pictures", await text('[data-count="icons"]'), "4");
await shot("toggle-diagram");

await go("s-presets");
await page.click('[data-recipe="presets"] button[data-stall]');
expect("stall new frames", await text('[data-k="fresh"]'), "31");
expect("stall repeats", await text('[data-k="repeat"]'), "14");
await page.waitForTimeout(2200);
await shot("presets-stall");

await go("s-flip");
await page.click('[data-recipe="flip-cards"] .fc-all');
await page.waitForTimeout(700);
const flipped = await page.$$eval(".fc-card", cs => cs.every(c => c.getAttribute("aria-pressed") === "true"));
if (!flipped) throw new Error("Flip all did not flip every card");
await shot("flip-all");

await go("s-stepper");
await page.click('[data-recipe="stepper"] .step-replay');
await page.waitForTimeout(3800);
expect("lit steps", await page.$$eval(".step.lit", s => s.length), 5);
await shot("stepper-lit");
