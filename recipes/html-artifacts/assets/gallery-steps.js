// Steps for check-artifact.mjs --steps: exercise every gallery control, assert, then photograph.
// In scope: page, go(sceneId, offsetPx), shot(name). A thrown error fails the check.
const text = sel => page.$eval(sel, el => el.textContent.trim());
const expect = (what, got, want) => { if (got !== want) throw new Error(`${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); };

await go("s-hero");
await page.click('[data-recipe="hero-mock"] .mock-play');
await page.$eval('[data-recipe="hero-mock"] input[type=range]', el => { el.value = 400; el.dispatchEvent(new Event("input")); });
expect("hero time", await text(".mock-time"), "4.00 s");
expect("hero caption", await text(".mock-caption"), "Contact. Lift and carry to the bin.");
await shot("hero-paused-4s");

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
