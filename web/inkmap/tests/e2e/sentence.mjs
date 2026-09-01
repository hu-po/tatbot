// End-to-end: the inklang sentence flow, no generator needed. Load the app,
// wait for body + atlas, say "a fine line snake on the left knee ditch",
// verify the placement lands inside the named region with a truthful caption,
// toggle the atlas overlay, and save a v3 file.
//   node tests/e2e/sentence.mjs [app url]
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = new URL("./out/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const t0 = Date.now();
const log = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${m}`);

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--ignore-gpu-blocklist"] });
const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
page.on("pageerror", (e) => log(`pageerror: ${e.message}`));
await page.goto(url);
await page.waitForFunction(() => window.__inkmap?.getState().body && window.__inkmap?.getState().atlas, null, { timeout: 120_000 });
log("body + atlas loaded");

await page.fill(".sentence input", "a fine line snake on the left knee ditch");
await page.click(".sentence button.primary");
await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1, null, { timeout: 10_000 });
const p = await page.evaluate(() => window.__inkmap.getState().placements[0]);
if (!p.site || p.site.id !== "knee_ditch" || p.site.laterality !== "left") {
  log(`ERROR: placement site is ${JSON.stringify(p.site)}`);
  process.exit(1);
}
if (!p.language || !/snake on the left knee ditch/.test(p.language.sentence)) {
  log(`ERROR: caption is ${JSON.stringify(p.language)}`);
  process.exit(1);
}
log(`placed: “${p.language.sentence}” at face ${p.anchor.face}`);

// Atlas overlay toggles.
await page.evaluate(() => window.__inkmap.getState().toggleAtlas());
await page.waitForFunction(() => window.__inkmap.getState().showAtlas, null, { timeout: 5_000 });
log("atlas overlay on");
await page.screenshot({ path: `${out}/sentence.png` });

// Accept, then save: the file must be schema v3 with site + language.
await page.evaluate(() => window.__inkmap.getState().accept());
const file = await page.evaluate(() => window.__inkmap.getState().toFile());
if (file.schema_version !== 3 || !file.placements[0].site || !file.placements[0].language) {
  log(`ERROR: file is not a v3 site+language file: ${JSON.stringify(file).slice(0, 200)}`);
  process.exit(1);
}
writeFileSync(`${out}/sentence-placement.json`, JSON.stringify(file, null, 2));
log("saved v3 placement file with site + language");

// A hand-placed design must get auto-captioned too.
await page.evaluate(() => {
  const s = window.__inkmap.getState();
  const design = s.designs.find((d) => d.name === "Star");
  s.placeAt(design.id, s.atlas.anchorFor({ id: "sternum", laterality: null }), null);
});
const p2 = await page.evaluate(() => window.__inkmap.getState().placements[1]);
if (!p2.language || !/star on the sternum/.test(p2.language.sentence)) {
  log(`ERROR: auto-caption is ${JSON.stringify(p2.language)}`);
  process.exit(1);
}
log(`auto-captioned: “${p2.language.sentence}”`);

// A relative phrase grounds too: two inches below the left collarbone.
await page.evaluate(() => window.__inkmap.getState().setPending(null));
await page.fill(".sentence input", "a snake two inches below the left collarbone");
await page.click(".sentence button.primary");
await page.waitForFunction(() => window.__inkmap.getState().placements.length === 3, null, { timeout: 10_000 });
const p3 = await page.evaluate(() => window.__inkmap.getState().placements[2]);
if (!p3.language || !/two inches below the left collarbone/.test(p3.language.program.site ? JSON.stringify(p3.language.program) + p3.language.sentence : p3.language.sentence)) {
  log(`ERROR: relative placement language is ${JSON.stringify(p3.language)}`);
  process.exit(1);
}
log(`relative: “${p3.language.sentence}” at face ${p3.anchor.face} (site: ${p3.site?.id})`);

await browser.close();
log("PASS");
