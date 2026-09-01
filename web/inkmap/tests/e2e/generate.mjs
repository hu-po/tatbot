// End-to-end: real generation through an inkgen service, traced in the
// browser, placed on the body, saved as a v2 placement file.
//   node tests/e2e/generate.mjs [app url] [api url]
// Defaults: http://127.0.0.1:4181/ (vite preview) and the api from the app's
// config.json. Headless system Chrome; no GPU needed in the browser.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const api = process.argv[3];
const out = new URL("./out/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const t0 = Date.now();
const log = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${m}`);

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--ignore-gpu-blocklist"] });
const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
page.on("pageerror", (e) => log(`pageerror: ${e.message}`));
await page.goto(api ? `${url}${url.includes("?") ? "&" : "?"}api=${encodeURIComponent(api)}` : url);
await page.waitForFunction(() => window.__inkmap?.getState().body, null, { timeout: 120_000 });
log("body loaded");
await page.waitForFunction(() => /generator:/.test(document.querySelector(".generate > p")?.textContent ?? ""), null, { timeout: 60_000 })
  .catch(async () => { log(`generator not reachable: ${await page.textContent(".generate > p")}`); await browser.close(); process.exit(2); });
log(await page.textContent(".generate > p"));

await page.fill(".generate input", "a swallow carrying a rose");
await page.click(".generate button.primary");
await page.waitForFunction(() => document.querySelector(".generate .result") || document.querySelector(".generate .error"), null, { timeout: 300_000 });
const err = await page.textContent(".generate .error").catch(() => null);
if (err) { log(`ERROR: ${err}`); await page.screenshot({ path: `${out}/error.png` }); await browser.close(); process.exit(1); }
log(`generated: ${await page.textContent(".generate figcaption")}`);
await page.screenshot({ path: `${out}/generated.png` });
const svg = await page.evaluate(() => decodeURIComponent(document.querySelector(".generate .pair figure:nth-child(2) img").src.split(",")[1]));
writeFileSync(`${out}/design.svg`, svg);
log(`svg ${svg.length} chars, ${(svg.match(/<path/g) ?? []).length} paths`);

await page.click(".generate .genactions button.primary");
await page.waitForFunction(() => window.__inkmap.getState().placing?.startsWith("gen-"));
const box = await page.locator("canvas").boundingBox();
const x = box.x + box.width * 0.52, y = box.y + box.height * 0.42;
await page.mouse.move(x, y); await page.waitForTimeout(400);
await page.mouse.move(x + 1, y + 1);
await page.mouse.down(); await page.mouse.up();
await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1, null, { timeout: 10_000 });
await page.waitForTimeout(1000);
await page.screenshot({ path: `${out}/placed.png` });
const file = await page.evaluate(() => window.__inkmap.getState().toFile());
writeFileSync(`${out}/placement.json`, JSON.stringify(file, null, 2));
log(`placement file: version ${file.schema_version}, designs ${Object.keys(file.designs ?? {}).join(",")}`);
await browser.close();
log("ok");
