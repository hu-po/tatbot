// Source art in the browser: the collection loads as images, Inkmap's crop of each matches the art pipeline's,
// imported art keeps its generation sidecar, a placed design is sized freely and exported as a brief, and a
// saved project of another version gives way to a new one instead of blocking the save.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { base64ToBytes, jpegOrientation, validateArtRecord } from "../../src/core/art.ts";
import { validateBrief } from "../../src/core/brief.ts";

const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "art") : mkdtempSync(join(tmpdir(), "inkmap-art-"));
mkdirSync(out, { recursive: true });
const collection = JSON.parse(readFileSync(new URL("../../public/designs/manifest.json", import.meta.url), "utf8"));
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const ready = () => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length >= 3; }, null, { timeout: 120_000 });
  const state = (fn, arg) => page.evaluate(fn, arg);
  await page.goto(url); await ready();
  assert.deepEqual(await state(() => window.__inkmap.getState().designs.filter(d => d.library !== false).map(d => d.id)), ["swallow-poppies", "sun-and-moon", "snake-and-dagger"]);
  assert.deepEqual(await state(() => window.__inkmap.getState().designs.map(d => d.embedded.schema)), Array(3).fill("tatbot.inkmap-art/1"));

  // Inkmap's crop of each stock image is the art pipeline's art_box (the manifest's art_px, from Python).
  const input = page.locator("input[type=file][multiple]").first();
  for (const entry of collection.designs) {
    await input.setInputFiles({ name: `${entry.id}.jpg`, mimeType: "image/jpeg", buffer: readFileSync(new URL(`../../public/${entry.image}`, import.meta.url)) });
    await page.waitForFunction(id => window.__inkmap.getState().placing?.startsWith("art-") && window.__inkmap.getState().designs.some(d => d.name === id), entry.id);
    const box = await page.evaluate(id => window.__inkmap.getState().designs.find(d => d.name === id).embedded.art_px, entry.id);
    box.forEach((value, i) => assert.ok(Math.abs(value - entry.art_px[i]) <= 2, `${entry.id}: art_px ${box} vs the pipeline's ${entry.art_px}`));
    await state(() => window.__inkmap.getState().cancelPlacing());
  }

  // Imported art keeps its generation sidecar's prompt; placing it sizes it at its aspect, the width to 0.1 mm.
  const ring = readFileSync(new URL("../fixtures/art-ring.png", import.meta.url));
  const sidecar = { schema: "tatbot.art-generation/1", prompt: "a ring of ink", model: "test-model", generator: "test", subject: "Ring" };
  await input.setInputFiles([{ name: "ring.png", mimeType: "image/png", buffer: ring },
    { name: "ring.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(sidecar)) }]);
  await page.waitForFunction(() => window.__inkmap.getState().designs.some(d => d.name === "Ring"));
  const imported = await state(() => window.__inkmap.getState().designs.find(d => d.name === "Ring").embedded);
  assert.deepEqual([imported.source.kind, imported.source.generation.prompt, imported.art_px], ["generated", "a ring of ink", [6, 6, 52, 52]]);
  await page.fill(".sentence input", "on the left forearm");
  await page.press(".sentence input", "Enter");
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1);
  assert.deepEqual(await state(() => window.__inkmap.getState().placements[0].size_mm), [60, 60]);
  await state(() => { const s = window.__inkmap.getState(); s.nudgeSize(1.0371); });
  assert.deepEqual(await state(() => window.__inkmap.getState().placements[0].size_mm), [62.2, 62.2], "resized freely, snapped to 0.1 mm");
  await page.locator("button.accept").first().click();
  await page.waitForFunction(() => window.__inkmap.getState().selected === null);
  await page.screenshot({ path: join(out, "ring-placed.png") });

  // The brief carries the art and the width to separate it at.
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 15_000 }), page.locator(".dock-body.ready").getByRole("button", { name: "Export brief", exact: true }).click()]);
  const brief = await validateBrief(JSON.parse(readFileSync(await download.path(), "utf8")));
  assert.equal(brief.placements[0].width_mm, 62.2);
  assert.equal(brief.art[brief.placements[0].art_id].content_sha256, imported.content_sha256);
  writeFileSync(join(out, "brief.json"), JSON.stringify(brief, null, 2));

  // Both readers see one picture: these art rectangles are what the pipeline measures (test_art_brief.py).
  const fixture = (name) => readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  const importOne = async (name, mimeType) => {
    if (await page.getByRole("button", { name: "Add artwork", exact: true }).count()) await page.getByRole("button", { name: "Add artwork", exact: true }).click();
    await page.locator("input[type=file][multiple]").first().setInputFiles({ name, mimeType, buffer: fixture(name) });
    const stem = name.replace(/\.[^.]+$/, "");
    await page.waitForFunction(id => window.__inkmap.getState().designs.some(d => d.name === id), stem);
    return state(id => { const s = window.__inkmap.getState(); const record = s.designs.find(d => d.name === id).embedded; s.cancelPlacing(); return record; }, stem);
  };
  const near = (box, pipeline, why) => box.forEach((value, i) => assert.ok(Math.abs(value - pipeline[i]) <= 2, `${why}: ${box} vs ${pipeline}`));
  const clear = await importOne("art-alpha.png", "image/png");
  near(clear.art_px, [54, 64, 93, 93], "transparent pixels are paper, not black ink");
  const photo = await importOne("art-exif6.jpg", "image/jpeg");
  assert.deepEqual([photo.image.width_px, photo.image.height_px], [100, 200], "a turned photo is kept upright");
  near(photo.art_px, [23, 14, 53, 53], "the upright photo's art");
  assert.equal(jpegOrientation(base64ToBytes(photo.image.data_base64)), 1, "nothing is left to turn it twice");
  await validateArtRecord(photo);

  // A reload keeps the placement and its art; an SVG is refused by name.
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 20_000 });
  await page.reload(); await ready();
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1, null, { timeout: 30_000 });
  assert.equal(await state(() => { const s = window.__inkmap.getState(); return s.designs.find(d => d.id === s.placements[0].design_id).embedded.content_sha256; }), imported.content_sha256);
  if (await page.getByRole("button", { name: "Add artwork", exact: true }).count()) await page.getByRole("button", { name: "Add artwork", exact: true }).click();
  await page.locator("input[type=file][multiple]").first().setInputFiles({ name: "legacy.svg", mimeType: "image/svg+xml", buffer: Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"/>') });
  await page.getByRole("alert").filter({ hasText: "PNG or JPEG" }).waitFor();
  await page.locator("input[type=file][multiple]").first().setInputFiles({ name: "artwork.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ schema: "tatbot.inkmap-artwork/2" })) });
  await page.getByRole("alert").filter({ hasText: "artwork.json is DBV3 stroke artwork" }).waitFor();

  // A saved project of another version does not open, and the next save takes its place.
  await page.evaluate(() => new Promise((resolve, reject) => {
    const request = indexedDB.open("inkmap-projects", 1);
    request.onsuccess = () => {
      const tx = request.result.transaction("projects", "readwrite"), store = tx.objectStore("projects");
      const get = store.get("current");
      get.onsuccess = () => store.put({ revision: (get.result?.revision ?? 0) + 1, project: { schema: "tatbot.inkmap-project/2", name: "September", chart: null } }, "current");
      tx.oncomplete = () => { request.result.close(); resolve(null); };
      tx.onabort = () => reject(tx.error);
    };
    request.onerror = () => reject(request.error);
  }));
  await page.reload(); await ready();
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 20_000 });
  assert.equal(await state(() => window.__inkmap.getState().placements.length), 0);
  assert.deepEqual(errors, []);
  console.log("PASS source art: collection crop parity, import with sidecar, free sizing, brief, reload, refusals");
} finally { await browser.close(); }
