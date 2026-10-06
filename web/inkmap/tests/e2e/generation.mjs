// Native generation is acquired by tatbot drawingbot; this browser exercise imports its unchanged output.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "generation") : mkdtempSync(join(tmpdir(), "inkmap-dbv3-"));
mkdirSync(out, { recursive: true });
const artwork = JSON.parse(readFileSync(new URL("../../public/designs/dbv3-orbit/artwork.json", import.meta.url), "utf8"));
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(url);
  await page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length === 3; }, null, { timeout: 120_000 });
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().designs.map(d => d.id)), ["dbv3-orbit", "dbv3-sprout", "dbv3-ridges"]);
  assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), 0);
  await page.getByRole("tab", { name: "Generate" }).click();
  assert.match(await page.getByLabel("Generate artwork").innerText(), /DrawingBot V3/);
  assert.equal(await page.getByRole("textbox", { name: "Subject" }).count(), 0);
  await page.getByRole("tab", { name: "Paper", exact: true }).click();
  await page.getByRole("tab", { name: "Generate" }).click();
  const file = join(out, "artwork.json"); writeFileSync(file, JSON.stringify(artwork));
  await page.getByLabel("Import artwork").setInputFiles(file);
  await page.waitForFunction(() => window.__inkmap.getState().chartPlacing?.startsWith("art-"));
  // Placement geometry comes from the unchanged acquired record.
  await page.evaluate(() => window.__inkmap.getState().chartCommit([0, 0]));
  await page.waitForFunction(() => window.__inkmap.getState().chart.items.length === 1);
  await page.getByRole("button", { name: "✓ Accept", exact: true }).click();
  const record = await page.evaluate(() => window.__inkmap.getState().chart.items[0].artwork);
  assert.deepEqual(record, artwork);
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved");
  await page.screenshot({ path: join(out, "acquired-placement.png") });
  await page.reload();
  await page.waitForFunction(() => window.__inkmap?.getState().chart.items.length === 1 && window.__inkmap.getState().projectReady);
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().chart.items[0].artwork), artwork);
  // A legacy artwork record with valid hashes must require regeneration, even inside a current saved project.
  const rejected = await page.evaluate(async () => {
    const s = window.__inkmap.getState();
    const project = await s.toProject();
    project.chart.artwork[Object.keys(project.chart.artwork)[0]].conversion = { adapter: "tatbot-svg-paint/1", recipe_sha256: null, chord_error_m: .000005 };
    try { s.restoreProject(project); return null; } catch (error) { return error.message; }
  });
  assert.match(rejected, /Generate with DrawingBot V3/);
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().chart.items[0].artwork), artwork);
  await page.getByRole("tab", { name: "Paper", exact: true }).click();
  if (await page.getByRole("button", { name: "Add artwork", exact: true }).count()) await page.getByRole("button", { name: "Add artwork", exact: true }).click();
  await page.getByRole("tab", { name: "Library" }).click();
  const svg = join(out, "legacy.svg"); writeFileSync(svg, '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>');
  await page.getByLabel("Import artwork").setInputFiles(svg);
  await page.getByRole("alert").filter({ hasText: "Supply source SVG" }).waitFor();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().chart.items.length), 1);
  writeFileSync(join(out, "report.json"), JSON.stringify({ result: "pass", checks: ["fresh DBV3 catalogue", "native acquisition import", "frozen metric placement", "save restore identity", "legacy project refusal", "SVG refusal"] }, null, 2));
} finally { await browser.close(); }
