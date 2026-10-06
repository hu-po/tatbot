// Layout across viewports, fresh and recovered: the stock body path is three
// primary actions (choose, place, accept) with the body and the primary
// action on screen together, nothing overlapping, no horizontal overflow.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "layout") : mkdtempSync(join(tmpdir(), "inkmap-layout-"));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];
const ready = (page) => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
const rect = (page, selector) => page.evaluate((sel) => { const el = document.querySelector(sel); if (!el) return null; const r = el.getBoundingClientRect(); return r.width && r.height ? { x: r.x, y: r.y, w: r.width, h: r.height, r: r.right, b: r.bottom } : null; }, selector);
const overlaps = (a, b) => Boolean(a && b) && a.x < b.r && b.x < a.r && a.y < b.b && b.y < a.b;
const VIEWPORTS = [[390, 844], [522, 900], [1440, 900], [1024, 560], [390, 600]];
const report = [];

async function journey(page, width, height, label) {
  const mobile = width <= 768;
  const tap = async (locator) => (mobile ? locator.tap() : locator.click());
  const noOverflow = async () => assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width, `${label}: horizontal overflow`);
  await noOverflow();
  // 1. choose
  const tile = page.getByRole("button", { name: "dbv3-sprout", exact: true });
  if (!(await tile.count())) { await tap(page.getByRole("button", { name: "Add artwork", exact: true })); }
  await tap(tile);
  await page.waitForFunction(() => window.__inkmap.getState().placing === "dbv3-sprout");
  let canvas = await rect(page, "canvas");
  assert.ok(canvas.h >= height * 0.35, `${label}: placing leaves ${canvas.h} px of body in ${height}`);
  assert.ok(!overlaps(await rect(page, ".hud"), await rect(page, ".dock")), `${label}: hint overlaps the tray`);
  await page.screenshot({ path: join(out, `${label}-place.png`) });
  // 2. place
  const before = await page.evaluate(() => window.__inkmap.getState().placements.length);
  for (const [fx, fy] of [[0.5, 0.42], [0.5, 0.5], [0.45, 0.42], [0.55, 0.42], [0.5, 0.6], [0.5, 0.3]]) {
    const x = canvas.x + canvas.w * fx, y = canvas.y + canvas.h * fy;
    if (mobile) await page.touchscreen.tap(x, y);
    else { await page.mouse.move(x, y); await page.waitForTimeout(120); await page.mouse.move(x + 1, y + 1); await page.mouse.down(); await page.mouse.up(); }
    if ((await page.evaluate(() => window.__inkmap.getState().placements.length)) > before) break;
  }
  assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), before + 1, `${label}: placement did not land`);
  await page.waitForTimeout(400);
  // 3. adjust → accept, with the body and the toolbar on screen together
  const accept = page.getByRole("button", { name: "✓ Accept", exact: true });
  const acceptBox = await accept.boundingBox();
  assert.ok(acceptBox && acceptBox.y >= 0 && acceptBox.y + acceptBox.height <= height && acceptBox.x + acceptBox.width <= width, `${label}: Accept is off screen ${JSON.stringify(acceptBox)}`);
  canvas = await rect(page, "canvas");
  assert.ok(canvas.h >= height * 0.3, `${label}: adjusting leaves ${canvas.h} px of body`);
  const bar = await rect(page, ".adjustbar"), hud = await rect(page, ".hud"), msgs = await rect(page, ".messages");
  assert.ok(!overlaps(bar, hud), `${label}: toolbar overlaps hint`);
  assert.ok(!overlaps(bar, msgs) || !(await page.locator(".messages > *").count()), `${label}: toolbar overlaps messages`);
  await noOverflow();
  // Keyboard: focus is visible on the toolbar, Enter in a number field does not accept.
  await page.getByRole("spinbutton", { name: "Width in millimeters", exact: true }).focus();
  await page.keyboard.press("Enter");
  assert.notEqual(await page.evaluate(() => window.__inkmap.getState().selected), null, `${label}: Enter in a field accepted the placement`);
  await page.screenshot({ path: join(out, `${label}-adjust.png`) });
  await tap(accept);
  await page.waitForFunction(() => window.__inkmap.getState().selected === null);
  await page.screenshot({ path: join(out, `${label}-ready.png`) });
  report.push({ label, width, height, canvas_height_adjusting: canvas.h, accept: acceptBox, actions: ["choose", "place", "accept"] });
}

try {
  for (const [width, height] of VIEWPORTS) {
    const context = await browser.newContext({ viewport: { width, height }, hasTouch: width <= 768 });
    const page = await context.newPage();
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url); await ready(page);
    await journey(page, width, height, `fresh-${width}x${height}`);
    // Recovered: the same tab reloads onto its saved project and goes again.
    await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 15_000 });
    await page.reload(); await ready(page);
    assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), 1, "recovery lost the placement");
    await journey(page, width, height, `recovered-${width}x${height}`);
    await context.close();
  }
  assert.deepEqual(errors, []);
  writeFileSync(join(out, "report.json"), JSON.stringify({ schema: "tatbot.inkmap-layout-report/1", checked_at: new Date().toISOString(), url, viewports: report, result: "pass" }, null, 2) + "\n");
  console.log(`PASS layout across ${VIEWPORTS.length} viewports, fresh and recovered ${out}`);
} finally { await browser.close(); }
