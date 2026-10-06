import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "project") : mkdtempSync(join(tmpdir(), "inkmap-project-"));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];
const ready = async page => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
const saved = async page => page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 15_000 });
try {
  const context = await browser.newContext({ viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("dialog", dialog => dialog.accept());
  await page.goto(url); await ready(page);
  await page.fill(".sentence input", "an dbv3-ridges on the left forearm"); await page.click(".sentence button.primary");
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1);
  assert.equal(await page.evaluate(() => window.__inkmap.getState().toFile()), null, "pending edit exported");
  await page.getByRole("button", { name: "✓ Accept", exact: true }).click(); await saved(page);
  const original = await page.evaluate(() => window.__inkmap.getState().placements[0]);
  await page.evaluate(id => { const s = window.__inkmap.getState(); s.select(id); s.nudgeRotation(0.05); s.nudgeRotation(0.05); }, original.id);
  await page.getByRole("button", { name: "✕ Cancel", exact: true }).click();
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().placements[0]), original, "cancel did not restore accepted tattoo");
  // The same through the toolbar's numbers: width, rotation and mirror edit
  // the selected tattoo, and Cancel restores the accepted values exactly.
  await page.evaluate(id => window.__inkmap.getState().select(id), original.id);
  await page.getByRole("spinbutton", { name: "Width in millimeters", exact: true }).fill("45");
  await page.getByRole("spinbutton", { name: "Rotation in degrees", exact: true }).fill("12");
  await page.getByRole("checkbox", { name: "Mirror", exact: true }).check();
  const numeric = await page.evaluate(() => window.__inkmap.getState().placements[0]);
  assert.equal(numeric.size_mm[0], 45); assert.equal(numeric.mirror, true);
  assert.ok(Math.abs(numeric.rotation_rad - 12 * Math.PI / 180) < 1e-9);
  await page.keyboard.press("Escape");
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().placements[0]), original, "Escape did not restore accepted values");
  await page.evaluate(id => { const s = window.__inkmap.getState(); s.select(id); s.nudgeRotation(0.05); s.nudgeRotation(0.05); s.accept(); }, original.id);
  const edited = await page.evaluate(() => window.__inkmap.getState().placements[0]);
  assert.notEqual(edited.rotation_rad, original.rotation_rad);
  assert.equal(await page.evaluate(() => window.__inkmap.getState().past.length), 2, "one edit should produce one history entry");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().placements[0]), original);
  await page.getByRole("button", { name: "Redo", exact: true }).click();
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().placements[0]), edited);
  await page.evaluate(() => window.__inkmap.getState().setPoseId("reclined-left-arm-supported")); await ready(page);
  await page.mouse.move(1150, 300); await page.mouse.down(); await page.mouse.move(1200, 330, { steps: 8 }); await page.mouse.up();
  await page.waitForFunction(() => window.__inkmap.getState().cameraSnapshot !== null);
  await page.evaluate(id => { const s = window.__inkmap.getState(); s.setSkinTone("#804030"); s.select(id); s.nudgeRotation(0.03); }, original.id);
  await saved(page);
  const recovery = await page.evaluate(() => window.__inkmap.getState().toProject());
  await page.reload(); await ready(page); await saved(page);
  const recovered = await page.evaluate(() => window.__inkmap.getState().toProject());
  assert.deepEqual(recovered, recovery, "reload changed project or pending edit");
  await page.getByRole("button", { name: "✕ Cancel", exact: true }).click(); await saved(page);
  assert.deepEqual(await page.evaluate(() => window.__inkmap.getState().placements[0]), edited);
  // A second tab reads revision N; the first advances to N+1. A later write
  // from the second tab must refuse instead of replacing the newer document.
  const second = await context.newPage(); second.on("dialog", dialog => dialog.accept());
  await second.goto(url); await ready(second); await saved(second);
  const rename = async (tab, value) => {
    await tab.getByRole("button", { name: "File", exact: true }).click();
    await tab.getByRole("textbox", { name: "Project name", exact: true }).fill(value);
    await tab.keyboard.press("Escape");
    await tab.getByRole("textbox", { name: "Project name", exact: true }).waitFor({ state: "detached" });
  };
  await rename(page, "First tab wins"); await saved(page);
  await rename(second, "Second tab recovery copy");
  await second.waitForFunction(() => window.__inkmap.getState().saveStatus === "failed");
  assert.match(await second.evaluate(() => window.__inkmap.getState().saveError), /project_conflict/);
  assert.equal((await second.evaluate(() => window.__inkmap.getState().toProject())).name, "Second tab recovery copy");
  await second.screenshot({ path: join(out, "conflict.png") }); await second.close();
  await page.evaluate(() => {
    const put = IDBObjectStore.prototype.put;
    window.restoreProjectWrites = () => { IDBObjectStore.prototype.put = put; };
    IDBObjectStore.prototype.put = function(value, key) {
      if (key === "current") throw new DOMException("Storage quota exhausted", "QuotaExceededError");
      return put.call(this, value, key);
    };
  });
  await rename(page, "Recoverable quota edit");
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "failed");
  assert.match(await page.evaluate(() => window.__inkmap.getState().saveError), /quota/i);
  assert.equal((await page.evaluate(() => window.__inkmap.getState().toProject())).name, "Recoverable quota edit");
  await page.evaluate(() => window.restoreProjectWrites());
  // The failure is conspicuous beside the work, with recovery at hand.
  await page.locator(".messages").getByText(/Local save failed/).waitFor();
  await page.locator(".messages").getByRole("button", { name: "Download project backup", exact: true }).waitFor();
  await page.locator(".messages").getByRole("button", { name: "Retry local save", exact: true }).click(); await saved(page);
  // A failed IndexedDB API leaves authoring and recovery downloads available.
  const failedContext = await browser.newContext();
  await failedContext.addInitScript(() => { Object.defineProperty(indexedDB, "open", { value: () => { throw new DOMException("Storage quota exhausted", "QuotaExceededError"); } }); });
  const failedPage = await failedContext.newPage(); await failedPage.goto(url); await ready(failedPage);
  assert.equal(await failedPage.evaluate(() => window.__inkmap.getState().saveStatus), "failed");
  assert.match(await failedPage.evaluate(() => window.__inkmap.getState().saveError), /quota/i);
  assert.ok(await failedPage.evaluate(() => window.__inkmap.getState().toProject()));
  await failedContext.close();
  for (const [width, height] of [[390, 844], [768, 1024]]) {
    const mobile = await browser.newContext({ viewport: { width, height }, hasTouch: true });
    const phone = await mobile.newPage(); await phone.goto(url); await ready(phone);
    // The tray sits under the stage rather than over it: no horizontal
    // overflow, a full-width canvas, and the canvas never under the tray.
    const layout = await phone.evaluate(() => ({ scroll: document.documentElement.scrollWidth, canvas: document.querySelector("canvas").getBoundingClientRect(), dock: document.querySelector(".dock").getBoundingClientRect() }));
    assert.equal(layout.scroll, width); assert.equal(layout.canvas.width, width);
    assert.ok(layout.canvas.bottom <= layout.dock.top + 1, `canvas ${layout.canvas.bottom} overlaps the tray at ${layout.dock.top}`);
    assert.ok(layout.canvas.height >= height * 0.4, `canvas height ${layout.canvas.height} leaves too little body`);
    assert.equal(await phone.locator(".dock-scroll").isVisible(), true);
    await phone.screenshot({ path: join(out, `tray-working-${width}.png`) });
    await phone.getByRole("button", { name: /more/ }).tap();
    assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth), width);
    await phone.screenshot({ path: join(out, `tray-expanded-${width}.png`) });
    await phone.getByRole("button", { name: /less/ }).tap();
    assert.equal(await phone.locator(".dock-scroll").isVisible(), false, "collapsing the tray left it open");
    await phone.waitForTimeout(400); // the canvas follows its container through a ResizeObserver
    const collapsed = await phone.evaluate(() => document.querySelector("canvas").getBoundingClientRect().height);
    assert.ok(collapsed > layout.canvas.height, "collapsing the tray gave the body no more room");
    await phone.screenshot({ path: join(out, `tray-collapsed-${width}.png`) });
    await mobile.close();
  }
  // New project: File offers a fresh, empty project; the confirmation names what it replaces.
  // (the page's dialog handler above accepts the confirmation)
  if (!(await page.locator(".menu-panel").count())) await page.getByRole("button", { name: "File", exact: true }).click();
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 0 && window.__inkmap.getState().projectName === "Untitled project");
  assert.deepEqual(errors, []);
  writeFileSync(join(out, "recovered-project.json"), JSON.stringify(recovered, null, 2));
  writeFileSync(join(out, "report.json"), JSON.stringify({ checked_at: new Date().toISOString(), url, result: "pass", checks: ["pending export refusal", "cancel preserves existing tattoo", "grouped undo redo", "pose appearance camera recovery", "draft recovery", "multi-tab compare-and-swap", "failed storage recovery", "transaction quota failure and retry", "390 and 768 px stacked tray layout", "numeric toolbar edit and cancel"], errors }, null, 2));
  console.log("PASS project recovery/history/storage", out);
} finally { await browser.close(); }
