// Inkgen's own page, driven in a browser against a running inkgen (the
// stand-in engine is enough: `INKGEN_FAKE_ENGINE=1 python web/inkgen/app.py`).
// Not part of the default runner because it needs the Python service.
//   node tests/e2e/inkgen_page.mjs [inkgen url]
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:8601/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "inkgen-page") : mkdtempSync(join(tmpdir(), "inkgen-page-"));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true });
const health = await (await fetch(new URL("/api/health", url))).json();
try {
  for (const [width, height] of [[1280, 900], [390, 844]]) {
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url);
    await page.getByText(/Inkgen draws tattoo artwork/).waitFor();
    assert.equal(await page.getByRole("checkbox", { name: /Random seed/ }).isChecked(), true, "random seed is the explicit default");
    assert.equal(await page.getByRole("spinbutton", { name: "Seed" }).isVisible().catch(() => false), false, "the seed field hides behind the random default");
    await page.getByRole("button", { name: "Generate artwork", exact: true }).click();
    await page.getByText(/^seed \d+ ·/).waitFor({ timeout: 120_000 });
    const first = await page.getByText(/^seed \d+ ·/).textContent();
    assert.match(first, health.fake_engine ? /stand-in engine/ : new RegExp(health.model.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    const firstSrc = await page.locator("img").first().getAttribute("src");
    assert.ok(firstSrc, "no image shown");
    await page.screenshot({ path: join(out, `generated-${width}.png`), fullPage: true });
    // The previous image stays while the next draws, and a variation has a new seed.
    await page.getByRole("button", { name: "New variation", exact: true }).click();
    assert.ok(await page.locator("img").first().getAttribute("src"), "the previous image was dropped while drawing");
    await page.waitForFunction((before) => !document.body.innerText.includes(before), first, { timeout: 120_000 });
    const second = await page.getByText(/^seed \d+ ·/).textContent();
    assert.notEqual(second.split(" · ")[0], first.split(" · ")[0], "a variation did not get a new seed");
    // Seed zero is a seed: an explicit zero is used and shown.
    await page.getByRole("checkbox", { name: /Random seed/ }).uncheck();
    await page.getByRole("spinbutton", { name: "Seed" }).fill("0");
    await page.getByRole("button", { name: "Generate artwork", exact: true }).click();
    await page.getByText(/^seed 0 ·/).waitFor({ timeout: 120_000 });
    const [png] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download PNG" }).click()]);
    const pngPath = join(out, `seed0-${width}.png`); await png.saveAs(pngPath);
    assert.ok(readFileSync(pngPath).subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])), "PNG download is not a PNG");
    const [meta] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download generation metadata" }).click()]);
    const metaPath = join(out, `seed0-${width}.json`); await meta.saveAs(metaPath);
    const document = JSON.parse(readFileSync(metaPath, "utf8"));
    assert.equal(document.seed, 0); assert.equal(document.seed_requested, true);
    assert.equal(document.model, health.model); assert.match(document.png_sha256, /^[0-9a-f]{64}$/);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, "horizontal overflow");
    await page.getByRole("button", { name: "Check service", exact: true }).click();
    await page.getByText(/^Ready — /).waitFor();
    await page.screenshot({ path: join(out, `seed0-${width}.png.screenshot.png`), fullPage: true });
    assert.deepEqual(errors, []);
    await page.close();
  }
  writeFileSync(join(out, "report.json"), JSON.stringify({ schema: "tatbot.inkgen-page-report/1", checked_at: new Date().toISOString(), url, backend: { model: health.model, model_revision: health.model_revision, device: health.device, fake_engine: health.fake_engine }, result: "pass" }, null, 2) + "\n");
  console.log(`PASS inkgen page (${health.fake_engine ? "stand-in engine" : health.model}) ${out}`);
} finally { await browser.close(); }
