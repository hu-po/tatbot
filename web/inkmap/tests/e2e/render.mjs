import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { canonicalDigest } from "../../src/core/human-representation/schema.ts";

// A robot render on the body: what `tatbot art separate --brief` gives back, held to the placement it was made for.
const url = process.argv[2] ?? "http://127.0.0.1:4186/";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];
const dir = mkdtempSync(join(tmpdir(), "inkmap-render-"));
const fixture = JSON.parse(readFileSync(new URL("../fixtures/render.json", import.meta.url), "utf8"));

/** The pipeline's record, for this art and size. */
async function renderFile(name, fields) {
  const render = { ...fixture, ...fields, content_sha256: "" };
  render.content_sha256 = await canonicalDigest(render);
  const path = join(dir, name);
  writeFileSync(path, JSON.stringify(render));
  return path;
}

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/designs/renders/**", route => route.abort());   // the estimate and Load render, without the stock's own
  const ready = () => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  await page.goto(url);
  await ready();
  await page.getByRole("textbox", { name: "Describe a location", exact: true }).fill("on the left forearm");
  await page.getByRole("button", { name: "Resolve", exact: true }).click();
  await page.getByRole("button", { name: "Swallow and poppies", exact: true }).click();
  const fit = page.getByRole("region", { name: "Fit and cost" });
  await fit.waitFor();
  assert.match(await fit.innerText(), /Estimate/);
  assert.match(await fit.innerText(), /tatbot art separate --brief/);

  const { art, height } = await page.evaluate(() => { const s = window.__inkmap.getState(); return { art: s.designs[0].embedded.content_sha256, height: s.placements[0].size_mm[1] }; });
  const load = page.getByLabel("Load robot render").first();

  // Art this project does not hold is refused by name, and nothing is shown.
  await load.setInputFiles(await renderFile("other.json", { art_sha256: "b".repeat(64), width_mm: 60, height_mm: height }));
  await page.getByRole("alert").filter({ hasText: "art this project does not hold" }).waitFor();

  // Its own render replaces the estimate with the pipeline's figures, and what it lost.
  await load.setInputFiles(await renderFile("mine.json", { art_sha256: art, width_mm: 60, height_mm: height, minutes: 137, dips: 68, rinses: 5 }));
  await fit.getByText("Separated").waitFor();
  const separated = await fit.innerText();
  assert.match(separated, /2 h 15 min, 68 dips, 5 rinses/);
  assert.match(separated, /Light Green drew nothing at this size/);
  assert.doesNotMatch(separated, /Estimate/);

  // A size it was not made for shows the art again, and says why.
  await page.getByRole("spinbutton", { name: "Width in millimeters", exact: true }).fill("50");
  await fit.getByText("The render is for 60.0 mm").waitFor();
  assert.match(await fit.innerText(), /Estimate/);
  await fit.getByRole("button", { name: "Use 60.0 mm" }).click();
  await fit.getByText("Separated").waitFor();

  // It can be switched off, and comes back; it is kept in this browser, not in the project.
  const shown = fit.getByRole("checkbox", { name: "Show it on the body" });
  await shown.uncheck();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().showRender), false);
  await shown.check();
  await page.getByRole("button", { name: /Accept/ }).click();
  const project = await page.evaluate(() => window.__inkmap.getState().toProject());
  assert.doesNotMatch(JSON.stringify(project), /inkmap-render/);
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 20_000 });

  // The card prints what will be drawn.
  const [card] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Placement card", exact: true }).click()]);
  const html = readFileSync(await card.path(), "utf8");
  assert.ok(html.includes(fixture.image.data_base64), "the card's figure is the render");
  assert.match(html, /2 h 15 min, 68 dips \(separated\)/);

  await page.reload();
  await ready();
  await page.waitForFunction(() => window.__inkmap.getState().renders.length === 1, null, { timeout: 20_000 });

  // New inks are new art: the render no longer stands for it.
  await page.evaluate(() => { const s = window.__inkmap.getState(); s.select(s.placements[0].id); });
  await fit.getByText("Separated").waitFor();
  await fit.getByRole("button", { name: "Remove Light Green" }).click();
  await fit.getByText("Estimate").waitFor();

  // A visitor with nothing to load: the collection's own renders, at the default width and the skin tile's.
  const visitor = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  visitor.on("pageerror", error => errors.push(error.message));
  const requested = [];
  visitor.on("request", request => { if (request.url().includes("/designs/renders/")) requested.push(request.url()); });
  await visitor.goto(url);
  await visitor.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  assert.deepEqual(requested, [], "nothing is fetched until a design is placed");
  await visitor.getByRole("textbox", { name: "Describe a location", exact: true }).fill("on the outer left biceps");
  await visitor.getByRole("button", { name: "Resolve", exact: true }).click();
  await visitor.getByRole("button", { name: "Swallow and poppies", exact: true }).click();
  const panel = visitor.getByRole("region", { name: "Fit and cost" });
  await panel.getByText("Separated").waitFor({ timeout: 30_000 });
  assert.match(await panel.innerText(), /dips, \d+ rinses/);
  const showIt = panel.getByRole("checkbox", { name: "Show it on the body" });
  assert.equal(await showIt.isChecked(), false, "a new visitor meets the art; the render is theirs to turn on");
  assert.equal(requested.length, 2, "only this design's two renders");
  await showIt.check();
  await panel.getByRole("button", { name: /Fit to Forearm/ }).click();
  await visitor.waitForFunction(() => window.__inkmap.getState().placements[0].size_mm[0] === 45);
  await panel.getByText("Light Green drew nothing at this size").waitFor();
  const widthBox = visitor.getByRole("spinbutton", { name: "Width in millimeters", exact: true });
  await widthBox.fill("50");
  await panel.getByText("The render is for 45.0 mm").waitFor();
  await widthBox.fill("58");
  await panel.getByText("The render is for 60.0 mm").waitFor();
  await panel.getByRole("button", { name: "Use 60.0 mm" }).click();
  await panel.getByText("Separated").waitFor();
  await visitor.getByRole("button", { name: /Accept/ }).click();
  await visitor.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 20_000 });
  assert.deepEqual(await visitor.evaluate(() => window.__inkmap.getState().renders), [], "the collection's renders are not the visitor's to save");
  await visitor.reload();
  await visitor.waitForFunction(() => window.__inkmap?.getState().stockRenders.length === 2, null, { timeout: 60_000 });
  assert.equal(await visitor.evaluate(() => window.__inkmap.getState().showRender), true, "the choice is remembered");
  assert.deepEqual(errors, []);
  console.log("render: ok");
} finally {
  await browser.close();
}
