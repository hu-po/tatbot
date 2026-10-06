import { acquiredArtwork } from "../acquired-artwork.ts";
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SvgDOMParser } from "../../tools/svg-dom.ts";
import { renderTattooProgramSvg } from "../../src/core/human-representation/program-svg.ts";
import { validateDesign } from "../../src/core/design.ts";
Object.assign(globalThis, { DOMParser: SvgDOMParser });
const out = mkdtempSync(join(tmpdir(), "inkmap-chart-"));
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(process.argv[2] ?? "http://127.0.0.1:4181/");
  const ready = () => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  const workspace = () => page.evaluate(() => window.__inkmapUi.getState().workspace);
  const download = async (name, click) => {
    // A download that never comes is reported with what the page said instead.
    const [event] = await Promise.all([page.waitForEvent("download", { timeout: 15_000 }).catch(async (error) => {
      const said = await page.evaluate(() => ({ messages: Array.from(document.querySelectorAll(".messages [role], .menu-panel [role=alert]")).map((e) => e.textContent),
        notice: window.__inkmapUi.getState().notice, error: window.__inkmap.getState().error }));
      throw new Error(`${error.message}\npage said: ${JSON.stringify(said)}`);
    }), click()]);
    const path = join(out, name); await event.saveAs(path); return path;
  };
  // Every file action lives under File; the menu closes after a download.
  const fileMenu = async () => { if (!(await page.locator(".menu-panel").count())) await page.getByRole("button", { name: "File", exact: true }).click(); };
  const openDesign = async (path) => { await fileMenu(); await page.getByLabel("Open design…").setInputFiles(path); await page.keyboard.press("Escape"); };
  const saveDesign = async name => {
    await fileMenu();
    const path = await download(`${name}.json`, () => page.getByRole("button", { name: /^Portable design \(JSON\)/ }).click());
    return { path, design: await validateDesign(JSON.parse(readFileSync(path, "utf8"))) };
  };
  const surface = async () => { if (!(await page.locator(".surface-details[open]").count())) await page.locator(".surface-details > summary").click(); };
  // Place artwork the way the body does it: pick it from the shared chooser,
  // then click the paper where it should go. The fixture's centre projects
  // to the middle of the stage, so a click there lands at (0, 0).
  const stage = async () => { const box = await page.locator(".chart-stage").boundingBox(); return { x: box.x + box.width / 2, y: box.y + box.height / 2 }; };
  const placeArtwork = async (name, dx = 0, dy = 0) => {
    const before = await page.evaluate(() => window.__inkmap.getState().chart.items.length);
    if (!(await page.getByRole("tab", { name: "Library" }).count())) await page.getByRole("button", { name: "Add artwork", exact: true }).click();
    await page.getByRole("tab", { name: "Library" }).click();
    if (await page.getByRole("button", { name: /^Browse all/ }).count()) await page.getByRole("button", { name: /^Browse all/ }).click();
    await page.getByRole("button", { name, exact: true }).click();
    await page.waitForFunction(() => window.__inkmap.getState().chartPlacing !== null);
    const at = await stage();
    await page.mouse.move(at.x + dx, at.y + dy);
    // The ghost under the pointer is the sign the paper has taken the pointer; a click before it rays into nothing.
    await page.waitForFunction(() => window.__inkmap.getState().chartHover !== null, null, { timeout: 15_000 });
    await page.mouse.click(at.x + dx, at.y + dy);
    // A refusal (artwork off the drawable area) is reported as such rather than as a silent timeout.
    await page.waitForFunction((n) => { const s = window.__inkmap.getState(); return s.chart.items.length === n || s.error; }, before + 1);
    const refused = await page.evaluate(() => window.__inkmap.getState().error);
    assert.equal(refused, null, `placing ${name} was refused: ${refused}`);
  };
  const accept = () => page.getByRole("button", { name: "✓ Accept", exact: true }).click();

  // One body design, authored the ordinary way, so the routing below is tested
  // against a real file rather than a checked-in fixture that could drift.
  await ready();
  await page.fill(".sentence input", "an dbv3-ridges on the left forearm");
  await page.click(".sentence button.primary");
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1);
  await page.getByRole("button", { name: "✓ Accept", exact: true }).click();
  const body = await saveDesign("body-design");
  const bodyPlacements = await page.evaluate(() => window.__inkmap.getState().placements);
  assert.equal(body.design.placements[0].placement.target.kind, "body");

  // The paper workspace: the same chooser as the body, the same click to
  // place, the same Accept. The tab is enabled only once the project has
  // loaded, so a recovered draft cannot be replaced by an empty pad.
  assert.equal(await page.getByRole("tab", { name: "Paper" }).isEnabled(), true);
  await page.getByRole("tab", { name: "Paper" }).click();
  assert.equal(await workspace(), "chart");
  await placeArtwork("dbv3-orbit");
  assert.equal(await page.getByLabel("Width in millimeters").isVisible(), true, "placing lands in the adjust toolbar");
  // A pending edit is refused at export, beside the action, like the body's.
  await fileMenu();
  await page.getByRole("button", { name: /^Portable design \(JSON\)/ }).click();
  await page.locator(".messages [role=alert]").filter({ hasText: /pending edits/ }).waitFor();
  if (await page.locator(".menu-panel").count()) await page.keyboard.press("Escape");
  await accept();
  await page.getByRole("button", { name: /^dbv3-orbit/ }).waitFor();
  const plane = await saveDesign("plane");
  assert.equal(plane.design.placements[0].placement.target.kind, "plane");
  assert.deepEqual(plane.design.placements[0].placement.target.canvas_m, [.1905, .2794]); // the 7.5 × 11 in pad
  const anchor = plane.design.placements[0].placement.target.anchor_uv_m;
  assert.ok(Math.hypot(anchor[0], anchor[1]) < 0.01, `a click on the middle of the pad landed at ${anchor}`);
  const preview = await page.getByLabel("Surface placement preview").boundingBox();
  assert.ok(preview.y >= 0 && preview.y + preview.height <= 1000, "complete preview fits the viewport");
  await page.screenshot({ path: join(out, "plane.png") });

  // Switching workspaces keeps every draft to itself: the body's placement,
  // the pad's item, and a cylinder that starts empty — nothing crosses.
  await page.getByRole("tab", { name: "Body" }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), 1);
  await page.getByRole("tab", { name: "Paper" }).click();
  assert.equal(await page.getByRole("button", { name: /^dbv3-orbit/ }).count(), 1);
  await page.getByRole("tab", { name: "Cylinder" }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().chart.items.length), 0, "the pad's item leaked onto the cylinder");
  await surface();
  assert.equal(await page.getByLabel("Surface", { exact: true }).inputValue(), "cylinder");
  assert.equal(Number(await page.getByLabel("Surface length along axis (mm)").inputValue()), 190.5, "the cylinder did not start at its own fixture size");
  await page.getByLabel("Cylinder radius (mm)").fill("60");
  // Placed on the cylinder the same way — a little above the middle of the
  // view, so the 90 mm dbv3-orbit sits near the crest rather than down the
  // flank where a 60 mm radius leaves it a hair outside the band — then
  // resized and turned through the same toolbar the body uses.
  await placeArtwork("dbv3-orbit", 0, -140);
  await page.getByLabel("Width in millimeters").fill("30");
  await page.getByLabel("Rotation in degrees").fill("15");
  await accept();
  const cylinder = await saveDesign("cylinder");
  // ...and back on the pad, the pad's own item is exactly as it was.
  await page.getByRole("tab", { name: "Paper" }).click();
  assert.deepEqual((await saveDesign("pad-again")).design, plane.design, "the cylinder's edit reached the pad");
  await page.getByRole("tab", { name: "Cylinder" }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().chart.items.length), 1, "the cylinder's item did not survive a switch");
  assert.equal(cylinder.design.placements[0].placement.target.radius_m, .06);
  assert.equal(cylinder.design.placements[0].placement.rotation_rad, Math.PI / 12);
  await openDesign(plane.path);
  await page.waitForFunction(() => window.__inkmap.getState().chart.kind === "plane");
  assert.equal(await page.getByRole("tab", { name: "Paper" }).getAttribute("aria-selected"), "true", "opening a paper design did not land on the Paper tab");
  const reopened = await saveDesign("reopened");
  assert.deepEqual(reopened.design, plane.design);

  // The surface draft is recovered like the body one: a reload lands back in
  // this editor's work rather than an empty pad.
  await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved", null, { timeout: 15_000 });
  await page.reload();
  await ready();
  await page.getByRole("tab", { name: "Paper" }).click();
  await page.getByRole("button", { name: /^dbv3-orbit/ }).waitFor();
  const recovered = await saveDesign("recovered");
  assert.deepEqual(recovered.design, plane.design, "reload lost or changed the surface draft");

  // One Open design: a body design opened from the chart goes to the body
  // editor by itself, and the chart draft is left exactly as it was.
  await openDesign(body.path);
  await page.waitForFunction(() => window.__inkmapUi.getState().workspace === "body");
  await page.waitForFunction(() => window.__inkmap.getState().placements.length === 1);
  const restored = await page.evaluate(() => window.__inkmap.getState().placements);
  assert.deepEqual(restored.map(p => [p.anchor, p.size_mm, p.rotation_rad, p.mirror]),
    bodyPlacements.map(p => [p.anchor, p.size_mm, p.rotation_rad, p.mirror]),
    "reopening a body design moved its placements");
  const again = await saveDesign("body-design-again");
  assert.equal(again.design.content_sha256, body.design.content_sha256, "an unedited body design changed identity on the way back out");
  await page.getByRole("tab", { name: "Paper" }).click();
  assert.equal(await page.getByRole("button", { name: /^dbv3-orbit/ }).count(), 1, "routing a body design changed the chart draft");
  // ...and a paper design opened from the body goes to the chart.
  await page.getByRole("tab", { name: "Body" }).click();
  await openDesign(plane.path);
  await page.waitForFunction(() => window.__inkmapUi.getState().workspace === "chart");
  assert.deepEqual((await saveDesign("routed")).design, plane.design);

  // An unsupported design (mixed targets) is refused by name, the working
  // draft is untouched, and the original bytes stay downloadable unchanged.
  const mixed = { ...plane.design, placements: [...plane.design.placements, ...body.design.placements], artworks: { ...plane.design.artworks, ...body.design.artworks } };
  const { canonicalDigest } = await import("../../src/core/human-representation/schema.ts");
  mixed.content_sha256 = await canonicalDigest({ ...mixed, content_sha256: "" });
  const mixedText = JSON.stringify(mixed);
  await fileMenu();
  await page.getByLabel("Open design…").setInputFiles({ name: "mixed.json", mimeType: "application/json", buffer: Buffer.from(mixedText) });
  await page.locator(".menu-panel [role=alert]").filter({ hasText: /mixes body and plane/ }).waitFor();
  const unchanged = await download("mixed-unchanged.json", () => page.getByRole("button", { name: "Download mixed.json unchanged", exact: true }).click());
  assert.equal(readFileSync(unchanged, "utf8"), mixedText, "the unsupported file was not kept byte for byte");
  await page.keyboard.press("Escape");
  assert.equal(await page.getByRole("button", { name: /^dbv3-orbit/ }).count(), 1, "a refused open changed the draft");
  assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), 1, "a refused open changed the body");

  // Cancel restores the accepted placement exactly, and a ghost that would
  // run off the paper is refused where it is clicked.
  await page.getByRole("button", { name: /^dbv3-orbit/ }).click();
  await page.getByLabel("Rotation in degrees").fill("90");
  await page.getByRole("button", { name: "✕ Cancel", exact: true }).click();
  assert.deepEqual((await saveDesign("cancelled")).design, plane.design, "Cancel did not restore the accepted placement");

  // Two distinct artworks: the second lands lower on the pad, is drawn
  // earlier, and the artwork SVG export is the *selected* one.
  await placeArtwork("dbv3-sprout", 0, 120);
  await page.getByRole("button", { name: "Draw earlier", exact: true }).click();
  const leafSvg = await download("leaf.svg", () => page.getByRole("button", { name: /^Download artwork SVG — dbv3-sprout/ }).click());
  await accept();
  const layered = await saveDesign("layered");
  assert.deepEqual(layered.design.placements.map(item => item.artwork_id), ["dbv3-sprout", "dbv3-orbit"]);
  assert.equal(readFileSync(leafSvg, "utf8"), renderTattooProgramSvg(layered.design.artworks["dbv3-sprout"].program));
  assert.ok(layered.design.placements[0].placement.target.anchor_uv_m[1] < -0.02, "the second artwork was not placed where the pad was clicked");
  await page.getByRole("button", { name: /^dbv3-orbit/ }).click();
  const jellySvg = await download("jelly.svg", () => page.getByRole("button", { name: /^Download artwork SVG — dbv3-orbit/ }).click());
  assert.equal(readFileSync(jellySvg, "utf8"), renderTattooProgramSvg(layered.design.artworks["dbv3-orbit"].program), "artwork export was not the selected artwork");
  assert.notEqual(readFileSync(jellySvg, "utf8"), readFileSync(leafSvg, "utf8"));
  await fileMenu();
  const viaMenu = await download("jelly-menu.svg", () => page.getByRole("button", { name: /^Artwork SVG — dbv3-orbit/ }).click());
  assert.equal(readFileSync(viaMenu, "utf8"), readFileSync(jellySvg, "utf8"));
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  // Undo takes the accepted step back, like the body's.
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().chart.items.length), 1, "undo did not remove the last placement");
  await page.getByRole("button", { name: "Redo", exact: true }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().chart.items.length), 2);

  // A desktop window dragged narrow keeps its dock beside the paper; the
  // phone tray is for touch screens only.
  await page.setViewportSize({ width: 640, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 640);
  assert.equal(await page.locator(".tray-toggle").count(), 0, "a narrow desktop window switched to the phone tray");
  assert.ok((await page.locator(".dock").boundingBox()).x > 300, "the dock left the side of the stage");
  await page.setViewportSize({ width: 1400, height: 1000 });

  // The phone tray: the chart and its panel stack; the adjust toolbar sits
  // in the tray, and collapsing it gives the paper the room.
  const phone = await (await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true })).newPage();
  await phone.goto(process.argv[2] ?? "http://127.0.0.1:4181/");
  await phone.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  await phone.getByRole("tab", { name: "Paper" }).tap();
  // The paper's canvas takes its size from its container a frame after it mounts; a tap before that rays into nothing.
  await phone.waitForFunction(() => { const c = document.querySelector(".chart-stage canvas"); return c && c.width > 300; });
  await phone.waitForTimeout(600);
  await phone.getByRole("button", { name: "dbv3-orbit", exact: true }).tap();
  await phone.waitForFunction(() => window.__inkmap.getState().chartPlacing !== null);
  const box = await phone.locator(".chart-stage").boundingBox();
  await phone.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
  await phone.waitForFunction(() => window.__inkmap.getState().chart.items.length === 1);
  assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth), 390);
  assert.equal(await phone.getByLabel("Width in millimeters").isVisible(), true);
  await phone.getByRole("button", { name: /less|more/ }).tap();
  await phone.getByRole("button", { name: /less/ }).tap();
  assert.equal(await phone.getByLabel("Width in millimeters").isVisible(), false);
  await phone.screenshot({ path: join(out, "chart-390.png") });
  await phone.context().close();
  // An external conversion must remain the same artwork after IndexedDB
  // recovery, including body imports and the chart that is parked on reload.
  const { embeddedFromArtwork, designFromBodyFile } = await import("../../src/core/body-design.ts");
  const { EMPTY_DRAFT, newChartItem, designFromChartDraft } = await import("../../src/core/chart-draft.ts");
  const external = await acquiredArtwork();
  const externalFile = JSON.parse(readFileSync(new URL("../../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8"));
  externalFile.designs["line-v1"] = embeddedFromArtwork(external);
  const atlas = JSON.parse(readFileSync(new URL("../../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8"));
  const targets = [await designFromBodyFile("External drawing", externalFile, atlas)];
  for (const kind of ["plane", "cylinder"]) targets.push(await designFromChartDraft({ ...EMPTY_DRAFT, kind,
    name: "External drawing", radius: 60, items: [newChartItem("external", "line-v1", external, [30, 30])] }));
  for (const design of targets) {
    const kind = design.placements[0].placement.target.kind;
    const path = join(out, `external-${kind}.json`); writeFileSync(path, JSON.stringify(design));
    await openDesign(path);
    await page.waitForFunction(({ kind, digest }) => {
      const s = window.__inkmap.getState();
      return kind === "body" ? s.designs.find(d => d.id === "line-v1")?.embedded?.content_sha256 === digest
        : s.chart.kind === kind && s.chart.items[0]?.artwork.content_sha256 === digest;
    }, { kind, digest: external.content_sha256 });
    await page.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved");
    await page.reload(); await ready();
    await page.getByRole("tab", { name: kind === "body" ? "Body" : kind === "plane" ? "Paper" : "Cylinder", exact: true }).click();
    assert.deepEqual((await saveDesign(`external-${kind}-recovered`)).design, design,
      `${kind} recovery changed the frozen artwork`);
  }
  // The paper artwork was parked while the cylinder was saved and reopened.
  await page.getByRole("tab", { name: "Paper", exact: true }).click();
  assert.deepEqual((await saveDesign("external-parked-paper")).design, targets[1]);
  // The file chooser accepts the acquisition itself without any source SVG.
  const artworkPath = join(out, "acquired-artwork.json");
  writeFileSync(artworkPath, JSON.stringify(external));
  await page.getByRole("button", { name: "Add artwork", exact: true }).click();
  await page.getByLabel("Import artwork").setInputFiles(artworkPath);
  await page.waitForFunction(() => window.__inkmap.getState().chartPlacing !== null);
  const imported = await page.evaluate(() => {
    const s = window.__inkmap.getState();
    return s.designs.find(d => d.id === s.chartPlacing)?.embedded;
  });
  assert.deepEqual(imported, external, "JSON import changed acquired geometry");
  assert.equal(errors.length, 0, JSON.stringify(errors));
  console.log(`PASS plane/cylinder placing, adjusting, undo, draft recovery, design routing, selected-artwork export and round-trips ${out}`);
} finally { await browser.close(); }
