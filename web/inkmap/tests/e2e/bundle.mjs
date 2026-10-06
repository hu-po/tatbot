import { renderTattooProgramSvg } from "../../src/core/human-representation/program-svg.ts";
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SvgDOMParser } from "../../tools/svg-dom.ts";
import { validateSimulationBundle } from "../../src/core/sim-bundle.ts";
import { canonicalDigest, canonicalDocumentDigest } from "../../src/core/human-representation/schema.ts";
import { POSE_CATALOG } from "../../src/core/pose.ts";
import { validateDesign } from "../../src/core/design.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const atlas = JSON.parse(readFileSync(new URL("../../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8"));
const designIds = JSON.parse(readFileSync(new URL("../../public/designs/manifest.json", import.meta.url), "utf8")).designs.map(design => design.id);
const url = process.argv[2] ?? "http://127.0.0.1:4181/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "bundles") : mkdtempSync(join(tmpdir(), "inkmap-bundles-"));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];
const browserMessages = [];
try {
  const context = await browser.newContext({ viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (["warning", "error"].includes(message.type())) browserMessages.push(message.text()); });
  await page.goto(url);
  await page.waitForFunction(ids => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady
    && s.designs.length === ids.length && ids.every(id => s.designs.some(design => design.id === id)); }, designIds, { timeout: 120_000 });
  const designs = await page.evaluate(() => window.__inkmap.getState().designs.map(d => d.id));
  const reports = [];
  let previewBundle;
  // Simulation formats live under File › Advanced; the menu closes on any
  // outside click, so it is reopened before each advanced action.
  const advanced = async () => {
    if (!(await page.locator(".menu-panel").count())) await page.getByRole("button", { name: "File", exact: true }).click();
    if (!(await page.locator(".menu-advanced[open]").count())) await page.locator(".menu-advanced > summary").click();
    if (!(await page.locator(".simulation-export[open]").count())) await page.locator(".simulation-export summary").click();
  };
  await advanced();
  await page.getByRole("spinbutton", { name: "Simulation seed", exact: true }).fill("314");
  await page.keyboard.press("Escape");
  for (const id of designs) {
    await page.evaluate(id => {
      const s = window.__inkmap.getState(); const d = s.designs.find(d => d.id === id);
      const file = s.snapshotFile();
      file.placements = [{ id: `placement-${id}`, design_id: id, anchor: { face: 8729, barycentric: [1/3, 1/3, 1/3] },
        rotation_rad: 0, size_mm: [25, 25 * d.default_size_mm[1] / d.default_size_mm[0]], mirror: false }];
      s.loadFile(file);
    }, id);
    assert.equal(await page.evaluate(() => window.__inkmap.getState().error), null);
    await advanced();
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 60_000 }),
      page.getByRole("button", { name: "Export simulation bundle", exact: true }).click(),
    ]);
    await page.keyboard.press("Escape");
    const path = join(out, `${id}.bundle.json`); await download.saveAs(path);
    const bundle = await validateSimulationBundle(JSON.parse(readFileSync(path, "utf8")), atlas);
    previewBundle ??= bundle;
    assert.equal(bundle.request.seed, 314); assert.equal(bundle.artworks[id].source.kind, "stock");
    assert.equal(bundle.surface_placements[0].id, `placement-${id}`);
    const [designDownload] = await Promise.all([
      page.waitForEvent("download", { timeout: 60_000 }),
      page.getByRole("button", { name: "Export design", exact: true }).click(),
    ]);
    const designPath = join(out, `${id}.design.json`); await designDownload.saveAs(designPath);
    const design = await validateDesign(JSON.parse(readFileSync(designPath, "utf8")));
    assert.deepEqual(design.artworks, bundle.artworks);
    assert.equal(design.placements[0].id, `placement-${id}`);
    assert.equal(design.placements[0].placement.target.kind, "body");
    assert.deepEqual(design.placements[0].placement.target.anchor, bundle.surface_placements[0].placement.anchor);
    reports.push({ design: id, content_sha256: bundle.content_sha256, bytes: readFileSync(path).length });
  }
  await page.evaluate(() => { const s = window.__inkmap.getState(); s.select(s.placements[0].id); });
  await page.getByRole("button", { name: "File", exact: true }).click();
  await page.getByRole("button", { name: /^Portable design \(JSON\)/ }).click();
  assert.match(await page.evaluate(() => window.__inkmap.getState().error), /Accept or cancel/);
  await page.locator(".messages [role=alert]").filter({ hasText: "Portable design" }).waitFor();
  await advanced();
  await page.getByRole("button", { name: "Export simulation bundle", exact: true }).click();
  assert.match(await page.evaluate(() => window.__inkmap.getState().error), /Accept or cancel/);
  await page.keyboard.press("Escape");
  await page.evaluate(() => window.__inkmap.getState().cancelPlacing());
  const inkProgram = JSON.parse(readFileSync(new URL("../../../../config/human-representation/examples/ink-program.json", import.meta.url), "utf8"));
  const placement = previewBundle.placement_file.placements[0];
  const artwork = previewBundle.artworks[placement.design_id];
  inkProgram.tattoo_program_sha256 = artwork.program.content_sha256;
  inkProgram.surface_placement_sha256 = previewBundle.surface_placements[0].placement.content_sha256;
  inkProgram.content_sha256 = await canonicalDigest(inkProgram);
  const strokes = inkProgram.events.filter(event => event.kind === "stroke").map(event =>
    event.curve.coordinates.map(point => ({ face: point.face_index, barycentric: point.barycentric })));
  const template = JSON.parse(readFileSync(new URL("../../public/showcase/supine-shin.scenario.json", import.meta.url), "utf8"));
  const pose = POSE_CATALOG.poses[previewBundle.request.pose_id];
  const compiled = {
    ...template, schema_version: 3, seed: previewBundle.request.seed,
    pose: { ...template.pose, id: previewBundle.request.pose_id,
      catalog_sha256: previewBundle.request.pose_catalog_sha256, posed_surface_sha256: pose.surface_sha256 },
    placement: { ...placement, source_sha256: await canonicalDigest(previewBundle.placement_file) },
    design: { id: placement.design_id, name: artwork.name, svg: renderTattooProgramSvg(artwork.program),
      sha256: artwork.source_sha256,
      source: previewBundle.placement_file.designs[placement.design_id].source ?? { kind: "embedded" } },
    trace: { compiler: "tatbot_sim.surface_trace", compiler_version: 3,
      sha256: await canonicalDocumentDigest(strokes), strokes },
    robot: { ...template.robot, tool_id: previewBundle.request.tool_id },
    support: { id: previewBundle.request.support_id, world_from_nominal: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
    program_binding: { bundle: previewBundle, placement_id: placement.id,
      ink_program: inkProgram, tool_profile_sha256: "a".repeat(64) },
  };
  await advanced();
  await page.getByLabel("Open compiled preview", { exact: true }).setInputFiles({
    name: "browser-compiled-v3.scenario.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(compiled)),
  });
  await page.waitForFunction(() => {
    const state = window.__inkmap.getState();
    return state.showcaseScenario?.schema_version === 3 || state.error;
  });
  assert.equal(await page.evaluate(() => window.__inkmap.getState().error), null);
  await page.getByRole("status").filter({ hasText: "browser-compiled-v3.scenario.json" }).waitFor();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().showcaseScenario?.schema_version), 3);
  assert.equal(await page.evaluate(() => window.__inkmap.getState().skinTone), previewBundle.request.skin_tone);
  assert.equal(await page.evaluate(id => {
    const path = window.__inkmap.getState().designs.find(design => design.id === id).path;
    return decodeURIComponent(path.slice(path.indexOf(",") + 1));
  }, placement.design_id), renderTattooProgramSvg(artwork.program));
  assert.deepEqual(errors, []);
  await page.screenshot({ path: join(out, "export.png") });
  writeFileSync(join(out, "report.json"), JSON.stringify({ result: "pass", url, checked_at: new Date().toISOString(), reports, errors }, null, 2));
  console.log("PASS shared stock simulation bundle downloads and pending-edit refusal", out);
} catch (error) {
  const page = browser.contexts()[0]?.pages()[0];
  const state = page && !page.isClosed() ? await page.evaluate(() => {
    const s = window.__inkmap?.getState();
    return { error: s?.error, selected: s?.selected, placing: s?.placing, placements: s?.placements };
  }).catch(() => null) : null;
  writeFileSync(join(out, "failure.json"), JSON.stringify({ message: String(error), errors, browserMessages, state }, null, 2));
  if (page && !page.isClosed()) await page.screenshot({ path: join(out, "failure.png") }).catch(() => {});
  throw error;
} finally { await browser.close(); }
