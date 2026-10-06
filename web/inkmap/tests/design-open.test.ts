import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { bodyFileFromDesign, designFromBodyFile } from "../src/core/body-design.ts";
import { chartDraftFromDesign, designFromChartDraft, EMPTY_DRAFT, newChartItem, type ChartDraft } from "../src/core/chart-draft.ts";
import { routeDesign } from "../src/core/design-open.ts";
import { acquiredArtwork } from "./acquired-artwork.ts";
import { parseDesign } from "../src/core/design.ts";
import type { AtlasData } from "../src/core/atlas.ts";
import type { PlacementFile } from "../src/core/schema.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const atlas = JSON.parse(readFileSync(new URL("../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8")) as AtlasData;
const file = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8")) as PlacementFile;
const cylinderFixture = readFileSync(new URL("./fixtures/python-cylinder-design.json", import.meta.url), "utf8");

async function draft(overrides: Partial<ChartDraft> = {}): Promise<ChartDraft> {
  const orbit = await acquiredArtwork();
  const sprout = await acquiredArtwork("dbv3-sprout");
  return {
    ...EMPTY_DRAFT, name: "Flash sheet", width: 200, height: 260,
    items: [
      { ...newChartItem("a", "dbv3-orbit", orbit, [30, 30]), uv: [-10, 12] as [number, number], rotation_rad: Math.PI / 6 },
      { ...newChartItem("b", "dbv3-sprout", sprout, [30, 30]), uv: [8, -20] as [number, number], rotation_rad: -Math.PI / 4, mirror: true },
    ],
    ...overrides,
  };
}

// ---- routing --------------------------------------------------------------
test("a design is routed by what it actually places artwork on", async () => {
  assert.equal(routeDesign(await designFromBodyFile("Body artwork", file, atlas)).kind, "body");
  assert.equal(routeDesign(await designFromChartDraft(await draft())).kind, "chart");
  assert.equal(routeDesign(await parseDesign(cylinderFixture)).kind, "chart");
});

test("a design no editor can show says which feature is unsupported", async () => {
  const chart = await designFromChartDraft(await draft());
  const body = await designFromBodyFile("Body artwork", file, atlas);
  const mixed = structuredClone(chart);
  mixed.placements.push(structuredClone(body.placements[0]));
  const route = routeDesign(mixed);
  assert.equal(route.kind, "unsupported");
  assert.match(route.kind === "unsupported" ? route.reason : "", /mixes body and plane targets/);
  // The original document is handed back untouched, never approximated.
  assert.deepEqual(route.design, mixed);
});

// ---- body round trip ------------------------------------------------------
test("a body design reopens as the same placements and exports the same identity", async () => {
  const design = await designFromBodyFile("Body artwork", file, atlas);
  const reopened = await bodyFileFromDesign(design, atlas);
  assert.deepEqual(reopened.placements.map(p => p.anchor), file.placements.map(p => p.anchor));
  assert.deepEqual(reopened.placements.map(p => p.size_mm), file.placements.map(p => p.size_mm));
  assert.deepEqual(reopened.placements.map(p => [p.rotation_rad, p.mirror]),
    file.placements.map(p => [p.rotation_rad, p.mirror]));
  // Export -> import -> export is the round trip that has to hold exactly.
  const again = await designFromBodyFile("Body artwork", reopened, atlas);
  assert.equal(again.content_sha256, design.content_sha256);
  assert.deepEqual(again.artworks, design.artworks);
});

test("a body design authored elsewhere is refused, not approximated onto this body", async () => {
  const design = await designFromBodyFile("Body artwork", file, atlas);
  const foreign = structuredClone(design);
  (foreign.placements[0].placement.target as Record<string, unknown>).body_identity_sha256 = "0".repeat(64);
  await assert.rejects(bodyFileFromDesign(foreign, atlas), /different body/);

  const offAtlas = structuredClone(design);
  (offAtlas.placements[0].placement.target as unknown as { anchor: { face_index: number } }).anchor.face_index = 0;
  await assert.rejects(bodyFileFromDesign(offAtlas, atlas), /supported atlas|does not put on that face/);
});

test("opening a chart design in the body editor is refused by name", async () => {
  await assert.rejects(bodyFileFromDesign(await designFromChartDraft(await draft()), atlas), /surface editor/);
});

// ---- chart round trip -----------------------------------------------------
test("a chart design reopens with the same metric placement and identity", async () => {
  for (const overrides of [{}, { kind: "cylinder" as const, radius: 60, height: 150 }]) {
    const original = await draft(overrides);
    const design = await designFromChartDraft(original);
    const reopened = chartDraftFromDesign(design);
    assert.deepEqual(reopened.items.map(item => [item.uv, item.size, item.rotation_rad, item.mirror]),
      original.items.map(item => [item.uv, item.size, item.rotation_rad, item.mirror]));
    assert.equal(reopened.kind, original.kind);
    assert.equal(reopened.margin, original.margin);
    if (overrides.kind === "cylinder") assert.equal(reopened.radius, original.radius);
    // Reopening and saving without an edit keeps the identity it arrived with.
    assert.equal((await designFromChartDraft(reopened)).content_sha256, design.content_sha256);
  }
});

test("an edit changes the design identity and only the artwork it touched", async () => {
  const original = await draft();
  const before = await designFromChartDraft(original);
  const moved = { ...original, items: original.items.map((item, index) =>
    index === 0 ? { ...item, uv: [item.uv[0] + 5, item.uv[1]] as [number, number] } : item) };
  const after = await designFromChartDraft(moved);
  assert.notEqual(after.content_sha256, before.content_sha256);
  // The artwork itself is immutable; only the placement moved.
  assert.deepEqual(after.artworks, before.artworks);
  assert.equal(after.placements[1].placement.content_sha256, before.placements[1].placement.content_sha256);
  assert.notEqual(after.placements[0].placement.content_sha256, before.placements[0].placement.content_sha256);
});

test("placement order is the drawing order and survives a round trip", async () => {
  const original = await draft();
  const swapped = { ...original, items: [original.items[1], original.items[0]] };
  const design = await designFromChartDraft(swapped);
  assert.deepEqual(design.placements.map(item => item.id), ["b", "a"]);
  assert.deepEqual(chartDraftFromDesign(design).items.map(item => item.id), ["b", "a"]);
});

test("a chart design spread over different charts is refused", async () => {
  const design = await designFromChartDraft(await draft());
  const split = structuredClone(design);
  (split.placements[1].placement.target as { canvas_m: [number, number] }).canvas_m = [0.5, 0.5];
  assert.throws(() => chartDraftFromDesign(split), /one shared plane or cylinder/);
});

test("the headless design CLI's cylinder output opens here unchanged", async () => {
  // The Python and browser paths are the same contract or they are not shared.
  const design = await parseDesign(cylinderFixture);
  const reopened = chartDraftFromDesign(design);
  assert.equal(reopened.kind, "cylinder");
  assert.equal((await designFromChartDraft(reopened)).content_sha256, design.content_sha256);
});
