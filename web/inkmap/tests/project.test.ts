import { acquiredArtwork } from "./acquired-artwork.ts";
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { makeProject, validateProject, parseProject } from "../src/core/project.ts";
import { chartDocumentFromDraft, chartDraftFromDocument, type ChartDocument } from "../src/core/chart-draft.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import type { PlacementFile } from "../src/core/schema.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const CHART: ChartDocument = {
  name: "Paper drawing", kind: "plane", width_mm: 100, height_mm: 150, radius_mm: 40, margin_mm: 5,
  selected_id: "item-1", artwork: { square: await acquiredArtwork() },
  items: [{ id: "item-1", artwork_id: "square", size_mm: [20, 20], uv_mm: [3, 4], rotation_rad: 0.2617993877991494, mirror: false,
    review: { status: "pending", reviewer: "unreviewed Inkmap export", evidence_sha256: "0".repeat(64) },
    provenance: { producer: "tatbot-inkmap", version: "1", created_utc: "1970-01-01T00:00:00Z", source_sha256: "a".repeat(64) } }],
};

const fixture = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8")) as PlacementFile;
const make = () => makeProject({ name: "Fixture", placement_file: fixture,
  editor: { pose_id: "supine", skin_tone: "#804030", show_atlas: true, camera: { position: [1, -2, 1], target: [0, 0, 1], fov: 38 } },
  history: { past: [[]], future: [] }, edit_before: null, selected_id: null });

test("strict project round-trip retains body, artwork, history, and editor settings", async () => {
  const p = await make();
  assert.deepEqual(await parseProject(JSON.stringify(p)), p);
  const changed = structuredClone(p); changed.editor.skin_tone = "#ffffff";
  changed.content_sha256 = await canonicalDigest(changed);
  assert.notEqual(changed.content_sha256, p.content_sha256);
  assert.deepEqual((await validateProject(changed)).placement_file, p.placement_file);
});

test("rehashed malformed histories, missing art, bad bodies, versions, and drafts fail closed", async () => {
  const p = await make();
  for (const mutate of [
    (r: any) => { r.unknown = true; },
    (r: any) => { r.schema = "tatbot.inkmap-project/99"; },
    (r: any) => { delete r.chart; },
    (r: any) => { r.chart = { ...CHART, items: [{ ...CHART.items[0], artwork_id: "missing" }] }; },
    (r: any) => { r.chart = { ...CHART, selected_id: "nobody" }; },
    (r: any) => { r.chart = { ...CHART, items: [CHART.items[0], CHART.items[0]] }; },
    (r: any) => { r.editor.pose_id = "unknown"; },
    (r: any) => { r.placement_file.body.identity_sha256 = "0".repeat(64); },
    (r: any) => { r.history.past = [Array(2).fill(fixture.placements[0])]; },
    (r: any) => { r.history.past = [[{ ...fixture.placements[0], design_id: "missing" }]]; },
    (r: any) => { r.selected_id = fixture.placements[0].id; },
    (r: any) => { r.placement_file.placements[0].size_mm[0] = Infinity; },
  ]) {
    const broken = structuredClone(p); mutate(broken);
    await assert.rejects(async () => { broken.content_sha256 = await canonicalDigest(broken); await validateProject(broken); });
  }
  await assert.rejects(parseProject('{"schema":1,"schema":2}'), /duplicate_key/);
});

test("obsolete project contracts require explicit migration", async () => {
  const current = await make();
  for (const schema of ["tatbot.inkmap-project/1", "tatbot.inkmap-project/2"]) {
    const old = { ...current, schema }; old.content_sha256 = await canonicalDigest(old);
    await assert.rejects(validateProject(old), /project_invalid/);
  }
});

test("a current project carries the surface draft through a save and a reload", async () => {
  const saved = await makeProject({ name: "Fixture", placement_file: fixture,
    editor: { pose_id: "supine", skin_tone: "#804030", show_atlas: true, camera: null },
    history: { past: [], future: [] }, edit_before: null, selected_id: null, chart: CHART });
  const reread = await parseProject(JSON.stringify(saved));
  assert.equal(reread.schema, "tatbot.inkmap-project/3");
  assert.deepEqual(reread.chart, CHART);
  // The frozen acquired geometry travels unchanged.
  const draft = await chartDraftFromDocument(reread.chart!);
  assert.deepEqual(draft.items[0].artwork, CHART.artwork.square);
  assert.deepEqual(draft.items[0].artwork.source, CHART.artwork.square.source);
  assert.equal(draft.items[0].uv[1], 4);
  const upgraded = chartDocumentFromDraft(draft);
  assert.equal(upgraded.artwork.square.content_sha256, draft.items[0].artwork.content_sha256);
  assert.deepEqual((await chartDraftFromDocument(upgraded)).items, draft.items);
});

test("an empty surface editor stores an explicit null, not a missing key", async () => {
  const saved = await make();
  assert.equal(saved.schema, "tatbot.inkmap-project/3");
  assert.equal(saved.chart, null);
});


test("chart and body projects reference the same artwork schema", () => {
  const schema = JSON.parse(readFileSync(new URL("../../../config/inkmap/project.schema.json", import.meta.url), "utf8"));
  assert.equal(schema.properties.chart.anyOf[1].properties.artwork.additionalProperties.$ref,
    "https://tatbot.dev/schemas/inkmap/artwork.schema.json");
});
