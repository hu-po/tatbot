import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { acquiredArtwork } from "./acquired-artwork.ts";
import { bodyFileFromDesign, designFromBodyFile, embeddedFromArtwork } from "../src/core/body-design.ts";
import { chartDocumentFromDraft, chartDraftFromDocument, chartDraftFromDesign, designFromChartDraft, EMPTY_DRAFT, newChartItem } from "../src/core/chart-draft.ts";
import { artworkSources, freezeDesign } from "../src/core/design-assets.ts";
import { artworkPreviewUrl } from "../src/core/svg.ts";
import { frozenArtwork } from "../src/core/frozen-artwork.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import { makeProject, parseProject, validateProject } from "../src/core/project.ts";
import { makeSimulationBundle, simulationRequest, validateSimulationBundle } from "../src/core/sim-bundle.ts";
import type { AtlasData } from "../src/core/atlas.ts";
import type { PlacementFile } from "../src/core/schema.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
const atlas = read("../public/bodies/mhr-soma-v1.regions.json") as AtlasData;
const fixture = read("../../../config/inkmap/examples/forearm-placement-v6.json") as PlacementFile;
const artwork = () => acquiredArtwork();
const project = (file: PlacementFile, charts = {}) => makeProject({ name: "Converted artwork", placement_file: file,
  editor: { pose_id: "supine", skin_tone: "#804030", show_atlas: false, camera: null },
  history: { past: [], future: [] }, edit_before: null, selected_id: null, ...charts });

async function convertedFile() {
  const file = structuredClone(fixture);
  file.designs!["line-v1"] = embeddedFromArtwork(await artwork());
  return file;
}

test("body, paper and parked cylinder preserve exact artwork through project JSON and typed export", async () => {
  const art = await artwork();
  const body = await designFromBodyFile("Converted artwork", await convertedFile(), atlas);
  const draft = { ...EMPTY_DRAFT, name: "Converted artwork", items: [newChartItem("item", "line-v1", art, [30, 30])] };
  const paper = await designFromChartDraft(draft);
  const cylinder = await designFromChartDraft({ ...draft, kind: "cylinder", radius: 60 });
  const saved = await project(await bodyFileFromDesign(body, atlas), {
    chart: chartDocumentFromDraft(chartDraftFromDesign(paper)),
    chart_parked: chartDocumentFromDraft(chartDraftFromDesign(cylinder)),
  });
  const reopened = await parseProject(JSON.stringify(saved));
  assert.deepEqual(await designFromBodyFile(body.name, reopened.placement_file, atlas), body);
  for (const [doc, original] of [[reopened.chart!, paper], [reopened.chart_parked!, cylinder]] as const) {
    assert.deepEqual(await designFromChartDraft(await chartDraftFromDocument(doc)), original);
  }
  const bundle = await makeSimulationBundle(reopened.placement_file, atlas, simulationRequest("supine", "#804030", null, "lutin-3rl-bugpin", 42));
  assert.deepEqual(bundle.artworks["line-v1"], art);
  const elements = art.program.layers.flatMap(layer => layer.elements);
  assert.ok(elements.length > 1);
  assert.ok(elements.every(element => element.kind === "path" && element.fill === false));
  const changed = structuredClone(bundle);
  changed.placement_file.designs!["line-v1"].program.layers[0].elements[0].width_m = .0007;
  changed.content_sha256 = await canonicalDigest(changed);
  await assert.rejects(validateSimulationBundle(changed, atlas), /frozen placement record/);
});

test("freezing and runtime collection provenance cannot overwrite a portable artwork record", async () => {
  const art = await artwork();
  const embedded = embeddedFromArtwork(art);
  assert.equal(embedded.source.generation, null);
  const design = await freezeDesign({ id: "line-v1", name: "Stale library name", default_size_mm: [50, 50],
    path: "designs/line-v1.svg", embedded, sha256: art.source_sha256, sourceSha256: art.source_sha256,
    source: { kind: "stock", identifier: "stale", license: null, attribution: null, generation: null } });
  assert.deepEqual(design.embedded, embedded);
  assert.equal(design.name, art.name);
  assert.deepEqual(design.default_size_mm, [30, 30]);
  assert.deepEqual(await frozenArtwork(design.id, design.embedded!, artworkSources([design])[design.id]), art);
});

test("rehashed projects cannot silently reinterpret frozen artwork on any saved surface", async () => {
  const art = await artwork();
  const doc = chartDocumentFromDraft({ ...EMPTY_DRAFT, items: [newChartItem("item", "line-v1", art, [30, 30])] });
  const saved = await project(await convertedFile(), { chart: doc, chart_parked: { ...doc, kind: "cylinder" } });
  for (const place of ["body", "chart", "parked"]) {
    for (const mutate of [
      (e: any) => { e.conversion.adapter = "changed"; },
      (e: any) => { e.program.layers[0].elements[0].width_m = .0007; },
      (e: any) => { e.source.attribution = "Changed"; },
      (e: any) => { e.content_sha256 = "0".repeat(64); },
      (e: any) => { e.program.canvas_m.width = .04; },
      (e: any) => { e.name = "Changed"; },
      (e: any) => { e.source_sha256 = "f".repeat(64); },
    ]) {
      const changed = structuredClone(saved);
      const embedded = place === "body" ? changed.placement_file.designs!["line-v1"]
        : (place === "chart" ? changed.chart! : changed.chart_parked!).artwork["line-v1"];
      mutate(embedded); changed.content_sha256 = await canonicalDigest(changed);
      await assert.rejects(validateProject(changed), /wrong_hash|record_invalid/);
    }
  }
  const inconsistent = structuredClone(saved);
  inconsistent.chart!.artwork["line-v1"].source.attribution = "Other source";
  inconsistent.content_sha256 = await canonicalDigest(inconsistent);
  await assert.rejects(validateProject(inconsistent), /wrong_hash/);
});

test("placement embeds the shared record directly with no derivation wrapper", async () => {
  const record = await artwork();
  assert.deepEqual(embeddedFromArtwork(record), record);
  const ref = read("../../../config/inkmap/placement.schema.json").properties.designs.additionalProperties;
  assert.equal(ref.$ref, "https://tatbot.dev/schemas/inkmap/artwork.schema.json");
});


test("placed previews scale the same centerlines on body and paper without changing pen width or frozen identity", async () => {
  const art = await artwork(), before = structuredClone(art);
  const design = { id: "line-v1", name: art.name, path: "unused", default_size_mm: [30, 30] as [number, number],
    embedded: embeddedFromArtwork(art) };
  for (const scale of [.4, 1, 2]) {
    const size = [30 * scale, 30 * scale] as [number, number];
    const chart = await artworkPreviewUrl(art, size), body = await artworkPreviewUrl(design, size);
    assert.equal(body, chart);
    const doc = new SvgDOMParser().parseFromString(decodeURIComponent(chart.split(",").slice(1).join(",")), "image/svg+xml");
    assert.equal(doc.documentElement!.getAttribute("width"), `${size[0]}mm`);
    assert.equal(doc.documentElement!.getAttribute("height"), `${size[1]}mm`);
    const line = doc.getElementsByTagName("polyline")[0];
    assert.equal(Number(line.getAttribute("stroke-width")), art.program.layers[0].elements[0].width_m);
    const points = line.getAttribute("points")!.trim().split(/[ ,]+/).map(Number);
    const expected = art.program.layers[0].elements[0].points_m!.flatMap(([x, y]) => [x * scale, (.03 - y) * scale]);
    points.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12));
  }
  assert.deepEqual(art, before);
  assert.deepEqual(await frozenArtwork(design.id, design.embedded), art);
});
