import { renderTattooProgramSvg } from "../src/core/human-representation/program-svg.ts";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import type { AtlasData } from "../src/core/atlas.ts";
import type { PlacementFile } from "../src/core/schema.ts";
import { canonicalDigest, canonicalDocumentDigest } from "../src/core/human-representation/schema.ts";
import { makeSimulationBundle, parseSimulationBundle, simulationRequest, validateCompiledScenario, validateSimulationBundle } from "../src/core/sim-bundle.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const atlas = JSON.parse(readFileSync(new URL("../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8")) as AtlasData;
const file = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8")) as PlacementFile;
const request = simulationRequest("supine", "#c07f57", null, "lutin-3rl-bugpin", 42);

test("simulation bundle binds portable art, typed surface, pose, seed and pinned assets", async () => {
  const bundle = await makeSimulationBundle(file, atlas, request);
  assert.deepEqual(await parseSimulationBundle(JSON.stringify(bundle), atlas), bundle);
  assert.deepEqual(await makeSimulationBundle(file, atlas, request), bundle);
  assert.deepEqual(bundle.artworks["line-v1"], file.designs!["line-v1"]);
  assert.equal(bundle.surface_placements[0].placement.review && (bundle.surface_placements[0].placement.review as {status:string}).status, "pending");
  assert.deepEqual(bundle.assets.map(asset => asset.key), ["body-rest", "body-poses", "body-exclusions"]);
});

test("rehashed tampering cannot reinterpret artwork, placements, or local asset keys", async () => {
  const original = await makeSimulationBundle(file, atlas, request);
  const mutations = [
    (b: typeof original) => { b.artworks["line-v1"].name += " "; },
    (b: typeof original) => { b.surface_placements[0].placement.rotation_rad = 1; },
    (b: typeof original) => { b.assets[0].sha256 = "0".repeat(64); },
    (b: typeof original) => { Object.assign(b.assets[0], { url: "https://invalid.example/asset" }); },
    (b: typeof original) => { b.request.support_id = "invented-support"; },
    (b: typeof original) => { b.atlas_sha256 = "0".repeat(64); },
    (b: typeof original) => { b.placement_file.body.identity_sha256 = "0".repeat(64); },
  ];
  for (const mutate of mutations) {
    const value = structuredClone(original); mutate(value); value.content_sha256 = await canonicalDigest(value);
    await assert.rejects(validateSimulationBundle(value, atlas));
  }
  await assert.rejects(parseSimulationBundle('{"schema":1,"schema":2}', atlas), /duplicate/);
});

test("all placements and their order survive; draft/history fields are not admitted", async () => {
  const multiple = structuredClone(file);
  multiple.placements.push({ ...structuredClone(multiple.placements[0]), id: "second", rotation_rad: .2 });
  const bundle = await makeSimulationBundle(multiple, atlas, request);
  assert.deepEqual(bundle.surface_placements.map(p => p.id), [multiple.placements[0].id, "second"]);
  bundle.surface_placements.reverse(); bundle.content_sha256 = await canonicalDigest(bundle);
  await assert.rejects(validateSimulationBundle(bundle, atlas), /order differs/);
  const draft = { ...await makeSimulationBundle(file, atlas, request), edit_before: [] };
  draft.content_sha256 = await canonicalDigest(draft);
  await assert.rejects(validateSimulationBundle(draft, atlas), /additional properties/);
});

test("compiled v3 preview revalidates its bundle, typed program and derived trace", async () => {
  const bundle = await makeSimulationBundle(file, atlas, request);
  const inkProgram = JSON.parse(readFileSync(
    new URL("../../../config/human-representation/examples/ink-program.json", import.meta.url), "utf8",
  ));
  inkProgram.tattoo_program_sha256 = bundle.artworks["line-v1"].program.content_sha256;
  inkProgram.surface_placement_sha256 = bundle.surface_placements[0].placement.content_sha256;
  inkProgram.content_sha256 = await canonicalDigest(inkProgram);
  const strokes = inkProgram.events.filter((event: { kind: string }) => event.kind === "stroke").map(
    (event: { curve: { coordinates: { face_index: number; barycentric: number[] }[] } }) =>
      event.curve.coordinates.map(point => ({ face: point.face_index, barycentric: point.barycentric })),
  );
  const template = JSON.parse(readFileSync(new URL("../public/showcase/supine-shin.scenario.json", import.meta.url), "utf8"));
  const placement = bundle.placement_file.placements[0];
  const artwork = bundle.artworks[placement.design_id];
  const scenario = {
    ...template,
    schema_version: 3,
    seed: request.seed,
    placement: { ...placement, source_sha256: await canonicalDigest(bundle.placement_file) },
    design: { id: placement.design_id, name: artwork.name, svg: renderTattooProgramSvg(artwork.program),
      sha256: artwork.source_sha256, source: bundle.placement_file.designs![placement.design_id].source },
    trace: { compiler: "tatbot_sim.surface_trace", compiler_version: 3,
      sha256: await canonicalDocumentDigest(strokes), strokes },
    robot: { ...template.robot, tool_id: request.tool_id },
    support: { id: request.support_id, world_from_nominal: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
    program_binding: { bundle, placement_id: placement.id, ink_program: inkProgram,
      tool_profile_sha256: "a".repeat(64) },
  };
  assert.deepEqual(await validateCompiledScenario(scenario, atlas), scenario);
  const badSupport = structuredClone(scenario);
  badSupport.support.world_from_nominal[1][3] = .01;
  await assert.rejects(validateCompiledScenario(badSupport, atlas), /support differs/);
  const badTrace = structuredClone(scenario);
  badTrace.trace.strokes[0][0].face += 1;
  badTrace.trace.sha256 = await canonicalDocumentDigest(badTrace.trace.strokes);
  await assert.rejects(validateCompiledScenario(badTrace, atlas), /trace differs/);
  const badProgram = structuredClone(scenario);
  badProgram.program_binding.ink_program.tattoo_program_sha256 = "0".repeat(64);
  badProgram.program_binding.ink_program.content_sha256 = await canonicalDigest(badProgram.program_binding.ink_program);
  await assert.rejects(validateCompiledScenario(badProgram, atlas), /different artwork/);
});
