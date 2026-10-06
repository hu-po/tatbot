import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020.js";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { validatePlacementFile, type PlacementFile } from "../src/core/schema.ts";
import { validateArtworkRecord, type ArtworkSource } from "../src/core/artwork-record.ts";
import { acquiredArtwork } from "./acquired-artwork.ts";
import { fixtureArtwork } from "./artwork-fixture.ts";
import { makeProject, parseProject } from "../src/core/project.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const read = (name: string) => JSON.parse(readFileSync(new URL(`../../../config/${name}`, import.meta.url), "utf8"));
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const name of ["inkmap/artwork.schema.json", "human-representation/tattoo-program.schema.json", "human-representation/common.schema.json"]) ajv.addSchema(read(name));
const jsonSchema = ajv.compile(read("inkmap/placement.schema.json"));
const SVG = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><rect x='1' y='1' width='8' height='8' fill='#111'/></svg>";
const REVISION = "b".repeat(40);
const full: ArtworkSource = { kind: "generated", identifier: null, license: null, attribution: null,
  generation: { model: "Tongyi-MAI/Z-Image-Turbo", prompt: "tattoo flash design of a swallow", seed: 0,
    model_revision: REVISION, tracing: "inkmap-vtracer-otsu-v2", request_sha256: "1".repeat(64), png_sha256: "2".repeat(64),
    settings: { model: "Tongyi-MAI/Z-Image-Turbo", model_revision: REVISION, width: 768, height: 768, steps: 8, guidance: 0 } } };
const unknown: ArtworkSource = { ...full, generation: { prompt: "p", model: "inkgen", model_revision: null, seed: 7, tracing: null } };
async function file(source: ArtworkSource): Promise<PlacementFile> {
  const file = read("inkmap/examples/forearm-placement-v6.json") as PlacementFile;
  file.designs!["line-v1"] = await fixtureArtwork(SVG, "swallow", [50, 50], source);
  return file;
}

test("generated provenance validates in-app and by JSON Schema with explicit unknowns", async () => {
  for (const source of [full, unknown]) {
    const f = await file(source);
    assert.doesNotThrow(() => validatePlacementFile(f));
    assert.equal(jsonSchema(f), true, ajv.errorsText(jsonSchema.errors));
    assert.deepEqual(f.designs!["line-v1"].source, source);
    await validateArtworkRecord(f.designs!["line-v1"]);
  }
});

test("malformed generation provenance fails both structural validators", async () => {
  const initial = await file(full);
  for (const mutate of [
    (source: any) => { source.generation.tracing = ""; },
    (source: any) => { source.generation.seed = -1; },
    (source: any) => { source.health_snapshot = { model: "later" }; },
  ]) {
    const f = structuredClone(initial); mutate(f.designs!["line-v1"].source);
    assert.throws(() => validatePlacementFile(f));
    assert.equal(jsonSchema(f), false);
  }
});

test("an acquired project retains the exact native record", async () => {
  const editor = { pose_id: "supine", skin_tone: "#804030", show_atlas: false, camera: null };
  const acquired = read("inkmap/examples/forearm-placement-v6.json") as PlacementFile;
  acquired.designs!["line-v1"] = await acquiredArtwork();
  const project = await makeProject({ name: "DBV3", placement_file: acquired, editor, history: { past: [], future: [] }, edit_before: null, selected_id: null });
  assert.deepEqual((await parseProject(JSON.stringify(project))).placement_file.designs, project.placement_file.designs);
  await assert.rejects(makeProject({ name: "Legacy", placement_file: await file(full), editor, history: { past: [], future: [] }, edit_before: null, selected_id: null }), /Generate with DrawingBot V3/);
});
