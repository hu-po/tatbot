import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validatePlacementFile, SCHEMA_VERSION } from "../src/core/schema.ts";

const schema = JSON.parse(readFileSync(new URL("../../../config/inkmap/placement.schema.json", import.meta.url), "utf8"));

const good = {
  schema_version: 3,
  units: { length: "m", tattoo_size: "mm", up: "+z" },
  body: { id: "kaykit-barbarian", sha256: "a".repeat(64), path: "bodies/kaykit-barbarian.glb" },
  placements: [
    { id: "p-1", design_id: "anchor", anchor: { face: 12, barycentric: [0.2, 0.5, 0.3] }, rotation_rad: 0.1, size_mm: [50, 60], mirror: false },
  ],
};

test("the in-app validator and the JSON Schema agree on the version", () => {
  assert.equal(schema.properties.schema_version.const, SCHEMA_VERSION);
});

test("a well-formed file validates", () => {
  assert.doesNotThrow(() => validatePlacementFile(structuredClone(good)));
});

test("bad files are rejected with a reason", () => {
  const bad = (mut: (f: any) => void, re: RegExp) => {
    const f = structuredClone(good) as any;
    mut(f);
    assert.throws(() => validatePlacementFile(f), re);
  };
  bad((f) => (f.schema_version = 4), /schema_version/);
  bad((f) => (f.placements[0].site = { id: "", laterality: "left", aspect: null, lexicon: "0.1" }), /site\.id/);
  bad((f) => (f.placements[0].site = { id: "knee_ditch", laterality: "both", aspect: null, lexicon: "0.1" }), /laterality/);
  bad((f) => (f.placements[0].language = { sentence: "", program: {} }), /sentence/);
  bad((f) => (f.designs = []), /designs/);
  bad((f) => (f.designs = { "gen-1": { name: "x", svg: "nope", default_size_mm: [10, 10] } }), /svg/);
  bad((f) => { f.placements[0].design_id = "gen-9"; f.designs = {}; }, /not embedded/);
  bad((f) => (f.units.up = "+y"), /units/);
  bad((f) => (f.body.sha256 = "nope"), /sha256/);
  bad((f) => (f.placements[0].anchor.barycentric = [0.5, 0.5, 0.5]), /sum to 1/);
  bad((f) => (f.placements[0].anchor.face = -1), /face/);
  bad((f) => (f.placements[0].size_mm = [0, 10]), /size_mm/);
  bad((f) => delete f.placements[0].mirror, /mirror/);
});

test("a v1 file (no designs) still loads; a v2 file with an embedded design validates", () => {
  const v1 = structuredClone(good) as any; v1.schema_version = 1;
  assert.doesNotThrow(() => validatePlacementFile(v1));
  const v2 = structuredClone(good) as any;
  v2.placements[0].design_id = "gen-1";
  v2.designs = { "gen-1": { name: "swallow", svg: "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M0 0h10v10z'/></svg>", default_size_mm: [60, 50], source: { kind: "generated", model: "sd-turbo", prompt: "p", seed: 7 } } };
  assert.doesNotThrow(() => validatePlacementFile(v2));
});

test("every field the JSON Schema requires is one the app writes", () => {
  const req = (o: any) => o.required as string[];
  assert.deepEqual(req(schema).sort(), Object.keys(good).sort());
  assert.deepEqual(req(schema.properties.placements.items).sort(), Object.keys(good.placements[0]).sort());
});
