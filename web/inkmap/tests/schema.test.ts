import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_ASSET_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "../src/core/body.ts";
import { validatePlacementFile, SCHEMA_VERSION } from "../src/core/schema.ts";

const schema = JSON.parse(readFileSync(new URL("../../../config/inkmap/placement.schema.json", import.meta.url), "utf8"));

const good = {
  schema_version: 6,
  units: { length: "m", tattoo_size: "mm", up: "+z" },
  body: {
    model_spec_id: MODEL_SPEC_ID,
    model_spec_sha256: MODEL_SPEC_SHA256,
    identity_sha256: REFERENCE_IDENTITY_SHA256,
    topology_sha256: TOPOLOGY_SHA256,
    rest_surface_sha256: REST_SURFACE_SHA256,
    asset_path: BODY_SPEC.path,
    asset_sha256: REST_ASSET_SHA256,
  },
  placements: [
    { id: "p-1", design_id: "anchor", anchor: { face: 12, barycentric: [0.2, 0.5, 0.3] }, rotation_rad: 0.1, size_mm: [50, 60], mirror: false },
  ],
};

test("the in-app validator and JSON Schema agree on placement v6", () => {
  assert.equal(schema.properties.schema_version.const, SCHEMA_VERSION);
  assert.equal(schema.properties.body.properties.model_spec_id.const, MODEL_SPEC_ID);
});

test("a well-formed v6 file validates", () => {
  assert.doesNotThrow(() => validatePlacementFile(structuredClone(good)));
});

test("old schemas, old models, and malformed geometry fail closed", () => {
  const bad = (mutate: (file: any) => void, pattern: RegExp) => {
    const file = structuredClone(good) as any;
    mutate(file);
    assert.throws(() => validatePlacementFile(file), pattern);
  };
  bad((file) => { file.schema_version = 5; }, /unsupported schema\/model/);
  bad((file) => { file.body.model_spec_id = "legacy-body"; }, /unsupported schema\/model/);
  bad((file) => { file.body.identity_sha256 = "0".repeat(64); }, /unsupported schema\/model/);
  bad((file) => { file.body.rest_surface_sha256 = "0".repeat(64); }, /unsupported schema\/model/);
  bad((file) => { delete file.body.topology_sha256; }, /body binding fields/);
  bad((file) => { file.body.asset_sha256 = "nope"; }, /unsupported schema\/model/);
  bad((file) => { file.placements[0].anchor.barycentric = [0.5, 0.5, 0.5]; }, /sum to 1/);
  bad((file) => { file.placements[0].anchor.face = -1; }, /face/);
  bad((file) => { file.placements[0].size_mm = [0, 10]; }, /size_mm/);
  bad((file) => { delete file.placements[0].mirror; }, /mirror/);
});

test("embedded designs contain the shared frozen artwork", () => {
  const file = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8"));
  assert.doesNotThrow(() => validatePlacementFile(file));
  file.designs["line-v1"].svg = "<svg/>";
  assert.throws(() => validatePlacementFile(file), /missing or unknown fields/);
});

test("every top-level and placement field required by JSON Schema is emitted", () => {
  const required = (value: any) => value.required as string[];
  assert.deepEqual(required(schema).sort(), Object.keys(good).sort());
  assert.deepEqual(required(schema.properties.placements.items).sort(), Object.keys(good.placements[0]).sort());
});
