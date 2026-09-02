import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SCENARIO_SCHEMA_VERSION, validateTattooScenario } from "../src/core/scenario.ts";

const fixtureUrl = new URL("../../../config/inkmap/examples/forearm-scenario-v1.json", import.meta.url);
const schemaUrl = new URL("../../../config/inkmap/tattoo-scenario.schema.json", import.meta.url);
const fixture = JSON.parse(readFileSync(fixtureUrl, "utf8"));
const schema = JSON.parse(readFileSync(schemaUrl, "utf8"));

test("the shared posed scenario fixture validates", () => {
  assert.equal(schema.properties.schema_version.const, SCENARIO_SCHEMA_VERSION);
  assert.doesNotThrow(() => validateTattooScenario(structuredClone(fixture)));
});

test("scenario geometry and transforms fail closed", () => {
  const brokenDigest = structuredClone(fixture);
  brokenDigest.body.surface_sha256 = "not-a-digest";
  assert.throws(() => validateTattooScenario(brokenDigest), /surface_sha256/);

  const brokenMatrix = structuredClone(fixture);
  brokenMatrix.pose.world_from_body = [[1, 0], [0, 1]];
  assert.throws(() => validateTattooScenario(brokenMatrix), /4x4/);

  const brokenAnchor = structuredClone(fixture);
  brokenAnchor.trace.strokes[0][0].barycentric = [0.5, 0.5, 0.5];
  assert.throws(() => validateTattooScenario(brokenAnchor), /normalized/);
});
