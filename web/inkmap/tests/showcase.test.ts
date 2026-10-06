import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  MODEL_SPEC_ID,
  REST_ASSET_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "../src/core/body.ts";
import { POSE_CATALOG } from "../src/core/pose.ts";
import { validateTattooScenario, type TattooScenario } from "../src/core/scenario.ts";

const root = new URL("../public/showcase/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));

test("the showcase uses one SOMA body and all five tattoo-session poses", () => {
  assert.equal(manifest.schema_version, 2);
  assert.equal(manifest.validation.scenarios, 5);
  assert.equal(manifest.validation.trace_compiler_version, 3);
  assert.equal(manifest.validation.body_asset_sha256, REST_ASSET_SHA256);
  assert.equal(manifest.validation.pose_asset_sha256, POSE_CATALOG.pose_asset.sha256);
  assert.equal(manifest.validation.reach_audited, false);
  assert.equal(manifest.validation.visual_review, "pending");
  assert.deepEqual(manifest.coverage, { bodies: 1, poses: 5, sites: 4, designs: 3 });

  const bodies = new Set<string>();
  const poses = new Set<string>();
  for (const slide of manifest.slides) {
    const scenario = JSON.parse(readFileSync(new URL(slide.scenario, root), "utf8")) as TattooScenario;
    assert.doesNotThrow(() => validateTattooScenario(scenario), slide.scenario);
    assert.equal(scenario.schema_version, 3);
    assert.equal(scenario.design.id, slide.artwork_id);
    assert.equal(scenario.design.sha256, slide.artwork_sha256);
    assert.ok(scenario.trace.strokes.flat().length > 20, `${slide.scenario}: trace is visible`);
    assert.equal(scenario.body.topology_sha256, TOPOLOGY_SHA256);
    assert.equal(scenario.body.rest_surface_sha256, REST_SURFACE_SHA256);
    assert.equal(scenario.body.asset_sha256, REST_ASSET_SHA256);
    bodies.add(scenario.body.model_spec_id);
    poses.add(scenario.pose.id);
  }
  assert.deepEqual([...bodies], [MODEL_SPEC_ID]);
  assert.deepEqual([...poses].sort(), [
    "prone",
    "reclined-left-arm-supported",
    "reclined-right-arm-supported",
    "reclined-seated",
    "supine",
  ]);
});
