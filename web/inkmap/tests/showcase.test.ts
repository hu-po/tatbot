import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateTattooScenario, type TattooScenario } from "../src/core/scenario.ts";

const root = new URL("../public/showcase/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
const rigConfig = JSON.parse(readFileSync(new URL("../../../config/inkmap/body-rig.json", import.meta.url), "utf8"));

test("the showcase spans both bodies and every tattoo-session pose with valid scenarios", () => {
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.validation.accepted, 64);
  assert.ok(manifest.validation.attempts >= manifest.validation.accepted);
  assert.ok(manifest.validation.rejection_rate < 0.25);
  assert.equal(manifest.validation.reach_audited, true);
  assert.ok(manifest.validation.pose_quality.max_joint_rotation_deg <= 120);
  assert.ok(manifest.validation.pose_quality.edge_length_ratio[0] >= rigConfig.quality_gates.edge_length_ratio_p001_min);
  assert.ok(manifest.validation.pose_quality.edge_length_ratio[1] <= rigConfig.quality_gates.edge_length_ratio_p99_max);
  assert.ok(manifest.validation.pose_quality.triangle_area_ratio[0] >= rigConfig.quality_gates.triangle_area_ratio_p01_min);
  assert.ok(manifest.validation.pose_quality.triangle_area_ratio[1] <= rigConfig.quality_gates.triangle_area_ratio_p99_max);
  assert.ok(manifest.validation.anatomy.knee_angle_deg[0] >= 145);
  assert.ok(manifest.validation.anatomy.knee_angle_deg[1] <= 170);
  assert.ok(manifest.validation.anatomy.knee_bend_offset_m[0] >= 0.045);
  assert.ok(manifest.validation.anatomy.max_knee_off_axis_m <= 0.01);
  assert.ok(manifest.validation.anatomy.supported_elbow_angle_deg[0] >= 110);
  assert.ok(manifest.validation.anatomy.supported_elbow_angle_deg[1] <= 145);
  assert.ok(manifest.validation.anatomy.supported_wrist_angle_deg[0] >= 170);
  assert.equal(manifest.slides.length, 5);

  const bodies = new Set<string>();
  const poses = new Set<string>();
  for (const slide of manifest.slides) {
    const scenario = JSON.parse(readFileSync(new URL(slide.scenario, root), "utf8")) as TattooScenario;
    assert.doesNotThrow(() => validateTattooScenario(scenario), slide.scenario);
    assert.ok(scenario.trace.strokes.flat().length > 20, `${slide.scenario}: trace is visible`);
    assert.ok(slide.probe_max_residual_mm <= 1, `${slide.scenario}: CPU reach gate`);
    assert.match(scenario.support.id, /^tattoo-(bed|chair)-/, `${slide.scenario}: session support`);
    bodies.add(scenario.body.id);
    poses.add(scenario.pose.id);
  }
  assert.equal(bodies.size, manifest.coverage.bodies);
  assert.equal(poses.size, manifest.coverage.poses);
  assert.deepEqual([...poses].sort(), [
    "prone",
    "reclined-left-arm-supported",
    "reclined-right-arm-supported",
    "reclined-seated",
    "supine",
  ]);
});
