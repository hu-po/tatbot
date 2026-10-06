import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import Ajv2020 from "ajv/dist/2020.js";
import { canonicalDigest, validateContract } from "../src/core/human-representation/schema.ts";
import { analyticFrame, makeSurfacePlacement, upgradeBodyPlacement, type SurfacePlacement } from "../src/core/surface-placement.ts";

const root = new URL("../../../config/human-representation/", import.meta.url);
const read = (path: string) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const ajv = new Ajv2020({ strict: false });
ajv.addSchema(read("common.schema.json"));
const schema = ajv.compile(read("surface-placement-v2.schema.json"));
ajv.addSchema(read("surface-curve-v2.schema.json"));
const inkSchema = ajv.compile(read("ink-program-v2.schema.json"));

test("plane, cylinder and body placements match Python digests and the published schema", async () => {
  for (const name of ["plane-placement", "cylinder-placement", "body-placement-v2"]) {
    const original = read(`examples/${name}.json`);
    assert.equal(schema(original), true, ajv.errorsText(schema.errors));
    assert.deepEqual(await validateContract(original, { expectedSchema: "tatbot.surface-placement/2" }), original);
    const { schema: _schema, content_sha256: _sha, ...input } = original;
    assert.deepEqual(await makeSurfacePlacement(input), original);
  }
  const old = read("examples/surface-placement.json");
  const snapshot = structuredClone(old);
  assert.deepEqual(await upgradeBodyPlacement(old), read("examples/body-placement-v2.json"));
  assert.deepEqual(old, snapshot);
});

test("placement refuses overflow after rotation, wrapping, wrong geometry fields and stale hashes", async () => {
  const mutations: ((v: any) => void)[] = [
    v => { v.target.anchor_uv_m = [0.1, 0]; },
    v => { v.target.canvas_m = [0.085, 0.055]; v.rotation_rad = Math.PI / 4; },
    v => { v.target.margin_m = -0.001; },
    v => { v.target.kind = "cylinder"; v.target.radius_m = 0.001; },
    v => { v.target.radius_m = 0.04; },
    v => { v.target.kind = "body"; },
    v => { v.target.robot_pose = [0, 0, 0]; },
    v => { v.warp = { kind: "unknown", max_displacement_m: 0.001, parameters: [] }; },
  ];
  for (const mutate of mutations) {
    const value = read("examples/plane-placement.json");
    mutate(value);
    value.content_sha256 = await canonicalDigest(value);
    await assert.rejects(validateContract(value, { expectedSchema: "tatbot.surface-placement/2" }));
  }
  const value = read("examples/body-placement-v2.json");
  value.target.supported_domain.face_indices = [42];
  value.content_sha256 = await canonicalDigest(value);
  await assert.rejects(validateContract(value), /anchor_outside_domain/);
  value.target.supported_domain.face_indices = [1200, 1201];
  await assert.rejects(validateContract(value), /wrong_hash/);
});

test("analytic frames use a surface crest origin and metric circumference", () => {
  const plane = read("examples/plane-placement.json") as SurfacePlacement;
  const cylinder = read("examples/cylinder-placement.json") as SurfacePlacement;
  assert.ok(plane.target.kind === "plane" && cylinder.target.kind === "cylinder");
  assert.deepEqual(analyticFrame(plane.target, [0.01, 0.02]), { point: [0.01, 0.02, 0], normal: [0, 0, 1] });
  const r = cylinder.target.radius_m;
  const { point, normal } = analyticFrame(cylinder.target, [0.01, Math.PI * r / 2]);
  assert.ok(Math.abs(point[0] - 0.01) < 1e-12);
  assert.ok(Math.abs(point[1] - r) < 1e-12 && Math.abs(point[2] + r) < 1e-12);
  assert.ok(Math.abs(normal[1] - 1) < 1e-12 && Math.abs(normal[2]) < 1e-12);
});

test("Python chart InkProgram retains target bindings, metric length and supply intent", async () => {
  const original = read("examples/chart-ink-program.json");
  assert.equal(inkSchema(original), true, ajv.errorsText(inkSchema.errors));
  assert.deepEqual(await validateContract(original, { expectedSchema: "tatbot.ink-program/2" }), original);
  assert.ok(original.events.some((event: any) => event.kind === "stroke"));
  assert.ok(!original.events.some((event: any) => event.kind === "dip"));
  for (const kind of ["length", "target", "mixed", "old_schema"]) {
    const value = structuredClone(original);
    const curve = value.events.find((event: any) => event.kind === "stroke").curve;
    if (kind === "length") {
      curve.rest_surface_arc_length_m += 0.001;
      value.total_material_path_length_m += 0.001;
    } else if (kind === "target") {
      for (const point of curve.coordinates) point.target_sha256 = "a".repeat(64);
    } else if (kind === "mixed") {
      curve.coordinates[1] = { topology_sha256: "b".repeat(64), face_index: 1, barycentric: [1, 0, 0] };
    } else value.schema = "tatbot.ink-program/1";
    value.content_sha256 = await canonicalDigest(value);
    await assert.rejects(validateContract(value));
  }
  // A program compiled with a non-default paint planner names it; the field
  // is an enum in both readers, and absent means concentric.
  const hatched = structuredClone(original);
  hatched.fill_style = "hatch";
  hatched.content_sha256 = await canonicalDigest(hatched);
  assert.equal(inkSchema(hatched), true, ajv.errorsText(inkSchema.errors));
  assert.deepEqual(await validateContract(hatched, { expectedSchema: "tatbot.ink-program/2" }), hatched);
  const unknown = structuredClone(original);
  unknown.fill_style = "stipple";
  unknown.content_sha256 = await canonicalDigest(unknown);
  assert.equal(inkSchema(unknown), false);
  await assert.rejects(validateContract(unknown), /fill_style/);
});
