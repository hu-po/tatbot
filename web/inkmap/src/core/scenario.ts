import type { Anchor } from "./anchor.ts";

export const SCENARIO_SCHEMA_VERSION = 1;

export interface TattooScenario {
  schema_version: 1;
  units: { length: "m"; tattoo_size: "mm"; angle: "rad"; up: "+z"; matrix_order: "row-major" };
  seed: number;
  body: {
    id: string; path: string; asset_sha256: string; surface_sha256: string;
    rig_id: string; rig_sha256: string;
  };
  pose: {
    id: string; catalog_sha256: string; source: "named" | "tracked";
    joint_rotations: Record<string, [number, number, number, number]>;
    world_from_body: number[][];
    capture_time_ns?: number;
    confidence?: number;
  };
  placement: {
    source_sha256: string; id: string; design_id: string; anchor: Anchor;
    rotation_rad: number; size_mm: [number, number]; mirror: boolean;
    [key: string]: unknown;
  };
  design: { id: string; name: string; svg: string; sha256: string; source: Record<string, unknown> };
  trace: {
    compiler: "tatbot_sim.surface_trace"; compiler_version: number; sha256: string;
    strokes: Anchor[][];
  };
  robot: { urdf_sha256: string; tool_id: string; world_from_robot: number[][] };
  support: { id: string };
  provenance: { created_at: string; git_sha: string; generator: string };
}

const SHA = /^[0-9a-f]{64}$/;

export function validateTattooScenario(x: unknown): asserts x is TattooScenario {
  const fail = (message: string): never => { throw new Error(`tattoo scenario: ${message}`); };
  if (!x || typeof x !== "object") fail("not an object");
  const s = x as Record<string, unknown>;
  if (s.schema_version !== SCENARIO_SCHEMA_VERSION) fail(`schema_version must be ${SCENARIO_SCHEMA_VERSION}`);
  const units = s.units as Record<string, unknown> | undefined;
  if (!units || units.length !== "m" || units.tattoo_size !== "mm" || units.angle !== "rad" || units.up !== "+z" || units.matrix_order !== "row-major") fail("units/frame contract mismatch");
  if (!Number.isSafeInteger(s.seed) || (s.seed as number) < 0) fail("seed must be a non-negative integer");

  const digest = (value: unknown, where: string) => {
    if (typeof value !== "string" || !SHA.test(value)) fail(`${where} must be a sha256 hex digest`);
  };
  const body0 = s.body as Record<string, unknown> | undefined;
  if (!body0 || typeof body0.id !== "string" || typeof body0.path !== "string" || typeof body0.rig_id !== "string") fail("body identity is incomplete");
  const body = body0 as Record<string, unknown>;
  digest(body.asset_sha256, "body.asset_sha256"); digest(body.surface_sha256, "body.surface_sha256"); digest(body.rig_sha256, "body.rig_sha256");

  const matrix = (value: unknown, where: string) => {
    if (!Array.isArray(value) || value.length !== 4 || !value.every((r) => Array.isArray(r) && r.length === 4 && r.every((v) => typeof v === "number" && Number.isFinite(v)))) fail(`${where} must be a finite row-major 4x4 matrix`);
  };
  const anchor = (value: unknown, where: string) => {
    const a = value as Record<string, unknown> | undefined;
    const bc = a?.barycentric;
    if (!a || !Number.isInteger(a.face) || (a.face as number) < 0 || !Array.isArray(bc) || bc.length !== 3 || !bc.every((v) => typeof v === "number" && v >= 0 && v <= 1) || Math.abs(bc.reduce((n, v) => n + (v as number), 0) - 1) > 1e-6) fail(`${where} is not a normalized face/barycentric anchor`);
  };

  const pose0 = s.pose as Record<string, unknown> | undefined;
  if (!pose0 || typeof pose0.id !== "string" || !["named", "tracked"].includes(pose0.source as string)) fail("pose identity/source is invalid");
  const pose = pose0 as Record<string, unknown>;
  digest(pose.catalog_sha256, "pose.catalog_sha256"); matrix(pose.world_from_body, "pose.world_from_body");
  if (!pose.joint_rotations || typeof pose.joint_rotations !== "object" || Array.isArray(pose.joint_rotations)) fail("pose.joint_rotations must be an object");
  for (const [bone, q] of Object.entries(pose.joint_rotations as Record<string, unknown>)) {
    if (!Array.isArray(q) || q.length !== 4 || !q.every((v) => typeof v === "number" && Number.isFinite(v))) fail(`pose.joint_rotations.${bone} must be [x,y,z,w]`);
  }

  const placement0 = s.placement as Record<string, unknown> | undefined;
  if (!placement0 || typeof placement0.id !== "string" || typeof placement0.design_id !== "string" || typeof placement0.rotation_rad !== "number" || typeof placement0.mirror !== "boolean") fail("placement is incomplete");
  const placement = placement0 as Record<string, unknown>;
  digest(placement.source_sha256, "placement.source_sha256"); anchor(placement.anchor, "placement.anchor");
  if (!Array.isArray(placement.size_mm) || placement.size_mm.length !== 2 || !placement.size_mm.every((v) => typeof v === "number" && v > 0)) fail("placement.size_mm must be positive [width,height]");

  const design0 = s.design as Record<string, unknown> | undefined;
  if (!design0 || typeof design0.id !== "string" || typeof design0.name !== "string" || typeof design0.svg !== "string" || !design0.svg.includes("<svg") || !design0.source || typeof design0.source !== "object") fail("design is incomplete");
  const design = design0 as Record<string, unknown>;
  digest(design.sha256, "design.sha256");

  const trace0 = s.trace as Record<string, unknown> | undefined;
  if (!trace0 || trace0.compiler !== "tatbot_sim.surface_trace" || !Number.isInteger(trace0.compiler_version) || (trace0.compiler_version as number) < 1 || !Array.isArray(trace0.strokes) || trace0.strokes.length === 0) fail("trace is incomplete");
  const trace = trace0 as Record<string, unknown>;
  digest(trace.sha256, "trace.sha256");
  for (const [i, stroke0] of (trace.strokes as unknown[]).entries()) {
    if (!Array.isArray(stroke0) || stroke0.length < 2) fail(`trace.strokes[${i}] needs at least two anchors`);
    const stroke = stroke0 as unknown[];
    stroke.forEach((a: unknown, j: number) => anchor(a, `trace.strokes[${i}][${j}]`));
  }

  const robot0 = s.robot as Record<string, unknown> | undefined;
  if (!robot0 || typeof robot0.tool_id !== "string") fail("robot is incomplete");
  const robot = robot0 as Record<string, unknown>;
  digest(robot.urdf_sha256, "robot.urdf_sha256"); matrix(robot.world_from_robot, "robot.world_from_robot");
  const support = s.support as Record<string, unknown> | undefined;
  if (!support || typeof support.id !== "string" || support.id.length === 0) fail("support.id is required");
  const provenance = s.provenance as Record<string, unknown> | undefined;
  if (!provenance || typeof provenance.created_at !== "string" || typeof provenance.git_sha !== "string" || typeof provenance.generator !== "string") fail("provenance is incomplete");
}
