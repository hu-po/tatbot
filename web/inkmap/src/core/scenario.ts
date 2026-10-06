import type { Anchor } from "./anchor.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_ASSET_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "./body.ts";
import { POSE_CATALOG, POSE_CATALOG_SHA256, poseRecord } from "./pose.ts";

export const SCENARIO_SCHEMA_VERSION = 2;
export const SUPPORTED_SCENARIO_SCHEMA_VERSIONS = [2, 3] as const;

export interface TattooScenario {
  schema_version: 2 | 3;
  units: { length: "m"; tattoo_size: "mm"; angle: "rad"; up: "+z"; matrix_order: "row-major" };
  seed: number;
  body: {
    model_spec_id: typeof MODEL_SPEC_ID;
    model_spec_sha256: typeof MODEL_SPEC_SHA256;
    identity_sha256: string;
    topology_sha256: typeof TOPOLOGY_SHA256;
    rest_surface_sha256: string;
    asset_path: string;
    asset_sha256: string;
    pose_asset_sha256: string;
  };
  pose: {
    id: string; catalog_sha256: string; source: "named";
    posed_surface_sha256: string;
    world_from_body: number[][];
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
  support: { id: string; world_from_nominal?: number[][] };
  placement_optimization?: {
    schema: "tatbot.placement-search/1";
    selected: { world_from_body: number[][]; world_from_support: number[][]; [key: string]: unknown };
    candidates: Record<string, unknown>[];
    [key: string]: unknown;
  };
  provenance: { created_at: string; git_sha: string; generator: string };
  program_binding?: Record<string, unknown>;
}

const SHA = /^[0-9a-f]{64}$/;

export function validateTattooScenario(x: unknown): asserts x is TattooScenario {
  const fail = (message: string): never => { throw new Error(`tattoo scenario: ${message}`); };
  if (!x || typeof x !== "object") fail("not an object");
  const s = x as Record<string, unknown>;
  if (!SUPPORTED_SCENARIO_SCHEMA_VERSIONS.includes(s.schema_version as 2 | 3)) {
    fail(`schema_version must be ${SUPPORTED_SCENARIO_SCHEMA_VERSIONS.join(" or ")}`);
  }
  if (s.schema_version === 3 && (!s.program_binding || typeof s.program_binding !== "object")) {
    fail("version 3 requires program_binding");
  }
  const units = s.units as Record<string, unknown> | undefined;
  if (!units || units.length !== "m" || units.tattoo_size !== "mm" || units.angle !== "rad" || units.up !== "+z" || units.matrix_order !== "row-major") fail("units/frame contract mismatch");
  if (!Number.isSafeInteger(s.seed) || (s.seed as number) < 0) fail("seed must be a non-negative integer");

  const digest = (value: unknown, where: string) => {
    if (typeof value !== "string" || !SHA.test(value)) fail(`${where} must be a sha256 hex digest`);
  };
  const body0 = s.body as Record<string, unknown> | undefined;
  if (
    !body0
    || body0.model_spec_id !== MODEL_SPEC_ID
    || body0.model_spec_sha256 !== MODEL_SPEC_SHA256
    || body0.identity_sha256 !== REFERENCE_IDENTITY_SHA256
    || body0.topology_sha256 !== TOPOLOGY_SHA256
    || body0.rest_surface_sha256 !== REST_SURFACE_SHA256
    || body0.asset_path !== BODY_SPEC.path
    || body0.asset_sha256 !== REST_ASSET_SHA256
    || body0.pose_asset_sha256 !== POSE_CATALOG.pose_asset.sha256
  ) fail("unsupported schema/model");
  const body = body0 as Record<string, unknown>;
  for (const field of ["identity_sha256", "rest_surface_sha256", "asset_sha256", "pose_asset_sha256"]) {
    digest(body[field], `body.${field}`);
  }

  const matrix = (value: unknown, where: string) => {
    if (!Array.isArray(value) || value.length !== 4 || !value.every((r) => Array.isArray(r) && r.length === 4 && r.every((v) => typeof v === "number" && Number.isFinite(v)))) fail(`${where} must be a finite row-major 4x4 matrix`);
  };
  const anchor = (value: unknown, where: string) => {
    const a = value as Record<string, unknown> | undefined;
    const bc = a?.barycentric;
    if (!a || !Number.isInteger(a.face) || (a.face as number) < 0 || !Array.isArray(bc) || bc.length !== 3 || !bc.every((v) => typeof v === "number" && v >= 0 && v <= 1) || Math.abs(bc.reduce((n, v) => n + (v as number), 0) - 1) > 1e-6) fail(`${where} is not a normalized face/barycentric anchor`);
  };

  const pose0 = s.pose as Record<string, unknown> | undefined;
  if (!pose0 || typeof pose0.id !== "string" || pose0.source !== "named" || !POSE_CATALOG.pose_ids.includes(pose0.id)) fail("pose identity/source is invalid");
  const pose = pose0 as Record<string, unknown>;
  digest(pose.catalog_sha256, "pose.catalog_sha256");
  digest(pose.posed_surface_sha256, "pose.posed_surface_sha256");
  if (pose.catalog_sha256 !== POSE_CATALOG_SHA256 || pose.posed_surface_sha256 !== poseRecord(pose.id as string).surface_sha256) fail("pose binding differs from the reviewed catalog");
  matrix(pose.world_from_body, "pose.world_from_body");

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
  const support0 = s.support as Record<string, unknown> | undefined;
  if (!support0 || typeof support0.id !== "string" || support0.id.length === 0) fail("support.id is required");
  const support = support0 as Record<string, unknown>;
  if (support.world_from_nominal !== undefined) matrix(support.world_from_nominal, "support.world_from_nominal");
  if (s.placement_optimization !== undefined) {
    const search = s.placement_optimization as Record<string, unknown>;
    if (search.schema !== "tatbot.placement-search/1" || !Array.isArray(search.candidates) || search.candidates.length === 0) fail("placement_optimization is incomplete");
    const selected = search.selected as Record<string, unknown> | undefined;
    if (!selected) fail("placement_optimization.selected is required");
    matrix(selected!.world_from_body, "placement_optimization.selected.world_from_body");
    matrix(selected!.world_from_support, "placement_optimization.selected.world_from_support");
  }
  const provenance = s.provenance as Record<string, unknown> | undefined;
  if (!provenance || typeof provenance.created_at !== "string" || typeof provenance.git_sha !== "string" || typeof provenance.generator !== "string") fail("provenance is incomplete");
}
