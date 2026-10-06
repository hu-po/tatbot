/** Target placement intent, independent of any measured robot frame.
 * Analytic charts use surface_model's metric convention: cylinder u is axial,
 * v is circumferential arc length; the origin lies on the crest, not the axis.
 */
import { canonicalDigest, validateContract, type JsonObject } from "./human-representation/schema.ts";

export type AnalyticTarget = {
  kind: "plane";
  canvas_m: [number, number];
  anchor_uv_m: [number, number];
  margin_m: number;
} | {
  kind: "cylinder";
  canvas_m: [number, number];
  anchor_uv_m: [number, number];
  margin_m: number;
  radius_m: number;
};

export interface BodyTarget extends JsonObject {
  kind: "body";
  body_identity_sha256: string;
  rest_surface_sha256: string;
  semantic_site: string;
  laterality: string;
  anchor: JsonObject;
  tangent_frame_rule: string;
  supported_domain: JsonObject;
}

export interface SurfacePlacement extends JsonObject {
  schema: "tatbot.surface-placement/2";
  content_sha256: string;
  tattoo_program_sha256: string;
  target: AnalyticTarget | BodyTarget;
  physical_scale_m: [number, number];
  rotation_rad: number;
  mirrored: boolean;
  warp: JsonObject | null;
  review: { status: "pending" | "accepted" | "rejected"; reviewer: string; evidence_sha256: string };
  provenance: JsonObject;
}

export type SurfacePlacementInput = Pick<SurfacePlacement,
  "tattoo_program_sha256" | "target" | "physical_scale_m" | "rotation_rad" | "mirrored" | "warp" | "review" | "provenance">;

export async function makeSurfacePlacement(input: SurfacePlacementInput): Promise<SurfacePlacement> {
  const value = { ...structuredClone(input), schema: "tatbot.surface-placement/2", content_sha256: "0".repeat(64) };
  value.content_sha256 = await canonicalDigest(value);
  return await validateContract(value, { expectedSchema: "tatbot.surface-placement/2" }) as SurfacePlacement;
}

/** Explicit import migration preserves every body-specific constraint. The
 * original v1 artifact remains untouched and retains its original digest.
 */
export async function upgradeBodyPlacement(value: unknown): Promise<SurfacePlacement> {
  const original = await validateContract(value, { expectedSchema: "tatbot.surface-placement/1" });
  const names = ["body_identity_sha256", "rest_surface_sha256", "semantic_site", "laterality", "anchor", "tangent_frame_rule", "supported_domain"];
  const target = { kind: "body", ...Object.fromEntries(names.map(name => [name, original[name]])) };
  const common = Object.fromEntries(Object.entries(original).filter(([key]) => !names.includes(key) && key !== "schema" && key !== "content_sha256"));
  return makeSurfacePlacement({ ...common, target } as SurfacePlacementInput);
}

/** Rendering geometry only. Physical placement must bind a fresh measured
 * surface separately; these nominal frames never grant motion authority.
 */
export function analyticFrame(target: AnalyticTarget, uv: [number, number]): {
  point: [number, number, number]; normal: [number, number, number];
} {
  if (!uv.every(Number.isFinite)) throw new Error("target coordinates must be finite");
  if (target.kind === "plane") return { point: [uv[0], uv[1], 0], normal: [0, 0, 1] };
  if (!Number.isFinite(target.radius_m) || target.radius_m <= 0) throw new Error("cylinder radius must be positive");
  const theta = uv[1] / target.radius_m;
  return {
    point: [uv[0], target.radius_m * Math.sin(theta), target.radius_m * (Math.cos(theta) - 1)],
    normal: [0, Math.sin(theta), Math.cos(theta)],
  };
}
