import catalogJson from "../../../../config/inkmap/body-poses.json" with { type: "json" };
import digestJson from "../../../../config/inkmap/body-poses.digest.json" with { type: "json" };
import {
  MID_FACE_COUNT,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "./body.ts";

export type QuaternionXyzw = [number, number, number, number];

export interface PoseRecord {
  label: string;
  support_id: string;
  body_rotation_xyzw: QuaternionXyzw;
  constraints: string[];
  surface_sha256: string;
  byte_offset: number;
  byte_length: number;
  chunk_sha256: string;
  quality: {
    max_joint_rotation_deg: number;
    edge_length_ratio_p001: number;
    edge_length_ratio_p99: number;
    triangle_area_ratio_p01: number;
    triangle_area_ratio_p99: number;
  };
}

interface PoseCatalog {
  schema: "tatbot.body-pose-catalog/2";
  model_spec_id: string;
  model_spec_sha256: string;
  identity_sha256: string;
  topology_sha256: string;
  rest_surface_sha256: string;
  rest_asset: { path: string; sha256: string; size: number };
  pose_asset: {
    path: string;
    sha256: string;
    size: number;
    format: string;
    face_count: number;
    vertices_per_pose: number;
  };
  exclusion_asset: {
    path: string;
    sha256: string;
    size: number;
    format: string;
    source_segments: string[];
    excluded_vertices: number;
    excluded_faces: number;
  };
  pose_ids: string[];
  poses: Record<string, PoseRecord>;
}

export const POSE_CATALOG = catalogJson as unknown as PoseCatalog;
// Exact sha256 of the catalog file bytes, written by tools/export-soma.py
// beside the catalog. Python derives the same value from the bytes and refuses
// a stale digest file; tests/pose.test.ts hashes the file here.
const digest = digestJson as { schema: string; catalog_path: string; sha256: string };
if (digest.schema !== "tatbot.body-pose-catalog-digest/1" || !/^[0-9a-f]{64}$/.test(digest.sha256)) {
  throw new Error("body_model_unpinned: pose catalog digest file is malformed");
}
export const POSE_CATALOG_SHA256 = digest.sha256;

const expected = {
  schema: "tatbot.body-pose-catalog/2",
  model_spec_id: MODEL_SPEC_ID,
  model_spec_sha256: MODEL_SPEC_SHA256,
  identity_sha256: REFERENCE_IDENTITY_SHA256,
  topology_sha256: TOPOLOGY_SHA256,
  rest_surface_sha256: REST_SURFACE_SHA256,
  face_count: MID_FACE_COUNT,
};
if (
  POSE_CATALOG.schema !== expected.schema
  || POSE_CATALOG.model_spec_id !== expected.model_spec_id
  || POSE_CATALOG.model_spec_sha256 !== expected.model_spec_sha256
  || POSE_CATALOG.identity_sha256 !== expected.identity_sha256
  || POSE_CATALOG.topology_sha256 !== expected.topology_sha256
  || POSE_CATALOG.rest_surface_sha256 !== expected.rest_surface_sha256
  || POSE_CATALOG.pose_asset.face_count !== expected.face_count
  || new Set(POSE_CATALOG.pose_ids).size !== POSE_CATALOG.pose_ids.length
  || POSE_CATALOG.pose_ids.some((id) => !(id in POSE_CATALOG.poses))
) {
  throw new Error("body_model_unpinned: pose catalog does not match the reviewed SOMA contract");
}

export function poseRecord(poseId: string): PoseRecord {
  const pose = POSE_CATALOG.poses[poseId];
  if (!pose) throw new Error(`pose_unsupported: ${poseId}`);
  return pose;
}

/** The poses the editor offers: standing, and reclined in the tattoo chair.
 *  The catalog keeps every named session pose for the showcase and the
 *  simulator; the picker shows these two, plus whatever pose a recovered
 *  project already holds. */
export const EDITOR_POSE_IDS = ["standing-neutral", "reclined-seated"] as const;
export const EDITOR_POSE_LABELS: Record<(typeof EDITOR_POSE_IDS)[number], string> = { "standing-neutral": "Standing", "reclined-seated": "Reclined seated" };
export function editorPoseOptions(current: string): { id: string; label: string }[] {
  const listed: { id: string; label: string }[] = EDITOR_POSE_IDS.filter((id) => id in POSE_CATALOG.poses).map((id) => ({ id, label: EDITOR_POSE_LABELS[id] }));
  if (current in POSE_CATALOG.poses && !listed.some((option) => option.id === current)) listed.push({ id: current, label: poseRecord(current).label });
  return listed;
}
