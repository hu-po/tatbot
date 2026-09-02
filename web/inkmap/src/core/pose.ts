import catalogJson from "../../../../config/inkmap/body-poses.json";

export type QuaternionXyzw = [number, number, number, number];

export interface PoseRecord {
  label: string;
  support_id: string;
  body_rotation_xyzw: QuaternionXyzw;
  joint_rotations: Record<string, QuaternionXyzw>;
  quality: {
    max_joint_rotation_deg: number;
    edge_length_ratio_p001: number;
    edge_length_ratio_p99: number;
    triangle_area_ratio_p01: number;
    triangle_area_ratio_p99: number;
  };
  /** Semantic joint metrics emitted by the named anatomy gates. */
  anatomy: Record<string, number>;
  /** Sparse Blender-authored positions used to gate cross-runtime skinning. */
  validation_vertices: [number, number, number][];
}

interface BodyPoseRecord {
  surface_sha256: string;
  rigged_path: string;
  rigged_asset_sha256: string;
  sidecar_path: string;
  validation_vertex_indices: number[];
  poses: Record<string, PoseRecord>;
}

interface PoseCatalog {
  schema_version: number;
  rig_id: string;
  pose_ids: string[];
  bodies: Record<string, BodyPoseRecord>;
}

export const POSE_CATALOG = catalogJson as unknown as PoseCatalog;

export function poseRecord(bodyId: string, poseId: string): PoseRecord {
  const body = POSE_CATALOG.bodies[bodyId];
  if (!body) throw new Error(`body ${bodyId}: missing from pose catalog`);
  const pose = body.poses[poseId];
  if (!pose) throw new Error(`body ${bodyId}: unknown pose "${poseId}"`);
  return pose;
}
