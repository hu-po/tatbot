#!/usr/bin/env node
/** Hermetic JSON adapter for Python/TypeScript intrinsic-walk parity tests. */
import { readFileSync, writeFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { applyBodyRotation, buildPosedSkin, buildSkin, type Skin } from "../src/core/body.ts";
import { anchorToPoint, frameAt, type Anchor } from "../src/core/anchor.ts";
import { buildDecal } from "../src/core/decal.ts";
import { POSE_CATALOG } from "../src/core/pose.ts";
import { mapChartPoints, unfoldSurfacePatch, type Vec2 } from "../src/core/unfold.ts";

interface CaseRequest {
  id?: string;
  anchor: { face: number; barycentric: [number, number, number] };
  rotation_rad: number;
  radius_m: number;
  points_m: Vec2[];
  footprint_m?: [number, number];
}

interface Request extends CaseRequest {
  cases?: CaseRequest[];
  pose_ids?: string[];
}

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("usage: surface_parity.ts INPUT.json OUTPUT.json");
}
const request = JSON.parse(readFileSync(inputPath, "utf8")) as Request;
const cases = request.cases ?? [request];
if (!Array.isArray(cases) || !cases.length || cases.length > 100) throw new Error("surface_coordinate_invalid: expected 1-100 cases");
for (const item of cases) if (!Array.isArray(item.points_m) || item.points_m.length > 20_000) {
  throw new Error("surface_coordinate_invalid: points_m must contain at most 20000 points");
}
const poseIds = request.pose_ids ?? [];
if (new Set(poseIds).size !== poseIds.length || poseIds.some(id => !POSE_CATALOG.pose_ids.includes(id))) {
  throw new Error("pose_unsupported: pose_ids must be unique reviewed poses");
}
const bodyPath = new URL("../public/bodies/mhr-soma-v1.glb", import.meta.url);
const bytes = readFileSync(bodyPath);
const gltf = await new GLTFLoader().parseAsync(
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  "",
);
const skin = buildSkin(gltf.scene);
const posedSkins = new Map<string, Skin>();
if (poseIds.length) {
  const posePath = new URL(`../public/${POSE_CATALOG.pose_asset.path}`, import.meta.url);
  const poseBytes = readFileSync(posePath);
  for (const id of poseIds) {
    const record = POSE_CATALOG.poses[id];
    const begin = poseBytes.byteOffset + record.byte_offset;
    const posed = buildPosedSkin(skin, poseBytes.buffer.slice(begin, begin + record.byte_length));
    applyBodyRotation(posed, record.body_rotation_xyzw);
    posedSkins.set(id, posed);
  }
}

function positions(anchors: Anchor[]): Record<string, number[][]> {
  return Object.fromEntries([...posedSkins].map(([id, posed]) => [
    id, anchors.map(anchor => anchorToPoint(posed.geometry, anchor).p.toArray()),
  ]));
}

function runCase(item: CaseRequest) {
  if (item.footprint_m) {
    const [width, height] = item.footprint_m;
    buildDecal(skin.geometry, skin.geometry, {
      anchor: item.anchor,
      rotationRad: item.rotation_rad,
      sizeMm: [width * 1000, height * 1000],
    });
  }
  const patch = unfoldSurfacePatch(skin.geometry, item.anchor, item.rotation_rad, item.radius_m);
  const mapped = mapChartPoints(skin.geometry, patch, item.points_m);
  const frame = frameAt(skin.geometry, item.anchor, item.rotation_rad);
  const position = skin.geometry.getAttribute("position");
  const seedTriangle = [0, 1, 2].map((corner) => {
    const index = item.anchor.face * 3 + corner;
    return [position.getX(index), position.getY(index), position.getZ(index)];
  });
  const anchors = mapped.map((entry) => entry.anchor);
  return {
    ...(item.id ? { id: item.id } : {}), status: "accepted",
    count: mapped.length, seed_triangle_uv: patch.seedTriangleUv,
    chart: {
      face_count: patch.faces.length,
      seam_count: patch.seams.length,
      maximum_seam_m: patch.seams.length ? Math.max(...patch.seams.map(seam => seam.mismatchM)) : 0,
      boundary_face_count: patch.boundaryFaces.length,
    },
    seed_triangle_xyz: seedTriangle,
    frame: { n: frame.n.toArray(), u: frame.u.toArray(), v: frame.v.toArray() },
    anchors, posed_positions_m: positions(anchors),
  };
}

if (request.cases) {
  const results = cases.map(item => {
    try { return runCase(item); }
    catch (error) { return { id: item.id, status: "rejected", reason: String(error) }; }
  });
  writeFileSync(outputPath, JSON.stringify({
    schema: "tatbot.surface-walk-parity-suite/1", requested: cases.length,
    accepted: results.filter(result => result.status === "accepted").length,
    rejected: results.filter(result => result.status === "rejected").length,
    pose_ids: poseIds, results,
  }) + "\n");
} else {
  writeFileSync(outputPath, JSON.stringify({ schema: "tatbot.surface-walk-parity/1", ...runCase(request) }) + "\n");
}
