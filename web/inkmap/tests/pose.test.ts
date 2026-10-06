import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  BODY_SPEC,
  MID_FACE_COUNT,
  applyBodyRotation,
  buildPosedSkin,
  buildSkin,
  canonicalSurfaceBytes,
} from "../src/core/body.ts";
import { POSE_CATALOG, POSE_CATALOG_SHA256, poseRecord } from "../src/core/pose.ts";
import { sha256Hex } from "../src/core/sha256.ts";

function arrayBuffer(bytes: Buffer): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

test("the browser reproduces every checked-in SOMA pose byte-for-byte", async () => {
  const restBytes = readFileSync(new URL(`../public/${BODY_SPEC.path}`, import.meta.url));
  assert.equal(await sha256Hex(arrayBuffer(restBytes)), POSE_CATALOG.rest_asset.sha256);
  const gltf = await new GLTFLoader().parseAsync(arrayBuffer(restBytes), "");
  const rest = buildSkin(gltf.scene);

  const poseBytes = readFileSync(new URL(`../public/${POSE_CATALOG.pose_asset.path}`, import.meta.url));
  assert.equal(await sha256Hex(arrayBuffer(poseBytes)), POSE_CATALOG.pose_asset.sha256);
  assert.equal(POSE_CATALOG.pose_asset.face_count, MID_FACE_COUNT);

  for (const poseId of POSE_CATALOG.pose_ids) {
    const pose = poseRecord(poseId);
    const chunk = poseBytes.subarray(pose.byte_offset, pose.byte_offset + pose.byte_length);
    assert.equal(chunk.byteLength, pose.byte_length, poseId);
    assert.equal(await sha256Hex(arrayBuffer(chunk)), pose.chunk_sha256, poseId);
    const skin = buildPosedSkin(rest, arrayBuffer(chunk));
    assert.equal(await sha256Hex(canonicalSurfaceBytes(skin.geometry)), pose.surface_sha256, poseId);
    assert.ok(pose.quality.max_joint_rotation_deg <= 120, poseId);
    assert.ok(pose.quality.edge_length_ratio_p001 > 0, poseId);
    assert.ok(pose.quality.triangle_area_ratio_p01 > 0, poseId);

    applyBodyRotation(skin, pose.body_rotation_xyzw);
    assert.ok(skin.bbox.min.toArray().every(Number.isFinite), poseId);
    assert.ok(skin.bbox.max.toArray().every(Number.isFinite), poseId);
    skin.geometry.dispose();
  }
  rest.geometry.dispose();
});

test("unknown poses fail closed", () => {
  assert.throws(() => poseRecord("legacy-pose"), /pose_unsupported/);
});

test("the published catalog digest is the sha256 of the catalog file itself", async () => {
  const catalogBytes = readFileSync(new URL("../../../config/inkmap/body-poses.json", import.meta.url));
  assert.equal(await sha256Hex(arrayBuffer(catalogBytes)), POSE_CATALOG_SHA256);
});
