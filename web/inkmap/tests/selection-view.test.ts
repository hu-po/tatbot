import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { selectionView } from "../src/core/selection-view.ts";
import { anchorToPoint, type Anchor } from "../src/core/anchor.ts";
import { BODY_SPEC, applyBodyRotation, buildPosedSkin, buildSkin } from "../src/core/body.ts";
import { AtlasIndex, parseAtlas } from "../src/core/atlas.ts";
import { POSE_CATALOG, poseRecord } from "../src/core/pose.ts";

function assertVisible(geometry: THREE.BufferGeometry, anchor: Anchor, eye: THREE.Vector3 | null) {
  assert.ok(eye, "a visible camera position must be found");
  const target = anchorToPoint(geometry, anchor).p;
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  try {
    const hit = new THREE.Raycaster(eye, target.clone().sub(eye).normalize())
      .intersectObject(new THREE.Mesh(geometry, material))[0];
    assert.ok(hit && hit.point.distanceTo(target) < .002, "the first visible surface must be the selected point");
  } finally { material.dispose(); }
}

test("focus keeps a clear normal view and searches around an occluding surface", () => {
  const plane = new THREE.PlaneGeometry(2, 2);
  const anchor: Anchor = { face: 0, barycentric: [1 / 3, 1 / 3, 1 / 3] };
  const { p, n } = anchorToPoint(plane, anchor);
  assert.deepEqual(selectionView(plane, anchor, .72)?.toArray(), p.clone().addScaledVector(n, .72).toArray());
  const obstacle = new THREE.PlaneGeometry(.2, .2).translate(p.x, p.y, .2);
  const occluded = mergeGeometries([plane, obstacle]);
  const eye = selectionView(occluded, anchor, .72);
  assertVisible(occluded, anchor, eye);
  assert.ok(eye!.distanceTo(p.clone().addScaledVector(n, .72)) > .1, "occlusion must change the view");
  plane.dispose(); obstacle.dispose(); occluded.dispose();
});

test("ordinary body placements focus visibly in both editor poses", async () => {
  const bytes = readFileSync(new URL(`../public/${BODY_SPEC.path}`, import.meta.url));
  const rest = buildSkin((await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "")).scene);
  const atlas = new AtlasIndex(parseAtlas(JSON.parse(readFileSync(new URL("../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8"))), rest.geometry, rest.centroids);
  const poses = readFileSync(new URL(`../public/${POSE_CATALOG.pose_asset.path}`, import.meta.url));
  for (const id of ["standing-neutral", "reclined-seated"]) {
    const pose = poseRecord(id), chunk = poses.subarray(pose.byte_offset, pose.byte_offset + pose.byte_length);
    const skin = buildPosedSkin(rest, chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
    applyBodyRotation(skin, pose.body_rotation_xyzw);
    for (const site of ["forearm", "shin", "bicep", "thigh"]) for (const side of ["left", "right"] as const) {
      const anchor = atlas.anchorFor({ id: site, laterality: side });
      assertVisible(skin.geometry, anchor, selectionView(skin.geometry, anchor, .72));
    }
    skin.geometry.dispose();
  }
  rest.geometry.dispose();
});
