import { test } from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { computeSmoothNormals, faceCentroids } from "../src/core/anchor.ts";
import { buildDecal, frameToEuler, subGeometry } from "../src/core/decal.ts";
import { frameAt } from "../src/core/anchor.ts";

function cylinder(): { g: THREE.BufferGeometry; c: Float32Array } {
  // A forearm-ish tube: 40 mm radius, 300 mm long, axis along Z (up).
  const g = new THREE.CylinderGeometry(0.04, 0.04, 0.3, 48, 12, true).toNonIndexed();
  g.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  return { g, c: faceCentroids(g) };
}

test("frameToEuler maps decal +Z onto the surface normal and +Y onto v", () => {
  const { g } = cylinder();
  const fr = frameAt(g, { face: 50, barycentric: [1 / 3, 1 / 3, 1 / 3] }, 0.3);
  const q = new THREE.Quaternion().setFromEuler(frameToEuler(fr));
  const z = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  const y = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
  assert.ok(z.distanceTo(fr.n) < 1e-6);
  assert.ok(y.distanceTo(fr.v) < 1e-6);
});

test("subGeometry keeps only nearby faces", () => {
  const { g, c } = cylinder();
  const center = new THREE.Vector3(c[0], c[1], c[2]);
  const sub = subGeometry(g, c, center, 0.03);
  const n = sub.getAttribute("position").count / 3;
  assert.ok(n > 0 && n < c.length / 3, `sub has ${n} faces`);
});

test("buildDecal produces triangles that sit on the surface within the requested size", () => {
  const { g, c } = cylinder();
  const anchor = { face: 200, barycentric: [0.3, 0.3, 0.4] as [number, number, number] };
  const { geometry, frame } = buildDecal(g, c, { anchor, rotationRad: 0, sizeMm: [40, 60] });
  const pos = geometry.getAttribute("position");
  assert.ok(pos.count >= 3, "decal has triangles");
  let maxOff = 0, maxRadial = 0;
  for (let i = 0; i < pos.count; i++) {
    const p = new THREE.Vector3().fromBufferAttribute(pos, i);
    const d = p.clone().sub(frame.p);
    maxOff = Math.max(maxOff, Math.abs(d.dot(frame.u)), Math.abs(d.dot(frame.v)));
    // every decal vertex lies on the (faceted) cylinder surface
    maxRadial = Math.max(maxRadial, Math.abs(Math.hypot(p.x, p.y) - 0.04));
  }
  assert.ok(maxOff <= 0.03 + 1e-6, `decal extends ${maxOff} m from centre`);
  assert.ok(maxRadial < 0.003, `decal vertices are ${maxRadial} m off the surface`);
  assert.ok(geometry.getAttribute("uv"), "decal has uvs");
});

test("a bigger design yields a bigger decal (more surface covered)", () => {
  const { g, c } = cylinder();
  const anchor = { face: 200, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  const small = buildDecal(g, c, { anchor, rotationRad: 0, sizeMm: [20, 20] }).geometry.getAttribute("position").count;
  const big = buildDecal(g, c, { anchor, rotationRad: 0, sizeMm: [60, 60] }).geometry.getAttribute("position").count;
  assert.ok(big > small, `${big} > ${small}`);
});
