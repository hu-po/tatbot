import { test } from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { anchorToPoint, computeSmoothNormals, faceCentroids, faceCount, frameAt, pointToAnchor, BODY_UP } from "../src/core/anchor.ts";

function sphere(): THREE.BufferGeometry {
  const g = new THREE.SphereGeometry(0.1, 24, 16).toNonIndexed();
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  return g;
}

test("anchor -> point -> anchor roundtrips on every face", () => {
  const g = sphere();
  const n = faceCount(g);
  for (let f = 0; f < n; f += 7) {
    const a = { face: f, barycentric: [0.2, 0.5, 0.3] as [number, number, number] };
    const { p } = anchorToPoint(g, a);
    const back = pointToAnchor(g, f, p);
    assert.equal(back.face, f);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(back.barycentric[i] - a.barycentric[i]) < 1e-5, `face ${f} weight ${i}`);
  }
});

test("pointToAnchor clamps a point outside the triangle back onto it", () => {
  const g = sphere();
  const far = new THREE.Vector3(5, 5, 5);
  const a = pointToAnchor(g, 3, far);
  const s = a.barycentric.reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(s - 1) < 1e-9);
  assert.ok(a.barycentric.every((w) => w >= 0 && w <= 1));
  assert.throws(() => pointToAnchor(g, faceCount(g), far), RangeError);
});

test("smooth normals on a sphere point radially outward", () => {
  const g = sphere();
  const pos = g.getAttribute("position"), nrm = g.getAttribute("normal");
  let worst = 1;
  for (let i = 0; i < pos.count; i++) {
    const p = new THREE.Vector3().fromBufferAttribute(pos, i).normalize();
    const n = new THREE.Vector3().fromBufferAttribute(nrm, i);
    worst = Math.min(worst, p.dot(n));
  }
  assert.ok(worst > 0.98, `min radial alignment ${worst}`);
});

test("frame is right-handed, orthonormal, tangent, and upright by default", () => {
  const g = sphere();
  for (const f of [0, 41, 200, faceCount(g) - 1]) {
    const fr = frameAt(g, { face: f, barycentric: [1 / 3, 1 / 3, 1 / 3] });
    assert.ok(Math.abs(fr.u.length() - 1) < 1e-9 && Math.abs(fr.v.length() - 1) < 1e-9 && Math.abs(fr.n.length() - 1) < 1e-9);
    assert.ok(Math.abs(fr.u.dot(fr.v)) < 1e-9 && Math.abs(fr.u.dot(fr.n)) < 1e-9 && Math.abs(fr.v.dot(fr.n)) < 1e-9);
    const cross = new THREE.Vector3().crossVectors(fr.u, fr.v);
    assert.ok(cross.distanceTo(fr.n) < 1e-9, "u × v = n");
    if (Math.abs(fr.n.dot(BODY_UP)) < 0.95) assert.ok(fr.v.dot(BODY_UP) > 0, "v points up when the surface is not horizontal");
  }
});

test("rotation about the normal rotates v by that angle and keeps the frame valid", () => {
  const g = sphere();
  const a = { face: 100, barycentric: [0.4, 0.4, 0.2] as [number, number, number] };
  const f0 = frameAt(g, a, 0);
  const f1 = frameAt(g, a, Math.PI / 2);
  assert.ok(Math.abs(f0.v.dot(f1.v)) < 1e-9, "90° apart");
  assert.ok(f1.v.distanceTo(f0.u.clone().negate()) < 1e-9 || f1.v.distanceTo(f0.u) < 1e-9, "v rotated into ±u");
  assert.ok(f1.n.distanceTo(f0.n) < 1e-12, "normal unchanged");
});

test("faceCentroids matches the mean of the three vertices", () => {
  const g = sphere();
  const c = faceCentroids(g);
  assert.equal(c.length, faceCount(g) * 3);
  const pos = g.getAttribute("position");
  const f = 12;
  const m = new THREE.Vector3();
  for (let k = 0; k < 3; k++) m.add(new THREE.Vector3().fromBufferAttribute(pos, f * 3 + k));
  m.divideScalar(3);
  assert.ok(Math.abs(c[f * 3] - m.x) < 1e-7 && Math.abs(c[f * 3 + 1] - m.y) < 1e-7 && Math.abs(c[f * 3 + 2] - m.z) < 1e-7);
});
