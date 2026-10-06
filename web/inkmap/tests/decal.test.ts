import { test } from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { computeSmoothNormals, faceCentroids } from "../src/core/anchor.ts";
import { buildDecal, frameToEuler, SurfaceChartError } from "../src/core/decal.ts";
import { frameAt } from "../src/core/anchor.ts";
import { unfoldSurfacePatch } from "../src/core/unfold.ts";

function cylinder(): { g: THREE.BufferGeometry; c: Float32Array } {
  // A forearm-ish tube: 40 mm radius, 300 mm long, axis along Z (up).
  const g = new THREE.CylinderGeometry(0.04, 0.04, 0.3, 48, 12, true).toNonIndexed();
  g.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  return { g, c: faceCentroids(g) };
}

function sphere(radiusM: number): THREE.BufferGeometry {
  const g = new THREE.SphereGeometry(radiusM, 64, 48).toNonIndexed();
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  return g;
}

/** A face on the sphere's equator, away from the polar fans. */
function equatorFace(g: THREE.BufferGeometry): number {
  const position = g.getAttribute("position");
  let best = 0;
  let bestScore = Infinity;
  for (let face = 0; face < position.count / 3; face++) {
    const z = (position.getZ(3 * face) + position.getZ(3 * face + 1) + position.getZ(3 * face + 2)) / 3;
    const y = (position.getY(3 * face) + position.getY(3 * face + 1) + position.getY(3 * face + 2)) / 3;
    const score = Math.abs(y) + Math.abs(z - Math.abs(z));
    if (score < bestScore) { bestScore = score; best = face; }
  }
  return best;
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

test("intrinsic unfold is deterministic and connected", () => {
  const { g } = cylinder();
  const anchor = { face: 200, barycentric: [0.3, 0.3, 0.4] as [number, number, number] };
  const first = unfoldSurfacePatch(g, anchor, 0.17, 0.04);
  const second = unfoldSurfacePatch(g, anchor, 0.17, 0.04);
  assert.deepEqual(first.faces, second.faces);
  assert.ok(first.faces.length > 10 && first.faces.length < g.getAttribute("position").count / 3);
  assert.ok(first.faces.every((entry) => entry.face === anchor.face || first.adjacency.get(entry.face)!.length > 0));
});

test("buildDecal produces triangles that sit on the surface within the requested size", () => {
  const { g } = cylinder();
  const anchor = { face: 200, barycentric: [0.3, 0.3, 0.4] as [number, number, number] };
  const { geometry, frame } = buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [40, 60] });
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
  const uv = geometry.getAttribute("uv");
  assert.ok(uv, "decal has uvs");
  const us = Array.from({ length: uv.count }, (_, index) => uv.getX(index));
  const vs = Array.from({ length: uv.count }, (_, index) => uv.getY(index));
  assert.ok(Math.max(...us) - Math.min(...us) >= 0.999, "preview spans the requested 40 mm chart width");
  assert.ok(Math.max(...vs) - Math.min(...vs) >= 0.999, "preview spans the requested 60 mm chart height");
});

test("a bigger design yields a bigger decal (more surface covered)", () => {
  const { g } = cylinder();
  const anchor = { face: 200, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  const small = buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [20, 20] }).geometry.getAttribute("position").count;
  const big = buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [60, 60] }).geometry.getAttribute("position").count;
  assert.ok(big > small, `${big} > ${small}`);
});

test("mapping quality reports metric area, posed stretch, and in-chart seam disagreement", () => {
  const { g } = cylinder();
  const posed = g.clone().applyMatrix4(new THREE.Matrix4().makeScale(1.1, 1.1, 1.1));
  computeSmoothNormals(posed);
  const anchor = { face: 200, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  const { geometry, quality } = buildDecal(g, posed, { anchor, rotationRad: 0, sizeMm: [30, 40] });
  assert.equal(quality.requestedAreaMm2, 1200);
  assert.ok(quality.mappedPercent > 95 && quality.mappedPercent < 105, `${quality.mappedPercent}% mapped`);
  assert.ok(Math.abs(quality.areaStretchRatio - 1.21) < 0.02, `${quality.areaStretchRatio} stretch`);
  assert.ok(quality.seamRatio >= 0 && quality.seamRatio < 0.5);
  assert.ok(quality.chartFaceCount > 0);
  geometry.dispose(); posed.dispose();
});

test("a chart wider than the cylinder circumference refuses overlap instead of painting the far side", () => {
  const { g } = cylinder();
  const anchor = { face: 200, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  assert.throws(
    () => buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [280, 40] }),
    (error: unknown) => error instanceof SurfaceChartError && error.code === "surface_chart_overlap",
  );
});

test("a palm-sized design on chest-like curvature builds despite the chart's curvature wedges", () => {
  // Chest/shoulder curvature is around a 150 mm radius. The developed chart of
  // a 50 mm design there leaves wedge gaps of about a percent of the area; a
  // flat stencil tolerates that and so must the preview.
  const g = sphere(0.15);
  const anchor = { face: equatorFace(g), barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  for (const size of [20, 50, 100]) {
    const { geometry } = buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [size, size] });
    assert.ok(geometry.getAttribute("position").count >= 3, `${size} mm design builds`);
  }
});

test("a design that would need more than a radian of angle defect is refused as unflattenable", () => {
  // A 120 mm design on a 40 mm ball encloses well over a hemisphere: no flat
  // chart represents it, and the seam mismatch exceeds half the diagonal.
  const g = sphere(0.04);
  const anchor = { face: equatorFace(g), barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  assert.throws(
    () => buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [120, 120] }),
    (error: unknown) => error instanceof SurfaceChartError && error.code === "surface_chart_overlap",
  );
});

test("a design that runs off the open end of the tube is refused", () => {
  const { g } = cylinder();
  // Row 0 of the open-ended tube borders the rim; a 60 mm tall design there crosses it.
  const anchor = { face: 20, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  assert.throws(
    () => buildDecal(g, g, { anchor, rotationRad: 0, sizeMm: [40, 60] }),
    (error: unknown) => error instanceof SurfaceChartError && /open edge/.test(error.message),
  );
});
