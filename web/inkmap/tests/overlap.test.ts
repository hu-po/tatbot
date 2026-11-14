import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { computeSmoothNormals, type Anchor } from "../src/core/anchor.ts";
import { overlaps, overlapShare } from "../src/core/overlap.ts";
import type { Placement } from "../src/core/schema.ts";

/** A flat sheet facing -y, 400 mm square, in faces 20 mm across. */
function sheet(): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(0.4, 0.4, 20, 20).toNonIndexed();
  g.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  return g;
}

/** The anchor at (x, z) mm on the sheet. */
function at(g: THREE.BufferGeometry, xMm: number, zMm: number): Anchor {
  const position = g.getAttribute("position");
  const target = new THREE.Vector3(xMm / 1000, 0, zMm / 1000), triangle = new THREE.Triangle(), weights = new THREE.Vector3();
  for (let face = 0; face < position.count / 3; face++) {
    triangle.setFromAttributeAndIndices(position, 3 * face, 3 * face + 1, 3 * face + 2);
    if (triangle.containsPoint(target)) {
      triangle.getBarycoord(target, weights);
      return { face, barycentric: [weights.x, weights.y, weights.z] };
    }
  }
  throw new Error("off the sheet");
}

const placed = (id: string, anchor: Anchor, size: [number, number], rotation = 0): Placement =>
  ({ id, design_id: id, anchor, rotation_rad: rotation, size_mm: size, mirror: false });

test("a design's overlap is the share of it inside the other, on one face or across several", () => {
  const g = sheet();
  const a = placed("a", at(g, 1, 1), [40, 40]);
  assert.ok(overlapShare(g, a, placed("b", at(g, 1, 1), [40, 40])) > 0.95, "the same spot");
  assert.ok(Math.abs(overlapShare(g, a, placed("b", at(g, 21, 1), [40, 40])) - 0.5) < 0.08, "half a width along");
  assert.ok(Math.abs(overlapShare(g, a, placed("b", at(g, 1, 1), [20, 20])) - 0.25) < 0.08, "a smaller design inside it");
  assert.equal(overlapShare(g, a, placed("b", at(g, 90, 1), [40, 40])), 0, "well apart");
});

test("rotation counts: a long design turned away no longer reaches its neighbour", () => {
  const g = sheet();
  const long = (rotation: number) => placed("a", at(g, 1, 1), [160, 10], rotation);
  const neighbour = placed("b", at(g, 60, 1), [20, 20]);
  assert.ok(overlapShare(g, long(0), neighbour) > 0);
  assert.equal(overlapShare(g, long(Math.PI / 2), neighbour), 0);
});

test("touching rectangles are no notice, a real overlap names the pair", () => {
  const g = sheet();
  const a = placed("a", at(g, 1, 1), [40, 40]), touching = placed("b", at(g, 41, 1), [40, 40]), over = placed("c", at(g, 11, 1), [40, 40]);
  assert.deepEqual(overlaps(g, [a, touching]), []);
  const [pair] = overlaps(g, [a, over]);
  assert.deepEqual([pair.a.id, pair.b.id], ["a", "c"]);
  assert.ok(pair.share > 0.6 && pair.share < 0.85);
});
