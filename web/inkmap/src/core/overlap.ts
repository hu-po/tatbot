/** Whether two placed designs share skin: each one's art rectangle, charted about its own anchor, sampled against the
 *  other's. Skin is curved and a mesh face can be 60 mm long, so sharing a face says nothing; a sample's place on a
 *  shared face carries it from one chart to the other. This is the rectangles, not their ink: advice, as the art can
 *  leave corners empty. */
import type * as THREE from "three";
import { anchorToPoint } from "./anchor.ts";
import type { Placement } from "./schema.ts";
import { unfoldSurfacePatch, type Vec2 } from "./unfold.ts";

const SAMPLES_PER_SIDE = 16;
/** Less of a design than this overlapping is two rectangles touching, not two designs sharing skin. */
export const OVERLAP_NOTICE_SHARE = 0.03;

const reachM = (placement: Placement) => Math.hypot(...placement.size_mm) / 2000;

const chart = (rest: THREE.BufferGeometry, placement: Placement) => new Map(
  unfoldSurfacePatch(rest, placement.anchor, placement.rotation_rad, reachM(placement) + 0.002).faces.map((entry) => [entry.face, entry.triangleUv]));

/** `point`'s barycentric weights in the triangle, when it lies inside. */
function inTriangle([a, b, c]: [Vec2, Vec2, Vec2], [x, y]: Vec2): [number, number, number] | null {
  const area = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  if (Math.abs(area) < 1e-18) return null;
  const u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / area;
  const v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / area;
  return u >= 0 && v >= 0 && u + v <= 1 ? [u, v, 1 - u - v] : null;
}

/** The share of `a`'s rectangle that lies inside `b`'s, 0 to 1. */
export function overlapShare(rest: THREE.BufferGeometry, a: Placement, b: Placement): number {
  if (anchorToPoint(rest, a.anchor).p.distanceTo(anchorToPoint(rest, b.anchor).p) > reachM(a) + reachM(b)) return 0;
  const inB = chart(rest, b);
  const shared = [...chart(rest, a)].filter(([face]) => inB.has(face));
  const [halfW, halfH] = [b.size_mm[0] / 2000, b.size_mm[1] / 2000];
  let hits = 0;
  for (let row = 0; row < SAMPLES_PER_SIDE; row++) for (let column = 0; column < SAMPLES_PER_SIDE; column++) {
    const sample: Vec2 = [((column + 0.5) / SAMPLES_PER_SIDE - 0.5) * a.size_mm[0] / 1000, ((row + 0.5) / SAMPLES_PER_SIDE - 0.5) * a.size_mm[1] / 1000];
    for (const [face, triangle] of shared) {
      const weights = inTriangle(triangle, sample);
      if (!weights) continue;
      const corners = inB.get(face)!;
      const [x, y] = [0, 1].map((axis) => weights.reduce((sum, weight, corner) => sum + weight * corners[corner][axis], 0));
      if (Math.abs(x) <= halfW && Math.abs(y) <= halfH) hits++;
      break;
    }
  }
  return hits / SAMPLES_PER_SIDE ** 2;
}

/** Every pair of placements sharing skin, with the larger share either gives up to the other. */
export function overlaps(rest: THREE.BufferGeometry, placements: readonly Placement[]): { a: Placement; b: Placement; share: number }[] {
  return placements.flatMap((a, index) => placements.slice(index + 1).flatMap((b) => {
    const share = Math.max(overlapShare(rest, a, b), overlapShare(rest, b, a));
    return share >= OVERLAP_NOTICE_SHARE ? [{ a, b, share }] : [];
  }));
}
