// One renderer for a placement: project the design onto the skin as a decal.
// Fine on torso/upper arm, stretches on tightly wrapping limbs — the phase-2
// geodesic wrap replaces this file and nothing else.
import * as THREE from "three";
import { DecalGeometry } from "three/examples/jsm/geometries/DecalGeometry.js";
import { frameAt, faceVertexIndices, type Anchor, type Frame } from "./anchor.ts";

export interface DecalParams {
  anchor: Anchor;
  rotationRad: number;
  /** [width, height] on the surface, millimetres. */
  sizeMm: [number, number];
}

/** Rotation that maps decal-local (+X, +Y, +Z) onto (u, v, n). */
export function frameToEuler(f: Frame): THREE.Euler {
  const m = new THREE.Matrix4().makeBasis(f.u, f.v, f.n);
  return new THREE.Euler().setFromRotationMatrix(m);
}

/**
 * Build the decal geometry for a placement. Only faces whose centroid lies
 * within the design's reach are handed to DecalGeometry, so a per-frame rebuild
 * while dragging stays cheap on a 16k-triangle skin.
 */
export function buildDecal(skin: THREE.BufferGeometry, centroids: Float32Array, params: DecalParams): { geometry: THREE.BufferGeometry; frame: Frame } {
  const frame = frameAt(skin, params.anchor, params.rotationRad);
  const [wMm, hMm] = params.sizeMm;
  const w = wMm / 1000, h = hMm / 1000;
  // Depth: deep enough to follow curvature under the design, shallow enough
  // not to pick up the far side of a limb.
  const depth = 0.6 * Math.max(w, h);
  const reach = 0.5 * Math.hypot(w, h, depth) * 1.5;
  const sub = subGeometry(skin, centroids, frame.p, reach);
  const mesh = new THREE.Mesh(sub);
  mesh.updateMatrixWorld(true);
  const geometry = new DecalGeometry(mesh, frame.p, frameToEuler(frame), new THREE.Vector3(w, h, depth));
  sub.dispose();
  return { geometry, frame };
}

/** Non-indexed copy of the faces whose centroid is within `r` of `center` (position + normal only). */
export function subGeometry(skin: THREE.BufferGeometry, centroids: Float32Array, center: THREE.Vector3, r: number): THREE.BufferGeometry {
  const pos = skin.getAttribute("position");
  const nrm = skin.getAttribute("normal");
  const r2 = r * r;
  const faces: number[] = [];
  for (let f = 0; f < centroids.length / 3; f++) {
    const dx = centroids[f * 3] - center.x, dy = centroids[f * 3 + 1] - center.y, dz = centroids[f * 3 + 2] - center.z;
    if (dx * dx + dy * dy + dz * dz <= r2) faces.push(f);
  }
  const p = new Float32Array(faces.length * 9);
  const n = new Float32Array(faces.length * 9);
  let o = 0;
  for (const f of faces) {
    for (const i of faceVertexIndices(skin, f)) {
      p[o] = pos.getX(i); p[o + 1] = pos.getY(i); p[o + 2] = pos.getZ(i);
      n[o] = nrm.getX(i); n[o + 1] = nrm.getY(i); n[o + 2] = nrm.getZ(i);
      o += 3;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(n, 3));
  return g;
}
