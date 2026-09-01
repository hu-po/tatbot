// A placed tattoo is a *surface anchor*, not pixels: a face index plus
// barycentric weights on the body's skin geometry. Everything else — the
// point, the normal, the tangent frame the design is drawn in — is derived
// here, deterministically, from the geometry.
//
// Frame convention (Z-up scene, metres):
//   n = interpolated surface normal (outward)
//   v = body up-vector projected onto the tangent plane, rotated by
//       `rotationRad` about n  -> the design's +Y ("up" in the SVG)
//   u = v × n                 -> the design's +X, so (u, v, n) is right-handed
// Only the scalar rotation is stored; the frame is recomputed on every load so
// a re-projected anchor still yields a valid, upright frame.
import * as THREE from "three";

export type Barycentric = [number, number, number];

export interface Anchor {
  face: number;
  barycentric: Barycentric;
}

export interface Frame {
  p: THREE.Vector3;
  n: THREE.Vector3;
  u: THREE.Vector3;
  v: THREE.Vector3;
}

export const BODY_UP = new THREE.Vector3(0, 0, 1);

export function faceCount(g: THREE.BufferGeometry): number {
  const idx = g.getIndex();
  return idx ? idx.count / 3 : g.getAttribute("position").count / 3;
}

export function faceVertexIndices(g: THREE.BufferGeometry, face: number): [number, number, number] {
  const idx = g.getIndex();
  const i = face * 3;
  return idx ? [idx.getX(i), idx.getX(i + 1), idx.getX(i + 2)] : [i, i + 1, i + 2];
}

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _bc = new THREE.Vector3();

/** Barycentric weights of `point` (geometry-local) on `face`, clamped to the triangle and renormalised. */
export function pointToAnchor(g: THREE.BufferGeometry, face: number, point: THREE.Vector3): Anchor {
  if (!Number.isInteger(face) || face < 0 || face >= faceCount(g)) throw new RangeError(`face ${face} out of range`);
  const pos = g.getAttribute("position");
  const [ia, ib, ic] = faceVertexIndices(g, face);
  _a.fromBufferAttribute(pos, ia);
  _b.fromBufferAttribute(pos, ib);
  _c.fromBufferAttribute(pos, ic);
  THREE.Triangle.getBarycoord(point, _a, _b, _c, _bc);
  let w0 = Math.max(0, _bc.x);
  let w1 = Math.max(0, _bc.y);
  let w2 = Math.max(0, _bc.z);
  const s = w0 + w1 + w2 || 1;
  w0 /= s; w1 /= s; w2 /= s;
  return { face, barycentric: [w0, w1, w2] };
}

/** Surface point and (interpolated, normalised) normal at an anchor. */
export function anchorToPoint(g: THREE.BufferGeometry, anchor: Anchor): { p: THREE.Vector3; n: THREE.Vector3 } {
  const pos = g.getAttribute("position");
  const nrm = g.getAttribute("normal");
  if (!nrm) throw new Error("geometry has no normals — call computeSmoothNormals first");
  const [ia, ib, ic] = faceVertexIndices(g, anchor.face);
  const [w0, w1, w2] = anchor.barycentric;
  const p = new THREE.Vector3()
    .addScaledVector(_a.fromBufferAttribute(pos, ia), w0)
    .addScaledVector(_b.fromBufferAttribute(pos, ib), w1)
    .addScaledVector(_c.fromBufferAttribute(pos, ic), w2);
  const n = new THREE.Vector3()
    .addScaledVector(_a.fromBufferAttribute(nrm, ia), w0)
    .addScaledVector(_b.fromBufferAttribute(nrm, ib), w1)
    .addScaledVector(_c.fromBufferAttribute(nrm, ic), w2);
  if (n.lengthSq() < 1e-12) {
    // Degenerate interpolation (opposing normals): fall back to the face normal.
    THREE.Triangle.getNormal(_a.fromBufferAttribute(pos, ia), _b.fromBufferAttribute(pos, ib), _c.fromBufferAttribute(pos, ic), n);
  }
  n.normalize();
  return { p, n };
}

/** The design's drawing frame at an anchor. */
export function frameAt(g: THREE.BufferGeometry, anchor: Anchor, rotationRad = 0, up: THREE.Vector3 = BODY_UP): Frame {
  const { p, n } = anchorToPoint(g, anchor);
  const v = up.clone().addScaledVector(n, -up.dot(n));
  if (v.lengthSq() < 1e-8) {
    // Surface is horizontal (top of the head, sole of the foot): any tangent is "up".
    v.set(1, 0, 0).addScaledVector(n, -n.x);
    if (v.lengthSq() < 1e-8) v.set(0, 1, 0).addScaledVector(n, -n.y);
  }
  v.normalize().applyAxisAngle(n, rotationRad);
  const u = new THREE.Vector3().crossVectors(v, n).normalize();
  return { p, n, u, v };
}

/**
 * Smooth vertex normals keyed by *position*, so a non-indexed mesh (which is
 * what Loop subdivision hands back) still gets a continuous normal field across
 * the seams that per-face UVs force into the vertex buffer.
 */
export function computeSmoothNormals(g: THREE.BufferGeometry, weld = 1e-6): void {
  const pos = g.getAttribute("position");
  const n = pos.count;
  const acc = new Map<string, THREE.Vector3>();
  const key = (i: number) => {
    const q = 1 / weld;
    return `${Math.round(pos.getX(i) * q)},${Math.round(pos.getY(i) * q)},${Math.round(pos.getZ(i) * q)}`;
  };
  const keys = new Array<string>(n);
  for (let i = 0; i < n; i++) keys[i] = key(i);
  const fn = new THREE.Vector3();
  for (let f = 0; f < faceCount(g); f++) {
    const [ia, ib, ic] = faceVertexIndices(g, f);
    _a.fromBufferAttribute(pos, ia);
    _b.fromBufferAttribute(pos, ib);
    _c.fromBufferAttribute(pos, ic);
    // Cross product is area-weighted; that is the weighting we want.
    fn.subVectors(_c, _b).cross(_a.clone().sub(_b));
    for (const i of [ia, ib, ic]) {
      const k = keys[i];
      const v = acc.get(k);
      if (v) v.add(fn); else acc.set(k, fn.clone());
    }
  }
  const out = new Float32Array(n * 3);
  const tmp = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    tmp.copy(acc.get(keys[i])!).normalize();
    out[i * 3] = tmp.x; out[i * 3 + 1] = tmp.y; out[i * 3 + 2] = tmp.z;
  }
  g.setAttribute("normal", new THREE.BufferAttribute(out, 3));
}

/** Per-face centroids, packed xyz — precomputed once so decal building can cull to a neighbourhood. */
export function faceCentroids(g: THREE.BufferGeometry): Float32Array {
  const pos = g.getAttribute("position");
  const fc = faceCount(g);
  const out = new Float32Array(fc * 3);
  for (let f = 0; f < fc; f++) {
    const [ia, ib, ic] = faceVertexIndices(g, f);
    out[f * 3] = (pos.getX(ia) + pos.getX(ib) + pos.getX(ic)) / 3;
    out[f * 3 + 1] = (pos.getY(ia) + pos.getY(ib) + pos.getY(ic)) / 3;
    out[f * 3 + 2] = (pos.getZ(ia) + pos.getZ(ib) + pos.getZ(ic)) / 3;
  }
  return out;
}
