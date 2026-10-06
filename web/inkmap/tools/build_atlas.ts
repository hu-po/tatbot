// Build the inklang region atlas for a body: label every skin face with a
// leaf site id + laterality, deterministically, from geometry alone.
//
//   node --experimental-strip-types tools/build_atlas.ts
//
// Writes public/bodies/<body-id>.regions.json and prints a per-region audit.
// The atlas is data, not code: this tool is one way to author it (geometric
// rules on the exported GLB); a Blender vertex-group pass can replace any
// region later by writing the same file. Frame facts (checked at runtime):
// app frame is Z-up, the body faces -Y, anatomical LEFT is +X (EyeL is at
// +x). Face order matches src/core/body.ts exactly because this tool calls
// the same buildSkin on the same GLB.
import { readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { buildRegionRecords } from "../src/core/atlas.ts";
import {
  BODY_SPEC,
  buildSkin,
  canonicalSurfaceBytes,
  canonicalTopologyBytes,
} from "../src/core/body.ts";
import { sha256Hex } from "../src/core/sha256.ts";
import { INKLANG_VERSION, SITES } from "../src/core/inklang/lexicon.ts";
import { ATLAS_SCHEMA_VERSION } from "../src/core/inklang/types.ts";

const POSE_CATALOG = JSON.parse(
  readFileSync(new URL("../../../config/inkmap/body-poses.json", import.meta.url), "utf8"),
);
const ELIGIBILITY = JSON.parse(
  readFileSync(new URL("../../../config/body-models/mhr-soma-v1/eligibility.json", import.meta.url), "utf8"),
);
const SUPPORTED_SITES = new Set<string>(ELIGIBILITY.initial_supported_domain.sites);

const spec = BODY_SPEC;
const bytes = readFileSync(new URL(`../public/${spec.path}`, import.meta.url));
const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const gltf = await new GLTFLoader().parseAsync(buf, "");
const skin = buildSkin(gltf.scene);
const pos = skin.geometry.getAttribute("position") as THREE.BufferAttribute;
const nFaces = pos.count / 3;
const bodyFaces = nFaces;

// Per-face centroid and flat normal, in the app (Z-up) frame.
const C = new Float32Array(nFaces * 3);
const N = new Float32Array(nFaces * 3);
{
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3(), n = new THREE.Vector3();
  for (let f = 0; f < nFaces; f++) {
    a.fromBufferAttribute(pos, 3 * f); b.fromBufferAttribute(pos, 3 * f + 1); c.fromBufferAttribute(pos, 3 * f + 2);
    C[3 * f] = (a.x + b.x + c.x) / 3; C[3 * f + 1] = (a.y + b.y + c.y) / 3; C[3 * f + 2] = (a.z + b.z + c.z) / 3;
    n.copy(ab.subVectors(b, a)).cross(ac.subVectors(c, a)).normalize();
    N[3 * f] = n.x; N[3 * f + 1] = n.y; N[3 * f + 2] = n.z;
  }
}
const minZ = skin.bbox.min.z;
const cx = (f: number) => C[3 * f], cy = (f: number) => C[3 * f + 1], cz = (f: number) => C[3 * f + 2] - minZ;
const nx = (f: number) => N[3 * f], ny = (f: number) => N[3 * f + 1], nz = (f: number) => N[3 * f + 2];

// Frame sanity for the reviewed Tatbot axes: left is +X, front is -Y, up is +Z.
{
  if (!(skin.bbox.max.x > 0 && skin.bbox.min.x < 0 && skin.bbox.max.z > skin.bbox.min.z)) {
    throw new Error("frame check: left/right or up sentinel failed");
  }
  const topFaces = Array.from({ length: nFaces }, (_, face) => face).filter((face) => cz(face) > 0.82 * (skin.bbox.max.z - minZ));
  if (Math.min(...topFaces.map(cy)) >= 0) throw new Error("frame check: facial surface is not toward -y");
}

const H = skin.bbox.max.z - minZ;
const eyeZ = 0.91 * H;

// Stable semantic landmarks for the reviewed reference identity. Fractions
// are checked against the generated joint audit, not inherited mesh vertices.
const crotchZ = 0.55 * H;
const armpitZ = 0.82 * H;
const shoulderHalf = Math.max(Math.abs(skin.bbox.min.x), Math.abs(skin.bbox.max.x));
const neckZ = 0.85 * H;
const chinZ = 0.88 * H;
const isArm = (f: number): "left" | "right" | null => {
  if (cz(f) < 0.74 * H || cz(f) > 0.87 * H || Math.abs(cx(f)) < 0.09 * H) return null;
  if (cx(f) > 0) return "left";
  if (cx(f) < 0) return "right";
  return null;
};

// Face adjacency (shared edge, keyed by rounded vertex positions): the arm
// bends at the elbow, so a straight-chord parameter under-measures the distal
// end — the arm coordinate must be geodesic along the surface.
const adj: number[][] = Array.from({ length: nFaces }, () => []);
{
  const edgeOwner = new Map<string, number>();
  const vkey = (i: number) => `${Math.round(pos.getX(i) * 1e5)},${Math.round(pos.getY(i) * 1e5)},${Math.round(pos.getZ(i) * 1e5)}`;
  for (let f = 0; f < nFaces; f++) {
    const k = [vkey(3 * f), vkey(3 * f + 1), vkey(3 * f + 2)];
    for (let e = 0; e < 3; e++) {
      const ek = [k[e], k[(e + 1) % 3]].sort().join("|");
      const other = edgeOwner.get(ek);
      if (other === undefined) edgeOwner.set(ek, f);
      else if (other !== f) { adj[f].push(other); adj[other].push(f); }
    }
  }
}

/** Multi-source Dijkstra over face centroids within an allowed face set. */
function geodesic(sources: number[], allowed: Set<number>): Map<number, number> {
  const dist = new Map<number, number>();
  // Binary min-heap of [d, face].
  const heap: [number, number][] = [];
  const push = (d: number, f: number) => {
    heap.push([d, f]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]]; i = p;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0], last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]]; i = m;
      }
    }
    return top;
  };
  for (const s of sources) { dist.set(s, 0); push(0, s); }
  while (heap.length) {
    const [d, f] = pop();
    if (d > (dist.get(f) ?? Infinity)) continue;
    for (const g of adj[f]) {
      if (!allowed.has(g)) continue;
      const w = Math.hypot(cx(g) - cx(f), cy(g) - cy(f), cz(g) - cz(f));
      const nd = d + w;
      if (nd < (dist.get(g) ?? Infinity)) { dist.set(g, nd); push(nd, g); }
    }
  }
  return dist;
}

// Arm coordinate per side: t = geodesic distance from the shoulder end,
// normalized by the arm's own maximum (the fingertip).
function armParam(side: "left" | "right") {
  const faces: number[] = [];
  for (let f = 0; f < bodyFaces; f++) if (isArm(f) === side) faces.push(f);
  const allowed = new Set(faces);
  // The band-cut arm set can carry disconnected slivers near the armpit; the
  // shoulder seed must come from the arm's LARGEST connected component, or
  // the geodesic never leaves the sliver.
  const compOf = new Map<number, number>();
  let nComps = 0;
  for (const f of faces) {
    if (compOf.has(f)) continue;
    const id = nComps++;
    const stack = [f];
    compOf.set(f, id);
    while (stack.length) {
      const g = stack.pop()!;
      for (const h of adj[g]) if (allowed.has(h) && !compOf.has(h)) { compOf.set(h, id); stack.push(h); }
    }
  }
  const sizes = new Array(nComps).fill(0);
  for (const id of compOf.values()) sizes[id]++;
  const mainId = sizes.indexOf(Math.max(...sizes));
  const main = faces.filter((f) => compOf.get(f) === mainId);
  const shoulderX = Math.min(...main.map((f) => Math.abs(cx(f))));
  const sources = main.filter((f) => Math.abs(cx(f)) < shoulderX + 0.02 * H);
  const dist = geodesic(sources, allowed);
  let max = 0;
  for (const d of dist.values()) max = Math.max(max, d);
  if (!(max > 0)) throw new Error(`arm ${side}: geodesic collapsed (0 reach)`);
  const t = new Map<number, number>();
  for (const f of main) t.set(f, dist.get(f)! / max);
  // Stray-sliver faces borrow the parameter of the euclidean-nearest main face.
  for (const f of faces) {
    if (t.has(f)) continue;
    let best = main[0], bd = Infinity;
    for (const g of main) {
      const d = (cx(f) - cx(g)) ** 2 + (cy(f) - cy(g)) ** 2 + (cz(f) - cz(g)) ** 2;
      if (d < bd) { bd = d; best = g; }
    }
    t.set(f, t.get(best)!);
  }
  return { faces, t };
}
const arms = { left: armParam("left"), right: armParam("right") };

// ---------------------------------------------------------------------------
// Joint calibration (inklang v0.3): joints are GIRTH MINIMA, not guessed
// fractions — the arm narrows at the elbow between the bicep and forearm
// bulges and bottoms out at the wrist; the leg at the knee and ankle. The
// v0.2 guessed bands put "elbow" ~14 cm distal of the real elbow.
function faceArea(f: number): number {
  const ax = pos.getX(3 * f), ay = pos.getY(3 * f), az = pos.getZ(3 * f);
  const bx1 = pos.getX(3 * f + 1) - ax, by1 = pos.getY(3 * f + 1) - ay, bz1 = pos.getZ(3 * f + 1) - az;
  const cx1 = pos.getX(3 * f + 2) - ax, cy1 = pos.getY(3 * f + 2) - ay, cz1 = pos.getZ(3 * f + 2) - az;
  const ix = by1 * cz1 - bz1 * cy1, iy = bz1 * cx1 - bx1 * cz1, iz = bx1 * cy1 - by1 * cx1;
  return 0.5 * Math.hypot(ix, iy, iz);
}

/** Area-per-bin girth profile over `coord(f)` ∈ [lo,hi]; returns the coord of
 *  the most PROMINENT local minimum inside the window — overall limb taper
 *  makes the plain minimum useless (it just finds a window edge). */
function girthMin(faces: number[], coord: (f: number) => number, lo: number, hi: number, winLo: number, winHi: number): number | null {
  const BINS = 60;
  const area = new Array(BINS).fill(0);
  for (const f of faces) {
    const t = (coord(f) - lo) / (hi - lo);
    if (t < 0 || t >= 1) continue;
    area[Math.floor(t * BINS)] += faceArea(f);
  }
  const sm = area.map((_, i) => (area[Math.max(0, i - 1)] + area[i] + area[Math.min(BINS - 1, i + 1)]) / 3);
  const meanA = sm.reduce((s, v) => s + v, 0) / BINS;
  let best = -1, bScore = 0;
  for (let i = 2; i < BINS - 2; i++) {
    const c = lo + ((i + 0.5) / BINS) * (hi - lo);
    if (c < winLo || c > winHi || sm[i] <= 0) continue;
    if (sm[i] > sm[i - 1] || sm[i] > sm[i + 1]) continue;
    let maxL = 0, maxR = 0;
    for (let k = Math.max(0, i - 8); k < i; k++) maxL = Math.max(maxL, sm[k]);
    for (let k = i + 1; k <= Math.min(BINS - 1, i + 8); k++) maxR = Math.max(maxR, sm[k]);
    const score = Math.min(maxL, maxR) - sm[i];
    if (score > bScore) { bScore = score; best = i; }
  }
  if (best < 0 || bScore < 0.04 * meanA) return null;
  return lo + ((best + 0.5) / BINS) * (hi - lo);
}

/** Where a joint actually is: its sculpted CREASE. Concave dihedral edges
 *  (a neighbour's centroid in front of this face's plane) concentrate in the
 *  elbow ditch and the back of the knee on any humanoid mesh, even a smooth
 *  stylized one where girth and centreline bends are ambiguous. Returns the
 *  coord of the most concave bin in the window. */
function creasePoint(faces: number[], coord: (f: number) => number, lo: number, hi: number, winLo: number, winHi: number): number | null {
  const BINS = 48;
  const conc = new Array(BINS).fill(0);
  const inSet = new Set(faces);
  for (const f of faces) {
    const t = (coord(f) - lo) / (hi - lo);
    if (t < 0 || t >= 1) continue;
    let c = 0;
    for (const g of adj[f]) {
      if (!inSet.has(g)) continue;
      const d = (cx(g) - cx(f)) * nx(f) + (cy(g) - cy(f)) * ny(f) + (cz(g) - cz(f)) * nz(f);
      if (d > 0) c += d; // concave: the neighbour sits in front of this face
    }
    conc[Math.floor(t * BINS)] += c;
  }
  const sm = conc.map((_, i) => (conc[Math.max(0, i - 1)] + conc[i] + conc[Math.min(BINS - 1, i + 1)]) / 3);
  let best = -1, bv = 0;
  for (let i = 0; i < BINS; i++) {
    const c = lo + ((i + 0.5) / BINS) * (hi - lo);
    if (c < winLo || c > winHi) continue;
    if (sm[i] > bv) { bv = sm[i]; best = i; }
  }
  return best < 0 || bv <= 0 ? null : lo + ((best + 0.5) / BINS) * (hi - lo);
}

const armJoints = { left: { elbowT: 0.45, wristT: 0.80 }, right: { elbowT: 0.45, wristT: 0.80 } };
for (const side of ["left", "right"] as const) {
  const a = arms[side];
  const tOf = (f: number) => a.t.get(f)!;
  // Anatomical prior fractions of the shoulder→fingertip path bound the
  // search tightly; the crease decides inside them. (A loose window latches
  // onto the forearm muscle concavity — verified against a render: the
  // visible elbow sits at t≈0.43 of this mesh's arm path.)
  const elbowT = creasePoint(a.faces, tOf, 0, 1, 0.38, 0.50);
  if (elbowT !== null) armJoints[side].elbowT = elbowT;
  const e = armJoints[side].elbowT;
  const wristT = creasePoint(a.faces, tOf, 0, 1, e + 0.18, 0.90) ?? girthMin(a.faces, tOf, 0, 1, e + 0.18, 0.90);
  armJoints[side].wristT = wristT ?? e + 0.62 * (1 - e);
}

const legFaces: number[] = [];
for (let f = 0; f < bodyFaces; f++) if (cz(f) < crotchZ && isArm(f) === null) legFaces.push(f);
// Anatomical priors bound the search: knee ~0.285·H, ankle ~0.055·H.
const kneeZ = creasePoint(legFaces, cz, 0, crotchZ, 0.22 * H, 0.35 * H) ?? girthMin(legFaces, cz, 0, crotchZ, 0.24 * H, 0.33 * H) ?? 0.285 * H;
const ankleZ = creasePoint(legFaces, cz, 0, crotchZ, 0.03 * H, 0.11 * H) ?? girthMin(legFaces, cz, 0, crotchZ, 0.03 * H, 0.10 * H) ?? 0.055 * H;
console.log(`joints: elbowT L=${armJoints.left.elbowT.toFixed(3)} R=${armJoints.right.elbowT.toFixed(3)}, wristT L=${armJoints.left.wristT.toFixed(3)} R=${armJoints.right.wristT.toFixed(3)}, kneeZ=${kneeZ.toFixed(3)}, ankleZ=${ankleZ.toFixed(3)}`);

// Torso half-width just below the armpit, measured directly on non-arm faces
// — the shoulder span includes the arms and over-shoots side-of-torso
// thresholds, and cluster picking is too brittle for this number.
const torsoHalf = (() => {
  let m = 0;
  for (let f = 0; f < bodyFaces; f++) {
    const z = cz(f);
    if (z > armpitZ - 0.06 * H && z < armpitZ - 0.03 * H && isArm(f) === null) m = Math.max(m, Math.abs(cx(f)));
  }
  return m;
})();

// The most forward point of the head (the nose tip's y), for nose carving.
let headFrontY = Infinity;
for (let f = 0; f < bodyFaces; f++) if (cz(f) > chinZ) headFrontY = Math.min(headFrontY, cy(f));

// ---------------------------------------------------------------------------
// Per-face labels. Empty = unlabeled (eyes stay unlabeled).
const label: (string | null)[] = new Array(nFaces).fill(null);
const lat: ("left" | "right" | null)[] = new Array(nFaces).fill(null);
const sideOf = (f: number): "left" | "right" => (cx(f) >= 0 ? "left" : "right");
const torsoT = (z: number) => (z - crotchZ) / (neckZ - crotchZ);

for (let f = 0; f < bodyFaces; f++) {
  const z = cz(f);
  const arm = isArm(f);

  if (arm) {
    const t = arms[arm].t.get(f)!;
    const { elbowT, wristT } = armJoints[arm];
    const front = ny(f) < 0 || cy(f) < -0.02 * H; // anterior of a hanging arm
    const handRel = (t - (wristT + 0.035)) / (1 - wristT - 0.035);
    let s: string;
    if (t < 0.06) s = "shoulder_cap";
    else if (t < elbowT - 0.05) s = front ? "bicep" : "tricep";
    else if (t < elbowT + 0.05) s = front ? "elbow_ditch" : "elbow";
    else if (t < wristT - 0.035) s = "forearm";
    else if (t < wristT + 0.035) s = "wrist";
    else if (handRel < 0.5) {
      // Hand: the rest surface is a T-pose with palms down, so the palm is
      // the -z face of the hand and the back of the hand faces +z.
      const out = nz(f);
      s = out < -0.35 ? "palm" : "hand";
    } else if (handRel < 0.65) {
      const out = nz(f);
      s = out > 0.35 ? "knuckles" : handRel > 0.58 ? "fingers" : (out < -0.35 ? "palm" : "hand");
    } else s = "fingers";
    label[f] = s; lat[f] = arm;
    continue;
  }

  if (z < crotchZ) {
    // Legs (anything not an arm below the crotch).
    const side = sideOf(f);
    let s: string;
    const inward = side === "left" ? nx(f) < -0.45 : nx(f) > 0.45;
    if (z < ankleZ - 0.02 * H) {
      // Foot: toes forward (-y), heel back, sole down, top up, arch inboard.
      if (nz(f) < -0.4) s = "sole";
      else if (cy(f) < -0.055 * H) s = "toes";
      else if (ny(f) > 0.35 && nz(f) < 0.35) s = "heel";
      else if (inward && nz(f) < 0.35) s = "instep";
      else s = "foot_top";
    } else if (z < ankleZ + 0.06 * H && ny(f) > 0.4 && z > ankleZ - 0.02 * H) s = "achilles";
    else if (z < ankleZ + 0.022 * H) s = "ankle";
    else if (z < kneeZ - 0.032 * H) s = ny(f) > 0.15 ? "calf" : "shin";
    else if (z < kneeZ + 0.032 * H) s = ny(f) > 0.15 ? "knee_ditch" : "knee";
    else s = "thigh";
    label[f] = s; lat[f] = side;
    continue;
  }

  if (z < neckZ) {
    // Torso. front = -y normals, back = +y, side = dominant |nx|.
    const tt = torsoT(f === -1 ? 0 : z);
    const side = sideOf(f);
    const front = ny(f) < -0.25;
    const back = ny(f) > 0.25;
    const lateral = Math.abs(nx(f)) > 0.75 && !front && !back;
    const nearMid = Math.abs(cx(f)) < 0.035 * H;
    let s: string; let sd: "left" | "right" | null = null;
    if (z >= armpitZ) {
      // Shoulder line: traps on top, caps outboard, blades and upper back
      // behind. In front, only the strip under the neck is collarbone — the
      // rest is still the chest panel, or it ends up with a 30-face "chest".
      if (nz(f) > 0.55 && Math.abs(cx(f)) < 0.6 * shoulderHalf && Math.abs(cx(f)) > 0.12 * shoulderHalf) { s = "traps"; sd = side; }
      else if (Math.abs(cx(f)) > 0.55 * shoulderHalf) { s = "shoulder_cap"; sd = side; }
      else if (front || ny(f) < 0) {
        if (z > neckZ - 0.045 * H) { s = "collarbone"; sd = side; }
        else {
          const ax = Math.abs(cx(f));
          s = ax < 0.18 * torsoHalf ? "sternum" : ax > 0.62 * torsoHalf ? "pec" : "chest";
          sd = s === "pec" ? side : null;
        }
      }
      else if (Math.abs(cx(f)) > 0.4 * shoulderHalf) { s = "shoulder_blade"; sd = side; }
      else { s = "upper_back"; }
    } else if (Math.abs(cx(f)) > 0.82 * torsoHalf && z > armpitZ - 0.035 * H && Math.abs(ny(f)) < 0.6) {
      s = "armpit"; sd = side;
    } else if (back || (!front && !lateral && cy(f) > 0)) {
      if (tt < 0.28 && Math.abs(cx(f)) < 0.28 * torsoHalf) s = "sacrum";
      else if (tt < 0.22) { s = "buttock"; sd = side; }
      else if (nearMid) s = "spine";
      else if (tt < 0.45) s = "lower_back";
      else if (tt < 0.62) s = "mid_back";
      else { s = Math.abs(cx(f)) > 0.38 * shoulderHalf ? "shoulder_blade" : "upper_back"; sd = s === "shoulder_blade" ? side : null; }
    } else if (lateral && tt > 0.3 && tt < 0.85) {
      // Side of the torso: obliques low, ribs high.
      s = tt < 0.5 ? "obliques" : "ribs"; sd = side;
    } else {
      // Front column.
      if (tt < 0.14) { s = Math.abs(cx(f)) < 0.16 * shoulderHalf ? "groin" : "hip"; sd = s === "hip" ? side : null; }
      else if (tt < 0.30) { s = Math.abs(cx(f)) > 0.72 * shoulderHalf ? "hip" : "stomach"; sd = s === "hip" ? side : null; }
      else if (tt < 0.52) {
        s = tt >= 0.30 && tt < 0.40 && Math.abs(cx(f)) < 0.14 * torsoHalf ? "navel" : "stomach";
      }
      else if (tt < 0.60) { s = Math.abs(cx(f)) < 0.3 * torsoHalf ? "sternum" : "underboob"; }
      else if (tt < 0.88) {
        // "Chest" is the panel people point at: keep the sternum a narrow
        // midline strip and the pecs the outer plates, chest the bulk between.
        const ax = Math.abs(cx(f));
        s = ax < 0.18 * torsoHalf ? "sternum" : ax > 0.62 * torsoHalf ? "pec" : "chest";
        sd = s === "pec" ? side : null;
      }
      else { s = "collarbone"; sd = side; }
    }
    label[f] = s; lat[f] = sd;
    continue;
  }

  if (z < chinZ) {
    // Neck: throat in front, nape behind, "neck" the sides.
    label[f] = ny(f) < -0.35 ? "throat" : ny(f) > 0.35 ? "nape" : "neck";
    continue;
  }

  // Head.
  {
    const side = sideOf(f);
    const headHalf = 0.055 * H;
    const front = ny(f) < -0.2 && cy(f) < 0;
    const lateralPos = Math.abs(cx(f)) > 0.62 * headHalf;
    const earBand = z < eyeZ + 0.02 * H && z > chinZ + 0.01 * H;
    let s: string; let sd: "left" | "right" | null = null;
    if (lateralPos && earBand && Math.abs(nx(f)) > 0.5) {
      s = cy(f) > 0.015 * H ? "behind_ear" : z < eyeZ - 0.035 * H ? "ear_lobe" : "ear"; sd = side;
    } else if (Math.abs(cx(f)) > 0.5 * headHalf && z >= eyeZ && z < eyeZ + 0.06 * H && cy(f) < 0.01 * H && Math.abs(nx(f)) > 0.35) {
      s = "temple"; sd = side;
    } else if (z < chinZ + 0.22 * (H - chinZ)) {
      if (front) { s = Math.abs(cx(f)) < 0.3 * headHalf ? "chin" : "jaw"; sd = s === "jaw" ? side : null; }
      else s = "jaw"; if (s === "jaw") sd = side;
    } else if (front && z < eyeZ + 0.35 * (H - eyeZ)) {
      if (Math.abs(cx(f)) < 0.15 * headHalf && z >= eyeZ - 0.045 * H && z < eyeZ + 0.008 * H && cy(f) < headFrontY + 0.012 * H) s = "nose";
      else if (z >= eyeZ + 0.008 * H && z < eyeZ + 0.03 * H && Math.abs(cx(f)) < 0.55 * headHalf) { s = "eyebrow"; sd = side; }
      else if (z >= eyeZ + 0.03 * H) s = "forehead";
      else { s = Math.abs(cx(f)) > 0.4 * headHalf && z < eyeZ ? "cheek" : "face"; sd = s === "cheek" ? side : null; }
    } else s = "scalp";
    label[f] = s; lat[f] = sd;
  }
}

// Thumb: within each hand, the sub-mass reaching forward of the hand's mean.
for (const side of ["left", "right"] as const) {
  const handSites = new Set(["hand", "palm", "knuckles", "fingers"]);
  const hf: number[] = [];
  for (let f = 0; f < bodyFaces; f++) if (lat[f] === side && label[f] !== null && handSites.has(label[f]!)) hf.push(f);
  if (hf.length === 0) continue;
  const meanY = hf.reduce((s2, f) => s2 + cy(f), 0) / hf.length;
  for (const f of hf) if (cy(f) < meanY - 0.022 * H) label[f] = "thumb";
}

// ---------------------------------------------------------------------------
// Clean the boundaries. Per-face normal/threshold rules leave salt-and-pepper
// edges and stray islands (a "ribs" face in the middle of the stomach). Two
// passes fix it without moving real boundaries far: (1) majority relaxation —
// a face whose neighbours mostly belong to another region joins it; (2)
// island absorption — each region keeps its largest connected component per
// side, smaller islands take their surrounding region's label. A size floor
// keeps every site alive.
const keyOf = (f: number) => (label[f] === null ? null : `${label[f]}|${lat[f] ?? ""}`);
{
  const counts0 = new Map<string, number>();
  for (let f = 0; f < bodyFaces; f++) { const k = keyOf(f); if (k) counts0.set(k, (counts0.get(k) ?? 0) + 1); }
  const floorOf = new Map([...counts0].map(([k, n]) => [k, Math.max(12, Math.floor(0.25 * n))]));
  const size = new Map(counts0);
  for (let iter = 0; iter < 4; iter++) {
    let changed = 0;
    for (let f = 0; f < bodyFaces; f++) {
      const own = keyOf(f);
      if (!own) continue;
      const tally = new Map<string, number>();
      for (const g of adj[f]) { if (g >= bodyFaces) continue; const k = keyOf(g); if (k) tally.set(k, (tally.get(k) ?? 0) + 1); }
      let best: string | null = null, bn = 0;
      for (const [k, n] of tally) if (n > bn) { best = k; bn = n; }
      if (best !== null && best !== own && bn >= 2 && (size.get(own) ?? 0) > (floorOf.get(own) ?? 12)) {
        const [s, l] = best.split("|");
        label[f] = s; lat[f] = l === "" ? null : (l as "left" | "right");
        size.set(own, size.get(own)! - 1); size.set(best, (size.get(best) ?? 0) + 1);
        changed++;
      }
    }
    if (changed === 0) break;
  }
  // Islands: connected components per (site, side); absorb all but the largest.
  const comp = new Int32Array(bodyFaces).fill(-1);
  const comps: { key: string; faces: number[] }[] = [];
  for (let f = 0; f < bodyFaces; f++) {
    if (comp[f] >= 0 || keyOf(f) === null) continue;
    const key = keyOf(f)!;
    const faces: number[] = [];
    const stack = [f];
    comp[f] = comps.length;
    while (stack.length) {
      const g = stack.pop()!;
      faces.push(g);
      for (const h of adj[g]) if (h < bodyFaces && comp[h] < 0 && keyOf(h) === key) { comp[h] = comps.length; stack.push(h); }
    }
    comps.push({ key, faces });
  }
  const largest = new Map<string, number>();
  for (const [i, c] of comps.entries()) {
    const cur = largest.get(c.key);
    if (cur === undefined || comps[cur].faces.length < c.faces.length) largest.set(c.key, i);
  }
  let absorbed = 0;
  for (const [i, c] of comps.entries()) {
    if (largest.get(c.key) === i) continue;
    const tally = new Map<string, number>();
    for (const f of c.faces) {
      for (const g of adj[f]) {
        if (g >= bodyFaces) continue;
        const k = keyOf(g);
        if (k && k !== c.key) tally.set(k, (tally.get(k) ?? 0) + 1);
      }
    }
    let best: string | null = null, bn = 0;
    for (const [k, n] of tally) if (n > bn) { best = k; bn = n; }
    if (best === null) continue;
    const [s, l] = best.split("|");
    for (const f of c.faces) { label[f] = s; lat[f] = l === "" ? null : (l as "left" | "right"); }
    absorbed += c.faces.length;
  }
  console.log(`smoothing: ${comps.length} components → ${largest.size} kept, ${absorbed} island faces absorbed`);
}

// ---------------------------------------------------------------------------
// Audit + write.
const exclusionBytes = readFileSync(new URL(`../public/${POSE_CATALOG.exclusion_asset.path}`, import.meta.url));
if (
  exclusionBytes.length !== nFaces
  || await sha256Hex(exclusionBytes.buffer.slice(exclusionBytes.byteOffset, exclusionBytes.byteOffset + exclusionBytes.byteLength)) !== POSE_CATALOG.exclusion_asset.sha256
) throw new Error("upstream exclusion asset does not match the pose catalog");
for (let face = 0; face < nFaces; face += 1) {
  if (exclusionBytes[face] === 1) {
    label[face] = null;
    lat[face] = null;
  } else if (exclusionBytes[face] !== 0) {
    throw new Error(`invalid exclusion byte on face ${face}`);
  }
}
const counts = new Map<string, { n: number; left: number; right: number; cz: number }>();
for (let f = 0; f < nFaces; f++) {
  const s = label[f];
  if (!s) continue;
  const e = counts.get(s) ?? { n: 0, left: 0, right: 0, cz: 0 };
  e.n++; e.cz += cz(f);
  if (lat[f] === "left") e.left++;
  if (lat[f] === "right") e.right++;
  counts.set(s, e);
}
const missing = Object.keys(SITES).filter((s) => !counts.has(s));
console.log(`# ${spec.id}: torsoHalf=${torsoHalf.toFixed(3)} shoulderHalf=${shoulderHalf.toFixed(3)} H=${H.toFixed(3)} crotch=${crotchZ.toFixed(3)} armpit=${armpitZ.toFixed(3)} neck=${neckZ.toFixed(3)} chin=${chinZ.toFixed(3)} eye=${eyeZ.toFixed(3)}`);
for (const [s, e] of [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
  const latRule = SITES[s]?.laterality;
  const balance = e.left + e.right > 0 ? ` L${e.left}/R${e.right}` : "";
  console.log(`${s.padEnd(15)} ${String(e.n).padStart(5)} faces  z̄=${(e.cz / e.n).toFixed(3)}${balance}${latRule === "sided" && (e.left === 0 || e.right === 0) ? "  !! sided but one-sided" : ""}`);
}
if (missing.length) console.log("MISSING:", missing.join(", "));
const unlabeled = label.slice(0, bodyFaces).filter((l) => l === null).length;
console.log(`unlabeled body faces: ${unlabeled}, eye faces: ${nFaces - bodyFaces}`);

const siteIds = Object.keys(SITES).sort();
const siteIndex = new Map(siteIds.map((s, i) => [s, i]));
const faces = new Array<number>(nFaces);
for (let f = 0; f < nFaces; f++) {
  const s = label[f];
  faces[f] = s === null ? -1 : siteIndex.get(s)! * 4 + (lat[f] === "left" ? 1 : lat[f] === "right" ? 2 : 0);
}
const eligibleFaces = faces.map((code) => (
  code >= 0 && SUPPORTED_SITES.has(siteIds[code >> 2]) ? 1 : 0
));
const toolBytes = readFileSync(new URL(import.meta.url));
const toolBuffer = toolBytes.buffer.slice(toolBytes.byteOffset, toolBytes.byteOffset + toolBytes.byteLength);
const out = {
  atlas_schema_version: ATLAS_SCHEMA_VERSION,
  inklang_version: INKLANG_VERSION,
  body: {
    model_spec_id: spec.id,
    model_spec_sha256: spec.modelSpecSha256,
    identity_sha256: spec.identitySha256,
    topology_sha256: await sha256Hex(canonicalTopologyBytes(skin.geometry)),
    rest_surface_sha256: await sha256Hex(canonicalSurfaceBytes(skin.geometry)),
    asset_sha256: await sha256Hex(buf),
  },
  builder: {
    name: "web/inkmap/tools/build_atlas.ts",
    version: 2,
    source_sha256: await sha256Hex(toolBuffer),
  },
  frame: { front: "-y", left: "+x", up: "+z" },
  encoding: "faces[i] = siteIndex*4 + laterality (0 none, 1 left, 2 right); -1 = not skin",
  sites: siteIds,
  site_status: Object.fromEntries(siteIds.map((site) => [
    site,
    counts.has(site) && SUPPORTED_SITES.has(site)
      ? { status: "mapped_supported", face_count: counts.get(site)!.n }
      : counts.has(site)
        ? { status: "mapped_unsupported", face_count: counts.get(site)!.n, reason: "outside the initial reviewed tattoo domain" }
      : { status: "unsupported", face_count: 0, reason: "no eligible SOMA faces after upstream exclusions" },
  ])),
  region_meaning_count: Object.values(SITES).reduce((total, site) => total + (site.laterality === "sided" ? 2 : 1), 0),
  upstream_exclusions: {
    asset_sha256: POSE_CATALOG.exclusion_asset.sha256,
    segments: POSE_CATALOG.exclusion_asset.source_segments,
    excluded_faces: POSE_CATALOG.exclusion_asset.excluded_faces,
  },
  faces,
  eligible_faces: eligibleFaces,
  regions: buildRegionRecords(siteIds, faces, skin.geometry, skin.centroids),
};
const dest = new URL(`../public/bodies/${spec.id}.regions.json`, import.meta.url);
writeFileSync(dest, JSON.stringify(out));
console.log(`wrote ${dest.pathname} (${counts.size}/${siteIds.length} sites mapped, 99 meanings declared)`);
