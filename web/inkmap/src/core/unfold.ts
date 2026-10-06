/** Deterministic intrinsic unfolding of one connected rest-surface patch.
 *
 * This is the TypeScript counterpart of tatbot_sim.inkmap.surface_trace. Face
 * identity always refers to the canonical SOMA mid surface. Texture UVs are
 * deliberately ignored: metric chart coordinates come only from triangle edge
 * lengths on the immutable rest mesh.
 */
import * as THREE from "three";
import { faceCount, faceVertexIndices, frameAt, type Anchor } from "./anchor.ts";

export const UNFOLD_VERSION = 2;
export const WALK_TOLERANCE = 2e-7;
export const VERTEX_WELD_M = 1e-6;

export type Vec2 = [number, number];

export interface UnfoldedFace {
  face: number;
  triangleUv: [Vec2, Vec2, Vec2];
}

/** Two developments of one mesh edge disagreeing after the walk closed a loop. */
export interface ChartSeam {
  face: number;
  neighbor: number;
  /** Largest chart distance between the two placements of a shared vertex, metres. */
  mismatchM: number;
}

export interface UnfoldedPatch {
  seedFace: number;
  seedTriangleUv: [Vec2, Vec2, Vec2];
  faces: UnfoldedFace[];
  adjacency: Map<number, number[]>;
  /**
   * Every adjacent pair the walk reached along two different paths. On a
   * developable surface both placements agree; on curved skin they differ by
   * the curvature enclosed between the paths (millimetres for a palm-sized
   * patch); once the patch closes around a limb they differ by its girth.
   */
  seams: ChartSeam[];
  /** Included faces that own a mesh boundary edge: the chart ran off the surface there. */
  boundaryFaces: number[];
}

export interface MappedChartPoint {
  anchor: Anchor;
  triangleUv: [Vec2, Vec2, Vec2];
}

const add2 = (a: Vec2, b: Vec2): Vec2 => [a[0] + b[0], a[1] + b[1]];
const sub2 = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
const mul2 = (a: Vec2, value: number): Vec2 => [a[0] * value, a[1] * value];
const norm2 = (a: Vec2): number => Math.hypot(a[0], a[1]);

function vertexKey(point: THREE.Vector3): string {
  return [point.x, point.y, point.z]
    .map((value) => Math.floor(value / VERTEX_WELD_M + 0.5))
    .join(",");
}

function trianglePoints(geometry: THREE.BufferGeometry, face: number): [THREE.Vector3, THREE.Vector3, THREE.Vector3] {
  const position = geometry.getAttribute("position");
  const indices = faceVertexIndices(geometry, face);
  return indices.map((index) => new THREE.Vector3().fromBufferAttribute(position, index)) as [
    THREE.Vector3,
    THREE.Vector3,
    THREE.Vector3,
  ];
}

interface SurfaceTopology {
  keys: [string, string, string][];
  adjacency: number[][];
}

const topologyCache = new WeakMap<THREE.BufferGeometry, SurfaceTopology>();

function topology(geometry: THREE.BufferGeometry): SurfaceTopology {
  const cached = topologyCache.get(geometry);
  if (cached) return cached;
  const count = faceCount(geometry);
  const keys: [string, string, string][] = new Array(count);
  const adjacency: number[][] = Array.from({ length: count }, () => []);
  const edgeFaces = new Map<string, number[]>();
  for (let face = 0; face < count; face++) {
    const points = trianglePoints(geometry, face);
    keys[face] = points.map(vertexKey) as [string, string, string];
    for (const [left, right] of [[0, 1], [1, 2], [2, 0]] as const) {
      const edge = [keys[face][left], keys[face][right]].sort().join("|");
      const owners = edgeFaces.get(edge) ?? [];
      owners.push(face);
      edgeFaces.set(edge, owners);
    }
  }
  for (const [edge, owners] of edgeFaces) {
    if (owners.length > 2) throw new Error(`surface_coordinate_invalid: non-manifold edge ${edge}`);
    if (owners.length === 2) {
      adjacency[owners[0]].push(owners[1]);
      adjacency[owners[1]].push(owners[0]);
    }
  }
  for (const neighbors of adjacency) neighbors.sort((left, right) => left - right);
  const result = { keys, adjacency };
  topologyCache.set(geometry, result);
  return result;
}

function unfoldNeighbor(
  geometry: THREE.BufferGeometry,
  keys: [string, string, string][],
  face: number,
  triangleUv: [Vec2, Vec2, Vec2],
  neighbor: number,
): [Vec2, Vec2, Vec2] {
  const known = new Map(keys[face].map((key, index) => [key, triangleUv[index]]));
  const shared = keys[neighbor].filter((key) => known.has(key));
  if (shared.length !== 2) throw new Error("surface_coordinate_invalid: adjacent faces do not share exactly two vertices");
  const [ka, kb] = shared;
  const qa = known.get(ka)!;
  const qb = known.get(kb)!;
  const kc = keys[neighbor].find((key) => key !== ka && key !== kb)!;
  const sourceKeys = keys[neighbor];
  const sourcePoints = trianglePoints(geometry, neighbor);
  const source = new Map(sourceKeys.map((key, index) => [key, sourcePoints[index]]));
  const edge = sub2(qb, qa);
  const edgeLength = norm2(edge);
  if (edgeLength <= 1e-14) throw new Error("surface_coordinate_invalid: collapsed rest edge");
  const unit = mul2(edge, 1 / edgeLength);
  const da = source.get(kc)!.distanceTo(source.get(ka)!);
  const db = source.get(kc)!.distanceTo(source.get(kb)!);
  const along = (da * da - db * db + edgeLength * edgeLength) / (2 * edgeLength);
  const height = Math.sqrt(Math.max(0, da * da - along * along));
  const perp: Vec2 = [-unit[1], unit[0]];
  const knownThird = keys[face].find((key) => key !== ka && key !== kb)!;
  const offset = sub2(known.get(knownThird)!, qa);
  const cross = edge[0] * offset[1] - edge[1] * offset[0];
  const side = Math.sign(cross) || 1;
  const qc = add2(add2(qa, mul2(unit, along)), mul2(perp, -side * height));
  const lookup = new Map<string, Vec2>([[ka, qa], [kb, qb], [kc, qc]]);
  return sourceKeys.map((key) => lookup.get(key)!) as [Vec2, Vec2, Vec2];
}

/** Unfold connected faces whose developed triangles can intersect radiusM. */
export function unfoldSurfacePatch(
  rest: THREE.BufferGeometry,
  anchor: Anchor,
  rotationRad: number,
  radiusM: number,
): UnfoldedPatch {
  if (!(radiusM > 0) || !Number.isFinite(radiusM)) throw new Error("surface_coordinate_invalid: patch radius must be positive");
  if (!Number.isInteger(anchor.face) || anchor.face < 0 || anchor.face >= faceCount(rest)) {
    throw new Error(`surface_coordinate_invalid: face ${anchor.face} is outside the canonical surface`);
  }
  const { keys, adjacency } = topology(rest);
  const frame = frameAt(rest, anchor, rotationRad);
  const triangle = trianglePoints(rest, anchor.face);
  const e01 = triangle[1].clone().sub(triangle[0]);
  const length = e01.length();
  if (length <= 1e-14) throw new Error("surface_coordinate_invalid: degenerate seed triangle");
  const projected: Vec2 = [e01.dot(frame.u), e01.dot(frame.v)];
  const projectedLength = norm2(projected);
  if (projectedLength <= 1e-14) throw new Error("surface_coordinate_invalid: seed edge has no tangent projection");
  const direction: Vec2 = mul2(projected, 1 / projectedLength);
  const q0: Vec2 = [0, 0];
  const q1 = mul2(direction, length);
  const d02 = triangle[2].distanceTo(triangle[0]);
  const d12 = triangle[2].distanceTo(triangle[1]);
  const along = (d02 * d02 - d12 * d12 + length * length) / (2 * length);
  const height = Math.sqrt(Math.max(0, d02 * d02 - along * along));
  const perpendicular: Vec2 = [-direction[1], direction[0]];
  const orientation = Math.sign(
    e01.clone().cross(triangle[2].clone().sub(triangle[0])).dot(frame.n),
  ) || 1;
  const q2 = add2(mul2(direction, along), mul2(perpendicular, orientation * height));
  const [b0, b1, b2] = anchor.barycentric;
  const origin: Vec2 = [
    b0 * q0[0] + b1 * q1[0] + b2 * q2[0],
    b0 * q0[1] + b1 * q1[1] + b2 * q2[1],
  ];
  const seed = [q0, q1, q2].map((point) => sub2(point, origin)) as [Vec2, Vec2, Vec2];

  const unfolded = new Map<number, [Vec2, Vec2, Vec2]>([[anchor.face, seed]]);
  const seamByPair = new Map<string, ChartSeam>();
  const queue = [anchor.face];
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const face = queue[cursor];
    const faceUv = unfolded.get(face)!;
    for (const neighbor of adjacency[face]) {
      const candidate = unfoldNeighbor(rest, keys, face, faceUv, neighbor);
      const placed = unfolded.get(neighbor);
      if (placed) {
        const mismatchM = Math.max(...candidate.map((point, corner) => norm2(sub2(point, placed[corner]))));
        const pair = face < neighbor ? `${face}:${neighbor}` : `${neighbor}:${face}`;
        const known = seamByPair.get(pair);
        if (!known || known.mismatchM < mismatchM) seamByPair.set(pair, { face, neighbor, mismatchM });
        continue;
      }
      // A triangle can reach into the disk only from within its own longest
      // edge of it. The mesh-wide longest edge (62 mm on the SOMA torso) would
      // walk a 16 mm patch on a finger around the whole hand and pile those
      // faces onto the chart.
      const longestEdge = Math.max(
        norm2(sub2(candidate[1], candidate[0])),
        norm2(sub2(candidate[2], candidate[1])),
        norm2(sub2(candidate[0], candidate[2])),
      );
      if (Math.min(...candidate.map(norm2)) <= radiusM + longestEdge) {
        unfolded.set(neighbor, candidate);
        queue.push(neighbor);
      }
    }
  }
  const faces = [...unfolded]
    .sort(([left], [right]) => left - right)
    .map(([face, triangleUv]) => ({ face, triangleUv }));
  const included = new Set(faces.map((entry) => entry.face));
  return {
    seedFace: anchor.face,
    seedTriangleUv: seed,
    faces,
    adjacency: new Map(faces.map((entry) => [
      entry.face,
      adjacency[entry.face].filter((neighbor) => included.has(neighbor)),
    ])),
    seams: [...seamByPair.values()].sort((left, right) => right.mismatchM - left.mismatchM),
    boundaryFaces: faces.map((entry) => entry.face).filter((face) => adjacency[face].length < 3),
  };
}

export function barycentric2d(point: Vec2, triangle: [Vec2, Vec2, Vec2]): [number, number, number] {
  const [a, b, c] = triangle;
  const v0 = sub2(b, a);
  const v1 = sub2(c, a);
  const v2 = sub2(point, a);
  const d00 = v0[0] * v0[0] + v0[1] * v0[1];
  const d01 = v0[0] * v1[0] + v0[1] * v1[1];
  const d11 = v1[0] * v1[0] + v1[1] * v1[1];
  const d20 = v2[0] * v0[0] + v2[1] * v0[1];
  const d21 = v2[0] * v1[0] + v2[1] * v1[1];
  const denominator = d00 * d11 - d01 * d01;
  if (Math.abs(denominator) <= 1e-16) throw new Error("surface_coordinate_invalid: degenerate chart triangle");
  const b1 = (d11 * d20 - d01 * d21) / denominator;
  const b2 = (d00 * d21 - d01 * d20) / denominator;
  return [1 - b1 - b2, b1, b2];
}

function pointAlong(left: Vec2, right: Vec2, amount: number): Vec2 {
  return [
    left[0] + amount * (right[0] - left[0]),
    left[1] + amount * (right[1] - left[1]),
  ];
}

/**
 * Walk an ordered chart trajectory over the same developed triangles used by
 * the exact Python compiler. The current face is carried between samples so a
 * path cannot jump to a disconnected or far-side triangle.
 */
export function mapChartPoints(
  rest: THREE.BufferGeometry,
  patch: UnfoldedPatch,
  points: Vec2[],
): MappedChartPoint[] {
  const topo = topology(rest);
  const included = new Set(patch.faces.map((entry) => entry.face));
  let face = patch.seedFace;
  let triangleUv = patch.seedTriangleUv.map((point) => [...point] as Vec2) as [Vec2, Vec2, Vec2];
  let point: Vec2 = [0, 0];
  let barycentric = barycentric2d(point, triangleUv);
  const output: MappedChartPoint[] = [];

  for (const target of points) {
    if (!Number.isFinite(target[0]) || !Number.isFinite(target[1])) {
      throw new Error("surface_coordinate_invalid: chart point is not finite");
    }
    let crossed = 0;
    while (true) {
      const targetBarycentric = barycentric2d(target, triangleUv);
      if (targetBarycentric.every((value) => value >= -WALK_TOLERANCE)) {
        const clipped = targetBarycentric.map((value) => Math.min(1, Math.max(0, value)));
        const total = clipped[0] + clipped[1] + clipped[2];
        barycentric = clipped.map((value) => value / total) as [number, number, number];
        output.push({
          anchor: { face, barycentric: [...barycentric] as [number, number, number] },
          triangleUv: triangleUv.map((value) => [...value] as Vec2) as [Vec2, Vec2, Vec2],
        });
        point = [...target];
        break;
      }
      if (++crossed > 512) throw new Error("surface_coordinate_invalid: surface walk exceeded 512 crossed faces");
      const crossings = targetBarycentric.flatMap((value, opposite) => {
        if (value >= -WALK_TOLERANCE) return [];
        const denominator = barycentric[opposite] - value;
        return denominator > 1e-14 ? [{ amount: barycentric[opposite] / denominator, opposite }] : [];
      }).sort((left, right) => left.amount - right.amount || left.opposite - right.opposite);
      if (!crossings.length) throw new Error("surface_coordinate_invalid: could not identify the next mesh edge");
      const { amount, opposite } = crossings[0];
      const edgeKeys = new Set(topo.keys[face].filter((_, index) => index !== opposite));
      const neighbor = topo.adjacency[face].find((candidate) => (
        included.has(candidate) && [...edgeKeys].every((key) => topo.keys[candidate].includes(key))
      ));
      if (neighbor === undefined) {
        throw new Error(`surface_coordinate_invalid: surface trace reached open patch edge on face ${face}`);
      }
      const intersection = pointAlong(point, target, amount);
      triangleUv = unfoldNeighbor(rest, topo.keys, face, triangleUv, neighbor);
      face = neighbor;
      const entered = barycentric2d(intersection, triangleUv).map((value) => Math.min(1, Math.max(0, value)));
      const enteredTotal = entered[0] + entered[1] + entered[2];
      barycentric = entered.map((value) => value / enteredTotal) as [number, number, number];
      point = intersection;
    }
  }
  return output;
}
