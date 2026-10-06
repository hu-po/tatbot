/** Intrinsic chart-clipped tattoo preview on the canonical SOMA surface. */
import * as THREE from "three";
import { faceVertexIndices, frameAt, type Anchor, type Frame } from "./anchor.ts";
import { unfoldSurfacePatch, type Vec2 } from "./unfold.ts";

export interface DecalParams {
  anchor: Anchor;
  rotationRad: number;
  /** [width, height] on the rest surface, millimetres. */
  sizeMm: [number, number];
}

export interface DecalQuality {
  /** Requested flat design area, before mapping to the surface. */
  requestedAreaMm2: number;
  /** Area represented by the clipped intrinsic chart. */
  mappedAreaMm2: number;
  mappedPercent: number;
  /** Posed 3D surface area divided by intrinsic chart area. */
  areaStretchRatio: number;
  /** Largest duplicate-edge disagreement relative to the design diagonal. */
  seamRatio: number;
  chartFaceCount: number;
}

interface ClipVertex {
  chart: Vec2;
  position: THREE.Vector3;
  normal: THREE.Vector3;
}

/**
 * A design is refused once two developments of one mesh edge inside it
 * disagree by more than this fraction of the design diagonal. Closing around a
 * limb disagrees by the limb's girth, which is at least the design's reach;
 * curvature alone reaches this only when the patch encloses about a radian of
 * angle defect, at which point no flat stencil represents the skin either.
 */
export const CHART_CLOSURE_LIMIT = 0.5;

/**
 * A design is also refused when the developed chart's area inside the design
 * differs from the design's own area by more than this fraction. Each mesh
 * vertex leaves a wedge gap (positive curvature) or overlap (negative) equal
 * to its angle defect, so the sum tracks the curvature the design encloses:
 * about 1% for a 50 mm design on the chest, 12-14% at the worst forearm and
 * wrist spots of the SOMA body at 100 mm, and 33% for a cap larger than a
 * hemisphere, which no flat stencil represents.
 */
export const CHART_AREA_BUDGET = 0.2;

export class SurfaceChartError extends Error {
  readonly code: "surface_chart_overlap" | "anchor_outside_domain";

  constructor(code: SurfaceChartError["code"], detail: string) {
    super(`${code}: ${detail}`);
    this.code = code;
  }
}

/** Retained as a public frame utility for callers and analytic tests. */
export function frameToEuler(frame: Frame): THREE.Euler {
  const matrix = new THREE.Matrix4().makeBasis(frame.u, frame.v, frame.n);
  return new THREE.Euler().setFromRotationMatrix(matrix);
}

function interpolate(left: ClipVertex, right: ClipVertex, amount: number): ClipVertex {
  return {
    chart: [
      left.chart[0] + amount * (right.chart[0] - left.chart[0]),
      left.chart[1] + amount * (right.chart[1] - left.chart[1]),
    ],
    position: left.position.clone().lerp(right.position, amount),
    normal: left.normal.clone().lerp(right.normal, amount).normalize(),
  };
}

function clipHalfPlane(
  polygon: ClipVertex[],
  axis: 0 | 1,
  bound: number,
  keepGreater: boolean,
): ClipVertex[] {
  if (!polygon.length) return polygon;
  const output: ClipVertex[] = [];
  const inside = (vertex: ClipVertex): boolean => keepGreater
    ? vertex.chart[axis] >= bound - 1e-12
    : vertex.chart[axis] <= bound + 1e-12;
  for (let index = 0; index < polygon.length; index++) {
    const current = polygon[index];
    const previous = polygon[(index + polygon.length - 1) % polygon.length];
    const currentInside = inside(current);
    const previousInside = inside(previous);
    if (currentInside !== previousInside) {
      const denominator = current.chart[axis] - previous.chart[axis];
      const amount = Math.abs(denominator) <= 1e-20 ? 0 : (bound - previous.chart[axis]) / denominator;
      output.push(interpolate(previous, current, amount));
    }
    if (currentInside) output.push(current);
  }
  return output;
}

function clipRectangle(polygon: ClipVertex[], halfWidth: number, halfHeight: number): ClipVertex[] {
  let output = clipHalfPlane(polygon, 0, -halfWidth, true);
  output = clipHalfPlane(output, 0, halfWidth, false);
  output = clipHalfPlane(output, 1, -halfHeight, true);
  return clipHalfPlane(output, 1, halfHeight, false);
}

function polygonArea(points: Vec2[]): number {
  let area = 0;
  for (let index = 0; index < points.length; index++) {
    const next = points[(index + 1) % points.length];
    area += points[index][0] * next[1] - points[index][1] * next[0];
  }
  return Math.abs(area) / 2;
}

function faceVertices(
  geometry: THREE.BufferGeometry,
  face: number,
  triangleUv: [Vec2, Vec2, Vec2],
): [ClipVertex, ClipVertex, ClipVertex] {
  const position = geometry.getAttribute("position");
  const normal = geometry.getAttribute("normal");
  if (!normal) throw new Error("surface_coordinate_invalid: posed geometry has no normals");
  const indices = faceVertexIndices(geometry, face);
  return indices.map((index, corner) => ({
    chart: triangleUv[corner],
    position: new THREE.Vector3().fromBufferAttribute(position, index),
    normal: new THREE.Vector3().fromBufferAttribute(normal, index).normalize(),
  })) as [ClipVertex, ClipVertex, ClipVertex];
}

/**
 * Clip the intrinsic rest chart to the design rectangle, while interpolating
 * positions and normals exclusively on the matching posed triangles.
 */
export function buildDecal(
  rest: THREE.BufferGeometry,
  posed: THREE.BufferGeometry,
  params: DecalParams,
): { geometry: THREE.BufferGeometry; frame: Frame; chartFaces: number[]; quality: DecalQuality } {
  const width = params.sizeMm[0] / 1000;
  const height = params.sizeMm[1] / 1000;
  if (!(width > 0 && height > 0)) throw new SurfaceChartError("anchor_outside_domain", "design size must be positive");
  const radius = Math.hypot(width, height) / 2 + 0.002;
  const patch = unfoldSurfacePatch(rest, params.anchor, params.rotationRad, radius);
  const clipped = patch.faces.flatMap((entry) => {
    const polygon = clipRectangle(faceVertices(posed, entry.face, entry.triangleUv), width / 2, height / 2);
    return polygon.length >= 3 && polygonArea(polygon.map((vertex) => vertex.chart)) > 1e-14
      ? [{ face: entry.face, chart: entry.triangleUv, polygon }]
      : [];
  });
  if (!clipped.length) throw new SurfaceChartError("anchor_outside_domain", "design rectangle contains no connected surface triangles");
  const inDesign = new Set(clipped.map((entry) => entry.face));
  const boundary = patch.boundaryFaces.find((face) => inDesign.has(face));
  if (boundary !== undefined) {
    throw new SurfaceChartError("surface_chart_overlap", `design runs off the open edge of the surface at face ${boundary}`);
  }
  // A flat chart of curved skin is never exact: around every vertex the
  // developed triangles leave a thin wedge or overlap by the angle defect, so
  // covered area and pairwise overlap both drift with curvature and cannot
  // separate ordinary skin from a genuine wrap. The seam mismatch can: it is
  // millimetres for curvature and a whole girth once the chart meets itself.
  const requestedArea = width * height;
  const coveredArea = clipped.reduce(
    (sum, entry) => sum + polygonArea(entry.polygon.map((vertex) => vertex.chart)),
    0,
  );
  if (Math.abs(coveredArea / requestedArea - 1) > CHART_AREA_BUDGET) {
    throw new SurfaceChartError(
      "surface_chart_overlap",
      `intrinsic chart covers ${(100 * coveredArea / requestedArea).toFixed(1)}% of the design area; the skin here is too curved to flatten`,
    );
  }
  const diagonal = Math.hypot(width, height);
  const seam = patch.seams.find((entry) => inDesign.has(entry.face) && inDesign.has(entry.neighbor)
    && entry.mismatchM > CHART_CLOSURE_LIMIT * diagonal);
  if (seam) {
    throw new SurfaceChartError(
      "surface_chart_overlap",
      `chart meets itself inside the design: faces ${seam.face} and ${seam.neighbor} develop ${(1000 * seam.mismatchM).toFixed(1)} mm apart (girth or seam reached)`,
    );
  }

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  let posedArea = 0;
  for (const entry of clipped) {
    for (let index = 1; index + 1 < entry.polygon.length; index++) {
      const triangle = [entry.polygon[0], entry.polygon[index], entry.polygon[index + 1]] as const;
      posedArea += new THREE.Triangle(triangle[0].position, triangle[1].position, triangle[2].position).getArea();
      for (const vertex of triangle) {
        positions.push(vertex.position.x, vertex.position.y, vertex.position.z);
        normals.push(vertex.normal.x, vertex.normal.y, vertex.normal.z);
        uvs.push(vertex.chart[0] / width + 0.5, vertex.chart[1] / height + 0.5);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    frame: frameAt(posed, params.anchor, params.rotationRad),
    chartFaces: clipped.map((entry) => entry.face),
    quality: {
      requestedAreaMm2: requestedArea * 1_000_000,
      mappedAreaMm2: coveredArea * 1_000_000,
      mappedPercent: 100 * coveredArea / requestedArea,
      areaStretchRatio: posedArea / coveredArea,
      seamRatio: patch.seams.reduce((largest, entry) => inDesign.has(entry.face) && inDesign.has(entry.neighbor)
        ? Math.max(largest, entry.mismatchM / diagonal)
        : largest, 0),
      chartFaceCount: clipped.length,
    },
  };
}
