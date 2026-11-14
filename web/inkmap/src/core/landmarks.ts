/** Where a placement sits, in terms a person can measure on a body: tape distances to bony landmarks, and which way
 *  the tile's top points. It reads the design's real direction on the surface, so it holds whatever the rotation
 *  number means. The body is one nominal model, so these are a starting point for a person of another size. */
import * as THREE from "three";
import { anchorToPoint, BODY_UP, faceCentroids, frameAt, type Anchor } from "./anchor.ts";
import type { AtlasIndex } from "./atlas.ts";
import { SITES } from "./inklang/lexicon.ts";
import type { Placement } from "./schema.ts";
import { unfoldSurfacePatch } from "./unfold.ts";

/** A place a tape can start from: the middle of a site, or its highest skin on the rest pose. */
interface Landmark { site: string; name: string; at?: "top" }

const LANDMARKS: Record<string, { marks: Landmark[]; top?: Landmark }> = {
  arm: { marks: [{ site: "elbow", name: "tip of the elbow" }, { site: "wrist", name: "wrist" }], top: { site: "shoulder_cap", name: "top of the shoulder", at: "top" } },
  leg: { marks: [{ site: "knee", name: "kneecap" }, { site: "ankle", name: "ankle" }], top: { site: "hip", name: "hip" } },
  torso_front: { marks: [{ site: "navel", name: "navel" }, { site: "sternum", name: "breastbone" }] },
  torso_back: { marks: [{ site: "sacrum", name: "base of the spine" }, { site: "nape", name: "nape" }] },
};
const SHOWN = 2;
const ROUND_MM = 5;

export interface PlacementGuide {
  landmarks: { name: string; distanceMm: number }[];
  /** What the tile's top is measured against, and how far it is turned from pointing there, as seen on the skin. */
  top: { towards: string; clockwiseDeg: number };
}

/** The turn from `towards` to the design's top, degrees, clockwise as seen from outside the skin. */
export function clockwiseFrom(rest: THREE.BufferGeometry, anchor: Anchor, rotationRad: number, towards: THREE.Vector3): number {
  const frame = frameAt(rest, anchor, rotationRad);
  const reference = towards.clone().addScaledVector(frame.n, -towards.dot(frame.n)).normalize();
  const anticlockwise = Math.atan2(new THREE.Vector3().crossVectors(reference, frame.v).dot(frame.n), reference.dot(frame.v));
  return -THREE.MathUtils.radToDeg(anticlockwise);
}

/** The stored rotation that stands a design upright at this anchor: none on most of the body, and on an arm the turn
 *  from the body's up to the limb's. It is measured for the anchor, so it follows the skin where the body's up, which
 *  stored rotations are measured from, stands nearly square to the surface. */
export function uprightRotation(atlas: AtlasIndex, rest: THREE.BufferGeometry, anchor: Anchor): number {
  const up = atlas.upAt(anchor.face);
  return up.z === 1 ? 0 : THREE.MathUtils.degToRad(clockwiseFrom(rest, anchor, 0, up));
}

/** The landmark's face on this side of the body. */
function landmarkFace(atlas: AtlasIndex, rest: THREE.BufferGeometry, mark: Landmark, side: "left" | "right" | null): number | null {
  const faces = atlas.facesOf(mark.site, SITES[mark.site].laterality === "sided" ? side : null);
  if (!faces.length) return null;
  const centroids = faceCentroids(rest);
  const point = (face: number) => new THREE.Vector3().fromArray(centroids, 3 * face);
  if (mark.at === "top") return faces.reduce((best, face) => (point(face).z > point(best).z ? face : best));
  const middle = faces.reduce((sum, face) => sum.add(point(face)), new THREE.Vector3()).divideScalar(faces.length);
  return faces.reduce((best, face) => (point(face).distanceTo(middle) < point(best).distanceTo(middle) ? face : best));
}

/** The distance a tape lies along the skin to `face`, mm: straight, in the surface unfolded about the anchor. A limb
 *  develops with almost no stretch, where walking face to face runs half as long again. */
function tapeMm(rest: THREE.BufferGeometry, anchor: Anchor, face: number): number | null {
  const barycentric: Anchor["barycentric"] = [1 / 3, 1 / 3, 1 / 3];
  const reach = anchorToPoint(rest, anchor).p.distanceTo(anchorToPoint(rest, { face, barycentric }).p);
  const corners = unfoldSurfacePatch(rest, anchor, 0, 1.6 * reach + 0.02).faces.find((entry) => entry.face === face)?.triangleUv;
  if (!corners) return null;
  return 1000 * Math.hypot(...[0, 1].map((axis) => corners.reduce((sum, corner) => sum + corner[axis] / 3, 0)));
}

export function placementGuide(atlas: AtlasIndex, rest: THREE.BufferGeometry, placement: Pick<Placement, "anchor" | "rotation_rad">): PlacementGuide | null {
  const site = atlas.describe(placement.anchor);
  const part = site && LANDMARKS[SITES[site.id].group];
  if (!site || !part) return null;
  const faceOf = (mark: Landmark) => (mark.site === site.id ? null : landmarkFace(atlas, rest, mark, site.laterality === "center" ? null : site.laterality));
  const landmarks = [...part.marks, ...part.top ? [part.top] : []].flatMap((mark) => {
    const face = faceOf(mark);
    const distanceMm = face === null ? null : tapeMm(rest, placement.anchor, face);
    return distanceMm === null ? [] : [{ name: mark.name, distanceMm: Math.round(distanceMm / ROUND_MM) * ROUND_MM }];
  }).sort((a, b) => a.distanceMm - b.distanceMm).slice(0, SHOWN);
  const topFace = part.top ? faceOf(part.top) : null;
  const towards = topFace === null ? BODY_UP.clone()
    : anchorToPoint(rest, { face: topFace, barycentric: [1 / 3, 1 / 3, 1 / 3] }).p.sub(anchorToPoint(rest, placement.anchor).p);
  return { landmarks, top: { towards: topFace === null ? "head" : part.top!.name, clockwiseDeg: Math.round(clockwiseFrom(rest, placement.anchor, placement.rotation_rad, towards)) } };
}
