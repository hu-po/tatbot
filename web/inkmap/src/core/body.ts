// The body: a GLB whose named skin nodes are merged into ONE static, Z-up,
// smooth "skin" geometry. Anchors are face indices into this geometry, so the
// build must be deterministic (sorted node order, fixed subdivision) and the
// GLB is content-hashed into every placement file.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { LoopSubdivision } from "three-subdivide";
import { computeSmoothNormals, faceCentroids } from "./anchor.ts";

export interface BodySpec {
  id: string;
  /** Short name for the body toggle. */
  label: string;
  /** One-character icon for the body toggle. */
  glyph: string;
  /** Relative to the site root (public/). */
  path: string;
  /** Mesh node names that are skin; everything else in the GLB is a prop and hidden. */
  skinNodes: string[];
  /** Loop subdivision iterations applied at load (toon meshes are flat-shaded and coarse). */
  subdivide: number;
  /** Uniform scale to bring the model to metres. */
  scale: number;
  /** Where the camera looks / the character stands, after scaling (m). */
  eyeHeight: number;
}

export const BODIES: BodySpec[] = [
  // Blender Studio "Human Base Meshes" bundle v1.4.1 (CC0), stylized bodies,
  // exported by tools/export-hbm.py (white body, dark eyes in COLOR_0). Already in
  // metres, feet on the floor, so scale is 1.
  { id: "hbm-male-stylized", label: "male", glyph: "\u2642", path: "bodies/hbm-male-stylized.glb", skinNodes: ["Body", "EyeL", "EyeR"], subdivide: 0, scale: 1, eyeHeight: 1.63 },
  { id: "hbm-female-stylized", label: "female", glyph: "\u2640", path: "bodies/hbm-female-stylized.glb", skinNodes: ["Body", "EyeL", "EyeR"], subdivide: 0, scale: 1, eyeHeight: 1.46 },
];

export function bodySpec(id: string): BodySpec {
  const spec = BODIES.find((b) => b.id === id);
  if (!spec) throw new Error(`unknown body "${id}"`);
  return spec;
}

/** glTF is Y-up; the robot, the URDF and this app are Z-up. Applied ONCE, here. */
export const Y_UP_TO_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);

export interface Skin {
  geometry: THREE.BufferGeometry;
  centroids: Float32Array;
  /** Texture from the first skin node's material, if any. */
  map: THREE.Texture | null;
  /** True when the GLB carries COLOR_0 (white skin / dark eyes; tinted by the skin-tone picker). */
  vertexColors: boolean;
  bbox: THREE.Box3;
}

/** Build the skin from a loaded glTF scene. Pure geometry work; no React, no DOM. */
export function buildSkin(scene: THREE.Object3D, spec: BodySpec): Skin {
  scene.updateMatrixWorld(true);
  const parts: THREE.BufferGeometry[] = [];
  let map: THREE.Texture | null = null;
  for (const name of [...spec.skinNodes].sort()) {
    const obj = scene.getObjectByName(name) as THREE.Mesh | undefined;
    if (!obj || !(obj as THREE.Mesh).isMesh) throw new Error(`body ${spec.id}: skin node "${name}" missing from GLB`);
    const src = obj.geometry;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", src.getAttribute("position").clone());
    const uv = src.getAttribute("uv");
    if (uv) g.setAttribute("uv", uv.clone());
    const color = src.getAttribute("color");
    if (color) g.setAttribute("color", color.clone());
    const idx = src.getIndex();
    if (idx) g.setIndex(idx.clone());
    // A SkinnedMesh at bind pose is exactly its geometry under matrixWorld.
    g.applyMatrix4(obj.matrixWorld);
    parts.push(g);
    if (!map) {
      const m = (Array.isArray(obj.material) ? obj.material[0] : obj.material) as THREE.MeshStandardMaterial;
      map = m?.map ?? null;
    }
  }
  // Optional attributes must be present on every part or on none, or the merge fails.
  for (const name of ["uv", "color"]) {
    if (!parts.every((p) => p.getAttribute(name))) for (const p of parts) p.deleteAttribute(name);
  }
  const vertexColors = parts[0].getAttribute("color") !== undefined;
  let merged = mergeGeometries(parts, false);
  if (!merged) throw new Error(`body ${spec.id}: skin parts have mismatched attributes`);
  merged = merged.toNonIndexed();
  merged.applyMatrix4(new THREE.Matrix4().makeScale(spec.scale, spec.scale, spec.scale).multiply(Y_UP_TO_Z_UP));
  if (spec.subdivide > 0) {
    // split:false keeps the face count a pure function of the input; uvSmooth:false keeps the atlas colours crisp.
    merged = LoopSubdivision.modify(merged, spec.subdivide, { split: false, uvSmooth: false, preserveEdges: false, flatOnly: false });
  }
  computeSmoothNormals(merged);
  merged.computeBoundingBox();
  merged.computeBoundingSphere();
  return { geometry: merged, centroids: faceCentroids(merged), map, vertexColors, bbox: merged.boundingBox!.clone() };
}

export { sha256Hex } from "./sha256.ts";
