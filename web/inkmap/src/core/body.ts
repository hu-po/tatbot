// Tatbot has one nominal-body path: the fixed MHR identity model transferred
// onto SOMA mid topology. Identity and pose are parameters of this body, not
// selectable body backends.
import * as THREE from "three";
import { computeSmoothNormals, faceCentroids } from "./anchor.ts";

export const MODEL_SPEC_ID = "mhr-soma-v1";
export const MODEL_SPEC_SHA256 = "e615b8485c367509833ee68b0405cd1e0ce6015604eaa4c1b2b699f4fc8d5144";
export const REFERENCE_IDENTITY_SHA256 = "800babcf48d09f4ff1da9d2e8a0ebe266869a19849557450c7210c9a9fb9a1b6";
export const TOPOLOGY_SHA256 = "e0ca7ee25dc0b4c8d841bb2626e364bb88b7af7fae037e30854728842e320a18";
export const REST_SURFACE_SHA256 = "caa66dff9b3625771c8f4c35bfe59556d30acdc3800880106f98f0ce75c49a95";
export const REST_ASSET_SHA256 = "c1d0ec25c6f4708bb3e8d48811711da0d309598a6b85cf256975d49d1d46e841";
export const MID_VERTEX_COUNT = 18_056;
export const MID_FACE_COUNT = 36_108;

export interface BodySpec {
  id: typeof MODEL_SPEC_ID;
  path: string;
  posePath: string;
  eyeHeight: number;
  modelSpecSha256: typeof MODEL_SPEC_SHA256;
  identitySha256: typeof REFERENCE_IDENTITY_SHA256;
  topologySha256: typeof TOPOLOGY_SHA256;
  restSurfaceSha256: typeof REST_SURFACE_SHA256;
  assetSha256: typeof REST_ASSET_SHA256;
}

export const BODY_SPEC: BodySpec = {
  id: MODEL_SPEC_ID,
  path: "bodies/mhr-soma-v1.glb",
  posePath: "bodies/mhr-soma-v1.poses.bin",
  eyeHeight: 1.72,
  modelSpecSha256: MODEL_SPEC_SHA256,
  identitySha256: REFERENCE_IDENTITY_SHA256,
  topologySha256: TOPOLOGY_SHA256,
  restSurfaceSha256: REST_SURFACE_SHA256,
  assetSha256: REST_ASSET_SHA256,
};

export interface Skin {
  geometry: THREE.BufferGeometry;
  centroids: Float32Array;
  map: THREE.Texture | null;
  vertexColors: boolean;
  bbox: THREE.Box3;
}

function concatBytes(header: Uint8Array, payload: Uint8Array): ArrayBuffer {
  const result = new Uint8Array(header.byteLength + payload.byteLength);
  result.set(header, 0);
  result.set(payload, header.byteLength);
  return result.buffer;
}

function roundTiesEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function sourceAttribute(geometry: THREE.BufferGeometry): THREE.BufferAttribute {
  const source = geometry.getAttribute("_soma_vertex") as THREE.BufferAttribute | undefined;
  if (!source || source.itemSize !== 1 || source.count !== MID_FACE_COUNT * 3) {
    throw new Error("body_topology_mismatch: missing canonical SOMA corner indices");
  }
  return source;
}

function canonicalIndexedPositions(geometry: THREE.BufferGeometry): Float32Array {
  const position = geometry.getAttribute("position") as THREE.BufferAttribute;
  const source = sourceAttribute(geometry);
  if (position.itemSize !== 3 || position.count !== MID_FACE_COUNT * 3) {
    throw new Error("body_topology_mismatch: browser surface has the wrong position count");
  }
  const result = new Float32Array(MID_VERTEX_COUNT * 3);
  const seen = new Uint8Array(MID_VERTEX_COUNT);
  for (let corner = 0; corner < source.count; corner += 1) {
    const index = source.getX(corner);
    if (!Number.isInteger(index) || index < 0 || index >= MID_VERTEX_COUNT) {
      throw new Error(`body_topology_mismatch: invalid SOMA vertex ${index}`);
    }
    const offset = index * 3;
    const xyz = [position.getX(corner), position.getY(corner), position.getZ(corner)];
    if (seen[index] && xyz.some((value, axis) => result[offset + axis] !== value)) {
      throw new Error(`body_topology_mismatch: SOMA vertex ${index} differs across faces`);
    }
    result.set(xyz, offset);
    seen[index] = 1;
  }
  if (seen.some((value) => value !== 1)) {
    throw new Error("body_topology_mismatch: browser surface omits a SOMA vertex");
  }
  return result;
}

/** Canonical bytes matching Python's signed int64 10-micrometre digest. */
export function canonicalSurfaceBytes(geometry: THREE.BufferGeometry): ArrayBuffer {
  const vertices = canonicalIndexedPositions(geometry);
  const payload = new ArrayBuffer(vertices.length * 8);
  const view = new DataView(payload);
  const quantum = Math.fround(0.00001);
  for (let index = 0; index < vertices.length; index += 1) {
    // NumPy's canonical implementation divides a float32 array by this
    // scalar in float32 before rint. Spell out both roundings in JavaScript.
    const units = Math.fround(Math.fround(vertices[index]) / quantum);
    view.setBigInt64(index * 8, BigInt(roundTiesEven(units)), true);
  }
  const header = new TextEncoder().encode(
    "dtype=<i8;shape=18056,3;order=C;quantization_m=0.00001;axes=x,-z,y\n",
  );
  return concatBytes(header, new Uint8Array(payload));
}

/** Canonical bytes matching Python's upstream-order int32 topology digest. */
export function canonicalTopologyBytes(geometry: THREE.BufferGeometry): ArrayBuffer {
  const source = sourceAttribute(geometry);
  const payload = new ArrayBuffer(source.count * 4);
  const view = new DataView(payload);
  for (let index = 0; index < source.count; index += 1) {
    view.setInt32(index * 4, source.getX(index), true);
  }
  const header = new TextEncoder().encode("dtype=<i4;shape=36108,3;order=C\n");
  return concatBytes(header, new Uint8Array(payload));
}

/** Build the direct SOMA rendering/picking view from its one named node. */
export function buildSkin(scene: THREE.Object3D): Skin {
  scene.updateMatrixWorld(true);
  const object = scene.getObjectByName("SOMA") as THREE.Mesh | undefined;
  if (!object?.isMesh) throw new Error("body_model_unsupported: SOMA mesh is missing");
  let geometry = object.geometry.clone();
  if (geometry.getIndex()) geometry = geometry.toNonIndexed();
  geometry.applyMatrix4(object.matrixWorld);
  sourceAttribute(geometry);
  computeSmoothNormals(geometry);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const material = (Array.isArray(object.material) ? object.material[0] : object.material) as THREE.MeshStandardMaterial;
  return {
    geometry,
    centroids: faceCentroids(geometry),
    map: material?.map ?? null,
    vertexColors: false,
    bbox: geometry.boundingBox!.clone(),
  };
}

/** Replace only face-expanded positions; address and UV order stay invariant. */
export function buildPosedSkin(rest: Skin, littleEndianFloat32: ArrayBuffer): Skin {
  const expectedBytes = MID_FACE_COUNT * 3 * 3 * Float32Array.BYTES_PER_ELEMENT;
  if (littleEndianFloat32.byteLength !== expectedBytes) {
    throw new Error(`body_asset_hash_mismatch: pose chunk has ${littleEndianFloat32.byteLength} bytes`);
  }
  const source = new DataView(littleEndianFloat32);
  const positions = new Float32Array(MID_FACE_COUNT * 3 * 3);
  for (let index = 0; index < positions.length; index += 1) {
    const value = source.getFloat32(index * 4, true);
    if (!Number.isFinite(value)) throw new Error("body_units_or_axes_invalid: pose contains non-finite data");
    positions[index] = value;
  }
  const geometry = rest.geometry.clone();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  computeSmoothNormals(geometry);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    centroids: faceCentroids(geometry),
    map: rest.map,
    vertexColors: false,
    bbox: geometry.boundingBox!.clone(),
  };
}

/** Apply a display/support transform after canonical pose digest validation. */
export function applyBodyRotation(skin: Skin, xyzw: [number, number, number, number]): void {
  const quaternion = new THREE.Quaternion(...xyzw);
  if (Math.abs(quaternion.lengthSq() - 1) > 1e-5) {
    throw new Error("pose_unsupported: body rotation is not unit length");
  }
  skin.geometry.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(quaternion));
  computeSmoothNormals(skin.geometry);
  skin.geometry.computeBoundingBox();
  skin.geometry.computeBoundingSphere();
  skin.centroids = faceCentroids(skin.geometry);
  skin.bbox = skin.geometry.boundingBox!.clone();
}

export { sha256Hex } from "./sha256.ts";
