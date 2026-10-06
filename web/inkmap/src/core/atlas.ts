// Canonical InkLang region-atlas index. The checked-in atlas carries the
// default anchors and chart parameters; consumers do not invent them again.
import * as THREE from "three";
import type { Anchor } from "./anchor.ts";
import { InkLangError } from "./inklang/errors.ts";
import { INKLANG_VERSION, SITES, ZONES } from "./inklang/lexicon.ts";
import {
  ATLAS_SCHEMA_VERSION,
  type Laterality,
  type Level,
  type RelativeWalk,
  type SitePhrase,
} from "./inklang/types.ts";

export interface AtlasRegionData {
  site_id: string;
  laterality: "left" | "right" | null;
  default_anchor: Anchor;
  chart: {
    mean: [number, number, number];
    normal: [number, number, number];
    u_axis: [number, number, number];
    v_axis: [number, number, number];
    u_range: [number, number];
    v_range: [number, number];
  };
}

export interface AtlasData {
  atlas_schema_version: number;
  inklang_version: string;
  body: {
    model_spec_id: string;
    model_spec_sha256: string;
    identity_sha256: string;
    topology_sha256: string;
    rest_surface_sha256: string;
    asset_sha256: string;
  };
  builder: { name: string; version: number; source_sha256?: string };
  frame: { front: "-y"; left: "+x"; up: "+z" };
  encoding: string;
  sites: string[];
  site_status: Record<string, { status: "mapped_supported" | "mapped_unsupported" | "unsupported"; face_count: number; reason?: string }>;
  region_meaning_count: number;
  upstream_exclusions: { asset_sha256: string; segments: string[]; excluded_faces: number };
  /** Per face: siteIndex*4 + laterality (0 none, 1 left, 2 right); -1 = not skin. */
  faces: number[];
  /** One only for faces inside the reviewed initial tattoo domain. */
  eligible_faces: number[];
  regions: Record<string, AtlasRegionData>;
}

const failAtlas = (message: string): never => {
  throw new InkLangError("INKLANG_VERSION_MISMATCH", `region atlas: ${message}`);
};

function isVec(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every((item) => typeof item === "number" && Number.isFinite(item));
}

function validAnchor(value: unknown): value is Anchor {
  if (typeof value !== "object" || value === null) return false;
  const anchor = value as Record<string, unknown>;
  return Number.isInteger(anchor.face)
    && (anchor.face as number) >= 0
    && isVec(anchor.barycentric, 3)
    && (anchor.barycentric as number[]).every((item) => item >= 0 && item <= 1)
    && Math.abs((anchor.barycentric as number[]).reduce((sum, item) => sum + item, 0) - 1) <= 1e-6;
}

export function parseAtlas(value: unknown, expectFaces?: number): AtlasData {
  if (typeof value !== "object" || value === null) failAtlas("not an object");
  const atlas = value as Record<string, unknown>;
  if (atlas.atlas_schema_version !== ATLAS_SCHEMA_VERSION) {
    failAtlas(`schema ${String(atlas.atlas_schema_version)} (reader accepts ${ATLAS_SCHEMA_VERSION})`);
  }
  if (atlas.inklang_version !== INKLANG_VERSION) {
    failAtlas(`lexicon ${String(atlas.inklang_version)} (app speaks ${INKLANG_VERSION})`);
  }
  const body = atlas.body as Record<string, unknown> | undefined;
  if (
    !body
    || typeof body.model_spec_id !== "string"
    || typeof body.model_spec_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.model_spec_sha256)
    || typeof body.identity_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.identity_sha256)
    || typeof body.topology_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.topology_sha256)
    || typeof body.rest_surface_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.rest_surface_sha256)
    || typeof body.asset_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.asset_sha256)
  ) {
    failAtlas("body needs the v2 model, identity, topology, surface, and asset digests");
  }
  const builder = atlas.builder as Record<string, unknown> | undefined;
  if (!builder || typeof builder.name !== "string" || !Number.isInteger(builder.version)) {
    failAtlas("builder needs name and integer version");
  }
  const rawSites = atlas.sites;
  if (!Array.isArray(rawSites) || !rawSites.every((site) => typeof site === "string" && site in SITES)) {
    failAtlas("sites must be known leaf ids");
  }
  const rawFaces = atlas.faces;
  if (!Array.isArray(rawFaces) || !rawFaces.every((code) => Number.isInteger(code) && (code as number) >= -1)) {
    failAtlas("faces must be integer region codes");
  }
  const sites = rawSites as string[];
  if (sites.length !== Object.keys(SITES).length || !Object.keys(SITES).every((site) => sites.includes(site))) {
    failAtlas("all semantic site names must be declared, including unsupported sites");
  }
  if (atlas.region_meaning_count !== 99) failAtlas("region_meaning_count must be 99");
  const faces = rawFaces as number[];
  const eligibleFaces = atlas.eligible_faces;
  if (
    !Array.isArray(eligibleFaces)
    || eligibleFaces.length !== faces.length
    || !eligibleFaces.every((value) => value === 0 || value === 1)
  ) failAtlas("eligible_faces must be a zero/one value for every face");
  if (expectFaces !== undefined && faces.length !== expectFaces) {
    failAtlas(`face count ${faces.length} does not match loaded body (${expectFaces})`);
  }
  const nSites = sites.length;
  if (!faces.every((code) => code === -1 || (code >> 2) < nSites)) {
    failAtlas("face code references an unknown site index");
  }
  if (typeof atlas.regions !== "object" || atlas.regions === null || Array.isArray(atlas.regions)) {
    failAtlas("regions must be an object");
  }
  for (const [key, raw] of Object.entries(atlas.regions as Record<string, unknown>)) {
    if (typeof raw !== "object" || raw === null) failAtlas(`region ${key} is not an object`);
    const record = raw as Record<string, unknown>;
    const chart = record.chart as Record<string, unknown> | undefined;
    if (
      typeof record.site_id !== "string"
      || !(record.site_id in SITES)
      || !["left", "right", null].includes(record.laterality as "left" | "right" | null)
      || !validAnchor(record.default_anchor)
      || !chart
      || !isVec(chart.mean, 3)
      || !isVec(chart.normal, 3)
      || !isVec(chart.u_axis, 3)
      || !isVec(chart.v_axis, 3)
      || !isVec(chart.u_range, 2)
      || !isVec(chart.v_range, 2)
    ) {
      failAtlas(`region ${key} has an invalid anchor or chart`);
    }
  }
  return value as AtlasData;
}

/** Bind atlas eligibility to the separately hashed upstream segment mask. */
export function validateExclusionMask(atlas: AtlasData, bytes: ArrayBuffer): void {
  const mask = new Uint8Array(bytes);
  if (mask.length !== atlas.faces.length) failAtlas("upstream exclusion mask has the wrong face count");
  let excluded = 0;
  for (let face = 0; face < mask.length; face++) {
    if (mask[face] !== 0 && mask[face] !== 1) failAtlas(`upstream exclusion mask byte ${face} is not zero or one`);
    if (mask[face] === 1) {
      excluded += 1;
      if (atlas.faces[face] !== -1 || atlas.eligible_faces[face] !== 0) {
        failAtlas(`upstream-excluded face ${face} remains labeled or eligible`);
      }
    }
  }
  if (excluded !== atlas.upstream_exclusions.excluded_faces) {
    failAtlas(`upstream exclusion count ${excluded} differs from ${atlas.upstream_exclusions.excluded_faces}`);
  }
}

export interface RegionRef {
  id: string;
  laterality: "left" | "right" | null;
}

interface Region extends RegionRef {
  key: string;
  faces: number[];
  mean: THREE.Vector3;
  normal: THREE.Vector3;
  uAxis: THREE.Vector3;
  vAxis: THREE.Vector3;
  uRange: [number, number];
  vRange: [number, number];
  defaultAnchor: Anchor;
}

const regionKey = (id: string, laterality: "left" | "right" | null): string => (
  laterality ? `${id}:${laterality}` : id
);

function faceNormals(geometry: THREE.BufferGeometry): Float32Array {
  const position = geometry.getAttribute("position") as THREE.BufferAttribute;
  const output = new Float32Array(position.count);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const normal = new THREE.Vector3();
  for (let face = 0; face < position.count / 3; face++) {
    a.fromBufferAttribute(position, 3 * face);
    b.fromBufferAttribute(position, 3 * face + 1);
    c.fromBufferAttribute(position, 3 * face + 2);
    normal.subVectors(b, a).cross(c.sub(a)).normalize();
    output.set([normal.x, normal.y, normal.z], 3 * face);
  }
  return output;
}

function centroidVector(centroids: Float32Array, face: number): THREE.Vector3 {
  return new THREE.Vector3(centroids[3 * face], centroids[3 * face + 1], centroids[3 * face + 2]);
}

function normalVector(normals: Float32Array, face: number): THREE.Vector3 {
  return new THREE.Vector3(normals[3 * face], normals[3 * face + 1], normals[3 * face + 2]);
}

function membership(sites: string[], faces: number[]): Map<string, RegionRef & { faces: number[] }> {
  const output = new Map<string, RegionRef & { faces: number[] }>();
  for (let face = 0; face < faces.length; face++) {
    const code = faces[face];
    if (code < 0) continue;
    const id = sites[code >> 2];
    const laterality = (code & 3) === 1 ? "left" as const : (code & 3) === 2 ? "right" as const : null;
    const key = regionKey(id, laterality);
    const region = output.get(key) ?? { id, laterality, faces: [] };
    region.faces.push(face);
    output.set(key, region);
  }
  return output;
}

function computeChart(
  entry: RegionRef & { faces: number[] },
  centroids: Float32Array,
  normals: Float32Array,
): Omit<AtlasRegionData, "site_id" | "laterality" | "default_anchor">["chart"] {
  const mean = new THREE.Vector3();
  const normal = new THREE.Vector3();
  for (const face of entry.faces) {
    mean.add(centroidVector(centroids, face));
    normal.add(normalVector(normals, face));
  }
  mean.divideScalar(entry.faces.length);
  if (normal.lengthSq() < 1e-12) normal.set(0, -1, 0);
  else normal.normalize();
  const covariance = [0, 0, 0, 0, 0, 0];
  const delta = new THREE.Vector3();
  for (const face of entry.faces) {
    delta.copy(centroidVector(centroids, face)).sub(mean);
    covariance[0] += delta.x * delta.x;
    covariance[1] += delta.x * delta.y;
    covariance[2] += delta.x * delta.z;
    covariance[3] += delta.y * delta.y;
    covariance[4] += delta.y * delta.z;
    covariance[5] += delta.z * delta.z;
  }
  const uAxis = new THREE.Vector3(0.3, 0.4, 0.87);
  for (let iteration = 0; iteration < 24; iteration++) {
    uAxis.set(
      covariance[0] * uAxis.x + covariance[1] * uAxis.y + covariance[2] * uAxis.z,
      covariance[1] * uAxis.x + covariance[3] * uAxis.y + covariance[4] * uAxis.z,
      covariance[2] * uAxis.x + covariance[4] * uAxis.y + covariance[5] * uAxis.z,
    );
    if (uAxis.lengthSq() < 1e-20) {
      uAxis.set(0, 0, -1);
      break;
    }
    uAxis.normalize();
  }
  if (Math.abs(uAxis.z) > 0.25 ? uAxis.z > 0 : uAxis.y > 0) uAxis.negate();
  const vAxis = new THREE.Vector3().crossVectors(normal, uAxis).normalize();
  if (vAxis.lengthSq() < 1e-12) vAxis.set(1, 0, 0);
  const side = (entry.laterality ?? (mean.x >= 0 ? "left" : "right")) === "left" ? 1 : -1;
  if (vAxis.x * side < 0) vAxis.negate();
  let uMin = Infinity;
  let uMax = -Infinity;
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const face of entry.faces) {
    delta.copy(centroidVector(centroids, face)).sub(mean);
    const u = delta.dot(uAxis);
    const v = delta.dot(vAxis);
    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }
  return {
    mean: mean.toArray() as [number, number, number],
    normal: normal.toArray() as [number, number, number],
    u_axis: uAxis.toArray() as [number, number, number],
    v_axis: vAxis.toArray() as [number, number, number],
    u_range: [uMin, uMax],
    v_range: [vMin, vMax],
  };
}

function defaultAnchor(
  entry: RegionRef & { faces: number[] },
  chart: AtlasRegionData["chart"],
  centroids: Float32Array,
): Anchor {
  const hint = SITES[entry.id]?.anchor;
  if (hint === "extremum_back" || hint === "extremum_front") {
    let best = entry.faces[0];
    let bestY = hint === "extremum_back" ? -Infinity : Infinity;
    for (const face of entry.faces) {
      const y = centroids[3 * face + 1];
      if (hint === "extremum_back" ? y > bestY : y < bestY) {
        best = face;
        bestY = y;
      }
    }
    return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
  }
  const mean = new THREE.Vector3(...chart.mean);
  if (SITES[entry.id]?.laterality === "midline") mean.x = 0;
  let best = entry.faces[0];
  let bestDistance = Infinity;
  for (const face of entry.faces) {
    const distance = centroidVector(centroids, face).distanceToSquared(mean);
    if (distance < bestDistance || (distance === bestDistance && face < best)) {
      best = face;
      bestDistance = distance;
    }
  }
  return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
}

/** Generate the data every runtime needs from one already-labeled atlas. */
export function buildRegionRecords(
  sites: string[],
  faces: number[],
  geometry: THREE.BufferGeometry,
  centroids: Float32Array,
): Record<string, AtlasRegionData> {
  const normals = faceNormals(geometry);
  const records: Record<string, AtlasRegionData> = {};
  for (const [key, entry] of [...membership(sites, faces)].sort(([left], [right]) => left.localeCompare(right))) {
    const chart = computeChart(entry, centroids, normals);
    records[key] = {
      site_id: entry.id,
      laterality: entry.laterality,
      default_anchor: defaultAnchor(entry, chart, centroids),
      chart,
    };
  }
  return records;
}

function buildAdjacency(geometry: THREE.BufferGeometry): number[][] {
  const position = geometry.getAttribute("position") as THREE.BufferAttribute;
  const adjacency: number[][] = Array.from({ length: position.count / 3 }, () => []);
  const owners = new Map<string, number>();
  const vertexKey = (index: number): string => [
    Math.round(position.getX(index) * 1e5),
    Math.round(position.getY(index) * 1e5),
    Math.round(position.getZ(index) * 1e5),
  ].join(",");
  for (let face = 0; face < position.count / 3; face++) {
    const keys = [vertexKey(3 * face), vertexKey(3 * face + 1), vertexKey(3 * face + 2)];
    for (let edge = 0; edge < 3; edge++) {
      const key = [keys[edge], keys[(edge + 1) % 3]].sort().join("|");
      const other = owners.get(key);
      if (other === undefined) owners.set(key, face);
      else if (other !== face) {
        adjacency[face].push(other);
        adjacency[other].push(face);
      }
    }
  }
  for (const neighbors of adjacency) neighbors.sort((left, right) => left - right);
  return adjacency;
}

class MinHeap {
  private values: [number, number][] = [];

  push(distance: number, face: number): void {
    this.values.push([distance, face]);
    let index = this.values.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.values[parent][0] < distance || (this.values[parent][0] === distance && this.values[parent][1] <= face)) break;
      [this.values[parent], this.values[index]] = [this.values[index], this.values[parent]];
      index = parent;
    }
  }

  pop(): [number, number] | undefined {
    if (!this.values.length) return undefined;
    const first = this.values[0];
    const last = this.values.pop()!;
    if (this.values.length) {
      this.values[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let best = index;
        for (const child of [left, right]) {
          if (child >= this.values.length) continue;
          const a = this.values[child];
          const b = this.values[best];
          if (a[0] < b[0] || (a[0] === b[0] && a[1] < b[1])) best = child;
        }
        if (best === index) break;
        [this.values[index], this.values[best]] = [this.values[best], this.values[index]];
        index = best;
      }
    }
    return first;
  }
}

/** Face-level index over one canonical rest surface. */
export class AtlasIndex {
  readonly atlas: AtlasData;
  private readonly regionByFace: (RegionRef | null)[];
  private readonly regions = new Map<string, Region>();
  private readonly faceNormals: Float32Array;
  private readonly centroids: Float32Array;
  private readonly adjacency: number[][];
  private readonly maxEdge: number;

  constructor(atlas: AtlasData, geometry: THREE.BufferGeometry, centroids: Float32Array) {
    this.atlas = atlas;
    this.centroids = centroids;
    if (atlas.faces.length !== centroids.length / 3) {
      throw new InkLangError(
        "INKLANG_SURFACE_MISMATCH",
        `region atlas has ${atlas.faces.length} faces for a ${centroids.length / 3}-face body`,
      );
    }
    this.faceNormals = faceNormals(geometry);
    this.adjacency = buildAdjacency(geometry);
    this.regionByFace = new Array(atlas.faces.length).fill(null);
    const members = membership(atlas.sites, atlas.faces);
    for (let face = 0; face < atlas.faces.length; face++) {
      const code = atlas.faces[face];
      if (code < 0) continue;
      this.regionByFace[face] = {
        id: atlas.sites[code >> 2],
        laterality: (code & 3) === 1 ? "left" : (code & 3) === 2 ? "right" : null,
      };
    }
    for (const [key, entry] of members) {
      const data = atlas.regions[key];
      if (!data || data.site_id !== entry.id || data.laterality !== entry.laterality) {
        throw new InkLangError("INKLANG_VERSION_MISMATCH", `atlas region record ${key} is missing or mismatched`);
      }
      if (!entry.faces.includes(data.default_anchor.face)) {
        throw new InkLangError("INKLANG_SEMANTIC_MISMATCH", `atlas default anchor for ${key} is outside the region`);
      }
      this.regions.set(key, {
        key,
        id: entry.id,
        laterality: entry.laterality,
        faces: entry.faces,
        mean: new THREE.Vector3(...data.chart.mean),
        normal: new THREE.Vector3(...data.chart.normal),
        uAxis: new THREE.Vector3(...data.chart.u_axis),
        vAxis: new THREE.Vector3(...data.chart.v_axis),
        uRange: data.chart.u_range,
        vRange: data.chart.v_range,
        defaultAnchor: data.default_anchor,
      });
    }
    let maxEdge = 0;
    for (let face = 0; face < this.adjacency.length; face++) {
      for (const neighbor of this.adjacency[face]) {
        maxEdge = Math.max(maxEdge, this.cvec(face).distanceTo(this.cvec(neighbor)));
      }
    }
    this.maxEdge = maxEdge;
  }

  private cvec(face: number): THREE.Vector3 {
    return centroidVector(this.centroids, face);
  }

  private nvec(face: number): THREE.Vector3 {
    return normalVector(this.faceNormals, face);
  }

  regionOf(face: number): RegionRef | null {
    return this.regionByFace[face] ?? null;
  }

  facesOf(id: string, laterality: "left" | "right" | null): number[] {
    const leaves = ZONES[id] ? ZONES[id].members : [id];
    const output: number[] = [];
    for (const leaf of leaves) {
      for (const region of this.regions.values()) {
        if (region.id !== leaf) continue;
        if (laterality !== null && region.laterality !== null && region.laterality !== laterality) continue;
        output.push(...region.faces.filter((face) => this.atlas.eligible_faces[face] === 1));
      }
    }
    return output;
  }

  uvOf(face: number): [number, number] | null {
    const ref = this.regionByFace[face];
    if (!ref) return null;
    const region = this.regions.get(regionKey(ref.id, ref.laterality))!;
    const delta = this.cvec(face).sub(region.mean);
    const u = (delta.dot(region.uAxis) - region.uRange[0]) / Math.max(1e-9, region.uRange[1] - region.uRange[0]);
    const v = (delta.dot(region.vAxis) - region.vRange[0]) / Math.max(1e-9, region.vRange[1] - region.vRange[0]);
    return [Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v))];
  }

  private aspectOf(face: number, ref: RegionRef): string | null {
    const allowed = SITES[ref.id]?.aspects ?? [];
    if (!allowed.length) return null;
    const normal = this.nvec(face);
    const scores: [string, number][] = allowed.map((aspect) => {
      let score = -1;
      if (aspect === "front") score = -normal.y;
      else if (aspect === "back") score = normal.y;
      // `top of the hand` means the dorsal hand, not world-up on a hanging
      // neutral pose. The atlas already separates dorsal `hand` from `palm`.
      else if (aspect === "top") score = ref.id === "hand" ? 1 : normal.z;
      else if (aspect === "side") score = Math.abs(normal.x) - 0.2;
      else if (aspect === "inner" || aspect === "outer") {
        const laterality = ref.laterality ?? (this.cvec(face).x >= 0 ? "left" : "right");
        const inward = laterality === "left" ? -normal.x : normal.x;
        score = aspect === "inner" ? inward : -inward;
      }
      return [aspect, score];
    });
    scores.sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
    return scores[0][1] > 0.35 ? scores[0][0] : null;
  }

  anchorFor(site: Pick<SitePhrase, "id"> & Partial<SitePhrase>): Anchor {
    const spec = SITES[site.id] ?? ZONES[site.id];
    if (!spec) throw new InkLangError("INKLANG_UNKNOWN_SITE", `unknown site "${site.id}"`);
    let laterality: "left" | "right" | null = site.laterality === "left" || site.laterality === "right"
      ? site.laterality
      : null;
    if (laterality === null && spec.laterality === "sided") {
      throw new InkLangError("INKLANG_AMBIGUOUS_LATERALITY", `${site.id} requires a left or right choice`);
    }
    let faces = this.facesOf(site.id, laterality);
    if (!faces.length) {
      throw new InkLangError("INKLANG_NO_REGION", `no faces for ${laterality ?? "center"} ${site.id}`);
    }
    const aspect = site.aspect ?? null;
    if (aspect) {
      const filtered = faces.filter((face) => this.aspectOf(face, this.regionByFace[face]!) === aspect);
      if (!filtered.length) {
        throw new InkLangError("INKLANG_NO_REGION", `no ${aspect} partition for ${laterality ?? "center"} ${site.id}`);
      }
      faces = filtered;
    }
    const level: Level | null = site.level ?? null;
    const regionUv = site.region_uv;
    if (!aspect && !level && !regionUv && !ZONES[site.id]) {
      const record = this.regions.get(regionKey(site.id, laterality));
      if (!record) throw new InkLangError("INKLANG_NO_REGION", `no region record for ${site.id}`);
      if (this.atlas.eligible_faces[record.defaultAnchor.face] === 1) return record.defaultAnchor;
      // Semantic labels remain available for review on unsupported anatomy,
      // but only reviewed eligible faces may become placement anchors.
    }
    const target: [number, number] = regionUv
      ?? [level === "upper" ? 0.22 : level === "lower" ? 0.78 : 0.5, 0.5];
    if (level) {
      const band = faces.filter((face) => {
        const uv = this.uvOf(face);
        if (!uv) return false;
        if (level === "upper") return uv[0] < 0.45;
        if (level === "lower") return uv[0] > 0.55;
        return uv[0] > 0.28 && uv[0] < 0.72;
      });
      if (!band.length) throw new InkLangError("INKLANG_NO_REGION", `no ${level} partition for ${site.id}`);
      faces = band;
    }
    let best = faces[0];
    let bestDistance = Infinity;
    for (const face of faces) {
      const uv = this.uvOf(face);
      if (!uv) continue;
      const distance = (uv[0] - target[0]) ** 2 + (uv[1] - target[1]) ** 2;
      if (distance < bestDistance || (distance === bestDistance && face < best)) {
        best = face;
        bestDistance = distance;
      }
    }
    return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
  }

  describe(anchor: Anchor): SitePhrase | null {
    const ref = this.regionByFace[anchor.face];
    if (!ref) return null;
    const regionUv = this.uvOf(anchor.face)!;
    const spec = SITES[ref.id];
    const laterality: Laterality | null = spec.laterality === "midline" ? null : ref.laterality;
    const level: Level | null = spec.geometry === "crease"
      ? null
      : regionUv[0] < 0.3
        ? "upper"
        : regionUv[0] > 0.7
          ? "lower"
          : null;
    return {
      id: ref.id,
      laterality,
      aspect: this.aspectOf(anchor.face, ref),
      level,
      region_uv: regionUv,
    };
  }

  contains(site: Pick<SitePhrase, "id" | "laterality">, anchor: Anchor): boolean {
    const ref = this.regionByFace[anchor.face];
    if (!ref) return false;
    const leaves = ZONES[site.id] ? ZONES[site.id].members : [site.id];
    const parent = SITES[ref.id]?.parent;
    if (!leaves.includes(ref.id) && (parent === undefined || !leaves.includes(parent))) return false;
    const requested = site.laterality === "left" || site.laterality === "right" ? site.laterality : null;
    return requested === null || ref.laterality === null || ref.laterality === requested;
  }

  isValidAnchor(anchor: Anchor): boolean {
    return Number.isInteger(anchor.face)
      && anchor.face >= 0
      && anchor.face < this.regionByFace.length
      && this.regionByFace[anchor.face] !== null
      && this.atlas.eligible_faces[anchor.face] === 1
      && anchor.barycentric.length === 3
      && anchor.barycentric.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
      && Math.abs(anchor.barycentric.reduce((sum, value) => sum + value, 0) - 1) <= 1e-6;
  }

  private shortestPath(start: number, stop?: number, limit = Infinity): {
    distance: Float64Array;
    previous: Int32Array;
  } {
    const distance = new Float64Array(this.regionByFace.length).fill(Infinity);
    const previous = new Int32Array(this.regionByFace.length).fill(-1);
    const queue = new MinHeap();
    distance[start] = 0;
    queue.push(0, start);
    for (;;) {
      const next = queue.pop();
      if (!next) break;
      const [currentDistance, face] = next;
      if (currentDistance !== distance[face]) continue;
      if (currentDistance > limit || face === stop) break;
      for (const neighbor of this.adjacency[face]) {
        if (!this.regionByFace[neighbor] || this.atlas.eligible_faces[neighbor] !== 1) continue;
        const candidate = currentDistance + this.cvec(face).distanceTo(this.cvec(neighbor));
        if (
          candidate < distance[neighbor]
          || (candidate === distance[neighbor] && face < previous[neighbor])
        ) {
          distance[neighbor] = candidate;
          previous[neighbor] = face;
          queue.push(candidate, neighbor);
        }
      }
    }
    return { distance, previous };
  }

  private walkDirection(
    start: number,
    direction: THREE.Vector3,
    offset: number,
  ): { anchor: Anchor; achieved_m: number } {
    direction.normalize();
    const limit = offset + 2 * this.maxEdge;
    const { distance } = this.shortestPath(start, undefined, limit);
    const origin = this.cvec(start);
    const displacement = new THREE.Vector3();
    let best = -1;
    let bestScore = Infinity;
    const minimumDistance = Math.max(offset * 0.65, offset - 1.5 * this.maxEdge);
    for (let face = 0; face < distance.length; face++) {
      const walked = distance[face];
      if (!Number.isFinite(walked) || walked < minimumDistance || walked > limit) continue;
      displacement.copy(this.cvec(face)).sub(origin);
      const projection = displacement.dot(direction);
      if (projection <= 0) continue;
      const perpendicular = Math.sqrt(Math.max(0, displacement.lengthSq() - projection * projection));
      const score = Math.abs(walked - offset) + 1.5 * perpendicular + 0.25 * Math.max(0, offset - projection);
      if (score < bestScore || (score === bestScore && face < best)) {
        best = face;
        bestScore = score;
      }
    }
    if (best < 0) {
      throw new InkLangError(
        "INKLANG_OFFSET_OUT_OF_BOUNDS",
        `surface walk from face ${start} cannot reach ${offset.toFixed(4)} m in the requested direction`,
      );
    }
    return { anchor: { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] }, achieved_m: distance[best] };
  }

  private between(start: number, stop: number): { anchor: Anchor; achieved_m: number } {
    const { distance, previous } = this.shortestPath(start, stop);
    if (!Number.isFinite(distance[stop])) {
      throw new InkLangError("INKLANG_OFFSET_OUT_OF_BOUNDS", `faces ${start} and ${stop} are disconnected`);
    }
    const path: number[] = [];
    for (let face = stop; face >= 0; face = previous[face]) {
      path.push(face);
      if (face === start) break;
    }
    if (path.at(-1) !== start) {
      throw new InkLangError("INKLANG_OFFSET_OUT_OF_BOUNDS", `no surface path between faces ${start} and ${stop}`);
    }
    const target = distance[stop] / 2;
    let best = start;
    let bestDifference = Infinity;
    for (const face of path) {
      const difference = Math.abs(distance[face] - target);
      if (difference < bestDifference || (difference === bestDifference && face < best)) {
        best = face;
        bestDifference = difference;
      }
    }
    return { anchor: { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] }, achieved_m: distance[best] };
  }

  anchorForPhrase(site: SitePhrase, besideDirection?: -1 | 1): Anchor {
    return this.anchorForPhraseDetailed(site, besideDirection).anchor;
  }

  /**
   * Resolve relations by a bounded walk over connected canonical-rest faces,
   * reporting the surface distance the walk actually covered so a caller can
   * check it against the distance that was asked for.
   */
  anchorForPhraseDetailed(
    site: SitePhrase,
    besideDirection?: -1 | 1,
  ): { anchor: Anchor; relative: RelativeWalk | null } {
    const relation = site.relation;
    if (!relation) return { anchor: this.anchorFor(site), relative: null };
    const base = this.anchorFor({
      id: site.id,
      laterality: site.laterality,
      aspect: site.aspect,
      level: site.level,
      region_uv: site.region_uv,
    });
    if (relation.kind === "between") {
      const other = this.anchorFor({
        id: relation.other!.id,
        laterality: relation.other!.laterality,
        aspect: null,
        level: null,
      });
      const midpoint = this.between(base.face, other.face);
      return {
        anchor: midpoint.anchor,
        relative: {
          kind: "between",
          requested_m: null,
          achieved_m: midpoint.achieved_m,
          reference_face: base.face,
        },
      };
    }
    const offset = relation.offset_m!;
    let direction: THREE.Vector3;
    if (relation.kind === "above") direction = new THREE.Vector3(0, 0, 1);
    else if (relation.kind === "below") direction = new THREE.Vector3(0, 0, -1);
    else if (relation.kind === "behind") direction = new THREE.Vector3(0, 1, 0);
    else if (relation.kind === "in_front") direction = new THREE.Vector3(0, -1, 0);
    else {
      const sign = besideDirection ?? Math.sign(this.cvec(base.face).x);
      if (sign === 0) {
        throw new InkLangError("INKLANG_AMBIGUOUS_RELATION", `beside ${site.id} needs a left or right direction`);
      }
      direction = new THREE.Vector3(sign, 0, 0);
    }
    const walked = this.walkDirection(base.face, direction, offset);
    return {
      anchor: walked.anchor,
      relative: {
        kind: relation.kind,
        requested_m: offset,
        achieved_m: walked.achieved_m,
        reference_face: base.face,
      },
    };
  }

  /**
   * Longest centroid-to-centroid step in the connected face graph. A relative
   * walk cannot resolve finer than this, so it bounds how closely `achieved_m`
   * can ever match a requested offset.
   */
  surfaceStepLimit(): number {
    return this.maxEdge;
  }

  presentSites(): string[] {
    return this.atlas.sites;
  }
}
