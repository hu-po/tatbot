// The region atlas: which named inklang site every skin face belongs to, and
// the per-region (u,v) chart that grounds "upper inner forearm" in an anchor.
// The atlas is data (public/bodies/<id>.regions.json, written by
// tools/build_atlas.ts today, a Blender pass tomorrow); this module only
// indexes and queries it. No React, no DOM.
import * as THREE from "three";
import type { Anchor } from "./anchor.ts";
import { SITES, ZONES, INKLANG_VERSION, type SitePhrase, type Laterality, type Level } from "./lang.ts";

export interface AtlasData {
  inklang: string;
  body: { id: string; sha256: string };
  sites: string[];
  /** Per face: siteIndex*4 + laterality (0 none, 1 left, 2 right); -1 = not skin. */
  faces: number[];
}

export function parseAtlas(x: unknown, expectFaces?: number): AtlasData {
  const fail = (m: string): never => { throw new Error(`region atlas: ${m}`); };
  if (typeof x !== "object" || x === null) fail("not an object");
  const a = x as Record<string, unknown>;
  if (a.inklang !== INKLANG_VERSION) fail(`lexicon ${String(a.inklang)} (app speaks ${INKLANG_VERSION}) — regenerate with tools/build_atlas.ts`);
  const body = a.body as Record<string, unknown> | undefined;
  if (!body || typeof body.id !== "string" || typeof body.sha256 !== "string") return fail("body needs id and sha256");
  if (!Array.isArray(a.sites) || !(a.sites as unknown[]).every((s) => typeof s === "string" && s in SITES)) fail("sites must be known leaf site ids");
  if (!Array.isArray(a.faces)) fail("faces must be an array");
  if (expectFaces !== undefined && (a.faces as unknown[]).length !== expectFaces) {
    fail(`face count ${(a.faces as unknown[]).length} does not match the loaded body (${expectFaces})`);
  }
  return a as unknown as AtlasData;
}

export interface RegionRef {
  id: string;
  laterality: "left" | "right" | null;
}

interface Region {
  key: string;
  id: string;
  laterality: "left" | "right" | null;
  faces: number[];
  mean: THREE.Vector3;
  normal: THREE.Vector3;
  /** Chart axes: u proximal→distal (down the body), v medial→lateral. */
  uAxis: THREE.Vector3;
  vAxis: THREE.Vector3;
  uRange: [number, number];
  vRange: [number, number];
}

const rkey = (id: string, lat: "left" | "right" | null) => (lat ? `${id}:${lat}` : id);

/** Face-level site index over one loaded body. */
export class AtlasIndex {
  readonly atlas: AtlasData;
  private readonly regionByFace: (RegionRef | null)[];
  private readonly regions = new Map<string, Region>();
  private readonly faceNormals: Float32Array;
  private readonly centroids: Float32Array;

  constructor(atlas: AtlasData, geometry: THREE.BufferGeometry, centroids: Float32Array) {
    this.atlas = atlas;
    this.centroids = centroids;
    const pos = geometry.getAttribute("position") as THREE.BufferAttribute;
    const nFaces = pos.count / 3;
    if (atlas.faces.length !== nFaces) throw new Error(`region atlas: ${atlas.faces.length} faces for a ${nFaces}-face body`);

    this.faceNormals = new Float32Array(nFaces * 3);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3();
    for (let f = 0; f < nFaces; f++) {
      a.fromBufferAttribute(pos, 3 * f); b.fromBufferAttribute(pos, 3 * f + 1); c.fromBufferAttribute(pos, 3 * f + 2);
      n.subVectors(b, a).cross(c.sub(a)).normalize();
      this.faceNormals[3 * f] = n.x; this.faceNormals[3 * f + 1] = n.y; this.faceNormals[3 * f + 2] = n.z;
    }

    this.regionByFace = new Array(nFaces).fill(null);
    for (let f = 0; f < nFaces; f++) {
      const v = atlas.faces[f];
      if (v < 0) continue;
      const id = atlas.sites[v >> 2];
      const laterality = (v & 3) === 1 ? "left" as const : (v & 3) === 2 ? "right" as const : null;
      this.regionByFace[f] = { id, laterality };
      const key = rkey(id, laterality);
      let r = this.regions.get(key);
      if (!r) {
        r = { key, id, laterality, faces: [], mean: new THREE.Vector3(), normal: new THREE.Vector3(), uAxis: new THREE.Vector3(), vAxis: new THREE.Vector3(), uRange: [0, 0], vRange: [0, 0] };
        this.regions.set(key, r);
      }
      r.faces.push(f);
    }
    for (const r of this.regions.values()) this.buildChart(r);
  }

  private cvec(f: number): THREE.Vector3 {
    return new THREE.Vector3(this.centroids[3 * f], this.centroids[3 * f + 1], this.centroids[3 * f + 2]);
  }
  private nvec(f: number): THREE.Vector3 {
    return new THREE.Vector3(this.faceNormals[3 * f], this.faceNormals[3 * f + 1], this.faceNormals[3 * f + 2]);
  }

  /** u along the region's dominant extent, oriented down the body (or forward
   *  when the region is horizontal, like a foot); v perpendicular in the
   *  tangent plane, oriented toward the body's outside. */
  private buildChart(r: Region): void {
    for (const f of r.faces) { r.mean.add(this.cvec(f)); r.normal.add(this.nvec(f)); }
    r.mean.divideScalar(r.faces.length);
    if (r.normal.lengthSq() < 1e-12) r.normal.set(0, -1, 0); else r.normal.normalize();
    // Dominant covariance axis by power iteration.
    const cov = [0, 0, 0, 0, 0, 0]; // xx, xy, xz, yy, yz, zz
    const d = new THREE.Vector3();
    for (const f of r.faces) {
      d.copy(this.cvec(f)).sub(r.mean);
      cov[0] += d.x * d.x; cov[1] += d.x * d.y; cov[2] += d.x * d.z;
      cov[3] += d.y * d.y; cov[4] += d.y * d.z; cov[5] += d.z * d.z;
    }
    const u = new THREE.Vector3(0.3, 0.4, 0.87);
    for (let i = 0; i < 24; i++) {
      u.set(
        cov[0] * u.x + cov[1] * u.y + cov[2] * u.z,
        cov[1] * u.x + cov[3] * u.y + cov[4] * u.z,
        cov[2] * u.x + cov[4] * u.y + cov[5] * u.z,
      );
      if (u.lengthSq() < 1e-20) { u.set(0, 0, -1); break; }
      u.normalize();
    }
    // Orient: down the body when the region is at all vertical, else forward.
    if (Math.abs(u.z) > 0.25 ? u.z > 0 : u.y > 0) u.negate();
    r.uAxis.copy(u);
    r.vAxis.crossVectors(r.normal, u).normalize();
    if (r.vAxis.lengthSq() < 1e-12) r.vAxis.set(1, 0, 0);
    // Toward the outside: +x on the left half, -x on the right.
    const lateral = (r.laterality ?? (r.mean.x >= 0 ? "left" : "right")) === "left" ? 1 : -1;
    if (r.vAxis.x * lateral < 0) r.vAxis.negate();
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const f of r.faces) {
      d.copy(this.cvec(f)).sub(r.mean);
      const pu = d.dot(r.uAxis), pv = d.dot(r.vAxis);
      u0 = Math.min(u0, pu); u1 = Math.max(u1, pu);
      v0 = Math.min(v0, pv); v1 = Math.max(v1, pv);
    }
    r.uRange = [u0, u1]; r.vRange = [v0, v1];
  }

  regionOf(face: number): RegionRef | null {
    return this.regionByFace[face] ?? null;
  }

  /** All faces of a leaf site (or zone) on one side; null side = every side. */
  facesOf(id: string, laterality: "left" | "right" | null): number[] {
    const leaves = ZONES[id] ? ZONES[id].members : [id];
    const out: number[] = [];
    for (const leaf of leaves) {
      for (const r of this.regions.values()) {
        if (r.id !== leaf) continue;
        if (laterality !== null && r.laterality !== null && r.laterality !== laterality) continue;
        out.push(...r.faces);
      }
    }
    return out;
  }

  /** The (u,v) chart coordinate of a face inside its own region, each in [0,1]. */
  uvOf(face: number): [number, number] | null {
    const ref = this.regionByFace[face];
    if (!ref) return null;
    const r = this.regions.get(rkey(ref.id, ref.laterality))!;
    const d = this.cvec(face).sub(r.mean);
    const nu = (d.dot(r.uAxis) - r.uRange[0]) / Math.max(1e-9, r.uRange[1] - r.uRange[0]);
    const nv = (d.dot(r.vAxis) - r.vRange[0]) / Math.max(1e-9, r.vRange[1] - r.vRange[0]);
    return [Math.min(1, Math.max(0, nu)), Math.min(1, Math.max(0, nv))];
  }

  /** Does this face's aspect (from its normal) match, for aspects the site declares? */
  private aspectOf(face: number, ref: RegionRef): string | null {
    const allowed = SITES[ref.id]?.aspects ?? [];
    if (allowed.length === 0) return null;
    const n = this.nvec(face);
    const scored: [string, number][] = [];
    for (const a of allowed) {
      let s = -1;
      if (a === "front") s = -n.y;
      else if (a === "back") s = n.y;
      else if (a === "top") s = n.z;
      else if (a === "side") s = Math.abs(n.x) - 0.2;
      else if (a === "inner" || a === "outer") {
        const lat = ref.laterality ?? (this.cvec(face).x >= 0 ? "left" : "right");
        const inward = lat === "left" ? -n.x : n.x;
        s = a === "inner" ? inward : -inward;
      }
      scored.push([a, s]);
    }
    scored.sort((x, y) => y[1] - x[1]);
    return scored[0][1] > 0.35 ? scored[0][0] : null;
  }

  /** Resolve a parsed site phrase to a concrete anchor. A sided site with no
   *  laterality defaults to the left (deterministic; the UI says so). */
  anchorFor(site: Pick<SitePhrase, "id"> & Partial<SitePhrase> & { uv?: [number, number] | null }): Anchor {
    const spec = SITES[site.id] ?? ZONES[site.id];
    if (!spec) throw new Error(`region atlas: unknown site "${site.id}"`);
    let lat: "left" | "right" | null = site.laterality === "left" || site.laterality === "right" ? site.laterality : null;
    if (lat === null && spec.laterality === "sided") lat = "left";
    let faces = this.facesOf(site.id, lat);
    if (faces.length === 0) throw new Error(`region atlas: no faces for ${lat ?? ""} ${site.id}`.trim());
    const aspect = site.aspect ?? null;
    if (aspect) {
      const filtered = faces.filter((f) => this.aspectOf(f, this.regionByFace[f]!) === aspect);
      if (filtered.length > 0) faces = filtered;
    }
    const level: Level | null = site.level ?? null;
    if (aspect === null && level === null && !site.uv) {
      // No refinement: an anchor hint places the point where people actually
      // point — "the elbow" is the olecranon (the backmost point), "the knee"
      // the kneecap — and everything else centres on the region.
      const hint = (SITES[site.id] as { anchor?: string } | undefined)?.anchor;
      if (hint === "extremum_back" || hint === "extremum_front") {
        let best = faces[0], bv = hint === "extremum_back" ? -Infinity : Infinity;
        for (const f of faces) {
          const y = this.centroids[3 * f + 1];
          if (hint === "extremum_back" ? y > bv : y < bv) { bv = y; best = f; }
        }
        return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
      }
      // The visually-centered spot is the face nearest the selection's 3D
      // centroid — not the (u,v) chart middle, which lands off on thin or
      // two-lobed regions (the chest panel around the sternum).
      const mean = new THREE.Vector3();
      for (const f of faces) mean.add(this.cvec(f));
      mean.divideScalar(faces.length);
      // A midline site's centre is on the midline, even when its lobes are
      // uneven (the chest panel flanks the sternum).
      if (spec.laterality === "midline") mean.x = 0;
      let best = faces[0], bd = Infinity;
      for (const f of faces) {
        const d = this.cvec(f).distanceToSquared(mean);
        if (d < bd) { bd = d; best = f; }
      }
      return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
    }
    const target: [number, number] = site.uv ?? [level === "upper" ? 0.22 : level === "lower" ? 0.78 : 0.5, 0.5];
    if (level) {
      const band = faces.filter((f) => {
        const uv = this.uvOf(f);
        if (!uv) return false;
        return level === "upper" ? uv[0] < 0.45 : level === "lower" ? uv[0] > 0.55 : uv[0] > 0.28 && uv[0] < 0.72;
      });
      if (band.length > 0) faces = band;
    }
    let best = faces[0], bd = Infinity;
    for (const f of faces) {
      const uv = this.uvOf(f);
      if (!uv) continue;
      const d = (uv[0] - target[0]) ** 2 + (uv[1] - target[1]) ** 2;
      if (d < bd) { bd = d; best = f; }
    }
    return { face: best, barycentric: [1 / 3, 1 / 3, 1 / 3] };
  }

  /** Name where an anchor actually is — the truthful caption's site phrase. */
  describe(anchor: Anchor): (SitePhrase & { uv: [number, number] }) | null {
    const ref = this.regionByFace[anchor.face];
    if (!ref) return null;
    const uv = this.uvOf(anchor.face)!;
    const spec = SITES[ref.id];
    const laterality: Laterality | null = spec.laterality === "midline" ? null : ref.laterality;
    const level: Level | null = spec.geometry === "crease" ? null : uv[0] < 0.3 ? "upper" : uv[0] > 0.7 ? "lower" : null;
    return { id: ref.id, laterality, aspect: this.aspectOf(anchor.face, ref), level, uv };
  }

  /** Is the anchor inside the named site (zone members count)? Laterality must
   *  match when both sides state one. */
  contains(site: Pick<SitePhrase, "id" | "laterality">, anchor: Anchor): boolean {
    const ref = this.regionByFace[anchor.face];
    if (!ref) return false;
    const leaves = ZONES[site.id] ? ZONES[site.id].members : [site.id];
    // A child leaf counts as inside its parent: the navel is on the stomach.
    const parent = SITES[ref.id]?.parent;
    if (!leaves.includes(ref.id) && (parent === undefined || !leaves.includes(parent))) return false;
    const want = site.laterality === "left" || site.laterality === "right" ? site.laterality : null;
    return want === null || ref.laterality === null || ref.laterality === want;
  }

  /** Nearest labeled skin face to a world point. */
  private nearestFace(p: THREE.Vector3): number {
    let best = 0, bd = Infinity;
    for (let f = 0; f < this.regionByFace.length; f++) {
      if (!this.regionByFace[f]) continue;
      const dx = this.centroids[3 * f] - p.x, dy = this.centroids[3 * f + 1] - p.y, dz = this.centroids[3 * f + 2] - p.z;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bd) { bd = d; best = f; }
    }
    return best;
  }

  /** Resolve a full site phrase, relations included: "two inches below the
   *  left collarbone" walks from the base anchor; "between" takes the
   *  midpoint of the two anchors. Plain phrases fall through to anchorFor. */
  anchorForPhrase(p: SitePhrase & { uv?: [number, number] | null }): Anchor {
    const rel = p.rel;
    if (!rel) return this.anchorFor(p);
    const base = this.anchorFor({ id: p.id, laterality: p.laterality, aspect: p.aspect, level: p.level });
    const basePos = this.cvec(base.face);
    let target: THREE.Vector3;
    if (rel.kind === "between") {
      const other = this.anchorFor({ id: rel.other!.id, laterality: rel.other!.laterality, aspect: null, level: null });
      target = basePos.add(this.cvec(other.face)).multiplyScalar(0.5);
    } else {
      const m = rel.offset_m ?? 0.05;
      const dir = rel.kind === "above" ? new THREE.Vector3(0, 0, m)
        : rel.kind === "below" ? new THREE.Vector3(0, 0, -m)
        : rel.kind === "behind" ? new THREE.Vector3(0, m, 0)
        : rel.kind === "in_front" ? new THREE.Vector3(0, -m, 0)
        : new THREE.Vector3(Math.sign(basePos.x || 1) * m, 0, 0); // beside: outboard
      target = basePos.add(dir);
    }
    return { face: this.nearestFace(target), barycentric: [1 / 3, 1 / 3, 1 / 3] };
  }

  /** Site ids present in this atlas (for UI palettes). */
  presentSites(): string[] {
    return this.atlas.sites;
  }
}
