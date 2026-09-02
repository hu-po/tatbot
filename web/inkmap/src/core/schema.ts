// The placement file: the contract between this app and the robot pipeline.
// Mirrors config/inkmap/placement.schema.json. Bump SCHEMA_VERSION
// on any breaking change and keep the JSON Schema in the same commit.
import type { Anchor } from "./anchor.ts";

export const SCHEMA_VERSION = 4;
/** Older versions a reader still accepts (v1 = no embedded designs, v2 = no inklang, v3 = whole-asset hash only). */
export const ACCEPTED_VERSIONS: readonly number[] = [1, 2, 3, 4];

export interface DesignMeta {
  id: string;
  name: string;
  /** Relative to the site root (public/), or a data: URL for a design made in the session. */
  path: string;
  /** Natural size when first placed, mm. Aspect matches the SVG viewBox. */
  default_size_mm: [number, number];
  /** Present only for designs generated in the app; the SVG text travels in the placement file. */
  embedded?: EmbeddedDesign;
}

/** A design that is not one of the site's files: carried inside the placement file so it stays self-contained. */
export interface EmbeddedDesign {
  name: string;
  svg: string;
  default_size_mm: [number, number];
  source?: { kind: "generated"; model: string; prompt: string; seed: number };
}

export interface Placement {
  id: string;
  design_id: string;
  anchor: Anchor;
  rotation_rad: number;
  size_mm: [number, number];
  mirror: boolean;
  /** v3: the named inklang body site this placement belongs to (ids from config/inkmap/sites.json). */
  site?: PlacementSite;
  /** v3: the tattoo program and its canonical sentence; the program is validated by the inklang core (lang.ts), not here. */
  language?: { sentence: string; program: Record<string, unknown> };
}

export interface PlacementSite {
  id: string;
  laterality: "left" | "right" | "center" | null;
  aspect: string | null;
  level?: "upper" | "lower" | "mid" | null;
  /** Chart coordinate inside the region (u proximal→distal, v medial→lateral). */
  uv?: [number, number];
  /** inklang lexicon version the site id is valid in. */
  lexicon: string;
}

export interface PlacementFile {
  schema_version: number;
  units: { length: "m"; tattoo_size: "mm"; up: "+z" };
  body: PlacementBody;
  placements: Placement[];
  /** v2: designs referenced by placements that are not site files, keyed by design id. */
  designs?: Record<string, EmbeddedDesign>;
}

/** v4 keeps the anchor's rest surface stable when rig/material bytes change. */
export interface PlacementBody {
  id: string;
  path: string;
  /** v1-v3 whole-GLB identity. */
  sha256?: string;
  /** v4 whole-asset provenance. */
  asset_sha256?: string;
  /** v4 canonical non-indexed rest-surface signed-10-micrometre XYZ digest. */
  surface_sha256?: string;
}

export function placementAssetSha(body: PlacementBody): string {
  return body.asset_sha256 ?? body.sha256 ?? "";
}

export function placementSurfaceSha(body: PlacementBody): string {
  return body.surface_sha256 ?? body.sha256 ?? "";
}

export function newPlacementId(): string {
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 8);
  return `p-${t}-${r}`;
}

/** Structural validation without a schema library; the JSON Schema is the authority, this is the in-app gate. */
export function validatePlacementFile(x: unknown): asserts x is PlacementFile {
  const fail = (m: string): never => { throw new Error(`placement file: ${m}`); };
  if (typeof x !== "object" || x === null) fail("not an object");
  const f = x as Record<string, unknown>;
  if (!ACCEPTED_VERSIONS.includes(f.schema_version as number)) fail(`schema_version ${String(f.schema_version)} not in ${ACCEPTED_VERSIONS.join("/")}`);
  const u = f.units as Record<string, unknown> | undefined;
  if (!u || u.length !== "m" || u.tattoo_size !== "mm" || u.up !== "+z") fail("units must be {length:m, tattoo_size:mm, up:+z}");
  const b = f.body as Record<string, unknown> | undefined;
  if (!b || typeof b.id !== "string" || typeof b.path !== "string") return fail("body needs id and path");
  const version = f.schema_version as number;
  if (version >= 4) {
    if (typeof b.asset_sha256 !== "string" || typeof b.surface_sha256 !== "string") {
      fail("v4 body needs asset_sha256 and surface_sha256");
    }
    if (!/^[0-9a-f]{64}$/.test(b.asset_sha256 as string)) fail("body.asset_sha256 is not a sha256 hex digest");
    if (!/^[0-9a-f]{64}$/.test(b.surface_sha256 as string)) fail("body.surface_sha256 is not a sha256 hex digest");
  } else {
    if (typeof b.sha256 !== "string") fail("v1-v3 body needs sha256");
    if (!/^[0-9a-f]{64}$/.test(b.sha256 as string)) fail("body.sha256 is not a sha256 hex digest");
  }
  if (!Array.isArray(f.placements)) fail("placements must be an array");
  for (const [i, p0] of (f.placements as unknown[]).entries()) {
    const p = p0 as Record<string, unknown>;
    const where = `placements[${i}]`;
    if (typeof p.id !== "string" || typeof p.design_id !== "string") fail(`${where}: id and design_id must be strings`);
    const a = p.anchor as Record<string, unknown> | undefined;
    if (!a || !Number.isInteger(a.face) || (a.face as number) < 0) return fail(`${where}.anchor.face must be a non-negative integer`);
    const bc = a.barycentric as unknown;
    if (!Array.isArray(bc) || bc.length !== 3 || !bc.every((w) => typeof w === "number" && w >= 0 && w <= 1)) fail(`${where}.anchor.barycentric must be three weights in [0,1]`);
    if (Math.abs((bc as number[]).reduce((s, w) => s + w, 0) - 1) > 1e-6) fail(`${where}.anchor.barycentric must sum to 1`);
    if (typeof p.rotation_rad !== "number" || !Number.isFinite(p.rotation_rad)) fail(`${where}.rotation_rad must be a finite number`);
    const s = p.size_mm as unknown;
    if (!Array.isArray(s) || s.length !== 2 || !s.every((v) => typeof v === "number" && v > 0)) fail(`${where}.size_mm must be two positive numbers`);
    if (typeof p.mirror !== "boolean") fail(`${where}.mirror must be a boolean`);
    if (p.site !== undefined) {
      const st = p.site as Record<string, unknown>;
      if (typeof st.id !== "string" || st.id.length === 0) fail(`${where}.site.id must be a non-empty string`);
      if (typeof st.lexicon !== "string" || st.lexicon.length === 0) fail(`${where}.site.lexicon must name the inklang lexicon version`);
      if (![null, "left", "right", "center"].includes(st.laterality as string | null)) fail(`${where}.site.laterality must be left/right/center/null`);
      if (st.aspect !== null && typeof st.aspect !== "string") fail(`${where}.site.aspect must be a string or null`);
      if (st.level !== undefined && ![null, "upper", "lower", "mid"].includes(st.level as string | null)) fail(`${where}.site.level must be upper/lower/mid/null`);
      if (st.uv !== undefined) {
        const uv = st.uv as unknown;
        if (!Array.isArray(uv) || uv.length !== 2 || !uv.every((v) => typeof v === "number" && v >= 0 && v <= 1)) fail(`${where}.site.uv must be two numbers in [0,1]`);
      }
    }
    if (p.language !== undefined) {
      const lg = p.language as Record<string, unknown>;
      if (typeof lg.sentence !== "string" || lg.sentence.length === 0) fail(`${where}.language.sentence must be a non-empty string`);
      if (typeof lg.program !== "object" || lg.program === null) fail(`${where}.language.program must be an object`);
    }
  }
  if (f.designs !== undefined) {
    if (typeof f.designs !== "object" || f.designs === null || Array.isArray(f.designs)) return fail("designs must be an object keyed by design id");
    for (const [id, d0] of Object.entries(f.designs as Record<string, unknown>)) {
      const d = d0 as Record<string, unknown>;
      const where = `designs[${id}]`;
      if (typeof d.name !== "string" || typeof d.svg !== "string" || !d.svg.includes("<svg")) fail(`${where}: needs name and svg text`);
      const sz = d.default_size_mm as unknown;
      if (!Array.isArray(sz) || sz.length !== 2 || !sz.every((v) => typeof v === "number" && v > 0)) fail(`${where}.default_size_mm must be two positive numbers`);
    }
    for (const p of f.placements as Placement[]) {
      if (p.design_id.startsWith("gen-") && !(p.design_id in (f.designs as object))) fail(`placement ${p.id} references generated design ${p.design_id} that is not embedded`);
    }
  }
}
