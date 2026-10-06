// The placement file: the contract between this app and the robot pipeline.
// Mirrors config/inkmap/placement.schema.json. Bump SCHEMA_VERSION
// on any breaking change and keep the JSON Schema in the same commit.
import { validateArtworkShape, type ArtworkRecord } from "./artwork-record.ts";
import type { Anchor } from "./anchor.ts";
import type { InkLangIntent, InkLangResolution } from "./inklang/index.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_ASSET_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "./body.ts";

export const SCHEMA_VERSION = 6;

export interface DesignMeta {
  id: string;
  name: string;
  /** Relative to the site root (public/), or a data: URL for a design made in the session. */
  path: string;
  /** Natural size when first placed, mm. Aspect matches the frozen program canvas. */
  default_size_mm: [number, number];
  /** Frozen shared artwork; previews are derived from its program. */
  embedded?: EmbeddedDesign;
  /** Runtime asset provenance. Portable simulation/project envelopes retain it. */
  sourcePath?: string;
  sourceSha256?: string;
  /** Collection identity is checked against the frozen SVG bytes. */
  sha256?: string;
  usage?: "artwork" | "preview";
  /** False keeps a stock design out of the picker: the simulator's artwork
   *  collection and the showcase still draw on it by id. */
  library?: boolean;
  family?: string;
  split?: "train" | "validation" | "test";
  source?: import("./artwork-record.ts").ArtworkSource;
}

/** Artwork carried inside the placement file so it stays self-contained. */
export type EmbeddedDesign = ArtworkRecord;

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
  language?: {
    sentence: string;
    program: Record<string, unknown>;
    intent?: InkLangIntent;
    resolution?: InkLangResolution;
    consumer_policy?: string;
  };
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
  schema_version: typeof SCHEMA_VERSION;
  units: { length: "m"; tattoo_size: "mm"; up: "+z" };
  body: PlacementBody;
  placements: Placement[];
  /** Every used design, keyed by design id. */
  designs?: Record<string, EmbeddedDesign>;
}

export interface PlacementBody {
  model_spec_id: typeof MODEL_SPEC_ID;
  model_spec_sha256: typeof MODEL_SPEC_SHA256;
  identity_sha256: typeof REFERENCE_IDENTITY_SHA256 | string;
  topology_sha256: typeof TOPOLOGY_SHA256;
  rest_surface_sha256: typeof REST_SURFACE_SHA256 | string;
  asset_path: string;
  asset_sha256: string;
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
  if (f.schema_version !== SCHEMA_VERSION) fail("unsupported schema/model");
  const u = f.units as Record<string, unknown> | undefined;
  if (!u || u.length !== "m" || u.tattoo_size !== "mm" || u.up !== "+z") fail("units must be {length:m, tattoo_size:mm, up:+z}");
  const b = f.body as Record<string, unknown> | undefined;
  if (!b) return fail("body binding is required");
  const bodyKeys = [
    "asset_path", "asset_sha256", "identity_sha256", "model_spec_id",
    "model_spec_sha256", "rest_surface_sha256", "topology_sha256",
  ];
  if (Object.keys(b).sort().join("|") !== bodyKeys.sort().join("|")) fail("body binding fields differ from v6");
  if (
    b.model_spec_id !== MODEL_SPEC_ID
    || b.model_spec_sha256 !== MODEL_SPEC_SHA256
    || b.identity_sha256 !== REFERENCE_IDENTITY_SHA256
    || b.topology_sha256 !== TOPOLOGY_SHA256
    || b.rest_surface_sha256 !== REST_SURFACE_SHA256
    || b.asset_path !== BODY_SPEC.path
    || b.asset_sha256 !== REST_ASSET_SHA256
  ) fail("unsupported schema/model");
  for (const field of ["model_spec_sha256", "identity_sha256", "topology_sha256", "rest_surface_sha256", "asset_sha256"] as const) {
    if (!/^[0-9a-f]{64}$/.test(b[field] as string)) fail(`body.${field} is not a sha256 hex digest`);
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
      if (lg.intent !== undefined && (typeof lg.intent !== "object" || lg.intent === null)) fail(`${where}.language.intent must be an object`);
      if (lg.resolution !== undefined) {
        if (typeof lg.resolution !== "object" || lg.resolution === null) fail(`${where}.language.resolution must be an object`);
        const resolution = lg.resolution as Record<string, unknown>;
        const resolvedAnchor = resolution.anchor as Record<string, unknown> | undefined;
        const resolvedBody = resolution.body as Record<string, unknown> | undefined;
        if (resolution.status !== "resolved") fail(`${where}.language.resolution must be resolved`);
        if (!resolvedAnchor) fail(`${where}.language.resolution must have an anchor`);
        if (!resolvedBody) fail(`${where}.language.resolution must have a body`);
        if (resolvedAnchor!.face !== a.face || JSON.stringify(resolvedAnchor!.barycentric) !== JSON.stringify(a.barycentric)) {
          fail(`${where}.language.resolution anchor differs from placement anchor`);
        }
        if (
          resolvedBody!.model_spec_id !== b.model_spec_id
          || resolvedBody!.model_spec_sha256 !== b.model_spec_sha256
          || resolvedBody!.identity_sha256 !== b.identity_sha256
          || resolvedBody!.topology_sha256 !== b.topology_sha256
          || resolvedBody!.rest_surface_sha256 !== b.rest_surface_sha256
          || resolvedBody!.asset_sha256 !== b.asset_sha256
        ) {
          fail(`${where}.language.resolution body differs from placement body`);
        }
      }
      if (lg.consumer_policy !== undefined && (typeof lg.consumer_policy !== "string" || lg.consumer_policy.length === 0)) {
        fail(`${where}.language.consumer_policy must be a non-empty string`);
      }
    }
  }
  if (f.designs !== undefined) {
    if (typeof f.designs !== "object" || f.designs === null || Array.isArray(f.designs)) return fail("designs must be an object keyed by design id");
    for (const [id, d0] of Object.entries(f.designs as Record<string, unknown>)) {
      if (!id) fail("artwork IDs must be nonempty");
      validateArtworkShape(d0);
    }
    for (const p of f.placements as Placement[]) {
      if (p.design_id.startsWith("gen-") && !(p.design_id in (f.designs as object))) fail(`placement ${p.id} references generated design ${p.design_id} that is not embedded`);
    }
  }
}
