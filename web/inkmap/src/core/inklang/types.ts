import type { Anchor } from "../anchor.ts";

export const INTENT_SCHEMA_VERSION = 1;
export const ATLAS_SCHEMA_VERSION = 2;
export const RESOLUTION_SCHEMA_VERSION = 2;
export const RESOLVER_VERSION = 1;

export type Laterality = "left" | "right" | "center";
export type Level = "upper" | "lower" | "mid";
export type RelKind = "above" | "below" | "behind" | "in_front" | "beside" | "between";
export type ResolutionStatus = "resolved" | "needs_choice" | "rejected";
export type ResolutionPolicyId = "interactive" | "seeded-v1";

export interface SiteReference {
  id: string;
  laterality: Laterality | null;
}

export interface SiteRelation {
  kind: RelKind;
  /** Surface offset in metres; absent for `between`. */
  offset_m?: number;
  /** The measure as spoken, retained for a faithful canonical phrase. */
  render?: string;
  /** The second site for `between`. */
  other?: SiteReference;
}

export interface SitePhrase {
  /** A leaf site id or zone id from config/inkmap/sites.json. */
  id: string;
  laterality: Laterality | null;
  aspect: string | null;
  level: Level | null;
  relation?: SiteRelation;
  region_uv?: [number, number];
}

export type InkLangIssueCode =
  | "INKLANG_PARSE_SYNTAX"
  | "INKLANG_UNKNOWN_SITE"
  | "INKLANG_INVALID_LATERALITY"
  | "INKLANG_INVALID_ASPECT"
  | "INKLANG_INVALID_LEVEL"
  | "INKLANG_INVALID_RELATION"
  | "INKLANG_AMBIGUOUS_LATERALITY"
  | "INKLANG_AMBIGUOUS_ZONE"
  | "INKLANG_AMBIGUOUS_RELATION"
  | "INKLANG_UNKNOWN_BODY"
  | "INKLANG_MISSING_ATLAS"
  | "INKLANG_BODY_MISMATCH"
  | "INKLANG_SURFACE_MISMATCH"
  | "INKLANG_VERSION_MISMATCH"
  | "INKLANG_NO_REGION"
  | "INKLANG_OFFSET_OUT_OF_BOUNDS"
  | "INKLANG_ANCHOR_INVALID"
  | "INKLANG_SEMANTIC_MISMATCH";

export interface InkLangIssue {
  code: InkLangIssueCode;
  message: string;
  field?: string;
}

export interface InkLangAmbiguity {
  field: "laterality" | "zone_member" | "relative_direction";
  candidates: string[];
}

export interface InkLangIntent {
  intent_schema_version: typeof INTENT_SCHEMA_VERSION;
  inklang_version: string;
  description: string;
  canonical_phrase: string | null;
  site: SitePhrase | null;
  ambiguities: InkLangAmbiguity[];
  unknown_terms: string[];
  issues: InkLangIssue[];
}

export interface ResolutionPolicy {
  id: ResolutionPolicyId;
  seed: number | null;
}

export interface ResolvedSite {
  site_id: string;
  laterality: Laterality | null;
  aspect: string | null;
  level: Level | null;
  region_uv: [number, number];
  canonical_phrase: string;
}

/**
 * What a relative placement's surface walk actually did. `achieved_m` is the
 * surface path length from the reference anchor to the resolved anchor, so a
 * caller can see how closely the walk met `requested_m` instead of assuming it.
 */
export interface RelativeWalk {
  kind: RelKind;
  /** Requested surface offset in metres; null for `between`, which has no offset. */
  requested_m: number | null;
  achieved_m: number;
  /** Face of the anchor the walk started from. */
  reference_face: number;
}

export interface ResolutionCandidate {
  label: string;
  site_id: string;
  laterality: Laterality | null;
  anchor: Anchor;
}

export interface InkLangResolution {
  resolution_schema_version: typeof RESOLUTION_SCHEMA_VERSION;
  status: ResolutionStatus;
  intent: InkLangIntent;
  body: {
    model_spec_id: string;
    model_spec_sha256: string | null;
    identity_sha256: string | null;
    topology_sha256: string | null;
    rest_surface_sha256: string | null;
    asset_sha256: string | null;
  };
  atlas_schema_version: number;
  resolver: {
    name: "inklang-typescript";
    version: typeof RESOLVER_VERSION;
    policy: ResolutionPolicy;
  };
  anchor: Anchor | null;
  actual: ResolvedSite | null;
  candidates: ResolutionCandidate[];
  issues: InkLangIssue[];
  /** Present when an interactive caller explicitly accepted one offered candidate. */
  choice?: ResolutionCandidate;
  /** Present when a resolved placement was reached by a relative surface walk. */
  relative?: RelativeWalk;
}
