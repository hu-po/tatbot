import sitesJson from "../../../../../config/inkmap/sites.json" with { type: "json" };
import { InkLangError } from "./errors.ts";
import type { Laterality, Level, SitePhrase } from "./types.ts";

export interface SiteEntry {
  name: string;
  group: string;
  laterality: "sided" | "midline" | "any";
  geometry: "flat" | "wrap" | "crease";
  ncic_site: string;
  aspects?: string[];
  aliases?: string[];
  plural?: boolean;
  snomed?: string;
  parent?: string;
  anchor?: string;
}

export interface ZoneEntry {
  name: string;
  laterality: "sided" | "midline";
  members: string[];
  aliases?: string[];
}

interface SiteHit {
  id: string;
  aspect: string | null;
  level: Level | null;
}

interface CompoundAlias {
  site: string;
  aspect?: string;
  level?: Level;
  canonical?: boolean;
}

export const INKLANG_VERSION: string = sitesJson.inklang;
export const SITES = sitesJson.sites as unknown as Record<string, SiteEntry>;
export const ZONES = sitesJson.zones as unknown as Record<string, ZoneEntry>;

export function normalizeWords(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[.,!?;:()\"]/g, " ")
    .replace(/-/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

export const wordKey = (words: string[]): string => words.join(" ");

const siteMap = new Map<string, SiteHit>();
const canonicalPhrase = new Map<string, string>();

for (const [id, entry] of Object.entries(SITES)) {
  for (const phrase of [id, entry.name, ...(entry.aliases ?? [])]) {
    siteMap.set(wordKey(normalizeWords(phrase)), { id, aspect: null, level: null });
  }
}
for (const [id, entry] of Object.entries(ZONES)) {
  for (const phrase of [id, entry.name, ...(entry.aliases ?? [])]) {
    siteMap.set(wordKey(normalizeWords(phrase)), { id, aspect: null, level: null });
  }
}
for (const [phrase, target] of Object.entries(
  sitesJson.compound_aliases as Record<string, CompoundAlias>,
)) {
  siteMap.set(wordKey(normalizeWords(phrase)), {
    id: target.site,
    aspect: target.aspect ?? null,
    level: target.level ?? null,
  });
  if (target.canonical) {
    canonicalPhrase.set(`${target.site}|${target.aspect ?? ""}|${target.level ?? ""}`, phrase);
  }
}

export const ASPECT_WORDS = new Map<string, string>();
for (const [id, aliases] of Object.entries(sitesJson.aspects as Record<string, string[]>)) {
  ASPECT_WORDS.set(id, id);
  for (const alias of aliases) ASPECT_WORDS.set(alias, id);
}

export const LATERALITY_WORDS = new Map<string, Laterality>();
for (const [id, aliases] of Object.entries(
  sitesJson.laterality_words as Record<string, string[]>,
)) {
  LATERALITY_WORDS.set(id, id as Laterality);
  for (const alias of aliases) LATERALITY_WORDS.set(alias, id as Laterality);
}

export const LEVEL_WORDS = new Map<string, Level>();
for (const [id, aliases] of Object.entries(sitesJson.levels as Record<string, string[]>)) {
  LEVEL_WORDS.set(id, id as Level);
  for (const alias of aliases) LEVEL_WORDS.set(alias, id as Level);
}

export function isZone(id: string): boolean {
  return id in ZONES;
}

export function findSiteSuffix(tokens: string[], from = 0): { hit: SiteHit; start: number } | null {
  for (let start = from; start < tokens.length; start++) {
    const hit = siteMap.get(wordKey(tokens.slice(start)));
    if (hit) return { hit, start };
  }
  return null;
}

export function exactSiteHit(tokens: string[]): SiteHit | null {
  return siteMap.get(wordKey(tokens)) ?? null;
}

export function preferredSiteWords(site: Pick<SitePhrase, "id" | "laterality" | "aspect" | "level">): string {
  const spec = SITES[site.id] ?? ZONES[site.id];
  if (!spec) throw new InkLangError("INKLANG_UNKNOWN_SITE", `unknown site "${site.id}"`, "site.id");
  const parts: string[] = [];
  if (site.laterality !== null) parts.push(site.laterality);
  const compound = canonicalPhrase.get(`${site.id}|${site.aspect ?? ""}|${site.level ?? ""}`);
  if (compound) {
    parts.push(compound);
  } else {
    if (site.level !== null) parts.push(site.level);
    if (site.aspect !== null) parts.push(site.aspect);
    parts.push(spec.name);
  }
  return parts.join(" ");
}

export function validateSitePhrase(sitePhrase: SitePhrase): void {
  const site = SITES[sitePhrase.id];
  const zone = ZONES[sitePhrase.id];
  if (!site && !zone) {
    throw new InkLangError("INKLANG_UNKNOWN_SITE", `unknown site "${sitePhrase.id}"`, "site.id");
  }
  const lateralityRule = (site ?? zone).laterality;
  if (sitePhrase.laterality === "center" && lateralityRule === "sided") {
    throw new InkLangError(
      "INKLANG_INVALID_LATERALITY",
      `${sitePhrase.id} is a sided site; center does not apply`,
      "site.laterality",
    );
  }
  if (
    (sitePhrase.laterality === "left" || sitePhrase.laterality === "right")
    && lateralityRule === "midline"
  ) {
    throw new InkLangError(
      "INKLANG_INVALID_LATERALITY",
      `${sitePhrase.id} is a midline site; left/right do not apply`,
      "site.laterality",
    );
  }
  if (sitePhrase.aspect !== null) {
    if (zone) {
      throw new InkLangError(
        "INKLANG_INVALID_ASPECT",
        `zones take no aspect ("${sitePhrase.aspect}" on ${sitePhrase.id})`,
        "site.aspect",
      );
    }
    if (!(site.aspects ?? []).includes(sitePhrase.aspect)) {
      throw new InkLangError(
        "INKLANG_INVALID_ASPECT",
        `aspect "${sitePhrase.aspect}" not allowed on ${sitePhrase.id}`,
        "site.aspect",
      );
    }
  }
  if (sitePhrase.level !== null && zone) {
    throw new InkLangError(
      "INKLANG_INVALID_LEVEL",
      `zones take no level ("${sitePhrase.level}" on ${sitePhrase.id})`,
      "site.level",
    );
  }
  if (sitePhrase.region_uv) {
    if (
      sitePhrase.region_uv.length !== 2
      || sitePhrase.region_uv.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
    ) {
      throw new InkLangError("INKLANG_PARSE_SYNTAX", "region_uv must contain two values in [0,1]", "site.region_uv");
    }
  }
  const relation = sitePhrase.relation;
  if (!relation) return;
  if (relation.kind === "between") {
    if (!relation.other || !(relation.other.id in SITES)) {
      throw new InkLangError(
        "INKLANG_INVALID_RELATION",
        "between needs a second known leaf site",
        "site.relation.other",
      );
    }
    validateSitePhrase({
      id: relation.other.id,
      laterality: relation.other.laterality,
      aspect: null,
      level: null,
    });
  } else if (
    typeof relation.offset_m !== "number"
    || !(relation.offset_m > 0)
    || relation.offset_m > 0.5
  ) {
    throw new InkLangError(
      "INKLANG_INVALID_RELATION",
      "relation offset must be in (0, 0.5] m",
      "site.relation.offset_m",
    );
  }
}
