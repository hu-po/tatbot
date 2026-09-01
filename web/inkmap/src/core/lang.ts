// Inklang: the tattoo placement + style language. Program-first, like the sim
// drawing language: a TattooProgram is the source of truth; realize() compiles
// it up to one canonical, truthful sentence and parseSentence() inverts it.
// The lexicons are config, not code: config/inkmap/{sites,styles}.json.
// Deterministic grammar + alias table only — no model in the loop (locked
// 2026-08-31). No React, no DOM.
import sitesJson from "../../../../config/inkmap/sites.json" with { type: "json" };
import stylesJson from "../../../../config/inkmap/styles.json" with { type: "json" };

export const INKLANG_VERSION: string = sitesJson.inklang;

export type Laterality = "left" | "right" | "center";
export type Level = "upper" | "lower" | "mid";

export type RelKind = "above" | "below" | "behind" | "in_front" | "beside" | "between";

/** v0.3: a placement stated relative to a site — "two inches below the left
 *  collarbone", "between the shoulder blades". */
export interface SiteRel {
  kind: RelKind;
  /** Offset in metres; absent for "between". */
  offset_m?: number;
  /** The measure exactly as spoken ("two inches", "just", ""), for faithful realization. */
  render?: string;
  /** The second site, for "between". */
  other?: { id: string; laterality: Laterality | null };
}

export interface SitePhrase {
  /** A leaf site id or a zone id from sites.json. */
  id: string;
  laterality: Laterality | null;
  /** Canonical aspect id ("inner", "back", …), null when unspecified. */
  aspect: string | null;
  /** Position along the site, proximal to distal ("upper inner forearm"). */
  level: Level | null;
  /** Present only for relative placements; the base site is the fields above. */
  rel?: SiteRel;
}

/** The language half of a tattoo: everything a sentence states, nothing it does not. */
export interface TattooProgram {
  inklang: string;
  /** Free noun phrase: "octopus", "snake wrapped around a dagger". */
  motif: string;
  /** Primary style id, null when the sentence names none. */
  style: string | null;
  /** Secondary style ids — "with tribal themes". */
  secondary: string[];
  /** Technique id, null = unspecified (machine is the default and never realized). */
  technique: string | null;
  /** Color id, null = unspecified. */
  color: string | null;
  site: SitePhrase;
}

interface SiteEntry {
  name: string;
  group: string;
  laterality: "sided" | "midline" | "any";
  geometry: "flat" | "wrap" | "crease";
  ncic_site: string;
  aspects?: string[];
  aliases?: string[];
  plural?: boolean;
  snomed?: string;
  /** v0.3: this leaf is a sub-part of another leaf; containment rolls up. */
  parent?: string;
  /** v0.3: where the default anchor goes — centroid (default) | extremum_front | extremum_back. */
  anchor?: string;
}
interface ZoneEntry {
  name: string;
  laterality: "sided" | "midline";
  members: string[];
  aliases?: string[];
}
interface TermEntry { name: string; aliases: string[]; default?: boolean; prompt?: string }

export const SITES: Record<string, SiteEntry> = sitesJson.sites as unknown as Record<string, SiteEntry>;
export const ZONES: Record<string, ZoneEntry> = sitesJson.zones as unknown as Record<string, ZoneEntry>;
export const STYLES: Record<string, TermEntry> = stylesJson.styles;
export const TECHNIQUES: Record<string, TermEntry> = stylesJson.techniques;
export const COLORS: Record<string, TermEntry> = stylesJson.colors;

// ---------------------------------------------------------------------------
// Normalization and alias tables. Keys are normalized token strings; longest
// match always wins, so "watercolor wash" (color) beats "watercolor" (style).

function normalize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[.,!?;:()"]/g, " ")
    .replace(/-/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}
const key = (words: string[]) => words.join(" ");

function buildTermMap(table: Record<string, TermEntry>): Map<string, string> {
  const m = new Map<string, string>();
  for (const [id, e] of Object.entries(table)) {
    for (const phrase of [id, e.name, ...e.aliases]) m.set(key(normalize(phrase)), id);
  }
  return m;
}
const styleMap = buildTermMap(STYLES);
const techMap = buildTermMap(TECHNIQUES);
const colorMap = buildTermMap(COLORS);

interface SiteHit { id: string; aspect: string | null; level: Level | null; zone: boolean }
interface CompoundAlias { site: string; aspect?: string; level?: Level; canonical?: boolean }
const siteMap = new Map<string, SiteHit>();
/** `${site}|${aspect}|${level}` → the compound phrase the realizer prefers ("forehead" over "upper face"). */
const canonicalPhrase = new Map<string, string>();
for (const [id, e] of Object.entries(SITES)) {
  for (const phrase of [id, e.name, ...(e.aliases ?? [])]) {
    siteMap.set(key(normalize(phrase)), { id, aspect: null, level: null, zone: false });
  }
}
for (const [id, e] of Object.entries(ZONES)) {
  for (const phrase of [id, e.name, ...(e.aliases ?? [])]) {
    siteMap.set(key(normalize(phrase)), { id, aspect: null, level: null, zone: true });
  }
}
for (const [phrase, target] of Object.entries(sitesJson.compound_aliases as Record<string, CompoundAlias>)) {
  siteMap.set(key(normalize(phrase)), { id: target.site, aspect: target.aspect ?? null, level: target.level ?? null, zone: false });
  if (target.canonical) canonicalPhrase.set(`${target.site}|${target.aspect ?? ""}|${target.level ?? ""}`, phrase);
}

const aspectMap = new Map<string, string>();
for (const [id, aliases] of Object.entries(sitesJson.aspects as Record<string, string[]>)) {
  aspectMap.set(id, id);
  for (const a of aliases) aspectMap.set(a, id);
}
const latMap = new Map<string, Laterality>();
for (const [id, aliases] of Object.entries(sitesJson.laterality_words as Record<string, string[]>)) {
  latMap.set(id, id as Laterality);
  for (const a of aliases) latMap.set(a, id as Laterality);
}
const levelMap = new Map<string, Level>();
for (const [id, aliases] of Object.entries(sitesJson.levels as Record<string, string[]>)) {
  levelMap.set(id, id as Level);
  for (const a of aliases) levelMap.set(a, id as Level);
}

/** Longest alias match starting at `from`; returns [id, tokens consumed] or null. */
function matchAt<T>(map: Map<string, T>, tokens: string[], from: number): [T, number] | null {
  for (let end = tokens.length; end > from; end--) {
    const hit = map.get(key(tokens.slice(from, end)));
    if (hit !== undefined) return [hit, end - from];
  }
  return null;
}

// ---------------------------------------------------------------------------
// Validation: a program must be realizable without lying.

export function isZone(id: string): boolean {
  return id in ZONES;
}

export function validateProgram(p: TattooProgram): void {
  const fail = (m: string): never => { throw new Error(`inklang program: ${m}`); };
  if (typeof p.motif !== "string" || normalize(p.motif).length === 0) fail("motif must be a non-empty phrase");
  if (p.style !== null && !(p.style in STYLES)) fail(`unknown style "${p.style}"`);
  for (const s of p.secondary) if (!(s in STYLES)) fail(`unknown secondary style "${s}"`);
  if (p.technique !== null && !(p.technique in TECHNIQUES)) fail(`unknown technique "${p.technique}"`);
  if (p.color !== null && !(p.color in COLORS)) fail(`unknown color "${p.color}"`);
  const site = SITES[p.site.id];
  const zone = ZONES[p.site.id];
  if (!site && !zone) return fail(`unknown site "${p.site.id}"`);
  const latRule = (site ?? zone).laterality;
  if (p.site.laterality === "center" && latRule === "sided") fail(`${p.site.id} is a sided site; "center" does not apply`);
  if ((p.site.laterality === "left" || p.site.laterality === "right") && latRule === "midline") {
    fail(`${p.site.id} is a midline site; left/right do not apply`);
  }
  if (p.site.aspect !== null) {
    if (zone) return fail(`zones take no aspect ("${p.site.aspect}" on ${p.site.id})`);
    if (!(site.aspects ?? []).includes(p.site.aspect)) fail(`aspect "${p.site.aspect}" not allowed on ${p.site.id} (allowed: ${(site.aspects ?? []).join(", ") || "none"})`);
  }
  if (p.site.level !== null) {
    if (zone) return fail(`zones take no level ("${p.site.level}" on ${p.site.id})`);
    if (!["upper", "lower", "mid"].includes(p.site.level)) fail(`unknown level "${p.site.level}"`);
  }
  const rel = p.site.rel;
  if (rel !== undefined) {
    if (!["above", "below", "behind", "in_front", "beside", "between"].includes(rel.kind)) fail(`unknown relation "${rel.kind}"`);
    if (rel.kind === "between") {
      if (!rel.other || !(rel.other.id in SITES)) return fail(`"between" needs a second known site`);
    } else {
      if (typeof rel.offset_m !== "number" || !(rel.offset_m > 0) || rel.offset_m > 0.5) fail(`relation offset must be in (0, 0.5] m`);
    }
  }
}

/** The style slots as an image-generator prompt fragment; undefined when the
 *  program asks for nothing beyond the generator's default look. */
export function stylePrompt(p: TattooProgram): string | undefined {
  const bits: string[] = [];
  const pick = (table: Record<string, TermEntry>, id: string | null) => {
    const e = id === null ? undefined : table[id];
    if (e?.prompt) bits.push(e.prompt);
  };
  pick(STYLES, p.style);
  for (const s of p.secondary) pick(STYLES, s);
  pick(TECHNIQUES, p.technique);
  pick(COLORS, p.color);
  return bits.length > 0 ? bits.join(", ") : undefined;
}

// ---------------------------------------------------------------------------
// Realizer: program → the one canonical sentence. Deterministic; omits
// defaults (machine technique, unspecified color) so the sentence never
// carries words the program did not.

function article(word: string): "a" | "an" {
  return /^[aeiou]/.test(word) ? "an" : "a";
}

export function realize(p: TattooProgram): string {
  validateProgram(p);
  const mods: string[] = [];
  if (p.color !== null) mods.push(COLORS[p.color].name);
  if (p.technique !== null && p.technique !== "machine") mods.push(TECHNIQUES[p.technique].name);
  if (p.style !== null) mods.push(STYLES[p.style].name);
  mods.push(normalize(p.motif).join(" "));
  const np = mods.join(" ");
  let s = `${article(np)} ${np}`;
  if (p.secondary.length > 0) {
    s += ` with ${p.secondary.map((id) => STYLES[id].name).join(" and ")} themes`;
  }
  const rel = p.site.rel;
  if (rel !== undefined && rel.kind === "between") {
    const other = rel.other!;
    return `${s} between the ${sitePartString(p.site)} and the ${sitePartString({ id: other.id, laterality: other.laterality, aspect: null, level: null })}`;
  }
  if (rel !== undefined) {
    const word = { above: "above", below: "below", behind: "behind", in_front: "in front of", beside: "beside" }[rel.kind as Exclude<RelKind, "between">];
    const measure = rel.render ? `${rel.render} ` : "";
    return `${s} ${measure}${word} the ${sitePartString(p.site)}`;
  }
  return `${s} on the ${sitePartString(p.site)}`;
}

/** The site phrase without its preposition: "left upper inner forearm", "forehead". */
function sitePartString(site: Pick<SitePhrase, "id" | "laterality" | "aspect" | "level">): string {
  const parts: string[] = [];
  if (site.laterality === "left" || site.laterality === "right") parts.push(site.laterality);
  // A compound alias marked canonical realizes as itself: "throat", not
  // "front neck". The parser maps it straight back, so the form is stable.
  const compound = canonicalPhrase.get(`${site.id}|${site.aspect ?? ""}|${site.level ?? ""}`);
  if (compound !== undefined) {
    parts.push(compound);
  } else {
    if (site.level !== null) parts.push(site.level);
    if (site.aspect !== null) parts.push(site.aspect);
    parts.push((SITES[site.id] ?? ZONES[site.id]).name);
  }
  return parts.join(" ");
}

// ---------------------------------------------------------------------------
// Parser: sentence → program. Grammar:
//   <article> [color] [technique] [style]* <motif> [with <style> (and <style>)* themes]
//     ( on | [<measure>] above|below|behind|beside|in front of | between ) the
//     [laterality] [level] [aspect] <site> [and the <site>]
// Longest alias match wins at every position; extra leading styles beyond the
// first become secondary styles, so parse∘realize is stable.

const UNITS = new Map<string, number>([
  ["inch", 0.0254], ["inches", 0.0254],
  ["cm", 0.01], ["centimeter", 0.01], ["centimeters", 0.01], ["centimetre", 0.01], ["centimetres", 0.01],
  ["mm", 0.001], ["millimeter", 0.001], ["millimeters", 0.001], ["millimetre", 0.001], ["millimetres", 0.001],
]);
const NUMBER_WORDS = new Map<string, number>([
  ["one", 1], ["two", 2], ["three", 3], ["four", 4], ["five", 5], ["six", 6],
  ["seven", 7], ["eight", 8], ["nine", 9], ["ten", 10], ["eleven", 11], ["twelve", 12],
]);

/** [laterality] [level] [aspect] <site alias matched as a suffix>. */
function parseSitePart(tokens: string[], fail: (m: string) => never): SitePhrase {
  if (tokens.length === 0) fail("empty site phrase");
  let laterality: Laterality | null = null;
  let i = 0;
  const lat = latMap.get(tokens[i]);
  if (lat !== undefined) { laterality = lat; i++; }
  let hit: SiteHit | null = null;
  let siteStart = -1;
  for (let k = i; k < tokens.length; k++) {
    const m = siteMap.get(key(tokens.slice(k)));
    if (m !== undefined) { hit = m; siteStart = k; break; }
  }
  if (hit === null && i > 0) {
    // A laterality word may open a site alias ("middle of the back"): retry without it.
    const m = siteMap.get(key(tokens));
    if (m !== undefined) { hit = m; siteStart = 0; laterality = null; i = 0; }
  }
  if (hit === null || siteStart < 0) fail(`unknown site "${tokens.slice(i).join(" ")}"`);
  let aspect: string | null = hit!.aspect;
  let level: Level | null = hit!.level;
  for (let k = i; k < siteStart; k++) {
    const tok = tokens[k];
    // "outside OF THE left shin": filler words pass, and laterality may come
    // after the aspect.
    if (tok === "of" || tok === "the") continue;
    const lat2 = latMap.get(tok);
    if (lat2 !== undefined) {
      if (laterality !== null && laterality !== lat2) fail(`conflicting lateralities "${laterality}" and "${lat2}"`);
      laterality = lat2;
      continue;
    }
    const lv = levelMap.get(tok);
    if (lv !== undefined) {
      if (level !== null && level !== lv) fail(`conflicting levels "${level}" and "${lv}"`);
      level = lv;
      continue;
    }
    const a = aspectMap.get(tok);
    if (a === undefined) fail(`"${tok}" is not an aspect or level word`);
    if (aspect !== null && aspect !== a) fail(`conflicting aspects "${aspect}" and "${a}"`);
    aspect = a;
  }
  return { id: hit!.id, laterality, aspect, level };
}

interface Marker { kind: "on" | RelKind; at: number; len: number }
/** The RIGHTMOST placement marker in the sentence. */
function findMarker(tokens: string[]): Marker | null {
  const pats: [string[], Marker["kind"]][] = [
    [["on", "the"], "on"], [["above", "the"], "above"], [["below", "the"], "below"],
    [["behind", "the"], "behind"], [["beside", "the"], "beside"],
    [["in", "front", "of", "the"], "in_front"], [["between", "the"], "between"],
  ];
  let best: Marker | null = null;
  for (const [pat, kind] of pats) {
    for (let i = tokens.length - pat.length; i >= 0; i--) {
      if (pat.every((w, j) => tokens[i + j] === w)) {
        if (best === null || i > best.at) best = { kind, at: i, len: pat.length };
        break;
      }
    }
  }
  return best;
}

export function parseSentence(sentence: string): TattooProgram {
  const fail = (m: string): never => { throw new Error(`inklang parse: ${m} — in "${sentence}"`); };
  const tokens = normalize(sentence);
  const marker = findMarker(tokens);
  if (marker === null || marker.at <= 0) return fail('expected "… on the <site>" (or above/below/behind/beside/in front of/between)');
  let right = tokens.slice(marker.at + marker.len);
  if (right.length === 0) fail(`no site after "${marker.kind}"`);

  // An optional measure before a relation marker: "two inches below the …".
  let measureLen = 0, offset = 0.05, render = "";
  if (marker.kind !== "on" && marker.kind !== "between") {
    const before = (k: number) => tokens[marker.at - k];
    const unit = UNITS.get(before(1));
    if (unit !== undefined) {
      const w = before(2);
      const amt = NUMBER_WORDS.get(w) ?? (w !== undefined && Number.isFinite(Number(w)) ? Number(w) : null);
      if (amt !== null && amt > 0) { measureLen = 2; offset = amt * unit; render = `${before(2)} ${before(1)}`; }
      else if (w === "a" || w === "an") {
        if (before(3) === "half") { measureLen = 3; offset = 0.5 * unit; render = `half ${w} ${before(1)}`; }
        else { measureLen = 2; offset = unit; render = `${w} ${before(1)}`; }
      }
    } else if (before(1) === "just") { measureLen = 1; offset = 0.03; render = "just"; }
  }

  let sitePhrase: SitePhrase;
  if (marker.kind === "on") {
    sitePhrase = parseSitePart(right, fail);
  } else if (marker.kind === "behind" && measureLen === 0 && (() => {
    // Bare "behind the ear" is a site of its own, not a relation ("3 cm
    // behind the ear" stays one): try the whole phrase (with any laterality
    // pulled out) as a site alias first.
    const latIn = latMap.has(right[0]) ? 1 : 0;
    return siteMap.has(key(["behind", "the", ...right.slice(latIn)]));
  })()) {
    const latIn = latMap.has(right[0]) ? right[0] : null;
    const hit2 = siteMap.get(key(["behind", "the", ...right.slice(latIn ? 1 : 0)]))!;
    sitePhrase = { id: hit2.id, laterality: latIn ? latMap.get(latIn)! : null, aspect: hit2.aspect, level: hit2.level };
  } else if (marker.kind === "between") {
    // "between the X and the Y", or "between the <sided site>s".
    let andAt = -1;
    for (let k = right.length - 2; k >= 1; k--) if (right[k] === "and") { andAt = k; break; }
    if (andAt > 0) {
      const a = parseSitePart(right.slice(0, andAt), fail);
      const bTokens = right[andAt + 1] === "the" ? right.slice(andAt + 2) : right.slice(andAt + 1);
      const b = parseSitePart(bTokens, fail);
      sitePhrase = { ...a, rel: { kind: "between", other: { id: b.id, laterality: b.laterality } } };
    } else {
      // Plural sugar: "between the shoulder blades" = between left and right.
      const sing = [...right.slice(0, -1), right[right.length - 1].replace(/s$/, "")];
      const a = parseSitePart(sing, fail);
      if (SITES[a.id]?.laterality !== "sided") fail(`"between the ${right.join(" ")}" needs two sites or a sided site`);
      sitePhrase = { ...a, laterality: "left", rel: { kind: "between", other: { id: a.id, laterality: "right" } } };
    }
  } else {
    const base = parseSitePart(right, fail);
    sitePhrase = { ...base, rel: { kind: marker.kind, offset_m: offset, render } };
  }

  const left = tokens.slice(0, marker.at - measureLen);
  if (left.length === 0) fail("no design before the placement");

  // Noun phrase: strip article, pull trailing "with … themes", scan modifiers.
  let np = left;
  if (np.length > 0 && (np[0] === "a" || np[0] === "an" || np[0] === "the")) np = np.slice(1);
  const secondary: string[] = [];
  const markers: string[] = stylesJson.secondary_markers;
  if (np.length >= 3 && markers.includes(np[np.length - 1])) {
    const w = np.lastIndexOf("with");
    if (w > 0) {
      const parts = key(np.slice(w + 1, np.length - 1)).split(/ and /);
      const ids = parts.map((part) => styleMap.get(part.trim()));
      if (ids.every((id): id is string => id !== undefined)) {
        // Only styles qualify as secondary; anything else ("with roses and
        // thorns") is part of the motif and stays there.
        secondary.push(...ids);
        np = np.slice(0, w);
      }
    }
  }
  let style: string | null = null;
  let technique: string | null = null;
  let color: string | null = null;
  let pos = 0;
  for (;;) {
    const c = matchAt(colorMap, np, pos);
    const t = matchAt(techMap, np, pos);
    const s = matchAt(styleMap, np, pos);
    const best = [c && ["color", ...c], t && ["technique", ...t], s && ["style", ...s]]
      .filter((x): x is [string, string, number] => x !== null)
      .sort((x, y) => y[2] - x[2])[0];
    if (best === undefined) break;
    const [kind, id, n] = best;
    if (kind === "color") { if (color !== null) fail(`two colors ("${color}", "${id}")`); color = id; }
    else if (kind === "technique") { if (technique !== null) fail(`two techniques ("${technique}", "${id}")`); technique = id; }
    else if (style === null) style = id;
    else secondary.push(id);
    pos += n;
  }
  const motif = np.slice(pos).join(" ");
  if (motif.length === 0) fail("no motif left after the style words");

  const program: TattooProgram = {
    inklang: INKLANG_VERSION,
    motif, style, secondary, technique, color,
    site: sitePhrase,
  };
  validateProgram(program);
  return program;
}
