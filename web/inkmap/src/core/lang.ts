// Compatibility adapter for the pre-unification full tattoo sentence.
// InkLang itself is placement-only and lives in ./inklang/. This module keeps
// accepted motif/style sentences working by composing their design fields with
// one canonical InkLang SitePhrase; it does not ground a second location.
import stylesJson from "../../../../config/inkmap/styles.json" with { type: "json" };
import {
  INKLANG_VERSION,
  SITES,
  ZONES,
  isZone,
  normalizeWords,
  parseMarkedPlacement,
  realizePlacement,
  validateSitePhrase,
  wordKey,
  type Laterality,
  type Level,
  type RelKind,
  type SitePhrase,
  type SiteRelation,
} from "./inklang/index.ts";

export {
  INKLANG_VERSION,
  SITES,
  ZONES,
  isZone,
  type Laterality,
  type Level,
  type RelKind,
  type SitePhrase,
  type SiteRelation,
};

/** Legacy design fields wrapped around one canonical InkLang placement. */
export interface TattooProgram {
  inklang: string;
  motif: string;
  style: string | null;
  secondary: string[];
  technique: string | null;
  color: string | null;
  site: SitePhrase;
}

interface TermEntry {
  name: string;
  aliases: string[];
  default?: boolean;
  prompt?: string;
}

export const STYLES: Record<string, TermEntry> = stylesJson.styles;
export const TECHNIQUES: Record<string, TermEntry> = stylesJson.techniques;
export const COLORS: Record<string, TermEntry> = stylesJson.colors;

function buildTermMap(table: Record<string, TermEntry>): Map<string, string> {
  const result = new Map<string, string>();
  for (const [id, entry] of Object.entries(table)) {
    for (const phrase of [id, entry.name, ...entry.aliases]) {
      result.set(wordKey(normalizeWords(phrase)), id);
    }
  }
  return result;
}

const styleMap = buildTermMap(STYLES);
const techniqueMap = buildTermMap(TECHNIQUES);
const colorMap = buildTermMap(COLORS);

function matchAt(map: Map<string, string>, tokens: string[], from: number): [string, number] | null {
  for (let end = tokens.length; end > from; end--) {
    const hit = map.get(wordKey(tokens.slice(from, end)));
    if (hit !== undefined) return [hit, end - from];
  }
  return null;
}

export function validateProgram(program: TattooProgram): void {
  const fail = (message: string): never => { throw new Error(`tattoo request: ${message}`); };
  if (typeof program.motif !== "string" || normalizeWords(program.motif).length === 0) {
    fail("motif must be a non-empty phrase");
  }
  if (program.style !== null && !(program.style in STYLES)) fail(`unknown style "${program.style}"`);
  for (const style of program.secondary) {
    if (!(style in STYLES)) fail(`unknown secondary style "${style}"`);
  }
  if (program.technique !== null && !(program.technique in TECHNIQUES)) {
    fail(`unknown technique "${program.technique}"`);
  }
  if (program.color !== null && !(program.color in COLORS)) fail(`unknown color "${program.color}"`);
  try {
    validateSitePhrase(program.site);
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error));
  }
}

/** Design-generator prompt assembly is deliberately outside InkLang. */
export function stylePrompt(program: TattooProgram): string | undefined {
  const fragments: string[] = [];
  const append = (table: Record<string, TermEntry>, id: string | null) => {
    const entry = id === null ? undefined : table[id];
    if (entry?.prompt) fragments.push(entry.prompt);
  };
  append(STYLES, program.style);
  for (const style of program.secondary) append(STYLES, style);
  append(TECHNIQUES, program.technique);
  append(COLORS, program.color);
  return fragments.length ? fragments.join(", ") : undefined;
}

const article = (word: string): "a" | "an" => (/^[aeiou]/.test(word) ? "an" : "a");

export function realize(program: TattooProgram): string {
  validateProgram(program);
  const modifiers: string[] = [];
  if (program.color !== null) modifiers.push(COLORS[program.color].name);
  if (program.technique !== null && program.technique !== "machine") {
    modifiers.push(TECHNIQUES[program.technique].name);
  }
  if (program.style !== null) modifiers.push(STYLES[program.style].name);
  modifiers.push(normalizeWords(program.motif).join(" "));
  const nounPhrase = modifiers.join(" ");
  let prefix = `${article(nounPhrase)} ${nounPhrase}`;
  if (program.secondary.length) {
    prefix += ` with ${program.secondary.map((id) => STYLES[id].name).join(" and ")} themes`;
  }
  return `${prefix} ${realizePlacement(program.site)}`;
}

export function parseSentence(sentence: string): TattooProgram {
  const fail = (message: string): never => {
    throw new Error(`tattoo request parse: ${message} — in "${sentence}"`);
  };
  const tokens = normalizeWords(sentence);
  let parsed: ReturnType<typeof parseMarkedPlacement>;
  try {
    parsed = parseMarkedPlacement(tokens);
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
  const left = tokens.slice(0, parsed.leftEnd);
  if (!left.length) fail("no design before the placement");

  let nounPhrase = left;
  if (["a", "an", "the"].includes(nounPhrase[0])) nounPhrase = nounPhrase.slice(1);
  const secondary: string[] = [];
  const secondaryMarkers: string[] = stylesJson.secondary_markers;
  if (nounPhrase.length >= 3 && secondaryMarkers.includes(nounPhrase.at(-1)!)) {
    const withIndex = nounPhrase.lastIndexOf("with");
    if (withIndex > 0) {
      const parts = wordKey(nounPhrase.slice(withIndex + 1, -1)).split(/ and /);
      const ids = parts.map((part) => styleMap.get(part.trim()));
      if (ids.every((id): id is string => id !== undefined)) {
        secondary.push(...ids);
        nounPhrase = nounPhrase.slice(0, withIndex);
      }
    }
  }

  let style: string | null = null;
  let technique: string | null = null;
  let color: string | null = null;
  let position = 0;
  for (;;) {
    const colorHit = matchAt(colorMap, nounPhrase, position);
    const techniqueHit = matchAt(techniqueMap, nounPhrase, position);
    const styleHit = matchAt(styleMap, nounPhrase, position);
    const candidates = [
      colorHit && ["color", ...colorHit],
      techniqueHit && ["technique", ...techniqueHit],
      styleHit && ["style", ...styleHit],
    ].filter((item): item is [string, string, number] => item !== null)
      .sort((leftItem, rightItem) => rightItem[2] - leftItem[2]);
    const best = candidates[0];
    if (!best) break;
    const [kind, id, length] = best;
    if (kind === "color") {
      if (color !== null) fail(`two colors ("${color}", "${id}")`);
      color = id;
    } else if (kind === "technique") {
      if (technique !== null) fail(`two techniques ("${technique}", "${id}")`);
      technique = id;
    } else if (style === null) {
      style = id;
    } else {
      secondary.push(id);
    }
    position += length;
  }
  const motif = nounPhrase.slice(position).join(" ");
  if (!motif) fail("no motif left after the style words");
  const program: TattooProgram = {
    inklang: INKLANG_VERSION,
    motif,
    style,
    secondary,
    technique,
    color,
    site: parsed.site,
  };
  validateProgram(program);
  return program;
}
