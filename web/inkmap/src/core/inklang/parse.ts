import { asIssue, InkLangError } from "./errors.ts";
import {
  ASPECT_WORDS,
  exactSiteHit,
  findSiteSuffix,
  INKLANG_VERSION,
  isZone,
  LATERALITY_WORDS,
  LEVEL_WORDS,
  normalizeWords,
  SITES,
  validateSitePhrase,
  wordKey,
  ZONES,
} from "./lexicon.ts";
import { realizePlacement } from "./realize.ts";
import {
  INTENT_SCHEMA_VERSION,
  type InkLangIntent,
  type Laterality,
  type RelKind,
  type SitePhrase,
} from "./types.ts";

const UNITS = new Map<string, number>([
  ["inch", 0.0254], ["inches", 0.0254],
  ["cm", 0.01], ["centimeter", 0.01], ["centimeters", 0.01],
  ["centimetre", 0.01], ["centimetres", 0.01],
  ["mm", 0.001], ["millimeter", 0.001], ["millimeters", 0.001],
  ["millimetre", 0.001], ["millimetres", 0.001],
]);
const NUMBER_WORDS = new Map<string, number>([
  ["one", 1], ["two", 2], ["three", 3], ["four", 4], ["five", 5], ["six", 6],
  ["seven", 7], ["eight", 8], ["nine", 9], ["ten", 10], ["eleven", 11], ["twelve", 12],
]);

interface Marker {
  kind: "on" | RelKind;
  at: number;
  len: number;
}

export interface MarkedPlacement {
  site: SitePhrase;
  /** Tokens before the placement, excluding a relation measure. */
  leftEnd: number;
}

function fail(code: ConstructorParameters<typeof InkLangError>[0], message: string, field?: string): never {
  throw new InkLangError(code, message, field);
}

function findMarker(tokens: string[]): Marker | null {
  const patterns: [string[], Marker["kind"]][] = [
    [["on"], "on"],
    [["above"], "above"],
    [["below"], "below"],
    [["behind"], "behind"],
    [["beside"], "beside"],
    [["in", "front", "of"], "in_front"],
    [["between"], "between"],
  ];
  let best: Marker | null = null;
  for (const [pattern, kind] of patterns) {
    for (let index = tokens.length - pattern.length; index >= 0; index--) {
      if (pattern.every((word, offset) => tokens[index + offset] === word)) {
        const article = ["the", "my", "your"].includes(tokens[index + pattern.length]) ? 1 : 0;
        if (!best || index > best.at) best = { kind, at: index, len: pattern.length + article };
        break;
      }
    }
  }
  return best;
}

function parseSiteWords(input: string[]): SitePhrase {
  const words = input.filter(word => word !== "my" && word !== "your");
  const tokens = words[0] === "the" ? words.slice(1) : words;
  if (tokens.length === 0) fail("INKLANG_UNKNOWN_SITE", "empty site phrase", "site.id");
  let laterality: Laterality | null = null;
  let first = 0;
  const leadingLaterality = LATERALITY_WORDS.get(tokens[first]);
  if (leadingLaterality) {
    laterality = leadingLaterality;
    first++;
  }
  let match = findSiteSuffix(tokens, first);
  if (!match && first > 0) {
    const exact = exactSiteHit(tokens);
    if (exact) {
      match = { hit: exact, start: 0 };
      laterality = null;
      first = 0;
    }
  }
  if (!match) {
    fail("INKLANG_UNKNOWN_SITE", `unknown site "${tokens.slice(first).join(" ")}"`, "site.id");
  }
  let aspect = match.hit.aspect;
  let level = match.hit.level;
  for (let index = first; index < match.start; index++) {
    const token = tokens[index];
    if (token === "of" || token === "the") continue;
    const otherLaterality = LATERALITY_WORDS.get(token);
    if (otherLaterality) {
      if (laterality !== null && laterality !== otherLaterality) {
        fail(
          "INKLANG_INVALID_LATERALITY",
          `conflicting lateralities "${laterality}" and "${otherLaterality}"`,
          "site.laterality",
        );
      }
      laterality = otherLaterality;
      continue;
    }
    const otherLevel = LEVEL_WORDS.get(token);
    if (otherLevel) {
      if (level !== null && level !== otherLevel) {
        fail("INKLANG_INVALID_LEVEL", `conflicting levels "${level}" and "${otherLevel}"`, "site.level");
      }
      level = otherLevel;
      continue;
    }
    const otherAspect = ASPECT_WORDS.get(token);
    if (!otherAspect) {
      fail("INKLANG_PARSE_SYNTAX", `"${token}" is not an aspect or level word`, "site");
    }
    if (aspect !== null && aspect !== otherAspect) {
      fail("INKLANG_INVALID_ASPECT", `conflicting aspects "${aspect}" and "${otherAspect}"`, "site.aspect");
    }
    aspect = otherAspect;
  }
  const site: SitePhrase = { id: match.hit.id, laterality, aspect, level };
  validateSitePhrase(site);
  return site;
}

function relationMeasure(tokens: string[], marker: Marker): { length: number; offset: number; render: string } {
  if (marker.kind === "on" || marker.kind === "between") return { length: 0, offset: 0, render: "" };
  const before = (count: number): string | undefined => tokens[marker.at - count];
  const unit = UNITS.get(before(1) ?? "");
  if (unit !== undefined) {
    const amountWord = before(2);
    const amount = NUMBER_WORDS.get(amountWord ?? "")
      ?? (amountWord !== undefined && Number.isFinite(Number(amountWord)) ? Number(amountWord) : null);
    if (amount !== null && amount > 0) {
      return { length: 2, offset: amount * unit, render: `${amountWord} ${before(1)}` };
    }
    if (amountWord === "a" || amountWord === "an") {
      if (before(3) === "half") {
        return { length: 3, offset: 0.5 * unit, render: `half ${amountWord} ${before(1)}` };
      }
      return { length: 2, offset: unit, render: `${amountWord} ${before(1)}` };
    }
  }
  if (before(1) === "just") return { length: 1, offset: 0.03, render: "just" };
  return { length: 0, offset: 0.05, render: "" };
}

export function parseMarkedPlacement(tokens: string[]): MarkedPlacement {
  const marker = findMarker(tokens);
  if (!marker) {
    fail(
      "INKLANG_PARSE_SYNTAX",
      "expected a placement site or an on the/above/below/behind/beside/in front of/between phrase",
    );
  }
  // The legacy adapter treats the prefix as artwork. Reject placement vetoes
  // before that fallback can reinterpret them as a motif. Negation within
  // artwork ("without leaves", "forget me not") still belongs to the artwork.
  const measure = relationMeasure(tokens, marker);
  const prefix = wordKey(tokens.slice(0, marker.at - measure.length)).replace(/[’‘]/g, "'");
  const endsInVeto = /(?:^|\s)(?:not|never|don't|dont|no|nowhere)(?:\s+anywhere)?$/.test(prefix)
    && !/(?:^|\s)forget me not$/.test(prefix);
  const negativeCommand = /(?:^|\s)(?:(?:don't|dont|do not|never)\s+(?:ever\s+)?(?:put|place|draw|apply|tattoo|ink)\b|avoid\s+(?:putting|placing|drawing|applying|tattooing|inking)\b)/.test(prefix);
  const noTattoo = /(?:^|\s)(?:no|without)\s+(?:(?:a|an|the)\s+)?(?:tattoo|design|ink)$/.test(prefix);
  if (endsInVeto || negativeCommand || noTattoo) {
    fail("INKLANG_PARSE_SYNTAX", "This describes where the tattoo should not go. Describe a location where it should go instead.");
  }
  const right = tokens.slice(marker.at + marker.len);
  if (right.length === 0) fail("INKLANG_UNKNOWN_SITE", `no site after "${marker.kind}"`, "site.id");
  let site: SitePhrase;
  if (marker.kind === "on") {
    site = parseSiteWords(right);
  } else if (
    marker.kind === "behind"
    && measure.length === 0
    && (() => {
      const lateralityOffset = LATERALITY_WORDS.has(right[0]) ? 1 : 0;
      return exactSiteHit(["behind", "the", ...right.slice(lateralityOffset)]) !== null;
    })()
  ) {
    const laterality = LATERALITY_WORDS.get(right[0]) ?? null;
    const hit = exactSiteHit(["behind", "the", ...right.slice(laterality ? 1 : 0)])!;
    site = { id: hit.id, laterality, aspect: hit.aspect, level: hit.level };
  } else if (marker.kind === "between") {
    let conjunction = -1;
    for (let index = right.length - 2; index >= 1; index--) {
      if (right[index] === "and") {
        conjunction = index;
        break;
      }
    }
    if (conjunction > 0) {
      const firstSite = parseSiteWords(right.slice(0, conjunction));
      const secondWords = right[conjunction + 1] === "the"
        ? right.slice(conjunction + 2)
        : right.slice(conjunction + 1);
      const secondSite = parseSiteWords(secondWords);
      site = {
        ...firstSite,
        relation: { kind: "between", other: { id: secondSite.id, laterality: secondSite.laterality } },
      };
    } else {
      const singular = [...right.slice(0, -1), right.at(-1)!.replace(/s$/, "")];
      const firstSite = parseSiteWords(singular);
      if (SITES[firstSite.id]?.laterality !== "sided") {
        fail(
          "INKLANG_INVALID_RELATION",
          `"between the ${right.join(" ")}" needs two sites or a sided site`,
          "site.relation",
        );
      }
      site = {
        ...firstSite,
        laterality: "left",
        relation: { kind: "between", other: { id: firstSite.id, laterality: "right" } },
      };
    }
  } else {
    const base = parseSiteWords(right);
    site = {
      ...base,
      relation: { kind: marker.kind, offset_m: measure.offset, render: measure.render },
    };
  }
  validateSitePhrase(site);
  return { site, leftEnd: marker.at - measure.length };
}

export function parsePlacementStrict(description: string): SitePhrase {
  const tokens = normalizeWords(description);
  if (tokens.length === 0) fail("INKLANG_PARSE_SYNTAX", "placement description is empty");
  const marker = findMarker(tokens);
  if (!marker) return parseSiteWords(tokens);
  const parsed = parseMarkedPlacement(tokens);
  if (parsed.leftEnd !== 0) {
    fail(
      "INKLANG_PARSE_SYNTAX",
      `placement-only InkLang does not include design words "${wordKey(tokens.slice(0, parsed.leftEnd))}"`,
    );
  }
  return parsed.site;
}

export function intentFromSite(description: string, site: SitePhrase): InkLangIntent {
  validateSitePhrase(site);
  const ambiguities: InkLangIntent["ambiguities"] = [];
  const issues: InkLangIntent["issues"] = [];
  const spec = SITES[site.id] ?? ZONES[site.id];
  if (spec.laterality === "sided" && site.laterality === null) {
    ambiguities.push({ field: "laterality", candidates: ["left", "right"] });
    issues.push({
      code: "INKLANG_AMBIGUOUS_LATERALITY",
      message: `${site.id} requires a left or right choice`,
      field: "site.laterality",
    });
  }
  if (isZone(site.id) && ZONES[site.id].members.length > 1) {
    const members = ZONES[site.id].members.flatMap((member) => {
      const memberSpec = SITES[member];
      if (memberSpec.laterality === "sided" && site.laterality === null) {
        return [`left:${member}`, `right:${member}`];
      }
      return [`${site.laterality ?? "center"}:${member}`];
    });
    ambiguities.push({ field: "zone_member", candidates: [...new Set(members)] });
    issues.push({
      code: "INKLANG_AMBIGUOUS_ZONE",
      message: `${site.id} contains several valid leaf-site locations`,
      field: "site.id",
    });
  }
  if (site.relation?.kind === "beside" && site.laterality === null) {
    ambiguities.push({ field: "relative_direction", candidates: ["left", "right"] });
    issues.push({
      code: "INKLANG_AMBIGUOUS_RELATION",
      message: `beside ${site.id} requires a left or right direction`,
      field: "site.relation",
    });
  }
  return {
    intent_schema_version: INTENT_SCHEMA_VERSION,
    inklang_version: INKLANG_VERSION,
    description,
    canonical_phrase: realizePlacement(site),
    site,
    ambiguities,
    unknown_terms: [],
    issues,
  };
}

export function parsePlacement(description: string): InkLangIntent {
  try {
    return intentFromSite(description, parsePlacementStrict(description));
  } catch (error) {
    const issue = asIssue(error);
    return {
      intent_schema_version: INTENT_SCHEMA_VERSION,
      inklang_version: INKLANG_VERSION,
      description,
      canonical_phrase: null,
      site: null,
      ambiguities: [],
      unknown_terms: issue.code === "INKLANG_UNKNOWN_SITE" ? [description.trim()] : [],
      issues: [issue],
    };
  }
}
