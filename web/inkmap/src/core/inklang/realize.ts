import { preferredSiteWords, validateSitePhrase } from "./lexicon.ts";
import type { RelKind, SitePhrase } from "./types.ts";

const RELATION_WORD: Record<Exclude<RelKind, "between">, string> = {
  above: "above",
  below: "below",
  behind: "behind",
  in_front: "in front of",
  beside: "beside",
};

export function realizePlacement(site: SitePhrase): string {
  validateSitePhrase(site);
  const relation = site.relation;
  if (!relation) return `on the ${preferredSiteWords(site)}`;
  if (relation.kind === "between") {
    const other = relation.other!;
    return `between the ${preferredSiteWords(site)} and the ${preferredSiteWords({
      id: other.id,
      laterality: other.laterality,
      aspect: null,
      level: null,
    })}`;
  }
  const measure = relation.render ? `${relation.render} ` : "";
  return `${measure}${RELATION_WORD[relation.kind]} the ${preferredSiteWords(site)}`;
}

export { preferredSiteWords as realizeSiteWords } from "./lexicon.ts";
