import type { Anchor } from "../anchor.ts";
import type { AtlasIndex } from "../atlas.ts";
import { asIssue, InkLangError } from "./errors.ts";
import { INKLANG_VERSION, SITES, ZONES } from "./lexicon.ts";
import { realizePlacement } from "./realize.ts";
import {
  RESOLUTION_SCHEMA_VERSION,
  RESOLVER_VERSION,
  type InkLangIntent,
  type InkLangResolution,
  type ResolutionCandidate,
  type ResolutionPolicy,
  type SitePhrase,
} from "./types.ts";

function chooseSeeded<T>(values: T[], intent: InkLangIntent, modelSpecId: string, seed: number): T {
  let hash = (2166136261 ^ seed) >>> 0;
  const value = `${modelSpecId}\u0000${intent.canonical_phrase ?? intent.description}`;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return values[hash % values.length];
}

function expandSite(site: SitePhrase): SitePhrase[] {
  const requiresSide = SITES[site.id]?.laterality === "sided" || ZONES[site.id]?.laterality === "sided";
  const sideVariants = requiresSide && site.laterality === null
    ? (["left", "right"] as const).map((laterality) => ({ ...site, laterality }))
    : [site];
  const zone = ZONES[site.id];
  if (!zone) return sideVariants;
  return sideVariants.flatMap((variant) => zone.members.flatMap((member) => {
    const memberSpec = SITES[member];
    if (memberSpec.laterality === "sided" && variant.laterality === null) {
      return (["left", "right"] as const).map((laterality) => ({
        ...variant,
        id: member,
        laterality,
      }));
    }
    return [{ ...variant, id: member }];
  }));
}

function policyRecord(policy: ResolutionPolicy): ResolutionPolicy {
  if (policy.id === "interactive") return { id: "interactive", seed: null };
  if (!Number.isInteger(policy.seed) || policy.seed === null || policy.seed < 0 || policy.seed > 2147483647) {
    throw new InkLangError("INKLANG_PARSE_SYNTAX", "seeded-v1 requires an integer seed in [0,2147483647]");
  }
  return { id: "seeded-v1", seed: policy.seed };
}

function baseResolution(
  intent: InkLangIntent,
  index: AtlasIndex,
  policy: ResolutionPolicy,
): Omit<InkLangResolution, "status" | "anchor" | "actual" | "candidates" | "issues"> {
  return {
    resolution_schema_version: RESOLUTION_SCHEMA_VERSION,
    intent,
    body: { ...index.atlas.body },
    atlas_schema_version: index.atlas.atlas_schema_version,
    resolver: {
      name: "inklang-typescript",
      version: RESOLVER_VERSION,
      policy,
    },
  };
}

function candidateFor(index: AtlasIndex, site: SitePhrase, besideDirection?: -1 | 1): ResolutionCandidate {
  const anchor = index.anchorForPhrase(site, besideDirection);
  return {
    label: besideDirection === undefined
      ? realizePlacement(site)
      : `${besideDirection < 0 ? "left" : "right"} of ${realizePlacement({ ...site, relation: undefined }).replace(/^on the /, "")}`,
    site_id: site.id,
    laterality: site.laterality,
    anchor,
  };
}

export function resolveIntent(
  intent: InkLangIntent,
  index: AtlasIndex,
  requestedPolicy: ResolutionPolicy = { id: "interactive", seed: null },
): InkLangResolution {
  let policy: ResolutionPolicy;
  try {
    policy = policyRecord(requestedPolicy);
  } catch (error) {
    const issue = asIssue(error);
    return {
      ...baseResolution(intent, index, { id: "interactive", seed: null }),
      status: "rejected",
      anchor: null,
      actual: null,
      candidates: [],
      issues: [issue],
    };
  }
  const base = baseResolution(intent, index, policy);
  if (intent.inklang_version !== INKLANG_VERSION || index.atlas.inklang_version !== INKLANG_VERSION) {
    return {
      ...base,
      status: "rejected",
      anchor: null,
      actual: null,
      candidates: [],
      issues: [{
        code: "INKLANG_VERSION_MISMATCH",
        message: `intent ${intent.inklang_version}, atlas ${index.atlas.inklang_version}, resolver ${INKLANG_VERSION}`,
      }],
    };
  }
  if (!intent.site) {
    return {
      ...base,
      status: "rejected",
      anchor: null,
      actual: null,
      candidates: [],
      issues: intent.issues.length ? intent.issues : [{
        code: "INKLANG_PARSE_SYNTAX",
        message: "placement intent has no site",
      }],
    };
  }
  const variants = expandSite(intent.site);
  if (variants.length === 0) {
    return {
      ...base,
      status: "rejected",
      anchor: null,
      actual: null,
      candidates: [],
      issues: [{ code: "INKLANG_NO_REGION", message: `${intent.site.id} has no resolvable leaf regions` }],
    };
  }
  const besideAmbiguous = intent.ambiguities.some((item) => item.field === "relative_direction");
  try {
    const directions: (undefined | -1 | 1)[] = besideAmbiguous ? [-1, 1] : [undefined];
    const options = variants.flatMap((variant) => directions.flatMap((besideDirection) => {
      try {
        return [{ variant, besideDirection, candidate: candidateFor(index, variant, besideDirection) }];
      } catch (error) {
        if (error instanceof InkLangError && error.code === "INKLANG_NO_REGION") return [];
        throw error;
      }
    }));
    if (!options.length) {
      throw new InkLangError(
        "INKLANG_NO_REGION",
        `no reviewed eligible surface for ${intent.canonical_phrase ?? intent.description}`,
      );
    }
    if (policy.id === "interactive" && intent.ambiguities.length > 0) {
      const candidates = options.map((option) => option.candidate);
      const unique = [...new Map(candidates.map((item) => [
        `${item.site_id}:${item.laterality}:${item.anchor.face}`,
        item,
      ])).values()];
      return {
        ...base,
        status: "needs_choice",
        anchor: null,
        actual: null,
        candidates: unique,
        issues: intent.issues,
      };
    }
    const selectedOption = policy.id === "seeded-v1"
      ? chooseSeeded(options, intent, index.atlas.body.model_spec_id, policy.seed!)
      : options[0];
    const { variant: selected, besideDirection } = selectedOption;
    const { anchor, relative } = index.anchorForPhraseDetailed(selected, besideDirection);
    if (!index.isValidAnchor(anchor)) {
      throw new InkLangError("INKLANG_ANCHOR_INVALID", `resolver produced invalid face ${anchor.face}`);
    }
    if (!selected.relation && !index.contains(selected, anchor)) {
      throw new InkLangError(
        "INKLANG_SEMANTIC_MISMATCH",
        `anchor ${anchor.face} is outside ${selected.laterality ?? ""} ${selected.id}`.trim(),
      );
    }
    const described = index.describe(anchor);
    if (!described) {
      throw new InkLangError("INKLANG_ANCHOR_INVALID", `anchor ${anchor.face} is not labeled skin`);
    }
    return {
      ...base,
      status: "resolved",
      anchor,
      actual: {
        site_id: described.id,
        laterality: described.laterality,
        aspect: described.aspect,
        level: described.level,
        region_uv: described.region_uv!,
        // The phrase for what was actually placed, not for what was asked.
        // These differ whenever a policy or a choice resolved an ambiguity:
        // the intent still says "forearm" or "quarter sleeve" after the
        // resolver has committed to a side or a leaf site.
        canonical_phrase: realizePlacement(selected),
      },
      candidates: [],
      issues: [],
      ...(relative ? { relative } : {}),
    };
  } catch (error) {
    const issue = asIssue(error);
    return {
      ...base,
      status: "rejected",
      anchor: null,
      actual: null,
      candidates: [],
      issues: [issue],
    };
  }
}

/** Turn a manual anchor or an explicitly accepted candidate into a canonical resolution. */
export function resolutionFromAnchor(
  intent: InkLangIntent,
  index: AtlasIndex,
  anchor: Anchor,
  choice?: ResolutionCandidate,
): InkLangResolution {
  const policy: ResolutionPolicy = { id: "interactive", seed: null };
  const base = baseResolution(intent, index, policy);
  if (!index.isValidAnchor(anchor)) {
    return {
      ...base, status: "rejected", anchor: null, actual: null, candidates: [],
      issues: [{ code: "INKLANG_ANCHOR_INVALID", message: `invalid manual anchor on face ${anchor.face}` }],
    };
  }
  const described = index.describe(anchor);
  if (!described) {
    return {
      ...base, status: "rejected", anchor: null, actual: null, candidates: [],
      issues: [{ code: "INKLANG_ANCHOR_INVALID", message: `face ${anchor.face} is not labeled skin` }],
    };
  }
  return {
    ...base,
    status: "resolved",
    anchor,
    actual: {
      site_id: described.id,
      laterality: described.laterality,
      aspect: described.aspect,
      level: described.level,
      region_uv: described.region_uv!,
      canonical_phrase: choice?.label ?? intent.canonical_phrase ?? realizePlacement(described),
    },
    candidates: [],
    issues: [],
    ...(choice ? { choice } : {}),
  };
}

export function acceptResolutionCandidate(
  resolution: InkLangResolution,
  candidateIndex: number,
  index: AtlasIndex,
): InkLangResolution {
  if (resolution.status !== "needs_choice") return resolution;
  const choice = resolution.candidates[candidateIndex];
  if (!choice) {
    return {
      ...resolution,
      status: "rejected",
      candidates: [],
      issues: [{ code: "INKLANG_PARSE_SYNTAX", message: `candidate ${candidateIndex} does not exist` }],
    };
  }
  return resolutionFromAnchor(resolution.intent, index, choice.anchor, choice);
}
