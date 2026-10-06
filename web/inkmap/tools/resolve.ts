#!/usr/bin/env node
// Offline, batch-capable InkLang resolver. Stdout is canonical JSON only.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { AtlasIndex, parseAtlas } from "../src/core/atlas.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
  buildSkin,
  canonicalSurfaceBytes,
} from "../src/core/body.ts";
import {
  ATLAS_SCHEMA_VERSION,
  RESOLUTION_SCHEMA_VERSION,
  RESOLVER_VERSION,
  InkLangError,
  asIssue,
  intentFromSite,
  parsePlacement,
  realizePlacement,
  resolveIntent,
  type InkLangIntent,
  type InkLangResolution,
  type ResolutionPolicy,
  type ResolutionPolicyId,
  type SitePhrase,
} from "../src/core/inklang/index.ts";
import { parseSentence } from "../src/core/lang.ts";
import { sha256Hex } from "../src/core/sha256.ts";

export interface ResolveRequest {
  /** Exactly one of prompt or site is required. Structured sites let non-language consumers avoid re-realizing text. */
  prompt?: string;
  site?: SitePhrase;
  /** Original caller text for a structured site; defaults to its canonical placement phrase. */
  description?: string;
  policy?: ResolutionPolicyId;
  seed?: number;
}

let loaded: Promise<AtlasIndex> | undefined;

async function loadIndex(): Promise<AtlasIndex> {
  if (loaded) return loaded;
  const pending = (async () => {
    const spec = BODY_SPEC;
    const bytes = readFileSync(new URL(`../public/${spec.path}`, import.meta.url));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const gltf = await new GLTFLoader().parseAsync(buffer, "");
    const skin = buildSkin(gltf.scene);
    const surfaceSha256 = await sha256Hex(canonicalSurfaceBytes(skin.geometry));
    const assetSha256 = await sha256Hex(buffer);
    let atlasRaw: unknown;
    try {
      atlasRaw = JSON.parse(readFileSync(
        new URL(`../public/bodies/${MODEL_SPEC_ID}.regions.json`, import.meta.url),
        "utf8",
      ));
    } catch (error) {
      throw new InkLangError(
        "INKLANG_MISSING_ATLAS",
        `missing region atlas for ${MODEL_SPEC_ID}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const atlas = parseAtlas(atlasRaw, skin.centroids.length / 3);
    if (
      atlas.body.model_spec_id !== MODEL_SPEC_ID
      || atlas.body.model_spec_sha256 !== MODEL_SPEC_SHA256
      || atlas.body.identity_sha256 !== REFERENCE_IDENTITY_SHA256
      || atlas.body.topology_sha256 !== TOPOLOGY_SHA256
      || atlas.body.rest_surface_sha256 !== REST_SURFACE_SHA256
    ) {
      throw new InkLangError("INKLANG_BODY_MISMATCH", "atlas body binding differs from the fixed SOMA contract");
    }
    if (atlas.body.rest_surface_sha256 !== surfaceSha256) {
      throw new InkLangError(
        "INKLANG_SURFACE_MISMATCH",
        `atlas surface ${atlas.body.rest_surface_sha256} does not match loaded ${surfaceSha256}`,
      );
    }
    if (atlas.body.asset_sha256 !== assetSha256) {
      throw new InkLangError(
        "INKLANG_SURFACE_MISMATCH",
        `atlas asset ${atlas.body.asset_sha256} does not match loaded ${assetSha256}`,
      );
    }
    return new AtlasIndex(atlas, skin.geometry, skin.centroids);
  })();
  loaded = pending;
  return pending;
}

function parseCompatiblePrompt(prompt: string): InkLangIntent {
  const placement = parsePlacement(prompt);
  if (placement.site) return placement;
  try {
    const legacy = parseSentence(prompt);
    return intentFromSite(prompt, legacy.site);
  } catch {
    return placement;
  }
}

function rejectedWithoutAtlas(
  request: ResolveRequest,
  intent: InkLangIntent,
  policy: ResolutionPolicy,
  error: unknown,
): InkLangResolution {
  return {
    resolution_schema_version: RESOLUTION_SCHEMA_VERSION,
    status: "rejected",
    intent,
    body: {
      model_spec_id: MODEL_SPEC_ID,
      model_spec_sha256: MODEL_SPEC_SHA256,
      identity_sha256: REFERENCE_IDENTITY_SHA256,
      topology_sha256: TOPOLOGY_SHA256,
      rest_surface_sha256: REST_SURFACE_SHA256,
      asset_sha256: null,
    },
    atlas_schema_version: ATLAS_SCHEMA_VERSION,
    resolver: { name: "inklang-typescript", version: RESOLVER_VERSION, policy },
    anchor: null,
    actual: null,
    candidates: [],
    issues: [asIssue(error)],
  };
}

export async function resolvePrompt(request: ResolveRequest): Promise<InkLangResolution> {
  let intent: InkLangIntent;
  try {
    const unknown = Object.keys(request as object).filter(
      (key) => !["prompt", "site", "description", "policy", "seed"].includes(key),
    );
    if (unknown.length) throw new InkLangError("INKLANG_UNKNOWN_BODY", "unsupported schema/model", unknown[0]);
    if ((request.prompt === undefined) === (request.site === undefined)) {
      throw new InkLangError("INKLANG_PARSE_SYNTAX", "request needs exactly one of prompt or site");
    }
    intent = request.site
      ? intentFromSite(request.description ?? realizePlacement(request.site), request.site)
      : parseCompatiblePrompt(request.prompt!);
  } catch (error) {
    intent = parsePlacement("");
    intent.description = request.prompt ?? "structured site";
    intent.issues = [asIssue(error)];
  }
  const policy: ResolutionPolicy = request.policy === "seeded-v1"
    ? { id: "seeded-v1", seed: request.seed ?? null }
    : { id: "interactive", seed: null };
  try {
    return resolveIntent(intent, await loadIndex(), policy);
  } catch (error) {
    return rejectedWithoutAtlas(request, intent, policy, error);
  }
}

export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

function usage(message?: string): never {
  if (message) process.stderr.write(`inklang resolve: ${message}\n`);
  process.stderr.write(
    "usage: resolve.ts --prompt TEXT [--policy interactive|seeded-v1 --seed N]\n"
    + "       resolve.ts --input FILE|-\n",
  );
  process.exit(2);
}

function parseArgs(argv: string[]): { request?: ResolveRequest; input?: string } {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index++) {
    const name = argv[index];
    if (name === "--json") continue;
    if (!["--prompt", "--policy", "--seed", "--input"].includes(name)) usage(`unknown argument ${name}`);
    const value = argv[++index];
    if (value === undefined) usage(`${name} needs a value`);
    values.set(name, value);
  }
  const input = values.get("--input");
  if (input) {
    if (values.has("--prompt")) usage("--input cannot be combined with --prompt");
    return { input };
  }
  const prompt = values.get("--prompt");
  if (!prompt) usage("--prompt is required");
  const policy = values.get("--policy") ?? "interactive";
  if (policy !== "interactive" && policy !== "seeded-v1") usage(`unknown policy ${policy}`);
  const seedText = values.get("--seed");
  const seed = seedText === undefined ? undefined : Number(seedText);
  if (seed !== undefined && !Number.isInteger(seed)) usage("--seed must be an integer");
  return { request: { prompt, policy, ...(seed !== undefined ? { seed } : {}) } };
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.request) {
    process.stdout.write(`${canonicalJson(await resolvePrompt(parsed.request))}\n`);
    return;
  }
  const text = parsed.input === "-"
    ? readFileSync(0, "utf8")
    : readFileSync(parsed.input!, "utf8");
  const decoded = JSON.parse(text) as ResolveRequest[] | { requests: ResolveRequest[] };
  const requests = Array.isArray(decoded) ? decoded : decoded.requests;
  if (!Array.isArray(requests)) usage("input must be an array or an object with a requests array");
  const results = [];
  for (const request of requests) results.push(await resolvePrompt(request));
  process.stdout.write(`${canonicalJson(results)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
