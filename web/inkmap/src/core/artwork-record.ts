/** Portable, reproducible artwork shared by project and simulation envelopes.
 * All validation is local: source identifiers are provenance, never fetch URLs.
 */
import { canonicalDigest, canonicalJson, ContractError } from "./human-representation/schema.ts";
import { validateTattooProgram, type TattooProgram } from "./human-representation/tattoo-program.ts";

export interface ArtworkSource {
  kind: "stock" | "imported" | "generated" | "fixture";
  identifier: string | null;
  license: string | null;
  attribution: string | null;
  generation: {
    prompt: string;
    model: string | null;
    model_revision: string | null;
    seed: number | null;
    tracing: string | null;
    request_sha256?: string;
    png_sha256?: string;
    settings?: { model: string; model_revision: string | null; width: number; height: number; steps: number; guidance: number };
  } | null;
}

/** Acquisition identity; the program contains the authoritative metric geometry. */
export interface ArtworkConversion {
  adapter: string;
  recipe_sha256: string | null;
  chord_error_m: number;
}

export interface ArtworkRecord {
  schema: "tatbot.inkmap-artwork/2";
  content_sha256: string;
  name: string;
  source_sha256: string;
  source: ArtworkSource;
  conversion: ArtworkConversion;
  program: TattooProgram;
}

export function artworkSizeM(record: ArtworkRecord): [number, number] {
  return [record.program.canvas_m.width, record.program.canvas_m.height];
}

/** Recipe acquisitions must be regenerated when placement changes either physical dimension. */
export function artworkNeedsRegeneration(record: ArtworkRecord, sizeMm: [number, number]): boolean {
  return record.conversion.recipe_sha256 !== null && artworkSizeM(record).some((m, axis) => Math.abs(m * 1000 - sizeMm[axis]) > .000001);
}

/** Production admission matches ROS preparation; generic contract fixtures stay readable. */
export function requireAcquiredArtwork(record: ArtworkRecord): void {
  if (record.conversion.adapter !== "dbv3-batik-paths/1" || record.conversion.recipe_sha256 === null
      || record.program.provenance.producer !== "dbv3-batik-paths/1"
      || record.program.layers.some(layer => layer.elements.some(element => element.kind !== "path" || element.fill))) {
    throw new Error("legacy_artwork: Generate with DrawingBot V3 and import its artwork.json. SVG sources must go through DBV3; saved legacy artwork cannot be traced or restored here.");
  }
}

export function artworkWidthM(record: ArtworkRecord): number {
  return record.program.layers.reduce((width, layer) => layer.elements.reduce((held, element) => Math.max(held, element.width_m), width), 0);
}

function fail(path: string, detail: string, code = "artwork_record_invalid"): never {
  throw new ContractError(code, path, detail);
}

function keys(value: unknown, required: string[], path: string, optional: string[] = []): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(path, "expected object");
  const actual = Object.keys(value).filter(key => !optional.includes(key)).sort();
  if (actual.join("|") !== [...required].sort().join("|")) fail(path, "missing or unknown fields");
}

function text(value: unknown, path: string, nullable = false): void {
  if (nullable && value === null) return;
  if (typeof value !== "string" || !value.trim() || value.length > 100_000) fail(path, "expected bounded nonempty string");
}

export function validateSource(value: unknown): asserts value is ArtworkSource {
  keys(value, ["kind", "identifier", "license", "attribution", "generation"], "$.source");
  if (!["stock", "imported", "generated", "fixture"].includes(String(value.kind))) fail("$.source.kind", "unknown source kind");
  for (const field of ["identifier", "license", "attribution"]) text(value[field], `$.source.${field}`, true);
  if (value.kind === "generated") {
    const gen = value.generation;
    keys(gen, ["prompt", "model", "model_revision", "seed", "tracing"], "$.source.generation", ["request_sha256", "png_sha256", "settings"]);
    for (const key of ["request_sha256", "png_sha256"]) {
      if (gen[key] !== undefined && (typeof gen[key] !== "string" || !/^[0-9a-f]{64}$/.test(gen[key] as string))) fail(`$.source.generation.${key}`, "expected SHA-256");
    }
    if (gen.settings !== undefined) {
      keys(gen.settings, ["model", "model_revision", "width", "height", "steps", "guidance"], "$.source.generation.settings");
      text(gen.settings.model, "$.source.generation.settings.model");
      text(gen.settings.model_revision, "$.source.generation.settings.model_revision", true);
      for (const key of ["width", "height", "steps", "guidance"]) {
        if (typeof gen.settings[key] !== "number" || !Number.isFinite(gen.settings[key])) fail(`$.source.generation.settings.${key}`, "expected finite number");
      }
    }
    text(gen.prompt, "$.source.generation.prompt");
    for (const field of ["model", "model_revision", "tracing"]) text(gen[field], `$.source.generation.${field}`, true);
    if (gen.seed !== null && (!Number.isSafeInteger(gen.seed) || (gen.seed as number) < 0)) fail("$.source.generation.seed", "expected nonnegative safe integer or null");
  } else if (value.generation !== null) fail("$.source.generation", "only generated sources carry generation metadata");
}

export function validateConversion(value: unknown): asserts value is ArtworkConversion {
  keys(value, ["adapter", "recipe_sha256", "chord_error_m"], "$.conversion");
  text(value.adapter, "$.conversion.adapter");
  if (value.recipe_sha256 !== null && (typeof value.recipe_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.recipe_sha256))) fail("$.conversion.recipe_sha256", "expected recipe digest or null");
  const tolerance = value.chord_error_m;
  if (typeof tolerance !== "number" || !Number.isFinite(tolerance) || tolerance <= 0 || tolerance > .0001) fail("$.conversion.chord_error_m", "expected chord error in (0, 0.1 mm]");
}

/** Synchronous structural admission for editor state; hash validation is async. */
export function validateArtworkShape(value: unknown): asserts value is ArtworkRecord {
  keys(value, ["schema", "content_sha256", "name", "source_sha256", "source", "conversion", "program"], "$");
  if (value.schema !== "tatbot.inkmap-artwork/2") fail("$.schema", "unsupported artwork version");
  text(value.name, "$.name");
  validateSource(value.source);
  validateConversion(value.conversion);
  for (const key of ["content_sha256", "source_sha256"]) {
    if (typeof value[key] !== "string" || !/^[0-9a-f]{64}$/.test(value[key] as string)) fail(`$.${key}`, "expected SHA-256");
  }
  canonicalJson(value);
}

/** Freeze acquired geometry. No SVG, conversion runtime, or source fetch occurs. */
export async function makeArtworkRecord(input: Omit<ArtworkRecord, "schema" | "content_sha256">): Promise<ArtworkRecord> {
  const record: ArtworkRecord = { ...structuredClone(input), schema: "tatbot.inkmap-artwork/2", content_sha256: "0".repeat(64) };
  record.content_sha256 = await canonicalDigest(record);
  return validateArtworkRecord(record);
}

export async function validateArtworkRecord(value: unknown): Promise<ArtworkRecord> {
  validateArtworkShape(value);
  if (value.content_sha256 !== await canonicalDigest(value)) fail("$.content_sha256", "record digest mismatch", "wrong_hash");
  const program = await validateTattooProgram(value.program);
  if (program.provenance.source_sha256 !== value.source_sha256) fail("$.source_sha256", "program binds different source bytes", "wrong_hash");
  if (artworkSizeM(value).some(dimension => dimension > 2)) fail("$.program.canvas_m", "dimensions exceed 2 m");
  if (artworkWidthM(value) > .02) fail("$.program", "planning width exceeds 20 mm");
  return structuredClone(value);
}
