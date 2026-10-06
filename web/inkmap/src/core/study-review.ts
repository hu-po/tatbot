/** Reviews transport existing artwork and compiler summaries, never another geometry format. */
import { requireAcquiredArtwork, validateArtworkRecord, type ArtworkRecord } from "./artwork-record.ts";
import { canonicalDigest, canonicalJson, parseJsonStrict } from "./human-representation/schema.ts";
import { renderTattooProgramSvg } from "./human-representation/program-svg.ts";
import { sha256Hex } from "./sha256.ts";

export const REVIEW_LIMIT = 20 * 1024 * 1024;
export const REVIEW_RENDERER = "inkmap-metric-svg/1";
export type Preference = "like" | "dislike" | "unrated";
export interface ReviewEntry {
  id: string; source_case: string; source_split: "train" | "validation"; artwork_sha256: string;
  recipe: Record<string, unknown>;
  preparation: {
    program_sha256: string; speed_m_s: number; identity: Record<string, unknown>;
    tool: { id: string; line_width_m: number | null; line_width_status: "unknown" | "assumed" | "measured" };
    stats: { paths: number; strokes: number; contact_m: number; travel_m: number; notes: string[];
      time_estimate: { modeled_s: number; scope: string; unknown_operations: Record<string, number> } };
  };
}
export interface StudyReview {
  schema: "tatbot.artwork-review/1"; content_sha256: string; name: string; study_sha256: string;
  artworks: Record<string, ArtworkRecord>; entries: ReviewEntry[];
}
export interface Observation {
  id: string; entry_id: string; artwork_sha256: string; program_sha256: string; preview_sha256: string;
  renderer: typeof REVIEW_RENDERER; preference: Preference; observed_at: string;
}
export interface LoadedReview { bundle: StudyReview; previews: Record<string, { svg: string; sha256: string }> }

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected a review object");
  return value as Record<string, unknown>;
}
function keys(value: unknown, names: string[]): Record<string, unknown> {
  const obj = object(value);
  if (Object.keys(obj).sort().join("|") !== [...names].sort().join("|")) throw new Error("Missing or unknown review fields");
  return obj;
}
function text(value: unknown, max = 200): asserts value is string {
  if (typeof value !== "string" || !value.trim() || value.length > max) throw new Error("Invalid review text");
}
function sha(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error("Invalid review digest");
}
function number(value: unknown): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error("Invalid review quantity");
}

function validateEntry(value: unknown, artworks: Record<string, ArtworkRecord>): ReviewEntry {
  const e = keys(value, ["id", "source_case", "source_split", "artwork_sha256", "recipe", "preparation"]);
  text(e.id); text(e.source_case); sha(e.artwork_sha256);
  if (!(e.artwork_sha256 in artworks) || !["train", "validation"].includes(String(e.source_split))) throw new Error("Unresolved artwork or held-out test case");
  text(object(e.recipe).pfm);
  const p = keys(e.preparation, ["program_sha256", "speed_m_s", "tool", "stats", "identity"]);
  sha(p.program_sha256); number(p.speed_m_s); object(p.identity);
  if (p.speed_m_s === 0) throw new Error("Drawing speed must be positive");
  const tool = object(p.tool); text(tool.id);
  if (tool.line_width_m !== null) { number(tool.line_width_m); if (tool.line_width_m === 0 || tool.line_width_m > .02) throw new Error("Invalid physical width"); }
  if (!["unknown", "assumed", "measured"].includes(String(tool.line_width_status)) || (tool.line_width_m === null) !== (tool.line_width_status === "unknown")) throw new Error("Invalid tool width evidence");
  const stats = object(p.stats), timing = object(stats.time_estimate);
  for (const name of ["paths", "strokes", "contact_m", "travel_m"]) number(stats[name]);
  for (const name of ["paths", "strokes"]) if (!Number.isSafeInteger(stats[name])) throw new Error("Path counts must be integers");
  number(timing.modeled_s); text(timing.scope, 2000);
  Object.values(object(timing.unknown_operations)).forEach(number);
  if (!Array.isArray(stats.notes) || stats.notes.length > 100) throw new Error("Invalid preparation notes");
  stats.notes.forEach(note => text(note, 2000));
  return value as ReviewEntry;
}

export async function loadReview(source: string): Promise<LoadedReview> {
  if (new TextEncoder().encode(source).length > REVIEW_LIMIT) throw new Error("Review exceeds 20 MiB");
  const value = keys(parseJsonStrict(source), ["schema", "content_sha256", "name", "study_sha256", "artworks", "entries"]);
  if (value.schema !== "tatbot.artwork-review/1") throw new Error("Expected an artwork review bundle");
  sha(value.content_sha256); sha(value.study_sha256); text(value.name);
  if (await canonicalDigest(value) !== value.content_sha256) throw new Error("Review digest differs");
  const artworks: Record<string, ArtworkRecord> = Object.create(null);
  const previews: LoadedReview["previews"] = Object.create(null);
  const raw = object(value.artworks);
  if (Object.keys(raw).length < 1 || Object.keys(raw).length > 64) throw new Error("Review requires 1–64 artworks");
  for (const [hash, record] of Object.entries(raw)) {
    sha(hash);
    const art = await validateArtworkRecord(record);
    requireAcquiredArtwork(art);
    if (art.content_sha256 !== hash) throw new Error("Artwork reference differs");
    artworks[hash] = art;
    const svg = renderTattooProgramSvg(art.program);
    previews[hash] = { svg, sha256: await sha256Hex(new TextEncoder().encode(svg).buffer) };
  }
  if (!Array.isArray(value.entries) || value.entries.length < 1 || value.entries.length > 64) throw new Error("Review requires 1–64 entries");
  const entries = value.entries.map(entry => validateEntry(entry, artworks));
  if (new Set(entries.map(e => e.id)).size !== entries.length) throw new Error("Duplicate review entry");
  if (new Set(entries.map(e => e.source_split)).size !== 1) throw new Error("Review must keep training and validation separate");
  if (new Set(entries.map(e => e.artwork_sha256)).size !== Object.keys(artworks).length) throw new Error("Unreferenced artwork in review");
  return { bundle: value as unknown as StudyReview, previews };
}

export function validateObservation(value: unknown, review: LoadedReview): Observation {
  const o = keys(value, ["id", "entry_id", "artwork_sha256", "program_sha256", "preview_sha256", "renderer", "preference", "observed_at"]);
  if (typeof o.id !== "string" || !/^[a-zA-Z0-9_-]{1,80}$/.test(o.id)) throw new Error("Invalid observation ID");
  const entry = review.bundle.entries.find(e => e.id === o.entry_id);
  if (!entry || o.artwork_sha256 !== entry.artwork_sha256 || o.program_sha256 !== entry.preparation.program_sha256 ||
      o.preview_sha256 !== review.previews[entry.artwork_sha256].sha256 || o.renderer !== REVIEW_RENDERER) throw new Error("Feedback belongs to a different artwork, preparation or preview");
  if (!["like", "dislike", "unrated"].includes(String(o.preference)) || typeof o.observed_at !== "string" ||
      !/Z$|[+-]\d\d:\d\d$/.test(o.observed_at) || !Number.isFinite(Date.parse(o.observed_at))) throw new Error("Invalid preference or observation time");
  return value as Observation;
}

export function observe(review: LoadedReview, entry: ReviewEntry, preference: Preference): Observation {
  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, "0")).join("");
  return validateObservation({ id, entry_id: entry.id, artwork_sha256: entry.artwork_sha256,
    program_sha256: entry.preparation.program_sha256, preview_sha256: review.previews[entry.artwork_sha256].sha256,
    renderer: REVIEW_RENDERER, preference, observed_at: new Date().toISOString() }, review);
}

export function mergeObservations(...groups: Observation[][]): Observation[] {
  const byId = new Map<string, Observation>();
  for (const o of groups.flat()) {
    if (byId.has(o.id) && canonicalJson(byId.get(o.id)) !== canonicalJson(o)) throw new Error("Conflicting feedback observation");
    byId.set(o.id, o);
  }
  return [...byId.values()].sort((a, b) => a.observed_at.localeCompare(b.observed_at) || a.id.localeCompare(b.id));
}

export function reviewStoragePrefix(review: LoadedReview): string { return `inkmap-review:${review.bundle.content_sha256}:`; }
export function readObservations(storage: Storage, review: LoadedReview): Observation[] {
  const prefix = reviewStoragePrefix(review), result: Observation[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)!;
    if (!key.startsWith(prefix)) continue;
    const observation = validateObservation(parseJsonStrict(storage.getItem(key)!), review);
    if (key !== prefix + observation.id) throw new Error("Feedback storage key differs");
    result.push(observation);
  }
  return mergeObservations(result);
}
export function saveObservation(storage: Storage, review: LoadedReview, observation: Observation): void {
  validateObservation(observation, review);
  const key = reviewStoragePrefix(review) + observation.id, existing = storage.getItem(key);
  if (existing && canonicalJson(validateObservation(parseJsonStrict(existing), review)) !== canonicalJson(observation)) throw new Error("Conflicting saved feedback");
  storage.setItem(key, JSON.stringify(observation));
}
export async function feedbackDocument(review: LoadedReview, observations: Observation[]) {
  const result = { schema: "tatbot.artwork-feedback/1", review_sha256: review.bundle.content_sha256,
    study_sha256: review.bundle.study_sha256, observations: mergeObservations(observations.map(o => validateObservation(o, review))) };
  return { ...result, content_sha256: await canonicalDigest(result) };
}
