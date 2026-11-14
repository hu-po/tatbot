/** The tattoo brief (`tatbot.inkmap-brief/1`, config/inkmap/brief.schema.json): what the art pipeline takes
 *  from Inkmap once the placement and art are settled.
 *
 * Each placement names its art and the size it was given here. `tatbot art separate --brief` separates that
 * image at that width (the art rectangle's width, as `art_px` measures it), and DrawingBot V3 computes the
 * strokes there. Rotation, mirror and the body site travel as the record of where the design goes; the
 * robot draws on the skin mount's printed page, so nothing here is a motion program or robot pose.
 */
import type { Anchor } from "./anchor.ts";
import { validateArtRecord, type ArtRecord } from "./art.ts";
import { canonicalDigest } from "./human-representation/schema.ts";
import type { PlacementBody, PlacementFile, PlacementSite } from "./schema.ts";

export const BRIEF_SCHEMA = "tatbot.inkmap-brief/1";

export interface BriefPlacement {
  id: string;
  art_id: string;
  /** The art rectangle's width to separate at, mm, to 0.1 mm. */
  width_mm: number;
  height_mm: number;
  rotation_rad: number;
  mirror: boolean;
  anchor: Anchor;
  site: PlacementSite | null;
  sentence: string | null;
}

export interface InkmapBrief {
  schema: typeof BRIEF_SCHEMA;
  content_sha256: string;
  name: string;
  body: PlacementBody;
  art: Record<string, ArtRecord>;
  placements: BriefPlacement[];
}

const fail = (detail: string): never => { throw new Error(`brief_invalid: ${detail}`); };

export async function validateBrief(value: unknown): Promise<InkmapBrief> {
  if (!value || typeof value !== "object") fail("not an object");
  const brief = value as InkmapBrief;
  if (Object.keys(brief).sort().join("|") !== "art|body|content_sha256|name|placements|schema") fail("missing or unknown fields");
  if (brief.schema !== BRIEF_SCHEMA) fail("unsupported brief version");
  if (typeof brief.name !== "string" || !brief.name.trim()) fail("needs a name");
  if (brief.content_sha256 !== await canonicalDigest(brief)) fail("content digest differs");
  if (!Array.isArray(brief.placements) || !brief.placements.length) fail("needs at least one placement");
  for (const record of Object.values(brief.art)) await validateArtRecord(record);
  const used = new Set(brief.placements.map(p => p.art_id));
  if (Object.keys(brief.art).length !== used.size || [...used].some(id => !Object.hasOwn(brief.art, id))) fail("art registry must hold exactly the placed art");
  if (new Set(brief.placements.map(p => p.id)).size !== brief.placements.length) fail("duplicate placement id");
  for (const p of brief.placements) {
    if (!(Number.isFinite(p.width_mm) && p.width_mm > 0 && Math.abs(p.width_mm * 10 - Math.round(p.width_mm * 10)) < 1e-9)) fail(`${p.id}: width_mm must be positive, to 0.1 mm`);
    if (!(Number.isFinite(p.height_mm) && p.height_mm > 0)) fail(`${p.id}: height_mm must be positive`);
  }
  return structuredClone(brief);
}

/** The accepted placements of a placement file as a brief. */
export async function makeBrief(name: string, file: PlacementFile): Promise<InkmapBrief> {
  const art = Object.fromEntries(file.placements.map(p => {
    const record = file.designs?.[p.design_id];
    if (!record) fail(`${p.id}: its art is not embedded`);
    return [p.design_id, record!];
  }));
  const brief: InkmapBrief = {
    schema: BRIEF_SCHEMA, content_sha256: "", name: name.trim() || "Inkmap brief", body: structuredClone(file.body), art,
    placements: file.placements.map(p => ({
      id: p.id, art_id: p.design_id, width_mm: Math.round(p.size_mm[0] * 10) / 10, height_mm: Number(p.size_mm[1].toFixed(3)),
      rotation_rad: p.rotation_rad, mirror: p.mirror, anchor: structuredClone(p.anchor),
      site: p.site ? structuredClone(p.site) : null, sentence: p.language?.sentence ?? null,
    })),
  };
  brief.content_sha256 = await canonicalDigest(brief);
  return validateBrief(brief);
}
