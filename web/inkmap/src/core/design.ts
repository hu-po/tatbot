/** The portable authoring handoff shared by every target and consumer. */
import Ajv2020 from "ajv/dist/2020.js";
import designSchema from "../../../../config/inkmap/design.schema.json" with { type: "json" };
import artworkSchema from "../../../../config/inkmap/artwork.schema.json" with { type: "json" };
import programSchema from "../../../../config/human-representation/tattoo-program.schema.json" with { type: "json" };
import placementSchema from "../../../../config/human-representation/surface-placement-v2.schema.json" with { type: "json" };
import commonSchema from "../../../../config/human-representation/common.schema.json" with { type: "json" };
import { validateArtworkRecord, type ArtworkRecord } from "./artwork-record.ts";
import type { SurfacePlacement } from "./surface-placement.ts";
import { canonicalDigest, canonicalJson, parseJsonStrict, validateContract } from "./human-representation/schema.ts";

export interface InkmapDesign {
  schema: "tatbot.inkmap-design/1";
  content_sha256: string;
  name: string;
  artworks: Record<string, ArtworkRecord>;
  placements: { id: string; artwork_id: string; placement: SurfacePlacement }[];
}
export const MAX_DESIGN_BYTES = 20_000_000;
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const schema of [artworkSchema, programSchema, placementSchema, commonSchema]) ajv.addSchema(schema);
const check = ajv.compile(designSchema);
const fail = (detail: string): never => { throw new Error(`design_invalid: ${detail}`); };

export async function validateDesign(value: unknown): Promise<InkmapDesign> {
  if (new TextEncoder().encode(canonicalJson(value)).length > MAX_DESIGN_BYTES) fail("maximum 20 MB");
  if (!check(value)) fail(ajv.errorsText(check.errors));
  const design = value as InkmapDesign;
  if (design.content_sha256 !== await canonicalDigest(design)) fail("content digest differs");
  const used = new Set(design.placements.map(item => item.artwork_id));
  if (Object.keys(design.artworks).length !== used.size || [...used].some(id => !Object.hasOwn(design.artworks, id))) {
    fail("artwork registry must contain exactly the used artwork");
  }
  if (new Set(design.placements.map(item => item.id)).size !== design.placements.length) fail("duplicate placement ID");
  for (const artwork of Object.values(design.artworks)) await validateArtworkRecord(artwork);
  for (const item of design.placements) {
    await validateContract(item.placement, { expectedSchema: "tatbot.surface-placement/2" });
    if (item.placement.tattoo_program_sha256 !== design.artworks[item.artwork_id].program.content_sha256) {
      fail(`placement ${item.id} binds different artwork`);
    }
  }
  return structuredClone(design);
}

export async function makeDesign(input: Omit<InkmapDesign, "schema" | "content_sha256">): Promise<InkmapDesign> {
  const design = { ...structuredClone(input), schema: "tatbot.inkmap-design/1" as const, content_sha256: "" };
  design.content_sha256 = await canonicalDigest(design);
  return validateDesign(design);
}

export async function parseDesign(text: string): Promise<InkmapDesign> {
  if (new TextEncoder().encode(text).length > MAX_DESIGN_BYTES) fail("maximum 20 MB");
  return validateDesign(parseJsonStrict(text));
}
