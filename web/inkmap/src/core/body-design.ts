/** Convert the existing body editor into shared artwork and surface contracts.
 * Simulation and physical design export use these identical derivations.
 */
import { type ArtworkRecord, type ArtworkSource } from "./artwork-record.ts";
import { canonicalDigest, validateContract, type JsonObject } from "./human-representation/schema.ts";
import { SCHEMA_VERSION, validatePlacementFile, type EmbeddedDesign, type PlacementFile } from "./schema.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_ASSET_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
} from "./body.ts";
import type { AtlasData } from "./atlas.ts";
import { makeDesign, type InkmapDesign } from "./design.ts";
import { upgradeBodyPlacement } from "./surface-placement.ts";
import { frozenArtwork } from "./frozen-artwork.ts";

/** The one body every placement in this app is authored against. */
const BODY = {
  model_spec_id: MODEL_SPEC_ID, model_spec_sha256: MODEL_SPEC_SHA256,
  identity_sha256: REFERENCE_IDENTITY_SHA256, topology_sha256: TOPOLOGY_SHA256,
  rest_surface_sha256: REST_SURFACE_SHA256,
} as const;

const fail = (detail: string): never => { throw new Error(`body_design_invalid: ${detail}`); };

export async function surfaceBindings(file: PlacementFile, artworks: Record<string, ArtworkRecord>, atlas: AtlasData) {
  const supported = atlas.eligible_faces.flatMap((eligible, face) => eligible === 1 ? [face] : []);
  const output = [];
  for (const item of file.placements) {
    const face = item.anchor.face;
    const code = atlas.faces[face];
    if (atlas.eligible_faces[face] !== 1 || code < 0) fail(`placement ${item.id} is outside the supported atlas`);
    const semanticSite = atlas.sites[code >> 2];
    const side = (code & 3) === 1 ? "left" : (code & 3) === 2 ? "right" : "midline";
    if (item.site && (item.site.id !== semanticSite || (item.site.laterality ?? "center") !== (side === "midline" ? "center" : side))) fail(`placement ${item.id} site disagrees with its actual face`);
    const placement: JsonObject = {
      schema: "tatbot.surface-placement/1", content_sha256: "", tattoo_program_sha256: artworks[item.design_id].program.content_sha256,
      body_identity_sha256: file.body.identity_sha256, rest_surface_sha256: file.body.rest_surface_sha256,
      semantic_site: semanticSite, laterality: side,
      anchor: { topology_sha256: file.body.topology_sha256, face_index: face, barycentric: [...item.anchor.barycentric] },
      tangent_frame_rule: "projected-body-up-then-oriented-normal", physical_scale_m: item.size_mm.map(mm => mm / 1000),
      rotation_rad: item.rotation_rad, mirrored: item.mirror, warp: null,
      supported_domain: { face_indices: supported, margin_m: 0 },
      review: { status: "pending", reviewer: "unreviewed Inkmap export", evidence_sha256: "0".repeat(64) },
      // The artwork record, not the enclosing placement file. A portable design
      // does not carry the editor's optional site/language annotations, so
      // binding to the file's digest made a design's identity depend on data the
      // design itself does not hold: exporting, reopening and exporting again
      // produced a different document although nothing had been edited. Every
      // other producer of this contract already binds the record.
      provenance: { producer: "tatbot-inkmap-sim-bundle", version: "1", created_utc: "2026-09-05T00:00:00Z", source_sha256: artworks[item.design_id].content_sha256 },
    };
    placement.content_sha256 = await canonicalDigest(placement);
    await validateContract(placement, { expectedSchema: "tatbot.surface-placement/1" });
    output.push({ id: item.id, placement });
  }
  return output;
}


export async function bodyArtworks(file: PlacementFile, sources: Record<string, ArtworkSource> = {}): Promise<Record<string, ArtworkRecord>> {
  validatePlacementFile(file);
  const artworks: Record<string, ArtworkRecord> = {};
  for (const id of new Set(file.placements.map(p => p.design_id))) {
    const embedded = file.designs?.[id];
    if (!embedded) return fail(`missing frozen artwork ${id}`);
    artworks[id] = await frozenArtwork(id, embedded, sources[id]);
  }
  return artworks;
}

export async function designFromBodyFile(name: string, file: PlacementFile, atlas: AtlasData,
  sources: Record<string, ArtworkSource> = {}): Promise<InkmapDesign> {
  validatePlacementFile(file);
  const comparable = Object.fromEntries(Object.entries(file.body).filter(([key]) => key !== "asset_path"));
  if (await canonicalDigest(comparable) !== await canonicalDigest(atlas.body)) fail("atlas/body binding differs");
  const artworks = await bodyArtworks(file, sources);
  const bindings = await surfaceBindings(file, artworks, atlas);
  const placements = await Promise.all(bindings.map(async ({ id, placement }, index) => ({
    id, artwork_id: file.placements[index].design_id, placement: await upgradeBodyPlacement(placement),
  })));
  return makeDesign({ name, artworks, placements });
}

/** The inverse: a portable design whose placements sit on this body, reopened.
 *
 * Same pinned identity, atlas, anchor and tangent conventions as the export
 * above, read backwards. A design authored on a different body, or on a face
 * this atlas does not support, is refused by name rather than approximated
 * onto the nearest thing that happens to be here.
 *
 * The portable contract carries artwork, anchors and metric placement; it does
 * not carry the editor's optional site/language annotations, so those are not
 * invented on the way in. Export -> import -> export is byte-identical, which
 * is the round trip that has to hold.
 */
export async function bodyFileFromDesign(design: InkmapDesign, atlas: AtlasData): Promise<PlacementFile> {
  const targets = design.placements.map(item => item.placement.target);
  if (!targets.length || targets.some(target => target.kind !== "body")) {
    fail("this design places artwork on a paper or cylinder chart; open it in the surface editor");
  }
  const designs: Record<string, EmbeddedDesign> = {};
  const placements = design.placements.map((item, index) => {
    const placement = item.placement;
    const target = placement.target as unknown as {
      body_identity_sha256: string; rest_surface_sha256: string; semantic_site: string;
      laterality: string; anchor: { topology_sha256: string; face_index: number; barycentric: number[] };
    };
    if (target.body_identity_sha256 !== BODY.identity_sha256 || target.rest_surface_sha256 !== BODY.rest_surface_sha256
      || target.anchor.topology_sha256 !== BODY.topology_sha256) {
      fail(`placement ${item.id} was authored on a different body`);
    }
    const face = target.anchor.face_index;
    const code = atlas.faces[face];
    if (atlas.eligible_faces[face] !== 1 || code === undefined || code < 0) {
      fail(`placement ${item.id} is outside the supported atlas`);
    }
    const side = (code & 3) === 1 ? "left" : (code & 3) === 2 ? "right" : "midline";
    if (atlas.sites[code >> 2] !== target.semantic_site || side !== target.laterality) {
      fail(`placement ${item.id} names a site this atlas does not put on that face`);
    }
    const record = design.artworks[item.artwork_id];
    designs[item.artwork_id] = embeddedFromArtwork(record);
    return {
      id: item.id || `p-${index}`, design_id: item.artwork_id,
      anchor: { face, barycentric: [...target.anchor.barycentric] as [number, number, number] },
      rotation_rad: placement.rotation_rad,
      size_mm: placement.physical_scale_m.map(value => value * 1000) as [number, number],
      mirror: placement.mirrored,
    };
  });
  const file = { schema_version: SCHEMA_VERSION, units: { length: "m", tattoo_size: "mm", up: "+z" },
    body: { ...BODY, asset_path: BODY_SPEC.path, asset_sha256: REST_ASSET_SHA256 },
    placements, designs } as unknown as PlacementFile;
  validatePlacementFile(file);
  return file;
}

/** The frozen bytes a project carries for one artwork record. */
export function embeddedFromArtwork(record: ArtworkRecord): EmbeddedDesign {
  return structuredClone(record);
}
