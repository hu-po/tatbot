/** One local, immutable authoring-to-simulator boundary. Never fetch by an ID
 * supplied in a bundle, and never treat an authoring click as safety approval.
 */
import Ajv2020 from "ajv/dist/2020.js";
import bundleSchema from "../../../../config/inkmap/sim-bundle.schema.json" with { type: "json" };
import scenarioV3Schema from "../../../../config/inkmap/tattoo-scenario-v3.schema.json" with { type: "json" };
import placementSchema from "../../../../config/inkmap/placement.schema.json" with { type: "json" };
import artworkSchema from "../../../../config/inkmap/artwork.schema.json" with { type: "json" };
import programSchema from "../../../../config/human-representation/tattoo-program.schema.json" with { type: "json" };
import inkProgramSchema from "../../../../config/human-representation/ink-program.schema.json" with { type: "json" };
import surfaceSchema from "../../../../config/human-representation/surface-placement.schema.json" with { type: "json" };
import commonSchema from "../../../../config/human-representation/common.schema.json" with { type: "json" };
import modelSpec from "../../../../config/body-models/mhr-soma-v1.json" with { type: "json" };
import { validateArtworkRecord, type ArtworkRecord, type ArtworkSource } from "./artwork-record.ts";
import { canonicalDigest, canonicalDocumentDigest, canonicalJson, parseJsonStrict, validateContract, type JsonObject } from "./human-representation/schema.ts";
import { validatePlacementFile, type PlacementFile } from "./schema.ts";
import type { AtlasData } from "./atlas.ts";
import type { ProjectCamera } from "./project.ts";
import { POSE_CATALOG, POSE_CATALOG_SHA256 } from "./pose.ts";
import { validateTattooScenario, type TattooScenario } from "./scenario.ts";

import { renderTattooProgramSvg } from "./human-representation/program-svg.ts";
import { bodyArtworks, surfaceBindings } from "./body-design.ts";

export const MAX_BUNDLE_BYTES = 20_000_000;
export interface SimulationRequest {
  pose_id: string; pose_catalog_sha256: string; support_id: string; tool_id: string; seed: number;
  skin_tone: string; camera: ProjectCamera | null;
  support_offset_m?: [number, number, number];
  target_world_m: [number, number, number]; align_patch_up: boolean; patch_yaw_rad: number;
}
export interface SimulationBundle {
  schema: "tatbot.inkmap-sim-bundle/1"; content_sha256: string;
  placement_file: PlacementFile; artworks: Record<string, ArtworkRecord>;
  surface_placements: { id: string; placement: JsonObject }[];
  atlas_sha256: string; request: SimulationRequest;
  assets: ReturnType<typeof pinnedAssets>;
}
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const schema of [placementSchema, artworkSchema, programSchema, surfaceSchema, commonSchema, inkProgramSchema]) ajv.addSchema(schema);
const check = ajv.compile(bundleSchema);
const checkScenarioV3 = ajv.compile(scenarioV3Schema);
const fail = (detail: string): never => { throw new Error(`sim_bundle_invalid: ${detail}`); };
const same = (a: unknown, b: unknown) => canonicalJson(a) === canonicalJson(b);

export function pinnedAssets() {
  const licenses = [...new Set(modelSpec.assets.map(asset => asset.license))];
  if (licenses.length !== 1) fail("body asset license ledger needs explicit per-asset mapping");
  return ([ ["body-rest", POSE_CATALOG.rest_asset], ["body-poses", POSE_CATALOG.pose_asset],
    ["body-exclusions", POSE_CATALOG.exclusion_asset] ] as const).map(([key, asset]) => ({
    key, sha256: asset.sha256, byte_length: asset.size, storage: "pinned-local-cache" as const,
    license: licenses[0], license_source_sha256: modelSpec.content_sha256,
  }));
}

export function simulationRequest(pose: string, skin: string, camera: ProjectCamera | null, tool: string, seed: number): SimulationRequest {
  const record = POSE_CATALOG.poses[pose];
  if (!record) fail("unknown pose");
  return { pose_id: pose, pose_catalog_sha256: POSE_CATALOG_SHA256, support_id: record.support_id,
    tool_id: tool, seed, skin_tone: skin, camera: structuredClone(camera),
    target_world_m: [.29, 0, .04], align_patch_up: true, patch_yaw_rad: Math.PI };
}


export async function validateSimulationBundle(value: unknown, atlas: AtlasData): Promise<SimulationBundle> {
  if (new TextEncoder().encode(canonicalJson(value)).length > MAX_BUNDLE_BYTES) fail("maximum 20 MB");
  if (!check(value)) fail(ajv.errorsText(check.errors));
  const bundle = value as SimulationBundle;
  if (bundle.content_sha256 !== await canonicalDigest(bundle)) fail("content digest differs");
  const file = bundle.placement_file;
  validatePlacementFile(file);
  if (file.body.model_spec_sha256 !== modelSpec.content_sha256 || modelSpec.content_sha256 !== await canonicalDigest(modelSpec)) fail("body model/license ledger digest differs");
  if (!file.placements.length || file.placements.length > 100 || new Set(file.placements.map(p => p.id)).size !== file.placements.length) fail("expected 1–100 unique placements");
  const comparable = Object.fromEntries(Object.entries(file.body).filter(([key]) => key !== "asset_path"));
  if (!same(atlas.body, comparable) || bundle.atlas_sha256 !== await canonicalDigest(atlas)) fail("atlas/body binding differs");
  const used = [...new Set(file.placements.map(p => p.design_id))].sort();
  if (!same(Object.keys(bundle.artworks).sort(), used) || !same(Object.keys(file.designs ?? {}).sort(), used)) fail("artwork registry must contain exactly the used designs");
  for (const id of used) {
    const art = await validateArtworkRecord(bundle.artworks[id]);
    const embedded = file.designs![id];
    if (!same(art, embedded)) fail(`artwork ${id} differs from the frozen placement record`);
  }
  const request = bundle.request;
  if (request.pose_catalog_sha256 !== POSE_CATALOG_SHA256 || !POSE_CATALOG.pose_ids.includes(request.pose_id)) fail("unknown pose/catalog");
  if (request.support_id !== POSE_CATALOG.poses[request.pose_id].support_id) fail("support does not match the named pose");
  if (request.camera && same(request.camera.position, request.camera.target)) fail("camera position and target coincide");
  if (!same(bundle.assets, pinnedAssets())) fail("asset manifest differs from pinned local cache");
  if (!same(bundle.surface_placements, await surfaceBindings(file, bundle.artworks, atlas))) fail("typed placement derivation/order differs");
  return structuredClone(bundle);
}

export async function makeSimulationBundle(file: PlacementFile, atlas: AtlasData, request: SimulationRequest,
  sources: Record<string, ArtworkSource> = {}): Promise<SimulationBundle> {
  validatePlacementFile(file);
  if (!file.placements.length || file.placements.length > 100 || new TextEncoder().encode(canonicalJson(file)).length > MAX_BUNDLE_BYTES) fail("placement input exceeds bundle limits or is empty");
  const artworks = await bodyArtworks(file, sources);
  const bundle: SimulationBundle = { schema: "tatbot.inkmap-sim-bundle/1", content_sha256: "",
    placement_file: structuredClone(file), artworks, request: structuredClone(request), assets: pinnedAssets(),
    atlas_sha256: await canonicalDigest(atlas), surface_placements: await surfaceBindings(file, artworks, atlas) };
  bundle.content_sha256 = await canonicalDigest(bundle);
  return validateSimulationBundle(bundle, atlas);
}

export async function parseSimulationBundle(text: string, atlas: AtlasData): Promise<SimulationBundle> {
  if (new TextEncoder().encode(text).length > MAX_BUNDLE_BYTES) fail("maximum 20 MB");
  return validateSimulationBundle(parseJsonStrict(text), atlas);
}

/** Validate a compiled v3 preview without claiming Python execution validity.
 * The browser can prove the portable bundle/program/trace bindings. Local
 * tool-profile bytes and compiler reconstruction remain the CLI's authority.
 */
export async function validateCompiledScenario(value: unknown, atlas: AtlasData): Promise<TattooScenario> {
  if (new TextEncoder().encode(canonicalJson(value)).length > MAX_BUNDLE_BYTES) fail("compiled preview exceeds 20 MB");
  if (!checkScenarioV3(value)) fail(`compiled preview: ${ajv.errorsText(checkScenarioV3.errors)}`);
  validateTattooScenario(value);
  const scenario = value as TattooScenario & { program_binding: {
    bundle: SimulationBundle; placement_id: string; ink_program: JsonObject; tool_profile_sha256: string;
  } };
  const binding = scenario.program_binding;
  const bundle = await validateSimulationBundle(binding.bundle, atlas);
  const placementIndex = bundle.placement_file.placements.findIndex(item => item.id === binding.placement_id);
  if (placementIndex < 0) fail("compiled preview placement ID is absent from its bundle");
  const placement = bundle.placement_file.placements[placementIndex];
  const surface = bundle.surface_placements[placementIndex].placement;
  const artwork = bundle.artworks[placement.design_id];
  await validateContract(binding.ink_program, { expectedSchema: "tatbot.ink-program/1" });
  if (binding.ink_program.tattoo_program_sha256 !== artwork.program.content_sha256
      || binding.ink_program.surface_placement_sha256 !== surface.content_sha256) {
    fail("compiled InkProgram is bound to different artwork or placement");
  }
  const expectedPlacement = { ...placement, source_sha256: await canonicalDigest(bundle.placement_file) };
  const expectedDesign = { id: placement.design_id, name: artwork.name, svg: renderTattooProgramSvg(artwork.program),
    sha256: artwork.source_sha256, source: artwork.source };
  const events = binding.ink_program.events as JsonObject[];
  const strokes = events.filter(event => event.kind === "stroke").map(event => {
    const curve = event.curve as JsonObject;
    return (curve.coordinates as JsonObject[]).map(point => ({
      face: point.face_index, barycentric: point.barycentric,
    }));
  });
  const expectedTrace = { compiler: "tatbot_sim.surface_trace", compiler_version: 3,
    sha256: await canonicalDocumentDigest(strokes), strokes };
  if (!same(scenario.placement, expectedPlacement)) fail("compiled placement differs from its binding");
  if (!same(scenario.design, expectedDesign)) fail("compiled design differs from its binding");
  if (!same(scenario.trace, expectedTrace)) fail("compiled trace differs from its binding");
  const request = bundle.request;
  if (scenario.seed !== request.seed || scenario.pose.id !== request.pose_id
      || scenario.pose.catalog_sha256 !== request.pose_catalog_sha256
      || scenario.support.id !== request.support_id || scenario.robot.tool_id !== request.tool_id) {
    fail("compiled request fields differ from the immutable bundle");
  }
  const offset = request.support_offset_m ?? [0, 0, 0];
  const support = [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]];
  if (!same(scenario.support.world_from_nominal, support)) fail("compiled support differs from bound request");
  return structuredClone(scenario);
}
