import commonSchema from "../../../../config/human-representation/common.schema.json" with { type: "json" };
import programSchema from "../../../../config/human-representation/tattoo-program.schema.json" with { type: "json" };
import artworkSchema from "../../../../config/inkmap/artwork.schema.json" with { type: "json" };
import Ajv2020 from "ajv/dist/2020.js";
import projectSchema from "../../../../config/inkmap/project.schema.json" with { type: "json" };
import placementSchema from "../../../../config/inkmap/placement.schema.json" with { type: "json" };
import { canonicalDigest, canonicalDocumentDigest, canonicalJson, parseJsonStrict } from "./human-representation/schema.ts";
import { validatePlacementFile, type Placement, type PlacementFile } from "./schema.ts";
import { requireAcquiredArtwork } from "./artwork-record.ts";
import { frozenArtwork } from "./frozen-artwork.ts";
import { POSE_CATALOG } from "./pose.ts";
import { MAX_CHART_ITEMS, type ChartDocument } from "./chart-draft.ts";

export const PROJECT_SCHEMA = "tatbot.inkmap-project/3";
export interface ProjectCamera { position: [number, number, number]; target: [number, number, number]; fov: number }
export interface InkmapProject {
  schema: "tatbot.inkmap-project/3";
  content_sha256: string;
  name: string;
  placement_file: PlacementFile;
  editor: { pose_id: string; skin_tone: string; show_atlas: boolean; camera: ProjectCamera | null };
  history: { past: Placement[][]; future: Placement[][] };
  edit_before: Placement[] | null;
  selected_id: string | null;
  chart: ChartDocument | null;
  chart_parked: ChartDocument | null;
}
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const schema of [artworkSchema, programSchema, commonSchema, placementSchema]) ajv.addSchema(schema);
const check = ajv.compile(projectSchema);
export const MAX_PROJECT_BYTES = 20_000_000;
// Autosave validates repeatedly as placements move. Cache only a bounded set
// of verified *input* hashes, never the untrusted claimed artwork digest.
const verifiedArtwork = new Set<string>();

export async function validateProject(value: unknown): Promise<InkmapProject> {
  if (!check(value)) throw new Error(`Regenerate legacy artwork with DrawingBot V3 and import artwork.json. project_invalid: ${ajv.errorsText(check.errors)}`);
  const p = value as unknown as InkmapProject;
  if (new TextEncoder().encode(canonicalJson(p)).length > MAX_PROJECT_BYTES) throw new Error("project_over_budget: maximum 20 MB");
  if (p.content_sha256 !== await canonicalDigest(p)) throw new Error("project_wrong_hash: content digest differs");
  if (p.chart) validateChart(p.chart);
  if (p.chart_parked) {
    validateChart(p.chart_parked);
    if (p.chart && p.chart_parked.kind === p.chart.kind) throw new Error("project_invalid: the parked draft is for the surface already showing");
  }
  if (!POSE_CATALOG.pose_ids.includes(p.editor.pose_id)) throw new Error("project_invalid: unknown pose");
  if (p.editor.camera && p.editor.camera.position.every((v, i) => v === p.editor.camera!.target[i])) throw new Error("project_invalid: camera position and target coincide");
  if ((p.edit_before === null) !== (p.selected_id === null) || (p.selected_id && !p.placement_file.placements.some(item => item.id === p.selected_id))) throw new Error("project_invalid: inconsistent pending edit");
  const frames = [p.placement_file.placements, ...p.history.past, ...p.history.future, ...(p.edit_before ? [p.edit_before] : [])];
  for (const placements of frames) {
    validatePlacementFile({ ...p.placement_file, placements });
    if (placements.length > 100 || new Set(placements.map(item => item.id)).size !== placements.length) throw new Error("project_invalid: duplicate IDs or more than 100 placements");
    for (const placement of placements) {
      if (placement.anchor.face >= 36108) throw new Error("project_invalid: face outside topology");
      if (!Object.hasOwn(p.placement_file.designs ?? {}, placement.design_id)) throw new Error(`project_missing_artwork: ${placement.design_id}`);
    }
  }
  // Verify the frozen geometry before replacing a working project. Shared artwork
  // may occur on body, paper and cylinder; validate each distinct record once.
  const held = [
    ...Object.entries(p.placement_file.designs ?? {}).map(([id, design]) => ({ id, design })),
    ...[p.chart, p.chart_parked].flatMap(chart => Object.entries(chart?.artwork ?? {}).map(([id, design]) => ({ id, design }))),
  ];
  for (const { id, design } of held) {
    const key = await canonicalDocumentDigest(design);
    if (verifiedArtwork.has(key)) continue;
    await frozenArtwork(id, design);
    requireAcquiredArtwork(design);
    if (verifiedArtwork.size >= 256) verifiedArtwork.delete(verifiedArtwork.values().next().value!);
    verifiedArtwork.add(key);
  }
  return structuredClone(p);
}

function validateChart(chart: ChartDocument): void {
  if (chart.items.length > MAX_CHART_ITEMS) throw new Error(`project_invalid: at most ${MAX_CHART_ITEMS} chart placements`);
  if (new Set(chart.items.map(item => item.id)).size !== chart.items.length) throw new Error("project_invalid: duplicate chart placement ID");
  for (const item of chart.items) {
    if (!Object.hasOwn(chart.artwork, item.artwork_id)) throw new Error(`project_missing_artwork: ${item.artwork_id}`);
  }
  if (chart.selected_id !== null && !chart.items.some(item => item.id === chart.selected_id)) {
    throw new Error("project_invalid: chart selection names no placement");
  }
}

export async function makeProject(input: Omit<InkmapProject, "schema" | "content_sha256" | "chart" | "chart_parked"> & { chart?: ChartDocument | null; chart_parked?: ChartDocument | null }): Promise<InkmapProject> {
  const project = { ...structuredClone(input), chart: input.chart ?? null, chart_parked: input.chart_parked ?? null,
    schema: PROJECT_SCHEMA as typeof PROJECT_SCHEMA, content_sha256: "" };
  project.content_sha256 = await canonicalDigest(project);
  return validateProject(project);
}

export async function parseProject(text: string): Promise<InkmapProject> {
  if (new TextEncoder().encode(text).length > MAX_PROJECT_BYTES) throw new Error("project_over_budget: maximum 20 MB");
  return validateProject(parseJsonStrict(text));
}
