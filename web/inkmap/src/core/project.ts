import artSchema from "../../../../config/inkmap/art.schema.json" with { type: "json" };
import Ajv2020 from "ajv/dist/2020.js";
import projectSchema from "../../../../config/inkmap/project.schema.json" with { type: "json" };
import placementSchema from "../../../../config/inkmap/placement.schema.json" with { type: "json" };
import { canonicalDigest, canonicalDocumentDigest, canonicalJson, parseJsonStrict } from "./human-representation/schema.ts";
import { validatePlacementFile, type Placement, type PlacementFile } from "./schema.ts";
import { validateArtRecord } from "./art.ts";
import { POSE_CATALOG } from "./pose.ts";
export const PROJECT_SCHEMA = "tatbot.inkmap-project/5";
export interface ProjectCamera { position: [number, number, number]; target: [number, number, number]; fov: number }
export interface InkmapProject {
  schema: "tatbot.inkmap-project/5";
  content_sha256: string;
  name: string;
  placement_file: PlacementFile;
  editor: { pose_id: string; skin_tone: string; show_atlas: boolean; camera: ProjectCamera | null };
  history: { past: Placement[][]; future: Placement[][] };
  edit_before: Placement[] | null;
  selected_id: string | null;
}
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const schema of [artSchema, placementSchema]) ajv.addSchema(schema);
const check = ajv.compile(projectSchema);
export const MAX_PROJECT_BYTES = 20_000_000;
// Autosave validates repeatedly as placements move. Cache only a bounded set
// of verified *input* hashes, never the untrusted claimed art digest.
const verifiedArt = new Set<string>();

export async function validateProject(value: unknown): Promise<InkmapProject> {
  if ((value as { schema?: unknown } | null)?.schema !== PROJECT_SCHEMA) throw new Error("project_invalid: this project is from another version of Inkmap and cannot open here");
  if (!check(value)) throw new Error(`project_invalid: ${ajv.errorsText(check.errors)}`);
  const p = value as unknown as InkmapProject;
  if (new TextEncoder().encode(canonicalJson(p)).length > MAX_PROJECT_BYTES) throw new Error("project_over_budget: maximum 20 MB");
  if (p.content_sha256 !== await canonicalDigest(p)) throw new Error("project_wrong_hash: content digest differs");
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
  // Verify the art before replacing a working project; validate each distinct record once.
  for (const design of Object.values(p.placement_file.designs ?? {})) {
    const key = await canonicalDocumentDigest(design);
    if (verifiedArt.has(key)) continue;
    await validateArtRecord(design);
    if (verifiedArt.size >= 256) verifiedArt.delete(verifiedArt.values().next().value!);
    verifiedArt.add(key);
  }
  return structuredClone(p);
}

export async function makeProject(input: Omit<InkmapProject, "schema" | "content_sha256">): Promise<InkmapProject> {
  const project = { ...structuredClone(input), schema: PROJECT_SCHEMA as typeof PROJECT_SCHEMA, content_sha256: "" };
  project.content_sha256 = await canonicalDigest(project);
  return validateProject(project);
}

export async function parseProject(text: string): Promise<InkmapProject> {
  if (new TextEncoder().encode(text).length > MAX_PROJECT_BYTES) throw new Error("project_over_budget: maximum 20 MB");
  return validateProject(parseJsonStrict(text));
}
