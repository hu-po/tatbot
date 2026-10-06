/** Files in and out, named by what they hold.
 *
 * Every export here says what it is: the artwork SVG *as this editor draws
 * it* (a preview, not a calibrated stencil and not a robot path), a portable
 * design, a project backup, and — under Advanced — the placement file and the
 * simulation bundle. A refusal is reported beside the action that produced
 * it; nothing is silently narrowed to something else.
 */
import { parseJsonStrict } from "./core/human-representation/schema.ts";
import { artworkSizeM, requireAcquiredArtwork, validateArtworkRecord } from "./core/artwork-record.ts";
import { tattooProgramToSvg } from "./core/human-representation/program-svg.ts";
import { artworkSources } from "./core/design-assets.ts";
import { frozenArtwork } from "./core/frozen-artwork.ts";
import { openDesignFile } from "./core/design-open.ts";
import { chartDraftFromDesign, designFromChartDraft } from "./core/chart-draft.ts";
import { downloadJson, downloadText } from "./core/download.ts";
import { parseProject, MAX_PROJECT_BYTES } from "./core/project.ts";
import { validatePlacementFile } from "./core/schema.ts";
import type { DesignMeta } from "./core/schema.ts";
import { useStore } from "./store.ts";
import { useUi } from "./ui.ts";

const notice = (action: string, error: unknown) =>
  useUi.getState().setNotice({ action, message: error instanceof Error ? error.message : String(error) });

/** A file kept for download exactly as it arrived, when it could not be opened. */
export interface UnsupportedFile { name: string; text: string; reason: string }

const safeName = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "artwork";

/** The artwork itself, before or after any placement: the frozen record's preview SVG. */
export async function exportArtworkSvg(design: DesignMeta): Promise<void> {
  const action = "Artwork SVG";
  useUi.getState().setNotice(null);
  try {
    if (!design.embedded) throw new Error("Artwork is still loading");
    const record = await frozenArtwork(design.id, design.embedded, artworkSources(useStore.getState().designs)[design.id]);
    downloadText(await tattooProgramToSvg(record.program), `inkmap-artwork-${safeName(design.name)}.svg`, "image/svg+xml");
  } catch (error) { notice(action, error); }
}

/** The body's accepted placements as one portable design. */
export async function exportBodyDesign(): Promise<void> {
  useUi.getState().setNotice(null);
  try { downloadJson(await useStore.getState().toDesign(), "inkmap-design.json"); }
  catch (error) { notice("Portable design", error); }
}

/** The paper/cylinder draft as one portable design. */
export async function exportChartDesign(): Promise<void> {
  useUi.getState().setNotice(null);
  const { chart, chartEditBefore, chartPlacing } = useStore.getState();
  if (chartEditBefore || chartPlacing) { notice("Portable design", new Error("Accept or cancel pending edits before exporting.")); return; }
  try { downloadJson(await designFromChartDraft(chart), "inkmap-design.json"); }
  catch (error) { notice("Portable design", error); }
}

/** The selected chart placement's artwork — the selected one, not the first. */
export async function exportChartArtworkSvg(): Promise<void> {
  useUi.getState().setNotice(null);
  try {
    const { chart } = useStore.getState();
    const item = chart.items.find((candidate) => candidate.id === chart.selected);
    if (!item) throw new Error("Select a placement to export its artwork");
    downloadText(await tattooProgramToSvg(item.artwork.program), `inkmap-artwork-${safeName(item.artwork.name)}.svg`, "image/svg+xml");
  } catch (error) { notice("Artwork SVG", error); }
}

export async function exportProjectBackup(): Promise<void> {
  useUi.getState().setNotice(null);
  try { downloadJson(await useStore.getState().toProject(), "inkmap-project.json"); }
  catch (error) { notice("Project backup", error); }
}

/** Advanced: the v6 body placement file. */
export function exportPlacementFile(): void {
  useUi.getState().setNotice(null);
  const file = useStore.getState().toFile();
  if (!file) { notice("Placement file", useStore.getState().error ?? "Nothing to export"); return; }
  downloadJson(file, `inkmap-${file.body.model_spec_id}.json`);
}

export async function openProject(file: File | undefined): Promise<void> {
  if (!file) return;
  useUi.getState().setNotice(null);
  try {
    if (file.size > MAX_PROJECT_BYTES) throw new Error("Project exceeds 20 MB");
    useStore.getState().restoreProject(await parseProject(await file.text()));
    useStore.getState().setToast(`Opened project “${file.name}”`);
  } catch (error) { notice("Open project", error); }
}

/** One Open design: the file says which editor can show it, and the shell
 *  goes there. An unsupported file leaves the working draft alone and stays
 *  available for download, byte for byte. */
export async function openDesign(file: File | undefined): Promise<UnsupportedFile | null> {
  if (!file) return null;
  useUi.getState().setNotice(null);
  try {
    const text = await file.text();
    const route = await openDesignFile(new File([text], file.name));
    if (route.kind === "unsupported") {
      notice("Open design", route.reason);
      return { name: file.name, text, reason: route.reason };
    }
    if (route.kind === "body") {
      await useStore.getState().loadDesign(route.design);
      useUi.getState().setWorkspace("body");
      return null;
    }
    const draft = chartDraftFromDesign(route.design);
    useStore.getState().openChartDraft(draft);
    useUi.getState().setWorkspace(draft.kind === "cylinder" ? "cylinder" : "paper");
    useStore.getState().setToast(`Opened design “${route.design.name}”`);
    return null;
  } catch (error) { notice("Open design", error); return null; }
}

/** Advanced: a raw v6 placement file into the body editor. */
export async function loadPlacementFile(file: File | undefined): Promise<void> {
  if (!file) return;
  useUi.getState().setNotice(null);
  try {
    if (file.size > MAX_PROJECT_BYTES) throw new Error("Placement file exceeds 20 MB");
    const value = parseJsonStrict(await file.text());
    validatePlacementFile(value);
    await Promise.all(Object.values(value.designs ?? {}).map(validateArtworkRecord));
    useStore.getState().loadFile(value);
    useStore.getState().setError(null);
  }
  catch (error) { notice("Load placement file", error); }
}

/** Import only the native acquisition record; previews and source SVGs are not artwork. */
export async function importArtwork(file: File | undefined): Promise<string | null> {
  if (!file) return null;
  useUi.getState().setNotice(null);
  try {
    if (file.size > MAX_PROJECT_BYTES) throw new Error("Artwork exceeds 20 MB");
    const text = await file.text();
    if (!text.trimStart().startsWith("{")) throw new Error("Generate with DrawingBot V3, then import its artwork.json. Supply source SVG or images to DBV3 first.");
    const artwork = await validateArtworkRecord(parseJsonStrict(text));
    requireAcquiredArtwork(artwork);
    const id = `art-${artwork.content_sha256}`;
    useStore.getState().addDesign({ id, name: artwork.name,
      path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(await tattooProgramToSvg(artwork.program))}`,
      default_size_mm: artworkSizeM(artwork).map(m => m * 1000) as [number, number], embedded: artwork });
    return id;
  } catch (error) { notice("Import artwork", error); return null; }
}
