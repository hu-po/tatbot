/** Files in and out, named by what they hold.
 *
 * Inkmap takes source art in (PNG or JPEG, with the art corpus's generation sidecar when there is one) and
 * gives out what it decided: the tattoo brief for the art pipeline (each design's art and the size to separate
 * it at), the project backup, the source image itself, and under Advanced the placement file. A refusal is
 * reported beside the action that produced it; nothing is silently narrowed to something else.
 */
import { parseJsonStrict } from "./core/human-representation/schema.ts";
import { base64ToBytes } from "./core/art.ts";
import { captureView } from "./capture.ts";
import { artFromFile, artProfile, artThumbnail, importNotes, knownInks } from "./core/art-image.ts";
import { saveRenders } from "./core/project-db.ts";
import { validateRender } from "./core/render-record.ts";
import { thinInks } from "./core/art-profile.ts";
import { cardHtml, type CardSection } from "./core/card.ts";
import { libraryEntry } from "./core/design-assets.ts";
import { downloadJson, downloadText } from "./core/download.ts";
import { estimateCost, formatMinutes } from "./core/estimate.ts";
import { inkById } from "./core/inks.ts";
import { placementGuide } from "./core/landmarks.ts";
import { overflowMm, STENCIL_TILES } from "./core/tiles.ts";
import { parseProject, MAX_PROJECT_BYTES } from "./core/project.ts";
import { validatePlacementFile } from "./core/schema.ts";
import type { DesignMeta } from "./core/schema.ts";
import { renderFor, useStore } from "./store.ts";
import { useUi } from "./ui.ts";

const notice = (action: string, error: unknown) =>
  useUi.getState().setNotice({ action, message: error instanceof Error ? error.message : String(error) });

const safeName = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "art";

/** The design's source image, byte for byte. */
export function exportArtImage(design: DesignMeta): void {
  useUi.getState().setNotice(null);
  try {
    const record = design.embedded;
    if (!record) throw new Error("The art is still loading");
    const blob = new Blob([base64ToBytes(record.image.data_base64) as BlobPart], { type: record.image.media_type });
    const url = URL.createObjectURL(blob);
    const link = Object.assign(document.createElement("a"), { href: url, download: `${safeName(record.name)}.${record.image.media_type === "image/png" ? "png" : "jpg"}` });
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { notice("Source image", error); }
}

/** The accepted placements as a tattoo brief, for `tatbot art separate --brief`. */
export async function exportBrief(): Promise<void> {
  useUi.getState().setNotice(null);
  try { downloadJson(await useStore.getState().toBrief(), "inkmap-brief.json"); }
  catch (error) { notice("Tattoo brief", error); }
}

/** `inkmap-render.json` files from `tatbot art separate --brief`, to show on the body. */
export async function importRenders(input: FileList | readonly File[] | null | undefined): Promise<void> {
  useUi.getState().setNotice(null);
  try {
    for (const file of input ? [...input] : []) {
      const render = await validateRender(parseJsonStrict(await file.text()));
      if (!useStore.getState().designs.some((design) => design.embedded?.content_sha256 === render.art_sha256)) {
        throw new Error(`${file.name} is of art this project does not hold, or whose inks have changed. Separate its brief again.`);
      }
      useStore.getState().addRender(render);
    }
    await saveRenders(useStore.getState().renders);
  } catch (error) { notice("Robot render", error); }
}

/** One printable page per placement, for whoever lays the stencil: `core/card.ts`. */
export async function exportCard(): Promise<void> {
  useUi.getState().setNotice(null);
  try {
    const state = useStore.getState();
    const brief = await state.toBrief();
    const tile = STENCIL_TILES.find((item) => item.id === useUi.getState().tile)!;
    const sections = await Promise.all(state.placements.map(async (placement): Promise<CardSection> => {
      const record = brief.art[placement.design_id];
      const profile = await artProfile(record);
      const inks = record.palette?.inks ?? [];
      const cost = estimateCost(profile.coverage, placement.size_mm, inks.length);
      const render = renderFor(state, record.content_sha256, placement.size_mm[0], placement.mirror);
      return {
        name: record.name, sentence: placement.language?.sentence ?? null, sizeMm: placement.size_mm, mirror: placement.mirror, flipArt: placement.mirror && !render,
        tile: { name: tile.name, sizeMm: tile.size_mm, fits: overflowMm(tile, placement.size_mm).every((over) => over === 0) },
        guide: state.atlas && state.body ? placementGuide(state.atlas, state.body.restSkin.geometry, placement) : null,
        inks: inks.map((id) => ({ name: inkById.get(id)?.name ?? id, rgb: inkById.get(id)?.rgb ?? null })),
        estimate: render ? `${formatMinutes(render.minutes)}, ${render.dips} dips (separated)` : `about ${formatMinutes(cost.minutes)}, ${Math.round(cost.dips / 5) * 5} dips`,
        detail: render ? render.lost_inks.map((id) => `${inkById.get(id)?.name ?? id} drew nothing at this size`)
          : thinInks(profile, knownInks(record), placement.size_mm[0]).map(({ ink, minWidthMm }) => `${ink.name} may be lost below about ${Math.ceil(minWidthMm)} mm`),
        artUrl: render ? `data:image/png;base64,${render.image.data_base64}` : await artThumbnail(record, 1024), mockupUrl: captureView(placement),
      };
    }));
    downloadText(cardHtml(brief.name, brief.content_sha256, sections), "inkmap-card.html", "text/html");
  } catch (error) { notice("Placement card", error); }
}

/** The current view of the body, as a picture to share. */
export function exportMockup(): void {
  useUi.getState().setNotice(null);
  try { Object.assign(document.createElement("a"), { href: captureView(), download: "inkmap-mockup.jpg" }).click(); }
  catch (error) { notice("Mockup image", error); }
}

export async function exportProjectBackup(): Promise<void> {
  useUi.getState().setNotice(null);
  try { downloadJson(await useStore.getState().toProject(), "inkmap-project.json"); }
  catch (error) { notice("Project backup", error); }
}

/** Advanced: the v7 body placement file. */
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

/** Advanced: a raw v7 placement file into the editor. */
export async function loadPlacementFile(file: File | undefined): Promise<void> {
  if (!file) return;
  useUi.getState().setNotice(null);
  try {
    if (file.size > MAX_PROJECT_BYTES) throw new Error("Placement file exceeds 20 MB");
    const value = parseJsonStrict(await file.text());
    validatePlacementFile(value);
    const { validateArtRecord } = await import("./core/art.ts");
    await Promise.all(Object.values(value.designs ?? {}).map(validateArtRecord));
    useStore.getState().loadFile(value);
    useStore.getState().setError(null);
  }
  catch (error) { notice("Load placement file", error); }
}

/** Import source art: PNG or JPEG images, each with its generation sidecar (`<image stem>.json`) when that is
 *  selected too. Returns the first new entry's id, to place next. */
export async function importArt(input: FileList | readonly File[] | File | null | undefined): Promise<string | null> {
  const files = !input ? [] : input instanceof File ? [input] : [...input];
  if (!files.length) return null;
  useUi.getState().setNotice(null);
  try {
    const stem = (file: File) => file.name.replace(/\.[^.]+$/, "");
    const sidecars = new Map<string, unknown>();
    for (const file of files.filter(f => /\.json$/i.test(f.name))) {
      if (file.size > 1_000_000) throw new Error(`${file.name} is too large for a generation sidecar`);
      const value = parseJsonStrict(await file.text()) as { schema?: unknown } | null;
      if (typeof value?.schema === "string" && /^tatbot\.inkmap-(artwork|design)\//.test(value.schema)) {
        throw new Error(`${file.name} is DBV3 stroke artwork. Inkmap places the source image; the art pipeline makes the strokes.`);
      }
      sidecars.set(stem(file), value);
    }
    const images = files.filter(f => !/\.json$/i.test(f.name));
    if (!images.length) throw new Error("Choose a PNG or JPEG image of the design (a generation sidecar alone is not art).");
    const ids: string[] = [], notes: string[] = [];
    for (const file of images) {
      const record = await artFromFile(file, sidecars.get(stem(file)));
      const entry = await libraryEntry(`art-${record.content_sha256}`, record);
      useStore.getState().addDesign(entry);
      ids.push(entry.id);
      notes.push(...await importNotes(record));
    }
    if (notes.length) useUi.getState().setNotice({ action: "Import art", message: notes.join(" "), advice: true });
    return ids[0];
  } catch (error) { notice("Import art", error); return null; }
}
