/** The paper/cylinder editor's working document, and its portable design.
 *
 * Two things that are easy to conflate and must not be: a *draft* is what you
 * are still moving around, recovered so a reload or a view switch does not
 * throw it away; a *design* is what you accepted and exported. Editing emits a
 * new design; opening one and saving it back without touching anything must
 * hand back the identity it arrived with.
 */
import { frozenArtwork } from "./frozen-artwork.ts";
import { embeddedFromArtwork } from "./body-design.ts";
import type { ArtworkRecord } from "./artwork-record.ts";
import { makeDesign, type InkmapDesign } from "./design.ts";
import { makeSurfacePlacement, type AnalyticTarget, type SurfacePlacement } from "./surface-placement.ts";
import { fixtureDimensions, PAPER_PAD } from "./fixtures.ts";

export const MAX_CHART_ITEMS = 100;
export const degrees = (radians: number): number => Number((radians * 180 / Math.PI).toFixed(4));
export const radians = (degrees: number): number => degrees * Math.PI / 180;
/** Fixed, like the artwork conversion epoch: a design's identity is its
 *  content, not the minute it happened to be exported. */
const EXPORT_EPOCH = "1970-01-01T00:00:00Z";

export type PlacementReview = SurfacePlacement["review"];
export type PlacementProvenance = { producer: string; version: string; created_utc: string; source_sha256: string };

const UNREVIEWED: PlacementReview = { status: "pending", reviewer: "unreviewed Inkmap export", evidence_sha256: "0".repeat(64) };

/** A placement authored here, as opposed to one that arrived in a file.
 *  An imported item keeps the review and provenance it came with, so opening a
 *  design and saving it back untouched hands back the identity it arrived with
 *  — including one written by the headless `tatbot design place`. */
const authored = (record: ArtworkRecord): PlacementProvenance =>
  ({ producer: "tatbot-inkmap", version: "1", created_utc: EXPORT_EPOCH, source_sha256: record.content_sha256 });

export interface ChartItem {
  id: string;
  artwork_id: string;
  artwork: ArtworkRecord;
  size: [number, number];
  uv: [number, number];
  /** Radians, like the placement contract. Degrees exist only at the UI edge:
   *  a degree round trip is lossy, and reopening a design must not move it. */
  rotation_rad: number;
  mirror: boolean;
  /** Carried through an import so an unedited resave keeps its identity. */
  review: PlacementReview;
  provenance: PlacementProvenance;
}

export interface ChartDraft {
  name: string;
  kind: "plane" | "cylinder";
  width: number;
  height: number;
  radius: number;
  margin: number;
  items: ChartItem[];
  selected: string | null;
}

/** What a project stores: frozen bytes, not a rebuilt record. */
export interface ChartDocument {
  name: string;
  kind: "plane" | "cylinder";
  width_mm: number;
  height_mm: number;
  radius_mm: number;
  margin_mm: number;
  selected_id: string | null;
  /** Frozen bytes *and* their provenance. Looking the source back up in the
   *  runtime collection would make a recovered draft depend on that collection
   *  still being loaded, and on it still saying the same thing. */
  artwork: Record<string, ArtworkRecord>;
  items: {
    id: string; artwork_id: string; size_mm: [number, number];
    uv_mm: [number, number]; rotation_rad: number; mirror: boolean;
    review: PlacementReview; provenance: PlacementProvenance;
  }[];
}

/** A fresh draft for one surface at its fixture's measured size (core/fixtures.ts). */
export function freshChartDraft(kind: ChartDraft["kind"]): ChartDraft {
  return { name: kind === "cylinder" ? "Cylinder drawing" : "Paper drawing", kind, ...fixtureDimensions(kind), margin: 5, items: [], selected: null };
}

/** A fresh chart is the paper pad on the bench: 7.5 × 11 in. */
export const EMPTY_DRAFT: ChartDraft = freshChartDraft(PAPER_PAD.kind);

const fail = (detail: string): never => { throw new Error(`chart_draft_invalid: ${detail}`); };

export function chartTarget(draft: ChartDraft, uv: [number, number]): AnalyticTarget {
  return {
    kind: draft.kind, canvas_m: [draft.width / 1000, draft.height / 1000],
    anchor_uv_m: [uv[0] / 1000, uv[1] / 1000], margin_m: draft.margin / 1000,
    ...(draft.kind === "cylinder" ? { radius_m: draft.radius / 1000 } : {}),
  } as AnalyticTarget;
}

/** The draft as a portable design. Placement order is the drawing order. */
export async function designFromChartDraft(draft: ChartDraft): Promise<InkmapDesign> {
  if (!draft.items.length) fail("a design needs at least one placement");
  const placements = await Promise.all(draft.items.map(async item => ({
    id: item.id, artwork_id: item.artwork_id,
    placement: await makeSurfacePlacement({
      tattoo_program_sha256: item.artwork.program.content_sha256, target: chartTarget(draft, item.uv),
      physical_scale_m: [item.size[0] / 1000, item.size[1] / 1000],
      rotation_rad: item.rotation_rad, mirrored: item.mirror, warp: null,
      review: structuredClone(item.review), provenance: structuredClone(item.provenance),
    }),
  })));
  return makeDesign({
    name: draft.name,
    artworks: Object.fromEntries(draft.items.map(item => [item.artwork_id, item.artwork])),
    placements,
  });
}

/** A portable analytic design reopened as an editable draft.
 *
 * This view shows one chart, so a design that spreads placements over several
 * different charts is refused by name; the file is left untouched for download
 * rather than flattened onto whichever chart happened to be first.
 */
export function chartDraftFromDesign(design: InkmapDesign, fallbackRadius = EMPTY_DRAFT.radius): ChartDraft {
  const first = design.placements[0]?.placement.target;
  if (!first || (first.kind !== "plane" && first.kind !== "cylinder")) {
    fail("this design places artwork on a body; open it in the body editor");
  }
  const chart = first as AnalyticTarget;
  const items = design.placements.map(item => {
    const placement = item.placement;
    const target = placement.target as AnalyticTarget;
    if (target.kind !== chart.kind || JSON.stringify(target.canvas_m) !== JSON.stringify(chart.canvas_m)
      || target.margin_m !== chart.margin_m || placement.warp !== null
      || (target.kind === "cylinder" && chart.kind === "cylinder" && target.radius_m !== chart.radius_m)) {
      fail("this view requires placements on one shared plane or cylinder without a warp");
    }
    return {
      id: item.id, artwork_id: item.artwork_id, artwork: design.artworks[item.artwork_id],
      size: placement.physical_scale_m.map(value => value * 1000) as [number, number],
      uv: target.anchor_uv_m.map(value => value * 1000) as [number, number],
      rotation_rad: placement.rotation_rad, mirror: placement.mirrored,
      review: structuredClone(placement.review),
      provenance: structuredClone(placement.provenance) as unknown as PlacementProvenance,
    };
  });
  return {
    name: design.name, kind: chart.kind, width: chart.canvas_m[0] * 1000,
    height: chart.canvas_m[1] * 1000,
    radius: chart.kind === "cylinder" ? chart.radius_m * 1000 : fallbackRadius,
    margin: chart.margin_m * 1000, items, selected: null,
  };
}

/** Draft -> stored document. The authoritative artwork record travels unchanged. */
export function chartDocumentFromDraft(draft: ChartDraft): ChartDocument {
  const artwork: ChartDocument["artwork"] = {};
  for (const item of draft.items) {
    artwork[item.artwork_id] = embeddedFromArtwork(item.artwork);
  }
  return {
    name: draft.name, kind: draft.kind, width_mm: draft.width, height_mm: draft.height,
    radius_mm: draft.radius, margin_mm: draft.margin, selected_id: draft.selected, artwork,
    items: draft.items.map(item => ({
      id: item.id, artwork_id: item.artwork_id, size_mm: [...item.size] as [number, number],
      uv_mm: [...item.uv] as [number, number], rotation_rad: item.rotation_rad, mirror: item.mirror,
      review: structuredClone(item.review), provenance: structuredClone(item.provenance),
    })),
  };
}

/** Stored document -> draft, validating each frozen record without conversion. */
export async function chartDraftFromDocument(document: ChartDocument): Promise<ChartDraft> {
  if (document.items.length > MAX_CHART_ITEMS) fail(`at most ${MAX_CHART_ITEMS} placements`);
  const items = await Promise.all(document.items.map(async item => {
    const held = document.artwork[item.artwork_id];
    if (!held) fail(`missing frozen artwork ${item.artwork_id}`);
    return {
      id: item.id, artwork_id: item.artwork_id,
      artwork: await frozenArtwork(item.artwork_id, held),
      size: [...item.size_mm] as [number, number], uv: [...item.uv_mm] as [number, number],
      rotation_rad: item.rotation_rad, mirror: item.mirror,
      review: structuredClone(item.review), provenance: structuredClone(item.provenance),
    };
  }));
  const selected = items.some(item => item.id === document.selected_id) ? document.selected_id : null;
  return {
    name: document.name, kind: document.kind, width: document.width_mm, height: document.height_mm,
    radius: document.radius_mm, margin: document.margin_mm, items, selected,
  };
}


/** The rotated, mirrored artwork's footprint in chart millimetres. */
export function chartItemCorners(item: Pick<ChartItem, "uv" | "size" | "rotation_rad">): [number, number][] {
  const [w, h] = item.size;
  const c = Math.cos(item.rotation_rad), s = Math.sin(item.rotation_rad);
  return ([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]] as [number, number][])
    .map(([a, b]) => [item.uv[0] + a * c - b * s, item.uv[1] + a * s + b * c]);
}

/** Why a placement cannot sit where it is, or null. The same rule the export
 *  applies (`makeSurfacePlacement` refuses a canvas overflow), asked early so
 *  the ghost turns red and a drag stops at the edge instead of the file being
 *  refused later. */
export function chartItemRefusal(draft: Pick<ChartDraft, "kind" | "width" | "height" | "margin">,
                                 item: Pick<ChartItem, "uv" | "size" | "rotation_rad">): string | null {
  if (!item.uv.every(Number.isFinite) || !item.size.every((v) => Number.isFinite(v) && v > 0)) return "chart_placement_invalid: size and offset must be finite";
  const limitU = draft.width / 2 - draft.margin, limitV = draft.height / 2 - draft.margin;
  let overU = 0, overV = 0;
  for (const [u, v] of chartItemCorners(item)) {
    overU = Math.max(overU, Math.abs(u) - limitU);
    overV = Math.max(overV, Math.abs(v) - limitV);
  }
  if (overU <= 1e-9 && overV <= 1e-9) return null;
  const where = draft.kind === "cylinder"
    ? (overU >= overV ? "past the end of the cylinder" : "too far around the cylinder for the drawable band")
    : "off the paper pad";
  return `placement_outside_canvas: artwork reaches ${Math.max(overU, overV).toFixed(1)} mm ${where} (margin ${draft.margin} mm)`;
}

/** A new placement of one artwork on the current chart, authored here. */
export function newChartItem(id: string, artworkId: string, record: ArtworkRecord,
                            size: [number, number], uv: [number, number] = [0, 0], rotation_rad = 0): ChartItem {
  return { id, artwork_id: artworkId, artwork: record, size: [...size], uv: [...uv],
           rotation_rad, mirror: false, review: structuredClone(UNREVIEWED),
           provenance: authored(record) };
}
