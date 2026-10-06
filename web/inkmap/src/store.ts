import { requireAcquiredArtwork } from "./core/artwork-record.ts";
import { create } from "zustand";
import type { Anchor } from "./core/anchor.ts";
import { BODY_SPEC, type BodySpec, type Skin } from "./core/body.ts";
import { DEFAULT_POSE_ID, DEFAULT_SKIN_TONE } from "./core/defaults.ts";
import { newPlacementId, SCHEMA_VERSION, validatePlacementFile, type DesignMeta, type Placement, type PlacementFile } from "./core/schema.ts";
import type { AtlasIndex } from "./core/atlas.ts";
import { INKLANG_VERSION, realize, type TattooProgram } from "./core/lang.ts";
import {
  acceptResolutionCandidate,
  intentFromSite,
  realizePlacement,
  resolutionFromAnchor,
  type InkLangIntent,
  type InkLangResolution,
} from "./core/inklang/index.ts";
import { POSE_CATALOG } from "./core/pose.ts";
import { type TattooScenario } from "./core/scenario.ts";
import { buildDecal } from "./core/decal.ts";
import { makeProject, type InkmapProject, type ProjectCamera } from "./core/project.ts";
import type { InkmapDesign } from "./core/design.ts";
import { chartDocumentFromDraft, chartDraftFromDocument, chartItemRefusal, EMPTY_DRAFT, freshChartDraft, MAX_CHART_ITEMS, newChartItem, type ChartDraft, type ChartItem } from "./core/chart-draft.ts";
import { frozenArtwork } from "./core/frozen-artwork.ts";
import { artworkSizeM, type ArtworkRecord } from "./core/artwork-record.ts";
import { renderTattooProgramSvg } from "./core/human-representation/program-svg.ts";
import { artworkSources } from "./core/design-assets.ts";

export interface LoadedBody {
  spec: BodySpec;
  /** Canonical rest skin used by InkLang and every durable anchor. */
  restSkin: Skin;
  /** Posed skin used only for rendering the same rest-surface anchors. */
  skin: Skin;
  assetSha256: string;
  surfaceSha256: string;
  topologySha256: string;
  poseAssetSha256: string;
  poseId: string;
  /** Bounds before the catalog display rotation; support proxies share this frame. */
  nominalBounds: { min: [number, number, number]; max: [number, number, number] };
}

export type CameraPreset = "reset" | "front" | "back" | "left" | "right" | "selection";
export type SurfaceInteraction = "move" | "rotate" | "size";

export interface PendingTattoo {
  /** Legacy full tattoo request, when the input also named a design. */
  program: TattooProgram | null;
  intent: InkLangIntent;
  resolution: InkLangResolution;
}

export interface State {
  projectReady: boolean;
  projectName: string;
  saveStatus: "loading" | "saved" | "saving" | "failed";
  saveError: string | null;
  cameraSnapshot: ProjectCamera | null;
  cameraRevision: number;
  cameraCommand: { revision: number; preset: CameraPreset } | null;
  surfaceInteraction: SurfaceInteraction | null;
  showQuality: boolean;
  neutralLight: boolean;
  toProject: () => Promise<InkmapProject>;
  restoreProject: (project: InkmapProject) => void;
  /** An empty project in place of this one: no placements, no chart items,
   *  no history, the default name. The library, pose and skin tone stay. */
  newProject: () => void;
  /** The paper/cylinder editor's working draft, kept beside the body one so a
   *  view switch or a reload does not throw it away. */
  chart: ChartDraft;
  /** Surface and items wholesale (a fixture change, an opened design). A new
   *  item list ends any edit in progress; the placement machine below edits
   *  items through its own actions. */
  setChart: (value: Partial<ChartDraft>) => void;
  /** The chart's own choose → place → adjust → ready machine, the body's
   *  shape exactly (placing ghost, edit snapshot for Cancel, history), kept
   *  as its own fields so neither editor can reach into the other's. */
  chartPlacing: string | null;
  chartDraft: { rotation_rad: number; size: [number, number] };
  chartHover: [number, number] | null;
  chartEditBefore: ChartItem[] | null;
  chartPast: ChartItem[][];
  chartFuture: ChartItem[][];
  chartInteraction: SurfaceInteraction | null;
  /** The other surface's draft and history, parked while `chart` shows.
   *  Paper and cylinder are two drafts in one project; a tab switch parks
   *  one and shows the other, and nothing crosses between them. */
  chartParked: { draft: ChartDraft; past: ChartItem[][]; future: ChartItem[][] } | null;
  switchChartKind: (kind: ChartDraft["kind"]) => void;
  /** A draft arriving from a file takes the place of the draft of its kind. */
  openChartDraft: (draft: ChartDraft) => void;
  chartStartPlacing: (artworkId: string) => void;
  chartCancelPlacing: () => void;
  chartSetHover: (uv: [number, number] | null) => void;
  chartCommit: (uv: [number, number]) => Promise<void>;
  chartSelect: (id: string | null) => void;
  chartUpdate: (id: string, patch: Partial<Pick<ChartItem, "uv" | "size" | "rotation_rad" | "mirror">>) => void;
  chartRemove: (id: string) => void;
  chartReorder: (id: string, index: number) => void;
  chartAccept: () => void;
  chartDiscard: () => void;
  chartNudgeRotation: (rad: number) => void;
  chartNudgeSize: (factor: number) => void;
  chartUndo: () => void;
  chartRedo: () => void;
  chartSetInteraction: (interaction: SurfaceInteraction | null) => void;
  /** The accepted state before the current edit; null means no pending edit. */
  editBefore: Placement[] | null;
  past: Placement[][];
  future: Placement[][];
  undo: () => void;
  redo: () => void;
  /** Shared named pose currently baked into the body's canonical surface. */
  poseId: string;
  body: LoadedBody | null;
  /** Skin colour as #rrggbb; tints the body's (white) vertex colour. Cosmetic, not part of a placement file. */
  skinTone: string;
  designs: DesignMeta[];
  placements: Placement[];
  selected: string | null;
  /** Design id currently being dragged onto the body, if any. */
  placing: string | null;
  /** Rotation/size the ghost carries while placing; applied at commit. Reset on each pick. */
  draft: { rotation_rad: number; size_mm: [number, number] };
  hover: Anchor | null;
  /** Required region atlas for the sole body; null only while loading. */
  atlas: AtlasIndex | null;
  /** Region overlay visibility (the toggle-able atlas). */
  showAtlas: boolean;
  /** A parsed request and its canonical body resolution, waiting for choice or design. */
  pending: PendingTattoo | null;
  /** Bumps whenever `pending` changes: work started against one intent checks it before acting on another. */
  intentRevision: number;
  /** Bumps whenever the project is replaced (open, restore, scenario): the same check for the project. */
  projectEpoch: number;
  error: string | null;
  /** Short-lived message shown over the viewport (cleared by App after a moment). */
  toast: string | null;
  /** Bumps once per accepted tattoo; the picker pulses on change. */
  accepted: number;
  /** Guided-showcase payload. Its trace uses the same canonical anchors as the visible placement. */
  showcaseScenario: TattooScenario | null;
  showcaseTraceVisible: boolean;
  showcaseFocus: boolean;

  setBody: (b: LoadedBody) => void;
  setPoseId: (id: string) => void;
  setSkinTone: (hex: string) => void;
  setDesigns: (d: DesignMeta[]) => void;
  /** Add a session design (generated) to the picker; returns its id. */
  addDesign: (d: DesignMeta) => void;
  setError: (e: string | null) => void;
  setAtlas: (a: AtlasIndex | null) => void;
  toggleAtlas: () => void;
  requestCamera: (preset: CameraPreset) => void;
  setSurfaceInteraction: (interaction: SurfaceInteraction | null) => void;
  toggleQuality: () => void;
  toggleNeutralLight: () => void;
  setPending: (p: PendingTattoo | null) => void;
  choosePending: (candidateIndex: number) => void;
  /** Place a design at a concrete anchor (the sentence flow); selects it for adjustment. */
  placeAt: (designId: string, anchor: Anchor, pending: PendingTattoo | null) => void;
  startPlacing: (designId: string) => void;
  cancelPlacing: () => void;
  setHover: (a: Anchor | null) => void;
  commit: (anchor: Anchor) => void;
  select: (id: string | null) => void;
  update: (id: string, patch: Partial<Omit<Placement, "id">>) => void;
  remove: (id: string) => void;
  reorder: (id: string, index: number) => void;
  /** Keyboard nudges: rotate the ghost (while placing) or the selected placement by `rad`. */
  /** Lock in the selected tattoo (deselect) and invite the next one. */
  accept: () => void;
  /** Throw away the selected tattoo. */
  discard: () => void;
  setToast: (t: string | null) => void;
  nudgeRotation: (rad: number) => void;
  /** Keyboard nudges: scale the ghost or the selected placement by `factor`, aspect locked, width clamped to [MIN_WIDTH_MM, MAX_WIDTH_MM]. */
  nudgeSize: (factor: number) => void;
  toFile: () => PlacementFile | null;
  snapshotFile: () => PlacementFile | null;
  loadFile: (raw: unknown) => void;
  /** The accepted body placements as one portable design, and back again. */
  toDesign: () => Promise<InkmapDesign>;
  loadDesign: (design: InkmapDesign) => Promise<void>;
  loadShowcaseScenario: (raw: unknown) => Promise<void>;
  toggleShowcaseTrace: () => void;
  toggleShowcaseFocus: () => void;
}

/** The design a sentence's motif refers to, if one already exists. */
export function findDesignForMotif(designs: DesignMeta[], motif: string): DesignMeta | undefined {
  const normal = (value: string) => value.toLowerCase().replace(/[-_]+/g, " ");
  const m = normal(motif);
  return designs.find((d) => normal(d.name) === m)
    ?? designs.find((d) => m.includes(normal(d.name)) || normal(d.name).includes(m));
}

function loadShowAtlas(): boolean {
  try { return localStorage.getItem("inkmap.showAtlas") === "1"; } catch { return false; }
}

/** Attach the truthful inklang site + caption to a fresh placement. The site
 *  comes from where the anchor actually IS (the atlas), never from what was
 *  asked for; the requested program keeps only its style/motif slots. */
function annotate(p: Placement, base: PendingTattoo | null, atlas: AtlasIndex | null, design: DesignMeta | undefined): Placement {
  if (!atlas) return p;
  const d = atlas.describe(p.anchor);
  if (!d) return p;
  p.site = {
    id: d.id,
    laterality: d.laterality,
    aspect: d.aspect,
    level: d.level,
    uv: d.region_uv,
    lexicon: INKLANG_VERSION,
  };
  // A relative request ("two inches below the collarbone") IS the precise
  // truthful description — the anchor was computed from it. Everything else
  // gets named after where the anchor actually is.
  const exactRequestedAnchor = base?.resolution.status === "resolved"
    && base.resolution.anchor?.face === p.anchor.face
    && base.resolution.anchor.barycentric.every((value, index) => Math.abs(value - p.anchor.barycentric[index]) <= 1e-9);
  const requestedSite = base?.intent.site;
  const site = exactRequestedAnchor && requestedSite?.relation
    ? requestedSite
    : { id: d.id, laterality: d.laterality, aspect: d.aspect, level: d.level };
  const program: TattooProgram = base?.program
    ? { ...base.program, site }
    : { inklang: INKLANG_VERSION, motif: (design?.name ?? p.design_id).toLowerCase(), style: null, secondary: [], technique: null, color: null, site };
  try {
    const intent = exactRequestedAnchor && base
      ? base.intent
      : intentFromSite(realizePlacement(site), site);
    const resolution = exactRequestedAnchor && base
      ? base.resolution
      : resolutionFromAnchor(intent, atlas, p.anchor);
    if (resolution.status !== "resolved") throw new Error(resolution.issues[0]?.message ?? "anchor did not resolve");
    p.language = {
      sentence: realize(program),
      program: program as unknown as Record<string, unknown>,
      intent,
      resolution,
    };
  } catch (e) {
    console.warn("[inkmap] caption skipped:", (e as Error).message);
  }
  return p;
}

function loadSkinTone(): string {
  try {
    const v = localStorage.getItem("inkmap.skinTone");
    if (v && /^#[0-9a-f]{6}$/i.test(v)) return v;
  } catch { /* no storage */ }
  return DEFAULT_SKIN_TONE;
}

export const MIN_WIDTH_MM = 10;
export const MAX_WIDTH_MM = 300;

function scaled(size: [number, number], factor: number): [number, number] {
  const w = Math.min(MAX_WIDTH_MM, Math.max(MIN_WIDTH_MM, size[0] * factor));
  return [w, (w * size[1]) / size[0]];
}

const wrap = (rad: number) => rad === 0 ? 0 : Math.atan2(Math.sin(rad), Math.cos(rad));

/** Exercise the exact render geometry before admitting a placement to state.
 *  Scene rendering is intentionally not a second, weaker validation path. */
function placementGeometryRefusal(body: LoadedBody | null, placement: Placement): string | null {
  if (!body) return "body_not_ready: wait for the canonical SOMA surface to load";
  try {
    const preview = buildDecal(body.restSkin.geometry, body.skin.geometry, {
      anchor: placement.anchor,
      rotationRad: placement.rotation_rad,
      sizeMm: placement.size_mm,
    });
    preview.geometry.dispose();
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

function historyRefusal(state: State, placements: Placement[]): string | null {
  if (!state.body || !state.atlas) return "Wait for the body and atlas before restoring history.";
  for (const placement of placements) {
    if (!state.atlas.isValidAnchor(placement.anchor)) return "History contains an unsupported surface anchor.";
    const reason = placementGeometryRefusal(state.body, placement);
    if (reason) return reason;
  }
  return null;
}

export const useStore = create<State>((set, get) => ({
  projectReady: false,
  projectName: "Untitled project",
  saveStatus: "loading",
  saveError: null,
  cameraSnapshot: null,
  cameraRevision: 0,
  cameraCommand: null,
  surfaceInteraction: null,
  showQuality: false,
  neutralLight: false,
  toProject: async () => {
    const s = get();
    const file = s.snapshotFile();
    if (!file) throw new Error(s.error ?? "Wait for the body before saving a project.");
    return makeProject({ name: s.projectName, placement_file: file,
      editor: { pose_id: s.poseId, skin_tone: s.skinTone, show_atlas: s.showAtlas, camera: s.cameraSnapshot },
      history: { past: s.past, future: s.future }, edit_before: s.editBefore, selected_id: s.selected,
      chart: s.chart.items.length ? chartDocumentFromDraft(s.chart) : null,
      chart_parked: s.chartParked?.draft.items.length ? chartDocumentFromDraft(s.chartParked.draft) : null });
  },
  chart: EMPTY_DRAFT,
  setChart: (value) => set(s => {
    const chart = { ...s.chart, ...value };
    for (const item of chart.items) requireAcquiredArtwork(item.artwork);
    // A new item list is a new document: no ghost, no history — and a
    // selection that arrived with it is an edit in progress, so Cancel has
    // the accepted state to return to.
    return { chart, ...("items" in value ? { chartPlacing: null, chartHover: null, chartEditBefore: chart.selected ? chart.items : null,
      chartPast: [], chartFuture: [], chartInteraction: null } : {}) };
  }),
  chartPlacing: null,
  chartDraft: { rotation_rad: 0, size: [50, 50] },
  chartHover: null,
  chartEditBefore: null,
  chartPast: [],
  chartFuture: [],
  chartInteraction: null,
  chartParked: null,
  switchChartKind: (kind) => {
    const s = get();
    if (s.chart.kind === kind) return;
    // A pending edit is settled first: accepted if it still fits, cancelled
    // otherwise; a ghost being placed is dropped. Nothing crosses surfaces.
    if (s.chartEditBefore) { if (s.chart.selected && !chartItemRefusal(s.chart, s.chart.items.find((item) => item.id === s.chart.selected)!)) s.chartAccept(); else s.chartDiscard(); }
    const settled = get();
    const parked = settled.chartParked?.draft.kind === kind ? settled.chartParked : null;
    set({
      chartParked: { draft: { ...settled.chart, selected: null }, past: settled.chartPast, future: settled.chartFuture },
      chart: parked?.draft ?? freshChartDraft(kind), chartPast: parked?.past ?? [], chartFuture: parked?.future ?? [],
      chartPlacing: null, chartHover: null, chartEditBefore: null, chartInteraction: null,
    });
  },
  openChartDraft: (draft) => {
    const s = get();
    if (s.chart.kind !== draft.kind) {
      if (s.chartEditBefore) s.chartDiscard();
      const current = get();
      set({ chartParked: { draft: { ...current.chart, selected: null }, past: current.chartPast, future: current.chartFuture } });
    }
    get().setChart(draft);
  },
  chartSetInteraction: (chartInteraction) => set({ chartInteraction }),
  chartStartPlacing: (artworkId) => set((s) => {
    if (s.chartEditBefore) return { error: "Accept or cancel the current edit before adding another artwork." };
    const design = s.designs.find((d) => d.id === artworkId);
    if (!design) return { error: `Unknown artwork ${artworkId}` };
    return { chartPlacing: artworkId, chartHover: null, chartDraft: { rotation_rad: 0, size: [...design.default_size_mm] },
      chart: { ...s.chart, selected: null }, error: null };
  }),
  chartCancelPlacing: () => { if (get().chartEditBefore) get().chartDiscard(); else set({ chartPlacing: null, chartHover: null }); },
  chartSetHover: (chartHover) => set({ chartHover }),
  chartCommit: async (uv) => {
    const s = get();
    if (!s.chartPlacing) return;
    if (s.chart.items.length >= MAX_CHART_ITEMS) { set({ error: `Design limit: ${MAX_CHART_ITEMS} placements` }); return; }
    if (s.chartEditBefore) { set({ error: "Accept or cancel the current edit before adding another artwork." }); return; }
    const design = s.designs.find((d) => d.id === s.chartPlacing);
    if (!design?.embedded) { set({ error: "Artwork is still loading" }); return; }
    const candidate = { uv, size: s.chartDraft.size, rotation_rad: s.chartDraft.rotation_rad };
    const refusal = chartItemRefusal(s.chart, candidate);
    if (refusal) { set({ error: refusal }); return; }
    const artworkId = s.chartPlacing;
    let item: ChartItem;
    try {
      // Freezing the artwork reads its bytes; the surface may have moved on meanwhile.
      const artwork = await frozenArtwork(artworkId, design.embedded, artworkSources(s.designs)[artworkId]);
      item = newChartItem(newPlacementId(), artworkId, artwork, candidate.size, uv, candidate.rotation_rad);
    } catch (failure) { set({ error: failure instanceof Error ? failure.message : String(failure) }); return; }
    set((now) => {
      if (now.chartPlacing !== artworkId || now.chartEditBefore) return {};
      return { chart: { ...now.chart, items: [...now.chart.items, item], selected: item.id }, chartEditBefore: now.chart.items,
        chartPlacing: null, chartHover: null, error: null };
    });
  },
  chartSelect: (id) => set((s) => {
    if (id === s.chart.selected) return {};
    if (s.chartEditBefore) return { error: "Accept or cancel the current edit before selecting another artwork." };
    if (id && !s.chart.items.some((item) => item.id === id)) return {};
    return { chart: { ...s.chart, selected: id }, chartEditBefore: id ? s.chart.items : null, chartPlacing: null, chartHover: null };
  }),
  chartUpdate: (id, patch) => set((s) => {
    if (s.chart.selected !== id || s.chartEditBefore === null) return { error: "Select the artwork before editing it." };
    const current = s.chart.items.find((item) => item.id === id);
    if (!current) return {};
    const next = { ...current, ...patch };
    const refusal = chartItemRefusal(s.chart, next);
    if (refusal) return { error: refusal };
    return { chart: { ...s.chart, items: s.chart.items.map((item) => (item.id === id ? next : item)) }, error: null };
  }),
  chartRemove: (id) => set((s) => {
    if (!s.chart.items.some((item) => item.id === id)) return {};
    if (s.chartEditBefore && s.chart.selected !== id) return { error: "Accept or cancel the current edit before deleting another artwork." };
    const before = s.chartEditBefore ?? s.chart.items;
    const next = before.filter((item) => item.id !== id);
    return { chart: { ...s.chart, items: next, selected: null }, chartPast: next.length === before.length ? s.chartPast : [...s.chartPast, before].slice(-50),
      chartFuture: [], chartEditBefore: null, chartInteraction: null, error: null };
  }),
  chartReorder: (id, index) => set((s) => {
    if (s.chart.selected !== id || !s.chartEditBefore || !Number.isInteger(index) || index < 0 || index >= s.chart.items.length) return {};
    const item = s.chart.items.find((candidate) => candidate.id === id);
    if (!item) return {};
    const items = s.chart.items.filter((candidate) => candidate.id !== id); items.splice(index, 0, item);
    return { chart: { ...s.chart, items }, error: null };
  }),
  chartAccept: () => set((s) => {
    if (!s.chart.selected) return {};
    const item = s.chart.items.find((candidate) => candidate.id === s.chart.selected);
    const refusal = item ? chartItemRefusal(s.chart, item) : "Selected artwork is missing";
    if (refusal) return { error: refusal };
    const changed = s.chartEditBefore && s.chartEditBefore !== s.chart.items;
    return { chartPast: changed ? [...s.chartPast, s.chartEditBefore!].slice(-50) : s.chartPast, chartFuture: changed ? [] : s.chartFuture,
      chartEditBefore: null, chartInteraction: null, chart: { ...s.chart, selected: null }, error: null, toast: `Artwork ${s.chart.items.length} placed on the ${s.chart.kind === "cylinder" ? "cylinder" : "pad"}` };
  }),
  chartDiscard: () => set((s) => {
    if (!s.chart.selected && !s.chartPlacing) return {};
    return { chart: { ...s.chart, items: s.chartEditBefore ?? s.chart.items, selected: null }, chartEditBefore: null, chartPlacing: null, chartHover: null,
      chartInteraction: null, error: null, toast: s.chartEditBefore ? "Edit cancelled" : null };
  }),
  chartNudgeRotation: (rad) => set((s) => {
    if (s.chartPlacing) return { chartDraft: { ...s.chartDraft, rotation_rad: wrap(s.chartDraft.rotation_rad + rad) } };
    if (!s.chart.selected) return {};
    const current = s.chart.items.find((item) => item.id === s.chart.selected);
    if (!current) return {};
    const next = { ...current, rotation_rad: wrap(current.rotation_rad + rad) };
    const refusal = chartItemRefusal(s.chart, next);
    if (refusal) return { error: refusal };
    return { chart: { ...s.chart, items: s.chart.items.map((item) => (item.id === current.id ? next : item)) }, error: null };
  }),
  chartNudgeSize: (factor) => set((s) => {
    if (s.chartPlacing) return { chartDraft: { ...s.chartDraft, size: scaled(s.chartDraft.size, factor) } };
    if (!s.chart.selected) return {};
    const current = s.chart.items.find((item) => item.id === s.chart.selected);
    if (!current) return {};
    const next = { ...current, size: scaled(current.size, factor) };
    const refusal = chartItemRefusal(s.chart, next);
    if (refusal) return { error: refusal };
    return { chart: { ...s.chart, items: s.chart.items.map((item) => (item.id === current.id ? next : item)) }, error: null };
  }),
  chartUndo: () => set((s) => {
    if (s.chartEditBefore) return { chart: { ...s.chart, items: s.chartEditBefore, selected: null }, chartEditBefore: null, chartInteraction: null, chartPlacing: null, chartHover: null };
    if (!s.chartPast.length) return {};
    const previous = s.chartPast[s.chartPast.length - 1];
    return { chart: { ...s.chart, items: previous, selected: null }, chartPast: s.chartPast.slice(0, -1), chartFuture: [s.chart.items, ...s.chartFuture] };
  }),
  chartRedo: () => set((s) => {
    if (s.chartEditBefore || !s.chartFuture.length) return {};
    const [next, ...rest] = s.chartFuture;
    return { chart: { ...s.chart, items: next, selected: null }, chartPast: [...s.chartPast, s.chart.items], chartFuture: rest };
  }),
  newProject: () => set(s => ({
    projectName: "Untitled project", placements: [], selected: null, placing: null, hover: null, draft: { rotation_rad: 0, size_mm: [50, 50] },
    pending: null, intentRevision: s.intentRevision + 1, projectEpoch: s.projectEpoch + 1,
    past: [], future: [], editBefore: null, surfaceInteraction: null, error: null, toast: "New project",
    chart: EMPTY_DRAFT, chartPlacing: null, chartHover: null, chartEditBefore: null, chartPast: [], chartFuture: [], chartInteraction: null, chartParked: null,
  })),
  restoreProject: (project) => {
    for (const record of Object.values(project.placement_file.designs ?? {})) requireAcquiredArtwork(record);
    for (const chart of [project.chart, project.chart_parked]) {
      for (const record of Object.values(chart?.artwork ?? {})) requireAcquiredArtwork(record);
    }
    get().loadFile(project.placement_file);
    get().setPoseId(project.editor.pose_id);
    set(s => ({ projectName: project.name, skinTone: project.editor.skin_tone, showAtlas: project.editor.show_atlas,
      cameraSnapshot: project.editor.camera, cameraRevision: s.cameraRevision + 1,
      past: project.history.past, future: project.history.future, editBefore: project.edit_before, selected: project.selected_id,
      chart: EMPTY_DRAFT, chartPlacing: null, chartHover: null, chartEditBefore: null, chartPast: [], chartFuture: [], chartInteraction: null, chartParked: null }));
    // Validate each retained record before restoring its chart state.
    if (project.chart) {
      void chartDraftFromDocument(project.chart)
        .then(chart => get().setChart(chart))
        .catch((error: Error) => get().setError(`Saved surface draft could not be reopened: ${error.message}`));
    }
    if (project.chart_parked) {
      void chartDraftFromDocument(project.chart_parked)
        .then(draft => set({ chartParked: { draft: { ...draft, selected: null }, past: [], future: [] } }))
        .catch((error: Error) => get().setError(`Saved surface draft could not be reopened: ${error.message}`));
    }
  },
  editBefore: null,
  past: [],
  future: [],
  undo: () => set((s) => {
    if (s.editBefore) return { placements: s.editBefore, editBefore: null, selected: null, placing: null, error: null };
    if (!s.past.length) return {};
    const refusal = historyRefusal(s, s.past[s.past.length - 1]);
    if (refusal) return { error: refusal };
    return { placements: s.past[s.past.length - 1], past: s.past.slice(0, -1), future: [s.placements, ...s.future].slice(0, 50), selected: null, placing: null, error: null };
  }),
  redo: () => set((s) => {
    if (s.editBefore || !s.future.length) return {};
    const refusal = historyRefusal(s, s.future[0]);
    if (refusal) return { error: refusal };
    return { placements: s.future[0], future: s.future.slice(1), past: [...s.past, s.placements].slice(-50), selected: null, placing: null, error: null };
  }),
  poseId: DEFAULT_POSE_ID,
  body: null,
  skinTone: loadSkinTone(),
  designs: [],
  placements: [],
  selected: null,
  placing: null,
  draft: { rotation_rad: 0, size_mm: [50, 50] },
  hover: null,
  atlas: null,
  showAtlas: loadShowAtlas(),
  pending: null,
  intentRevision: 0,
  projectEpoch: 0,
  error: null,
  toast: null,
  accepted: 0,
  showcaseScenario: null,
  showcaseTraceVisible: true,
  showcaseFocus: false,

  setBody: (body) => set((s) => {
    if (body.spec.id !== BODY_SPEC.id || body.poseId !== s.poseId) return {};
    const invalid = s.placements.find((placement) => placementGeometryRefusal(body, placement));
    if (!invalid) return { body, error: null };
    return {
      body,
      // Keep recoverable intent; a pose/render failure must never erase it.
      error: `placement ${invalid.id} refused: ${placementGeometryRefusal(body, invalid)}`,
    };
  }),
  setSkinTone: (skinTone) => {
    try { localStorage.setItem("inkmap.skinTone", skinTone); } catch { /* private mode etc. */ }
    set({ skinTone });
  },
  setPoseId: (poseId) => set((s) => {
    if (poseId === s.poseId) return {};
    if (!POSE_CATALOG.pose_ids.includes(poseId)) throw new Error(`unknown pose "${poseId}"`);
    return { poseId, body: null, atlas: null, hover: null };
  }),
  setDesigns: (designs) => {
    for (const design of designs) {
      if (!design.embedded) throw new Error("Load DBV3 artwork.json before selecting artwork");
      requireAcquiredArtwork(design.embedded);
    }
    set((s) => ({ designs: [
      ...designs.filter(design => !s.designs.some(existing => existing.embedded && existing.id === design.id)),
      ...s.designs.filter(design => design.embedded),
    ] }));
  },
  addDesign: (d) => {
    if (!d.embedded) throw new Error("Import DBV3 artwork.json before adding artwork");
    requireAcquiredArtwork(d.embedded);
    set((s) => ({ designs: [...s.designs.filter((x) => x.id !== d.id), d] }));
  },
  setError: (error) => set({ error }),
  setAtlas: (atlas) => set({ atlas }),
  toggleAtlas: () => set((s) => {
    try { localStorage.setItem("inkmap.showAtlas", s.showAtlas ? "0" : "1"); } catch { /* private mode etc. */ }
    return { showAtlas: !s.showAtlas };
  }),
  requestCamera: (preset) => set((s) => ({ cameraCommand: { revision: (s.cameraCommand?.revision ?? 0) + 1, preset } })),
  setSurfaceInteraction: (surfaceInteraction) => set({ surfaceInteraction }),
  toggleQuality: () => set((s) => ({ showQuality: !s.showQuality })),
  toggleNeutralLight: () => set((s) => ({ neutralLight: !s.neutralLight })),
  setPending: (pending) => set((s) => ({ pending, intentRevision: s.intentRevision + 1 })),
  choosePending: (candidateIndex) => set((s) => {
    if (!s.pending || !s.atlas) return {};
    const resolution = acceptResolutionCandidate(s.pending.resolution, candidateIndex, s.atlas);
    return { pending: { ...s.pending, resolution }, intentRevision: s.intentRevision + 1 };
  }),
  placeAt: (designId, anchor, pending) => {
    if (get().placements.length >= 100) { set({ error: "Project limit: 100 placements." }); return; }
    if (get().editBefore) { set({ error: "Accept or cancel the current edit before adding another tattoo." }); return; }
    const { body, designs, atlas } = get();
    if (!atlas?.isValidAnchor(anchor)) {
      set({ error: "surface_region_unsupported: anchor is outside the reviewed SOMA tattoo domain" });
      return;
    }
    const d = designs.find((x) => x.id === designId);
    if (!d) return;
    const p = annotate(
      { id: newPlacementId(), design_id: d.id, anchor, rotation_rad: 0, size_mm: [...d.default_size_mm], mirror: false },
      pending, atlas, d,
    );
    const refusal = placementGeometryRefusal(body, p);
    if (refusal) {
      set({ error: refusal });
      return;
    }
    set((s) => ({ editBefore: s.placements, placements: [...s.placements, p], placing: null, pending: null, intentRevision: s.intentRevision + 1, hover: null, selected: p.id, toast: p.language ? `“${p.language.sentence}”` : null, error: null,
      cameraCommand: { revision: (s.cameraCommand?.revision ?? 0) + 1, preset: "selection" } }));
  },
  startPlacing: (placing) => set((s) => {
    if (s.editBefore) return { error: "Accept or cancel the current edit before adding another tattoo." };
    const d = s.designs.find((x) => x.id === placing);
    return { placing, selected: null, hover: null, draft: { rotation_rad: 0, size_mm: d ? [...d.default_size_mm] : s.draft.size_mm } };
  }),
  cancelPlacing: () => { if (get().editBefore) get().discard(); else set({ placing: null, hover: null }); },
  setHover: (hover) => set({ hover }),
  commit: (anchor) => {
    if (get().placements.length >= 100) { set({ error: "Project limit: 100 placements." }); return; }
    if (get().editBefore) { set({ error: "Accept or cancel the current edit before adding another tattoo." }); return; }
    const { body, placing, designs, atlas, pending } = get();
    if (!placing) return;
    if (!atlas?.isValidAnchor(anchor)) {
      set({ error: "surface_region_unsupported: anchor is outside the reviewed SOMA tattoo domain" });
      return;
    }
    const d = designs.find((x) => x.id === placing);
    if (!d) return;
    const { draft } = get();
    const p = annotate(
      { id: newPlacementId(), design_id: d.id, anchor, rotation_rad: draft.rotation_rad, size_mm: [...draft.size_mm], mirror: false },
      pending, atlas, d,
    );
    const refusal = placementGeometryRefusal(body, p);
    if (refusal) {
      set({ error: refusal });
      return;
    }
    set((s) => ({ editBefore: s.placements, placements: [...s.placements, p], placing: null, pending: null, intentRevision: s.intentRevision + 1, hover: null, selected: p.id, error: null,
      cameraCommand: { revision: (s.cameraCommand?.revision ?? 0) + 1, preset: "selection" } }));
  },
  select: (selected) => set((s) => {
    if (selected === s.selected) return {};
    if (s.editBefore) return { error: "Accept or cancel the current edit before selecting another tattoo." };
    if (selected && !s.placements.some(p => p.id === selected)) return {};
    // Selecting never moves the camera: a click that starts a drag must leave
    // the body where it is. Focus is an explicit action in the toolbar.
    return { selected, editBefore: selected ? s.placements : null, placing: null, hover: null };
  }),
  update: (id, patch) => {
    const state = get();
    if (state.selected !== id || state.editBefore === null) { set({ error: "Select the tattoo before editing it." }); return; }
    const current = state.placements.find((placement) => placement.id === id);
    if (!current) return;
    let next = { ...current, ...patch };
    if (patch.anchor) {
      if (!state.atlas?.isValidAnchor(patch.anchor)) {
        set({ error: "surface_region_unsupported: anchor is outside the reviewed SOMA tattoo domain" });
        return;
      }
      const design = state.designs.find(item => item.id === current.design_id);
      const prior = current.language?.intent && current.language.resolution ? {
        program: current.language.program as unknown as TattooProgram,
        intent: current.language.intent,
        resolution: current.language.resolution,
      } : null;
      next = annotate(next, prior, state.atlas, design);
    }
    const refusal = placementGeometryRefusal(state.body, next);
    if (refusal) {
      set({ error: refusal });
      return;
    }
    set((s) => ({ placements: s.placements.map((placement) => (placement.id === id ? next : placement)), error: null }));
  },
  remove: (id) => set((s) => {
    if (!s.placements.some(p => p.id === id)) return {};
    if (s.editBefore && s.selected !== id) return { error: "Accept or cancel the current edit before deleting another tattoo." };
    const before = s.editBefore ?? s.placements;
    const next = before.filter(p => p.id !== id);
    return { placements: next, past: next.length === before.length ? s.past : [...s.past, before].slice(-50), future: [], editBefore: null, selected: null, error: null };
  }),
  reorder: (id, index) => set(s => {
    if (s.selected !== id || !s.editBefore || !Number.isInteger(index) || index < 0 || index >= s.placements.length) return {};
    const placement = s.placements.find(p => p.id === id);
    if (!placement) return {};
    const next = s.placements.filter(p => p.id !== id); next.splice(index, 0, placement);
    return { placements: next, error: null };
  }),
  accept: () => set((s) => {
    if (!s.selected) return {};
    const n = s.placements.length;
    const p = s.placements.find((x) => x.id === s.selected);
    const where = p?.language ? ` — “${p.language.sentence}”` : " — pick another design";
    const refusal = p ? placementGeometryRefusal(s.body, p) : "Selected tattoo is missing";
    if (refusal) return { error: refusal };
    const changed = s.editBefore && JSON.stringify(s.editBefore) !== JSON.stringify(s.placements);
    return { past: changed ? [...s.past, s.editBefore!].slice(-50) : s.past, future: changed ? [] : s.future, editBefore: null, selected: null, error: null, accepted: s.accepted + 1, toast: `Tattoo ${n} saved${where}` };
  }),
  discard: () => set((s) => {
    if (!s.selected) return {};
    return { placements: s.editBefore ?? s.placements, editBefore: null, selected: null, error: null, toast: "Edit cancelled" };
  }),
  setToast: (toast) => set({ toast }),
  nudgeRotation: (rad) => set((s) => {
    if (s.placing) return { draft: { ...s.draft, rotation_rad: wrap(s.draft.rotation_rad + rad) } };
    if (!s.selected) return {};
    const current = s.placements.find((placement) => placement.id === s.selected);
    if (!current) return {};
    const next = { ...current, rotation_rad: wrap(current.rotation_rad + rad) };
    const refusal = placementGeometryRefusal(s.body, next);
    if (refusal) return { error: refusal };
    return { placements: s.placements.map((placement) => (placement.id === s.selected ? next : placement)), error: null };
  }),
  nudgeSize: (factor) => set((s) => {
    if (s.placing) return { draft: { ...s.draft, size_mm: scaled(s.draft.size_mm, factor) } };
    if (!s.selected) return {};
    const current = s.placements.find((placement) => placement.id === s.selected);
    if (!current) return {};
    const next = { ...current, size_mm: scaled(current.size_mm, factor) };
    const refusal = placementGeometryRefusal(s.body, next);
    if (refusal) return { error: refusal };
    return { placements: s.placements.map((placement) => (placement.id === s.selected ? next : placement)), error: null };
  }),
  toFile: () => {
    if (get().editBefore || get().placing) { set({ error: "Accept or cancel pending edits before exporting." }); return null; }
    const refusal = historyRefusal(get(), get().placements);
    if (refusal) { set({ error: refusal }); return null; }
    const file = get().snapshotFile();
    if (!file) return null;
    const used = new Set(file.placements.map(p => p.design_id));
    file.designs = Object.fromEntries(Object.entries(file.designs ?? {}).filter(([id]) => used.has(id)));
    return file;
  },
  snapshotFile: () => {
    const { body, placements, designs } = get();
    if (!body) return null;
    const history = [...get().past, ...get().future, ...(get().editBefore ? [get().editBefore!] : [])];
    const used = new Set([...placements, ...history.flat()].map((p) => p.design_id));
    // Retain custom artwork and every undo/redo dependency. Unused catalogue
    // entries load from the library; serializing them overwhelms a small project.
    const embedded = Object.fromEntries(designs.filter((d) => d.embedded
      && (used.has(d.id) || d.sha256 !== d.embedded.source_sha256)).map((d) => [d.id, d.embedded!]));
    const missing = [...used].filter(id => !Object.hasOwn(embedded, id));
    if (missing.length) {
      set({ error: `Artwork unavailable for export: ${missing.join(", ")}. Reload or replace these designs.` });
      return null;
    }
    return {
      schema_version: SCHEMA_VERSION,
      units: { length: "m", tattoo_size: "mm", up: "+z" },
      body: {
        model_spec_id: body.spec.id,
        model_spec_sha256: body.spec.modelSpecSha256,
        identity_sha256: body.spec.identitySha256,
        topology_sha256: body.spec.topologySha256,
        rest_surface_sha256: body.surfaceSha256,
        asset_path: body.spec.path,
        asset_sha256: body.assetSha256,
      },
      placements: structuredClone(placements),
      ...(Object.keys(embedded).length ? { designs: embedded } : {}),
    };
  },
  toDesign: async () => {
    const { toFile, atlas, designs, projectName } = get();
    const file = toFile();
    if (!file || !atlas) throw new Error(get().error ?? "Wait for the body before exporting a design.");
    const { designFromBodyFile } = await import("./core/body-design.ts");
    const { artworkSources } = await import("./core/design-assets.ts");
    return designFromBodyFile(projectName, file, atlas.atlas, artworkSources(designs));
  },
  loadDesign: async (design) => {
    const { atlas } = get();
    if (!atlas) throw new Error("body_not_ready: wait for the canonical SOMA surface before loading a design");
    const { bodyFileFromDesign } = await import("./core/body-design.ts");
    const file = await bodyFileFromDesign(design, atlas.atlas);
    get().loadFile(file);
    set({ projectName: design.name, error: null, toast: `Opened design “${design.name}”` });
  },
  loadFile: (raw) => {
    if (get().editBefore) throw new Error("Accept or cancel pending edits before importing.");
    validatePlacementFile(raw);
    for (const record of Object.values(raw.designs ?? {})) requireAcquiredArtwork(record);
    const available = new Set([...get().designs.filter(d => d.embedded).map(d => d.id), ...Object.keys(raw.designs ?? {})]);
    for (const placement of raw.placements) if (!available.has(placement.design_id)) throw new Error(`legacy_artwork: ${placement.design_id} is unavailable. Generate with DrawingBot V3 and import artwork.json, then place it again.`);
    const { body } = get();
    if (!body) throw new Error("body_not_ready: wait for the canonical SOMA surface before loading placements");
    if (
      raw.body.model_spec_id !== body.spec.id
      || raw.body.model_spec_sha256 !== body.spec.modelSpecSha256
      || raw.body.identity_sha256 !== body.spec.identitySha256
      || raw.body.topology_sha256 !== body.topologySha256
      || raw.body.rest_surface_sha256 !== body.surfaceSha256
      || raw.body.asset_path !== body.spec.path
      || raw.body.asset_sha256 !== body.assetSha256
    ) {
      throw new Error("placement file body binding differs from the loaded SOMA surface — anchors would not line up");
    }
    const { atlas } = get();
    if (atlas) {
      for (const placement of raw.placements) {
        if (!atlas.isValidAnchor(placement.anchor)) throw new Error(`placement ${placement.id} has an invalid or unlabeled anchor`);
        const resolution = placement.language?.resolution;
        if (resolution) {
          const actual = atlas.describe(placement.anchor);
          const sameBarycentric = resolution.anchor?.barycentric.every(
            (value, index) => Math.abs(value - placement.anchor.barycentric[index]) <= 1e-9,
          );
          const sameRegionUv = resolution.actual?.region_uv.every(
            (value, index) => Math.abs(value - (actual?.region_uv?.[index] ?? Number.NaN)) <= 1e-6,
          );
          if (
            resolution.status !== "resolved"
            || resolution.body.model_spec_id !== raw.body.model_spec_id
            || resolution.body.model_spec_sha256 !== raw.body.model_spec_sha256
            || resolution.body.identity_sha256 !== raw.body.identity_sha256
            || resolution.body.topology_sha256 !== raw.body.topology_sha256
            || resolution.body.rest_surface_sha256 !== raw.body.rest_surface_sha256
            || resolution.body.asset_sha256 !== raw.body.asset_sha256
            || resolution.anchor?.face !== placement.anchor.face
            || !sameBarycentric
            || !actual
            || resolution.actual?.site_id !== actual.id
            || resolution.actual?.laterality !== actual.laterality
            || resolution.actual?.aspect !== actual.aspect
            || resolution.actual?.level !== actual.level
            || !sameRegionUv
            || placement.site?.id !== actual.id
            || placement.site?.laterality !== actual.laterality
          ) throw new Error(`placement ${placement.id} has inconsistent InkLang resolution provenance`);
        }
      }
    }
    for (const placement of raw.placements) {
      const refusal = placementGeometryRefusal(body, placement);
      if (refusal) throw new Error(`placement ${placement.id} refused: ${refusal}`);
    }
    const restored: DesignMeta[] = Object.entries(raw.designs ?? {}).map(([id, e]) => {
      // An ID alone is not stock provenance. Retain the known source only
      // when the incoming artwork is identical to the loaded record.
      const previous = get().designs.find(d => d.id === id && d.embedded?.content_sha256 === e.content_sha256);
      // A frozen record owns its name as well as its geometry and provenance.
      return { id, name: e.name, path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(renderTattooProgramSvg(e.program))}`,
        default_size_mm: artworkSizeM(e).map(m => m * 1000) as [number, number], embedded: e,
        ...(previous?.source ? { source: previous.source, sha256: previous.sha256,
          usage: previous.usage, family: previous.family, split: previous.split, library: previous.library } : {}),
        ...(previous?.sourcePath ? { sourcePath: previous.sourcePath, sourceSha256: previous.sourceSha256 } : {}),
      };
    });
    set((s) => ({
      // Restoring used catalogue entries must not change the library's order.
      designs: [...s.designs.map(d => restored.find(r => r.id === d.id) ?? d),
        ...restored.filter(r => !s.designs.some(d => d.id === r.id))],
      past: [...s.past, s.placements].slice(-50), future: [], editBefore: null,
      placements: structuredClone(raw.placements), selected: null, placing: null, hover: null, error: null,
      projectEpoch: s.projectEpoch + 1,
    }));
  },
  loadShowcaseScenario: async (raw) => {
    if ((raw as { schema_version?: unknown } | null)?.schema_version !== 3) {
      throw new Error("Legacy compiled preview: regenerate with acquired DBV3 artwork.json and the current simulator compiler.");
    }
    const atlas = get().atlas?.atlas;
    if (!atlas) throw new Error("compiled preview requires the reviewed body atlas to be loaded");
    const { validateCompiledScenario } = await import("./core/sim-bundle.ts");
    const scenario: TattooScenario = await validateCompiledScenario(raw, atlas);
    if (scenario.body.model_spec_id !== BODY_SPEC.id) {
      throw new Error("showcase scenario uses an unsupported schema/model");
    }
    if (!POSE_CATALOG.pose_ids.includes(scenario.pose.id)) {
      throw new Error(`showcase scenario uses unknown pose ${scenario.pose.id}`);
    }
    const placement = scenario.placement as unknown as Placement;
    const embedded = (scenario.program_binding as { bundle: { artworks: Record<string, ArtworkRecord> } }).bundle.artworks[scenario.design.id];
    requireAcquiredArtwork(embedded);
    const design: DesignMeta = { id: scenario.design.id, name: embedded.name,
      path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(renderTattooProgramSvg(embedded.program))}`,
      default_size_mm: artworkSizeM(embedded).map(m => m * 1000) as [number, number], embedded };
    set((s) => {
      const reloadBody = s.poseId !== scenario.pose.id;
      return {
        poseId: scenario.pose.id,
        ...(scenario.schema_version === 3
          ? { skinTone: ((scenario.program_binding as { bundle: { request: { skin_tone: string } } }).bundle.request.skin_tone) }
          : {}),
        body: reloadBody ? null : s.body,
        atlas: reloadBody ? null : s.atlas,
        designs: [...s.designs.filter((d) => d.id !== design.id), design],
        placements: [placement],
        selected: null,
        placing: null,
        hover: null,
        pending: null,
        intentRevision: s.intentRevision + 1,
        projectEpoch: s.projectEpoch + 1,
        error: null,
        showcaseScenario: scenario,
        showcaseFocus: false,
      };
    });
  },
  toggleShowcaseTrace: () => set((s) => ({ showcaseTraceVisible: !s.showcaseTraceVisible })),
  toggleShowcaseFocus: () => set((s) => ({ showcaseFocus: !s.showcaseFocus })),
}));
