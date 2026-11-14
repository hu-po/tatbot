import { create } from "zustand";
import type { Anchor } from "./core/anchor.ts";
import { BODY_SPEC, type BodySpec, type Skin } from "./core/body.ts";
import { DEFAULT_POSE_ID, DEFAULT_SKIN_TONE } from "./core/defaults.ts";
import { newPlacementId, SCHEMA_VERSION, validatePlacementFile, type DesignMeta, type EmbeddedDesign, type Placement, type PlacementFile } from "./core/schema.ts";
import type { AtlasIndex } from "./core/atlas.ts";
import { INKLANG_VERSION, realize, type TattooProgram } from "./core/lang.ts";
import {
  acceptResolutionCandidate,
  intentFromSite,
  realizePlacement,
  RESOLVER_VERSION,
  resolutionFromAnchor,
  type InkLangIntent,
  type InkLangResolution,
} from "./core/inklang/index.ts";
import { POSE_CATALOG } from "./core/pose.ts";
import { buildDecal } from "./core/decal.ts";
import { makeProject, type InkmapProject, type ProjectCamera } from "./core/project.ts";
import { artSizeMm, maxWidthMm, MIN_WIDTH_MM, validateArtShape } from "./core/art.ts";
import { withInks } from "./core/art-image.ts";
import { refreezeDesign } from "./core/design-assets.ts";
import { uprightRotation } from "./core/landmarks.ts";
import { makeBrief, type InkmapBrief } from "./core/brief.ts";
import { findRender, MAX_RENDERS, type InkmapRender } from "./core/render-record.ts";

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
  /** A width the sentence gave, mm. */
  widthMm?: number | null;
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
  /** Robot renders loaded in this browser, newest first; a placement shows the one made for it. */
  renders: InkmapRender[];
  /** The collection's own renders of designs placed so far. They come with the page, so nothing saves them. */
  stockRenders: InkmapRender[];
  /** Show a placement's robot render on the body in place of its source art. Off until someone asks: a new visitor
   *  meets the art, and turning this on, or loading a render, is remembered in this browser. */
  showRender: boolean;
  addRender: (render: InkmapRender) => void;
  addStockRender: (render: InkmapRender) => void;
  toggleRender: () => void;
  toProject: () => Promise<InkmapProject>;
  restoreProject: (project: InkmapProject) => void;
  /** An empty project in place of this one: no placements, no history, the
   *  default name. The library, pose and skin tone stay. */
  newProject: () => void;
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
  /** The guided showcase's current slide: one collection artwork placed on a named pose. */
  showcaseSlide: ShowcaseSlide | null;
  showcaseFocus: boolean;

  setBody: (b: LoadedBody) => void;
  setPoseId: (id: string) => void;
  setSkinTone: (hex: string) => void;
  setDesigns: (d: DesignMeta[]) => void;
  /** Add a session design (generated) to the picker; returns its id. */
  addDesign: (d: DesignMeta) => void;
  /** Change the inks a design's art names: what `tatbot art separate --brief` loads. */
  setArtInks: (designId: string, inks: string[]) => Promise<void>;
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
  /** Keyboard nudges: scale the ghost or the selected placement by `factor`, aspect locked, kept within the size limits. */
  nudgeSize: (factor: number) => void;
  toFile: () => PlacementFile | null;
  snapshotFile: () => PlacementFile | null;
  loadFile: (raw: unknown) => void;
  /** The accepted placements as a tattoo brief: each design's art and the size to separate it at. */
  toBrief: () => Promise<InkmapBrief>;
  loadShowcaseSlide: (slide: ShowcaseSlide) => void;
  toggleShowcaseFocus: () => void;
}

/** One slide of public/showcase/manifest.json. */
export interface ShowcaseSlide {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  artwork_id: string;
  /** The art's image sha256: the slide names the exact collection bytes it shows. */
  artwork_sha256: string;
  pose_id: string;
  placement: Placement;
}

/** The render made for this art, width and mirror: one the person loaded, else the collection's. */
export const renderFor = (s: Pick<State, "renders" | "stockRenders">, artSha: string | undefined, widthMm: number, mirror: boolean) =>
  findRender(s.renders, artSha, widthMm, mirror) ?? findRender(s.stockRenders, artSha, widthMm, mirror);

/** The design a sentence's motif refers to, if one already exists: only what the picker offers. */
export function findDesignForMotif(designs: DesignMeta[], motif: string): DesignMeta | undefined {
  const normal = (value: string) => value.toLowerCase().replace(/[-_]+/g, " ");
  const m = normal(motif);
  const offered = designs.filter((d) => d.library !== false);
  return offered.find((d) => normal(d.name) === m)
    ?? offered.find((d) => m.includes(normal(d.name)) || normal(d.name).includes(m));
}

function loadShowRender(): boolean {
  try { return localStorage.getItem("inkmap.showRender") === "1"; } catch { return false; }
}

function rememberShowRender(show: boolean): boolean {
  try { localStorage.setItem("inkmap.showRender", show ? "1" : "0"); } catch { /* private mode etc. */ }
  return show;
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

/** A uniform resize: the aspect stays, the width to 0.1 mm (the art pipeline's 10 px/mm). */
function scaled(size: [number, number], factor: number): [number, number] {
  const w = Math.round(Math.min(maxWidthMm(size), Math.max(MIN_WIDTH_MM, size[0] * factor)) * 10) / 10;
  return [w, Number(((w * size[1]) / size[0]).toFixed(3))];
}

const wrap = (rad: number) => rad === 0 ? 0 : Math.atan2(Math.sin(rad), Math.cos(rad));

/** The stored rotation that stands a design upright here. Placements store rotation from the body's up; a person
 *  sets it from upright, so a new design stands straight and a dragged one keeps its turn. */
export function uprightAt(anchor: Anchor): number {
  const { atlas, body } = useStore.getState();
  return atlas && body ? uprightRotation(atlas, body.restSkin.geometry, anchor) : 0;
}

/** A placement's rotation as a person reads it: how far it is turned from upright. */
export const turnFromUpright = (placement: Pick<Placement, "anchor" | "rotation_rad">) =>
  wrap(placement.rotation_rad - uprightAt(placement.anchor));

const priorRequest = (placement: Placement): PendingTattoo | null => placement.language?.intent && placement.language.resolution ? {
  program: placement.language.program as unknown as TattooProgram,
  intent: placement.language.intent,
  resolution: placement.language.resolution,
} : null;

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
  renders: [],
  stockRenders: [],
  showRender: loadShowRender(),
  addRender: (render) => set((s) => ({ showRender: rememberShowRender(true),
    renders: [render, ...s.renders.filter((held) => held.content_sha256 !== render.content_sha256)].slice(0, MAX_RENDERS) })),
  addStockRender: (render) => set((s) => ({ stockRenders: [...s.stockRenders, render] })),
  toggleRender: () => set((s) => ({ showRender: rememberShowRender(!s.showRender) })),
  toProject: async () => {
    const s = get();
    const file = s.snapshotFile();
    if (!file) throw new Error(s.error ?? "Wait for the body before saving a project.");
    return makeProject({ name: s.projectName, placement_file: file,
      editor: { pose_id: s.poseId, skin_tone: s.skinTone, show_atlas: s.showAtlas, camera: s.cameraSnapshot },
      history: { past: s.past, future: s.future }, edit_before: s.editBefore, selected_id: s.selected });
  },
  newProject: () => set(s => ({
    projectName: "Untitled project", placements: [], selected: null, placing: null, hover: null, draft: { rotation_rad: 0, size_mm: [50, 50] },
    pending: null, intentRevision: s.intentRevision + 1, projectEpoch: s.projectEpoch + 1,
    past: [], future: [], editBefore: null, surfaceInteraction: null, error: null, toast: "New project",
  })),
  restoreProject: (project) => {
    get().loadFile(project.placement_file);
    get().setPoseId(project.editor.pose_id);
    set(s => ({ projectName: project.name, skinTone: project.editor.skin_tone, showAtlas: project.editor.show_atlas,
      cameraSnapshot: project.editor.camera, cameraRevision: s.cameraRevision + 1,
      past: project.history.past, future: project.history.future, editBefore: project.edit_before, selected: project.selected_id }));
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
  showcaseSlide: null,
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
      if (!design.embedded) throw new Error("Import PNG or JPEG art before selecting it");
      validateArtShape(design.embedded);
    }
    // The same art restored from a project gives way to the collection's entry, which knows its renders.
    const same = (a: DesignMeta, b: DesignMeta) => a.id === b.id && a.embedded?.content_sha256 === b.embedded?.content_sha256;
    set((s) => ({ designs: [
      ...designs.filter(design => !s.designs.some(existing => existing.embedded && existing.id === design.id && !same(existing, design))),
      ...s.designs.filter(design => design.embedded && !designs.some(stock => same(stock, design))),
    ] }));
  },
  addDesign: (d) => {
    if (!d.embedded) throw new Error("Import PNG or JPEG art before adding it");
    validateArtShape(d.embedded);
    set((s) => ({ designs: [...s.designs.filter((x) => x.id !== d.id), d] }));
  },
  setArtInks: async (designId, inks) => {
    const record = get().designs.find((d) => d.id === designId)?.embedded;
    if (!record) return;
    const embedded = await withInks(record, inks);
    set((s) => ({ designs: s.designs.map((d) => (d.id === designId ? { ...d, embedded } : d)) }));
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
    if (get().editBefore) { set({ error: "Accept or cancel the current edit before adding another tattoo." }); return; }
    const { body, designs, atlas } = get();
    if (!atlas?.isValidAnchor(anchor)) {
      set({ error: "surface_region_unsupported: anchor is outside the reviewed SOMA tattoo domain" });
      return;
    }
    const d = designs.find((x) => x.id === designId);
    if (!d) return;
    if (get().placements.length >= 100) { set({ error: "Project limit: 100 placements." }); return; }
    const width = pending?.widthMm && d.embedded ? Math.min(maxWidthMm(d.default_size_mm), Math.max(MIN_WIDTH_MM, pending.widthMm)) : null;
    const size = width && d.embedded ? artSizeMm(d.embedded, width) : d.default_size_mm;
    const p = annotate({ id: newPlacementId(), design_id: d.id, anchor, rotation_rad: uprightAt(anchor), size_mm: [...size], mirror: false }, pending, atlas, d);
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
    if (get().placements.length >= 100) { set({ error: "Project limit: 100 placements." }); return; }
    const p = annotate({ id: newPlacementId(), design_id: d.id, anchor, rotation_rad: wrap(draft.rotation_rad + uprightAt(anchor)), size_mm: [...draft.size_mm], mirror: false }, pending, atlas, d);
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
    if (patch.anchor && !state.atlas?.isValidAnchor(patch.anchor)) {
      set({ error: "surface_region_unsupported: anchor is outside the reviewed SOMA tattoo domain" });
      return;
    }
    let next = { ...current, ...structuredClone(patch) };
    if (patch.anchor) {
      const turned = uprightAt(patch.anchor) - uprightAt(current.anchor);
      if (patch.rotation_rad === undefined && turned !== 0) next.rotation_rad = wrap(current.rotation_rad + turned);
      next = annotate(next, priorRequest(current), state.atlas, state.designs.find(item => item.id === current.design_id));
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
    return { placements: s.placements.map((placement) => (placement.id === current.id ? next : placement)), error: null };
  }),
  nudgeSize: (factor) => set((s) => {
    if (s.placing) return { draft: { ...s.draft, size_mm: scaled(s.draft.size_mm, factor) } };
    if (!s.selected) return {};
    const current = s.placements.find((placement) => placement.id === s.selected);
    if (!current) return {};
    const next = { ...current, size_mm: scaled(current.size_mm, factor) };
    const refusal = placementGeometryRefusal(s.body, next);
    if (refusal) return { error: refusal };
    return { placements: s.placements.map((placement) => (placement.id === current.id ? next : placement)), error: null };
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
    // Retain imported art and every undo/redo dependency. Unused collection
    // entries load from the collection; carrying them would bloat every save.
    const embedded = Object.fromEntries(designs.filter((d) => d.embedded
      && (used.has(d.id) || !d.sourcePath)).map((d) => [d.id, d.embedded!]));
    const missing = [...used].filter(id => !Object.hasOwn(embedded, id));
    if (missing.length) {
      set({ error: `Art unavailable for export: ${missing.join(", ")}. Reload or replace these designs.` });
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
  toBrief: async () => {
    const { toFile, projectName } = get();
    const file = toFile();
    if (!file) throw new Error(get().error ?? "Wait for the body before exporting a brief.");
    if (!file.placements.length) throw new Error("Place a design before exporting a brief.");
    return makeBrief(projectName, file);
  },
  loadFile: (raw) => {
    if (get().editBefore) throw new Error("Accept or cancel pending edits before importing.");
    validatePlacementFile(raw);
    const available = new Set([...get().designs.filter(d => d.embedded).map(d => d.id), ...Object.keys(raw.designs ?? {})]);
    for (const placement of raw.placements) if (!available.has(placement.design_id)) throw new Error(`art_unavailable: ${placement.design_id} is not in this file or the collection. Import its art, then place it again.`);
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
    // What an older lexicon or resolver called a place is out of date, not false: the anchor is the record, so its
    // words are made again below, and only a current resolution is held to the atlas.
    const outdated = (placement: Placement) => Boolean(placement.site && placement.site.lexicon !== INKLANG_VERSION)
      || Boolean(placement.language?.resolution && placement.language.resolution.resolver.version !== RESOLVER_VERSION);
    if (atlas) {
      for (const placement of raw.placements) {
        if (!atlas.isValidAnchor(placement.anchor)) throw new Error(`placement ${placement.id} has an invalid or unlabeled anchor`);
        const resolution = placement.language?.resolution;
        if (resolution && !outdated(placement)) {
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
    // An ID alone is not collection provenance: an entry identical to one already held stays that entry;
    // the rest show their image at once, and their ink-only thumbnail once it is decoded.
    const held = (id: string, e: EmbeddedDesign) => get().designs.find(d => d.id === id && d.embedded?.content_sha256 === e.content_sha256);
    const fromFile: DesignMeta[] = Object.entries(raw.designs ?? {}).filter(([id, e]) => !held(id, e))
      .map(([id, e]) => ({ id, name: e.name, path: `data:${e.image.media_type};base64,${e.image.data_base64}`, default_size_mm: artSizeMm(e), embedded: e }));
    const restored: DesignMeta[] = Object.entries(raw.designs ?? {}).map(([id, e]) => held(id, e) ?? fromFile.find(d => d.id === id)!);
    void Promise.all(fromFile.map(refreezeDesign))
      .then(fresh => set(s => ({ designs: s.designs.map(d => fresh.find(f => f.id === d.id && f.embedded?.content_sha256 === d.embedded?.content_sha256) ?? d) })))
      .catch(error => console.warn("[inkmap] thumbnail:", error));
    set((s) => ({
      // Restoring used collection entries must not change the library's order.
      designs: [...s.designs.map(d => restored.find(r => r.id === d.id) ?? d),
        ...restored.filter(r => !s.designs.some(d => d.id === r.id))],
      past: [...s.past, s.placements].slice(-50), future: [], editBefore: null,
      placements: structuredClone(raw.placements).map((placement) => {
        if (!outdated(placement)) return placement;
        const { site: _site, language: _language, ...kept } = placement;
        return annotate(kept, null, atlas, restored.find((d) => d.id === placement.design_id));
      }),
      selected: null, placing: null, hover: null, error: null,
      projectEpoch: s.projectEpoch + 1,
    }));
  },
  loadShowcaseSlide: (slide) => {
    if (!POSE_CATALOG.pose_ids.includes(slide.pose_id)) throw new Error(`showcase slide uses unknown pose ${slide.pose_id}`);
    if (slide.placement.design_id !== slide.artwork_id) throw new Error(`showcase slide ${slide.id} places another artwork`);
    const design = get().designs.find((d) => d.id === slide.artwork_id);
    if (!design?.embedded) throw new Error(`showcase artwork ${slide.artwork_id} is not in the collection`);
    if (design.embedded.image.sha256 !== slide.artwork_sha256) {
      throw new Error(`showcase artwork ${slide.artwork_id} differs from the bytes the slide names`);
    }
    const placements = [structuredClone(slide.placement)];
    set((s) => {
      const reloadBody = s.poseId !== slide.pose_id;
      return {
        poseId: slide.pose_id,
        body: reloadBody ? null : s.body,
        atlas: reloadBody ? null : s.atlas,
        placements,
        selected: null,
        placing: null,
        hover: null,
        pending: null,
        intentRevision: s.intentRevision + 1,
        projectEpoch: s.projectEpoch + 1,
        error: null,
        showcaseSlide: slide,
        showcaseFocus: false,
      };
    });
  },
  toggleShowcaseFocus: () => set((s) => ({ showcaseFocus: !s.showcaseFocus })),
}));
