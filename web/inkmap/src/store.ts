import { create } from "zustand";
import type * as THREE from "three";
import type { Anchor } from "./core/anchor.ts";
import { BODIES, type BodySpec, type Skin } from "./core/body.ts";
import { DEFAULT_BODY_ID, DEFAULT_POSE_ID, DEFAULT_SKIN_TONE } from "./core/defaults.ts";
import { newPlacementId, placementAssetSha, placementSurfaceSha, SCHEMA_VERSION, validatePlacementFile, type DesignMeta, type Placement, type PlacementFile } from "./core/schema.ts";
import type { AtlasIndex } from "./core/atlas.ts";
import { INKLANG_VERSION, realize, type TattooProgram } from "./core/lang.ts";
import { POSE_CATALOG } from "./core/pose.ts";
import { validateTattooScenario, type TattooScenario } from "./core/scenario.ts";

export interface LoadedBody {
  spec: BodySpec;
  skin: Skin;
  assetSha256: string;
  surfaceSha256: string;
  scene: THREE.Object3D;
  poseId: string;
}

export interface State {
  /** Which BodySpec is (being) shown; `body` is null while it loads. */
  bodyId: string;
  /** Shared named pose currently baked into the body's canonical surface. */
  poseId: string;
  body: LoadedBody | null;
  /** Placements parked per body id while another body is shown. */
  stash: Record<string, Placement[]>;
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
  /** Region atlas for the loaded body; null while loading or if the body has none. */
  atlas: AtlasIndex | null;
  /** Region overlay visibility (the toggle-able atlas). */
  showAtlas: boolean;
  /** A parsed sentence waiting for a design — its site glows until placed or cleared. */
  pending: TattooProgram | null;
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
  /** Switch bodies: parks the current placements, restores the target's. */
  setBodyId: (id: string) => void;
  setPoseId: (id: string) => void;
  setSkinTone: (hex: string) => void;
  setDesigns: (d: DesignMeta[]) => void;
  /** Add a session design (generated) to the picker; returns its id. */
  addDesign: (d: DesignMeta) => void;
  setError: (e: string | null) => void;
  setAtlas: (a: AtlasIndex | null) => void;
  toggleAtlas: () => void;
  setPending: (p: TattooProgram | null) => void;
  /** Place a design at a concrete anchor (the sentence flow); selects it for adjustment. */
  placeAt: (designId: string, anchor: Anchor, program: TattooProgram | null) => void;
  startPlacing: (designId: string) => void;
  cancelPlacing: () => void;
  setHover: (a: Anchor | null) => void;
  commit: (anchor: Anchor) => void;
  select: (id: string | null) => void;
  update: (id: string, patch: Partial<Omit<Placement, "id">>) => void;
  remove: (id: string) => void;
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
  loadFile: (raw: unknown) => void;
  loadShowcaseScenario: (raw: unknown) => void;
  toggleShowcaseTrace: () => void;
  toggleShowcaseFocus: () => void;
}

/** The design a sentence's motif refers to, if one already exists. */
export function findDesignForMotif(designs: DesignMeta[], motif: string): DesignMeta | undefined {
  const m = motif.toLowerCase();
  return designs.find((d) => d.name.toLowerCase() === m)
    ?? designs.find((d) => m.includes(d.name.toLowerCase()) || d.name.toLowerCase().includes(m));
}

function loadShowAtlas(): boolean {
  try { return localStorage.getItem("inkmap.showAtlas") === "1"; } catch { return false; }
}

/** Attach the truthful inklang site + caption to a fresh placement. The site
 *  comes from where the anchor actually IS (the atlas), never from what was
 *  asked for; the requested program keeps only its style/motif slots. */
function annotate(p: Placement, base: TattooProgram | null, atlas: AtlasIndex | null, design: DesignMeta | undefined): Placement {
  if (!atlas) return p;
  const d = atlas.describe(p.anchor);
  if (!d) return p;
  p.site = { id: d.id, laterality: d.laterality, aspect: d.aspect, level: d.level, uv: d.uv, lexicon: INKLANG_VERSION };
  // A relative request ("two inches below the collarbone") IS the precise
  // truthful description — the anchor was computed from it. Everything else
  // gets named after where the anchor actually is.
  const site = base?.site.rel
    ? base.site
    : { id: d.id, laterality: d.laterality, aspect: d.aspect, level: d.level };
  const program: TattooProgram = base
    ? { ...base, site }
    : { inklang: INKLANG_VERSION, motif: (design?.name ?? p.design_id).toLowerCase(), style: null, secondary: [], technique: null, color: null, site };
  try {
    p.language = { sentence: realize(program), program: program as unknown as Record<string, unknown> };
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

const wrap = (rad: number) => Math.atan2(Math.sin(rad), Math.cos(rad));

export const useStore = create<State>((set, get) => ({
  bodyId: DEFAULT_BODY_ID,
  poseId: DEFAULT_POSE_ID,
  body: null,
  stash: {},
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
  error: null,
  toast: null,
  accepted: 0,
  showcaseScenario: null,
  showcaseTraceVisible: true,
  showcaseFocus: false,

  setBody: (body) => set((s) => (body.spec.id === s.bodyId && body.poseId === s.poseId ? { body } : {})),
  setSkinTone: (skinTone) => {
    try { localStorage.setItem("inkmap.skinTone", skinTone); } catch { /* private mode etc. */ }
    set({ skinTone });
  },
  setBodyId: (id) => set((s) => {
    if (id === s.bodyId) return {};
    return {
      bodyId: id,
      body: null,
      // The atlas is per body; a stale one would mislabel every face.
      atlas: null,
      stash: { ...s.stash, [s.bodyId]: s.placements },
      placements: s.stash[id] ?? [],
      // A picked design is body-agnostic: keep it, so the next click on the new body places it.
      selected: null, hover: null,
    };
  }),
  setPoseId: (poseId) => set((s) => {
    if (poseId === s.poseId) return {};
    if (!POSE_CATALOG.pose_ids.includes(poseId)) throw new Error(`unknown pose "${poseId}"`);
    return { poseId, body: null, atlas: null, selected: null, hover: null };
  }),
  setDesigns: (designs) => set((s) => ({ designs: [...designs, ...s.designs.filter((d) => d.embedded)] })),
  addDesign: (d) => set((s) => ({ designs: [...s.designs.filter((x) => x.id !== d.id), d] })),
  setError: (error) => set({ error }),
  setAtlas: (atlas) => set({ atlas }),
  toggleAtlas: () => set((s) => {
    try { localStorage.setItem("inkmap.showAtlas", s.showAtlas ? "0" : "1"); } catch { /* private mode etc. */ }
    return { showAtlas: !s.showAtlas };
  }),
  setPending: (pending) => set({ pending }),
  placeAt: (designId, anchor, program) => {
    const { designs, atlas } = get();
    const d = designs.find((x) => x.id === designId);
    if (!d) return;
    const p = annotate(
      { id: newPlacementId(), design_id: d.id, anchor, rotation_rad: 0, size_mm: [...d.default_size_mm], mirror: false },
      program, atlas, d,
    );
    set((s) => ({ placements: [...s.placements, p], placing: null, pending: null, hover: null, selected: p.id, toast: p.language ? `“${p.language.sentence}”` : null }));
  },
  startPlacing: (placing) => set((s) => {
    const d = s.designs.find((x) => x.id === placing);
    return { placing, selected: null, hover: null, draft: { rotation_rad: 0, size_mm: d ? [...d.default_size_mm] : s.draft.size_mm } };
  }),
  cancelPlacing: () => set({ placing: null, hover: null }),
  setHover: (hover) => set({ hover }),
  commit: (anchor) => {
    const { placing, designs, atlas, pending } = get();
    if (!placing) return;
    const d = designs.find((x) => x.id === placing);
    if (!d) return;
    const { draft } = get();
    const p = annotate(
      { id: newPlacementId(), design_id: d.id, anchor, rotation_rad: draft.rotation_rad, size_mm: [...draft.size_mm], mirror: false },
      pending, atlas, d,
    );
    set((s) => ({ placements: [...s.placements, p], placing: null, pending: null, hover: null, selected: p.id }));
  },
  select: (selected) => set({ selected, placing: null, hover: null }),
  update: (id, patch) => set((s) => ({ placements: s.placements.map((p) => (p.id === id ? { ...p, ...patch } : p)) })),
  remove: (id) => set((s) => ({ placements: s.placements.filter((p) => p.id !== id), selected: s.selected === id ? null : s.selected })),
  accept: () => set((s) => {
    if (!s.selected) return {};
    const n = s.placements.length;
    const p = s.placements.find((x) => x.id === s.selected);
    const where = p?.language ? ` — “${p.language.sentence}”` : " — pick another design";
    return { selected: null, accepted: s.accepted + 1, toast: `Tattoo ${n} added${where}` };
  }),
  discard: () => set((s) => {
    if (!s.selected) return {};
    return { placements: s.placements.filter((p) => p.id !== s.selected), selected: null, toast: "Discarded" };
  }),
  setToast: (toast) => set({ toast }),
  nudgeRotation: (rad) => set((s) => {
    if (s.placing) return { draft: { ...s.draft, rotation_rad: wrap(s.draft.rotation_rad + rad) } };
    if (!s.selected) return {};
    return { placements: s.placements.map((p) => (p.id === s.selected ? { ...p, rotation_rad: wrap(p.rotation_rad + rad) } : p)) };
  }),
  nudgeSize: (factor) => set((s) => {
    if (s.placing) return { draft: { ...s.draft, size_mm: scaled(s.draft.size_mm, factor) } };
    if (!s.selected) return {};
    return { placements: s.placements.map((p) => (p.id === s.selected ? { ...p, size_mm: scaled(p.size_mm, factor) } : p)) };
  }),
  toFile: () => {
    const { body, placements, designs } = get();
    if (!body) return null;
    const used = new Set(placements.map((p) => p.design_id));
    const embedded = Object.fromEntries(designs.filter((d) => d.embedded && used.has(d.id)).map((d) => [d.id, d.embedded!]));
    return {
      schema_version: SCHEMA_VERSION,
      units: { length: "m", tattoo_size: "mm", up: "+z" },
      body: {
        id: body.spec.id,
        asset_sha256: body.assetSha256,
        surface_sha256: body.surfaceSha256,
        path: body.spec.path,
      },
      placements,
      ...(Object.keys(embedded).length ? { designs: embedded } : {}),
    };
  },
  loadFile: (raw) => {
    validatePlacementFile(raw);
    const { body } = get();
    const expected = raw.schema_version >= 4 ? placementSurfaceSha(raw.body) : placementAssetSha(raw.body);
    const loaded = raw.schema_version >= 4 ? body?.surfaceSha256 : body?.assetSha256;
    if (body && expected !== loaded) {
      const hint = BODIES.some((b) => b.id === raw.body.id && b.id !== body.spec.id) ? ` — switch to the ${raw.body.id} body and load again` : " — anchors would not line up";
      throw new Error(`placement file was made on body ${raw.body.id} (${expected.slice(0, 8)}…), loaded body is ${body.spec.id} (${loaded?.slice(0, 8)}…)${hint}`);
    }
    const restored: DesignMeta[] = Object.entries(raw.designs ?? {}).map(([id, e]) => ({
      id, name: e.name, path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(e.svg)}`, default_size_mm: [...e.default_size_mm], embedded: e,
    }));
    set((s) => ({
      designs: [...s.designs.filter((d) => !restored.some((r) => r.id === d.id)), ...restored],
      placements: raw.placements, selected: null, placing: null, hover: null,
    }));
  },
  loadShowcaseScenario: (raw) => {
    validateTattooScenario(raw);
    const scenario = raw;
    if (!BODIES.some((b) => b.id === scenario.body.id)) {
      throw new Error(`showcase scenario uses unknown body ${scenario.body.id}`);
    }
    if (!POSE_CATALOG.pose_ids.includes(scenario.pose.id)) {
      throw new Error(`showcase scenario uses unknown pose ${scenario.pose.id}`);
    }
    const placement = scenario.placement as unknown as Placement;
    const embedded = {
      name: scenario.design.name,
      svg: scenario.design.svg,
      default_size_mm: [...placement.size_mm] as [number, number],
    };
    const design: DesignMeta = {
      id: scenario.design.id,
      name: scenario.design.name,
      path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(scenario.design.svg)}`,
      default_size_mm: [...placement.size_mm],
      embedded,
    };
    set((s) => {
      const reloadBody = s.bodyId !== scenario.body.id || s.poseId !== scenario.pose.id;
      return {
        bodyId: scenario.body.id,
        poseId: scenario.pose.id,
        body: reloadBody ? null : s.body,
        atlas: reloadBody ? null : s.atlas,
        designs: [...s.designs.filter((d) => d.id !== design.id), design],
        placements: [placement],
        selected: null,
        placing: null,
        hover: null,
        pending: null,
        error: null,
        showcaseScenario: scenario,
        showcaseFocus: false,
      };
    });
  },
  toggleShowcaseTrace: () => set((s) => ({ showcaseTraceVisible: !s.showcaseTraceVisible })),
  toggleShowcaseFocus: () => set((s) => ({ showcaseFocus: !s.showcaseFocus })),
}));
