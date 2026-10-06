/** The paper pad and paper cylinder in 3D, at their measured size.
 *
 * One chart frame serves both fixtures (core/surface-placement's
 * `analyticFrame`): u across the pad or along the cylinder's axis, v up the
 * pad or as arc length around the cylinder from its crest, the crest itself
 * at the origin. The fixture is drawn in that frame, the artwork is bent onto
 * it through the same function the exported placement is checked with, and
 * pointer rays are intersected with the same analytic surface — so what the
 * eye sees sit on the paper is what the design file says.
 *
 * Interaction is the body editor's, on a different surface: while placing,
 * a ghost follows the pointer and a click commits it; a placed artwork is
 * selected by a click, dragged to move, and carries the ↻ ↔ handles.
 *
 * This is a preview of nominal geometry. Nothing here measures the bench or
 * grants motion authority.
 */
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js";
import { useStore } from "../store.ts";
import { chartItemCorners, chartItemRefusal, chartTarget, type ChartDraft, type ChartItem } from "../core/chart-draft.ts";
import { analyticFrame, type AnalyticTarget } from "../core/surface-placement.ts";
import { fixtureFor } from "../core/fixtures.ts";
import { artworkTexture, type PreviewArtwork } from "../core/svg.ts";

const M = 1 / 1000;
/** Artwork sits a hair above the paper so it never z-fights with the grid. */
const LIFT_M = 0.00025;
const GRID_PX_PER_MM = 4;
const BACKGROUND = "#1b1d22";
// One identity for the life of the module: react-three-fiber replaces the
// camera whenever this prop stops comparing equal, and a fresh literal on
// every render would throw away the framing and the controls' camera.
const CAMERA = { position: [0, -0.2, 0.3] as [number, number, number], up: [0, 0, 1] as [number, number, number], fov: 34, near: 0.002, far: 20 };
const GL = { antialias: true, powerPreference: "high-performance" as const };
const DPR: [number, number] = [1, 1.5];

type Vec3 = [number, number, number];
type Footprint = Pick<ChartItem, "uv" | "size" | "rotation_rad" | "mirror">;
type Grab = React.MutableRefObject<[number, number]>;

/** Chart (u, v) in millimetres to a point and normal in the scene, metres. */
function surfacePoint(target: AnalyticTarget, uvMm: [number, number], lift = 0): THREE.Vector3 {
  const { point, normal } = analyticFrame(target, [uvMm[0] * M, uvMm[1] * M]);
  return new THREE.Vector3(...point).addScaledVector(new THREE.Vector3(...normal), lift);
}

/** Where a pointer ray meets the fixture's analytic surface, in chart mm; null when it misses. */
export function rayToChart(ray: THREE.Ray, draft: Pick<ChartDraft, "kind" | "radius">): [number, number] | null {
  const o = ray.origin, d = ray.direction;
  if (draft.kind === "plane") {
    if (Math.abs(d.z) < 1e-9) return null;
    const t = -o.z / d.z;
    if (t <= 0) return null;
    return [(o.x + t * d.x) / M, (o.y + t * d.y) / M];
  }
  // A cylinder of radius r with its axis along x at z = -r: |(y, z + r)| = r.
  const r = draft.radius * M;
  const oy = o.y, oz = o.z + r;
  const a = d.y * d.y + d.z * d.z;
  const b = 2 * (oy * d.y + oz * d.z);
  const c = oy * oy + oz * oz - r * r;
  const disc = b * b - 4 * a * c;
  if (a < 1e-12 || disc < 0) return null;
  const t = (-b - Math.sqrt(disc)) / (2 * a);
  if (t <= 0) return null;
  const py = oy + t * d.y, pz = oz + t * d.z;
  const theta = Math.atan2(py, pz);
  return [(o.x + t * d.x) / M, (r * theta) / M];
}

/** One decimal of a millimetre, never a negative zero: the canonical JSON
 *  the design is digested from refuses -0, and (-0.02).toFixed(1) is one. */
const tenth = (value: number) => { const rounded = Number(value.toFixed(1)); return rounded === 0 ? 0 : rounded; };

/** White paper with a faint blue square grid, drawn at exact pitch and
 *  centred so one rule crosses the middle of the sheet. `spanU` runs along
 *  the texture's columns, `spanV` along its rows.
 *
 *  The pixels are copied out into a data texture and the canvas released at
 *  once: a canvas kept alive as a texture image counts against the browser's
 *  canvas memory budget, and past it Chrome silently blanks canvases to
 *  transparent black — which is how a white sheet turned black on the bench. */
function gridTexture(spanUMm: number, spanVMm: number, pitchMm: number, paper: string, rule: string): THREE.DataTexture {
  const w = Math.max(2, Math.round(spanUMm * GRID_PX_PER_MM));
  const h = Math.max(2, Math.round(spanVMm * GRID_PX_PER_MM));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = paper; ctx.fillRect(0, 0, w, h);
  // Faint comes from the colour, not from transparency: a half-millimetre rule at
  // full alpha still reads after mipmapping where a thin translucent one vanishes.
  ctx.strokeStyle = rule; ctx.lineWidth = Math.max(2, 0.5 * GRID_PX_PER_MM); ctx.globalAlpha = 1;
  const rules = (span: number, px: number) => {
    const out: number[] = [];
    const centre = px / 2;
    for (let k = -Math.ceil(span / pitchMm / 2) - 1; k <= Math.ceil(span / pitchMm / 2) + 1; k++) {
      const at = centre + k * pitchMm * GRID_PX_PER_MM;
      if (at >= -1 && at <= px + 1) out.push(at);
    }
    return out;
  };
  ctx.beginPath();
  for (const x of rules(spanUMm, w)) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (const y of rules(spanVMm, h)) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();
  const pixels = ctx.getImageData(0, 0, w, h);
  canvas.width = canvas.height = 0;
  if (pixels.data[3] === 0) throw new Error("grid canvas came back blank");
  const texture = new THREE.DataTexture(pixels.data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.flipY = true;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return texture;
}

/** The grid as a texture owned by an effect, not a memo: React may run an
 *  effect's cleanup and re-run it against the same render (StrictMode does),
 *  and a memoised texture disposed that way would come back black. Until it
 *  exists the paper is plain white. */
function useGridTexture(spanUMm: number, spanVMm: number, kind: ChartDraft["kind"]): THREE.DataTexture | null {
  const fixture = fixtureFor(kind);
  const [texture, setTexture] = useState<THREE.DataTexture | null>(null);
  useEffect(() => {
    const made = gridTexture(spanUMm, spanVMm, fixture.grid_pitch_mm, fixture.paper, fixture.rule);
    setTexture(made);
    return () => { made.dispose(); setTexture((current) => current === made ? null : current); };
  }, [spanUMm, spanVMm, fixture]);
  return texture;
}

type PointerHandlers = {
  onPointerMove: (e: ThreeEvent<PointerEvent>) => void;
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void;
  onPointerUp: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut: () => void;
};

/** The pad: a box whose top face carries the grid. */
function Pad({ draft, pointer }: { draft: ChartDraft; pointer: PointerHandlers }) {
  const fixture = fixtureFor("plane");
  const t = fixture.thickness_mm * M;
  const grid = useGridTexture(draft.width, draft.height, "plane");
  const sides = { color: "#f1efe9", roughness: 0.95 };
  return (
    <mesh position={[0, 0, -t / 2]} {...pointer}>
      <boxGeometry args={[draft.width * M, draft.height * M, t]} />
      <meshStandardMaterial attach="material-0" {...sides} />
      <meshStandardMaterial attach="material-1" {...sides} />
      <meshStandardMaterial attach="material-2" {...sides} />
      <meshStandardMaterial attach="material-3" {...sides} />
      {/* The paper itself is unlit and untonemapped: white paper reads as
          white, the rules as the faint blue they are printed, whatever the
          lights do — and if the texture ever fails to upload it stays white.
          The key remounts the material when the grid arrives: a program
          compiled without a map is not recompiled by assigning one later. */}
      <meshBasicMaterial key={grid ? "grid" : "blank"} attach="material-4" map={grid ?? undefined} color={fixture.paper} toneMapped={false} />
      <meshStandardMaterial attach="material-5" color="#d9d6cf" roughness={1} />
    </mesh>
  );
}

/** The cylinder: gridded all the way round, its axis along u at z = −r. */
function Tube({ draft, pointer }: { draft: ChartDraft; pointer: PointerHandlers }) {
  const fixture = fixtureFor("cylinder");
  const r = draft.radius * M;
  const length = draft.width * M;
  const grid = useGridTexture(2 * Math.PI * draft.radius, draft.width, "cylinder");
  return (
    // CylinderGeometry stands along +y with θ = 0 at +z; turning it −90° about z
    // lays the axis along +x and leaves θ = 0 — the texture's first column — on
    // the crest, so a rule runs straight along the top.
    <mesh position={[0, 0, -r]} rotation={[0, 0, -Math.PI / 2]} {...pointer}>
      <cylinderGeometry args={[r, r, length, 128, 1, false]} />
      <meshBasicMaterial key={grid ? "grid" : "blank"} attach="material-0" map={grid ?? undefined} color={fixture.paper} toneMapped={false} />
      <meshStandardMaterial attach="material-1" color="#f1efe9" roughness={0.95} />
      <meshStandardMaterial attach="material-2" color="#f1efe9" roughness={0.95} />
    </mesh>
  );
}

/** A polygon in chart mm, followed along the surface. */
function surfaceLoop(target: AnalyticTarget, corners: [number, number][], lift: number, steps = 24): Vec3[] {
  const points: Vec3[] = [];
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i], b = corners[(i + 1) % corners.length];
    for (let s = 0; s < steps; s++) {
      const f = s / steps;
      points.push(surfacePoint(target, [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f], lift).toArray() as Vec3);
    }
  }
  points.push(points[0]);
  return points;
}

const rect = (w: number, h: number): [number, number][] => [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]];

/** The chart's extent and its margin: where placements may go. */
function ChartBounds({ draft }: { draft: ChartDraft }) {
  const target = chartTarget(draft, [0, 0]);
  const outer = useMemo(() => surfaceLoop(target, rect(draft.width, draft.height), LIFT_M * 0.5), [draft.kind, draft.width, draft.height, draft.radius]);
  const inner = useMemo(() => surfaceLoop(target, rect(Math.max(0, draft.width - 2 * draft.margin), Math.max(0, draft.height - 2 * draft.margin)), LIFT_M * 0.5), [draft.kind, draft.width, draft.height, draft.radius, draft.margin]);
  return (
    <>
      {draft.kind === "cylinder" && <Line points={outer} color="#6f7686" lineWidth={1.2} />}
      {draft.margin > 0 && <Line points={inner} color="#8a90a0" lineWidth={1} dashed dashSize={0.004} gapSize={0.003} />}
    </>
  );
}

function itemToChart(item: Footprint, a: number, b: number): [number, number] {
  const x = item.mirror ? -a : a;
  const c = Math.cos(item.rotation_rad), s = Math.sin(item.rotation_rad);
  return [item.uv[0] + x * c - b * s, item.uv[1] + x * s + b * c];
}

/** The artwork bent onto the surface: a grid of the artwork's own plane,
 *  every vertex sent through the chart frame. */
function artworkGeometry(item: Footprint, target: AnalyticTarget, segments: number): THREE.BufferGeometry {
  const n = segments + 1;
  const positions = new Float32Array(n * n * 3);
  const normals = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);
  const [w, h] = item.size;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const fa = i / segments, fb = j / segments;
      const a = (fa - 0.5) * w, b = (fb - 0.5) * h;
      const { point, normal } = analyticFrame(target, itemToChart(item, a, b).map((v) => v * M) as [number, number]);
      const k = j * n + i;
      positions.set([point[0] + normal[0] * LIFT_M, point[1] + normal[1] * LIFT_M, point[2] + normal[2] * LIFT_M], k * 3);
      normals.set(normal, k * 3);
      uvs.set([fa, fb], k * 2);
    }
  }
  const index: number[] = [];
  for (let j = 0; j < segments; j++) {
    for (let i = 0; i < segments; i++) {
      const k = j * n + i;
      index.push(k, k + 1, k + n, k + 1, k + n + 1, k + n);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(index);
  return geometry;
}

/** One artwork on the surface — a placed item, or the ghost under the pointer. */
function Artwork({ item, id, source, draft, selected = false, ghost = false, invalid = false, grab }: {
  item: Footprint; id: string | null; source: PreviewArtwork; draft: ChartDraft; selected?: boolean; ghost?: boolean; invalid?: boolean; grab: Grab;
}) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const target = chartTarget(draft, [0, 0]);
  const segments = draft.kind === "cylinder" ? 24 : 1;
  const geometry = useMemo(() => artworkGeometry(item, target, segments),
    [item.size[0], item.size[1], item.uv[0], item.uv[1], item.rotation_rad, item.mirror, draft.kind, draft.radius, segments]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => {
    let live = true;
    artworkTexture(source, item.size).then((loaded) => {
      if (!live) { loaded.dispose(); return; }
      setTexture(loaded);
    }).catch(console.error);
    return () => { live = false; };
  }, [source, item.size[0], item.size[1]]);
  useEffect(() => () => texture?.dispose(), [texture]);
  const outline = useMemo(() => surfaceLoop(target, chartItemCorners(item), LIFT_M * 1.5, draft.kind === "cylinder" ? 16 : 1),
    [item.size[0], item.size[1], item.uv[0], item.uv[1], item.rotation_rad, draft.kind, draft.radius]);

  const onPointerDown = ghost || !id ? undefined : (e: ThreeEvent<PointerEvent>) => {
    const state = useStore.getState();
    // While placing, the paper under the pointer takes the click: a new
    // artwork may land over an old one without selecting it instead.
    if (state.chartPlacing) return;
    if (state.chart.selected !== id) state.chartSelect(id);
    if (useStore.getState().chart.selected === id) {
      const hit = rayToChart(e.ray, state.chart);
      grab.current = hit ? [hit[0] - item.uv[0], hit[1] - item.uv[1]] : [0, 0];
      useStore.getState().chartSetInteraction("move");
    }
  };
  return (
    <group>
      {ghost && <mesh geometry={geometry} renderOrder={1}>
        <meshBasicMaterial color={invalid ? "#ff2020" : "#5b7cff"} transparent opacity={invalid ? 0.45 : 0.22} side={THREE.DoubleSide} depthWrite={false} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} toneMapped={false} />
      </mesh>}
      <mesh geometry={geometry} renderOrder={2} onPointerDown={onPointerDown}
        onPointerUp={ghost ? undefined : () => useStore.getState().chartSetInteraction(null)}
        onClick={ghost ? undefined : (e) => { if (useStore.getState().chartPlacing) return; e.stopPropagation(); }}
        onPointerOver={ghost ? undefined : () => { document.body.style.cursor = "grab"; }} onPointerOut={ghost ? undefined : () => { document.body.style.cursor = ""; }}>
        <meshBasicMaterial key={texture ? "art" : "blank"} map={texture ?? undefined} color="#ffffff" transparent opacity={texture ? (ghost ? 0.65 : 1) : 0}
          side={THREE.DoubleSide} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} toneMapped={false} />
      </mesh>
      {selected && <Line points={outline} color="#5b7cff" lineWidth={2} />}
    </group>
  );
}

/** The ↻ ↔ handles and the width ruler beside the selected artwork, the
 *  body's SelectionControls on the chart's frame. */
function ChartSelectionControls() {
  const draft = useStore((s) => s.chart);
  const item = useStore((s) => s.chart.items.find((candidate) => candidate.id === s.chart.selected));
  const interaction = useStore((s) => s.chartInteraction);
  if (!item) return null;
  const target = chartTarget(draft, [0, 0]);
  const position = surfacePoint(target, itemToChart({ ...item, mirror: false }, item.size[0] / 2 + 25, 0), 0.008);
  const drag = (kind: "rotate" | "size") => ({
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      useStore.getState().chartSetInteraction(kind);
    },
    onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (useStore.getState().chartInteraction !== kind || event.movementX === 0) return;
      const current = useStore.getState().chart.items.find((candidate) => candidate.id === item.id);
      if (!current) return;
      if (kind === "rotate") useStore.getState().chartUpdate(current.id, { rotation_rad: current.rotation_rad + event.movementX * 0.012 });
      else useStore.getState().chartNudgeSize(Math.exp(event.movementX * 0.012));
    },
    onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      event.currentTarget.releasePointerCapture(event.pointerId);
      useStore.getState().chartSetInteraction(null);
    },
    onPointerCancel: () => useStore.getState().chartSetInteraction(null),
  });
  return (
    <Html center position={position} zIndexRange={[12, 0]}>
      <div className="surface-handles" onPointerDown={(event) => event.stopPropagation()}>
        <span className="ruler" aria-label={`Artwork width ${item.size[0].toFixed(0)} millimeters`}>{item.size[0].toFixed(0)} mm</span>
        <button type="button" className={interaction === "rotate" ? "active" : ""} aria-label="Drag to rotate artwork; A and D keys also rotate" title="Drag horizontally to rotate" {...drag("rotate")}>↻</button>
        <button type="button" className={interaction === "size" ? "active" : ""} aria-label="Drag to resize artwork; W and S keys also resize" title="Drag horizontally to resize" {...drag("size")}>↔</button>
      </div>
    </Html>
  );
}

/** Frames the fixture when its kind changes: a tilted top view of the pad,
 *  a three-quarter view of the cylinder, backed off until the whole fixture
 *  fits the viewport at its current aspect. Orbit, zoom and pan stay free after. */
function ChartCamera({ draft }: { draft: ChartDraft }) {
  const { camera, controls, invalidate, size, scene, gl } = useThree();
  // The browser suite and a debugger get at the live scene the same way they
  // get at the stores (`window.__inkmap`).
  useEffect(() => { (window as unknown as { __inkmapChart?: unknown }).__inkmapChart = { scene, camera, gl }; }, [scene, camera, gl]);
  const fitKey = `${draft.kind}`;
  useEffect(() => {
    const orbit = controls as OrbitControlsImpl | null;
    const perspective = camera as THREE.PerspectiveCamera;
    const w = draft.width * M, h = draft.height * M;
    const target = draft.kind === "plane" ? new THREE.Vector3(0, 0, 0) : new THREE.Vector3(0, 0, -draft.radius * M);
    const vertical = THREE.MathUtils.degToRad(perspective.fov ?? 34) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * Math.max(size.width, 1) / Math.max(size.height, 1));
    // What has to fit on screen: the pad seen from its tilted top view spans
    // its width across and most of its height up; the tube from three-quarters
    // spans most of its length across and its diameter up.
    const across = draft.kind === "plane" ? w / 2 : 0.8 * (w / 2) + draft.radius * M;
    const up = draft.kind === "plane" ? 0.85 * (h / 2) : 0.5 * (w / 2) + 1.5 * draft.radius * M;
    const distance = Math.max(across / Math.tan(horizontal), up / Math.tan(vertical)) * 1.12;
    const direction = draft.kind === "plane" ? new THREE.Vector3(0, -0.55, 0.85) : new THREE.Vector3(0.55, -0.7, 0.5);
    camera.position.copy(target).addScaledVector(direction.normalize(), distance);
    camera.up.set(0, 0, 1);
    orbit?.target.copy(target);
    camera.lookAt(target);
    orbit?.update();
    // Pointer rays are cast from the camera's world matrix, which the
    // renderer refreshes only when it draws; a click that lands before the
    // next frame must not be cast from where the camera was.
    camera.updateMatrixWorld(true);
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, camera, controls, invalidate]);
  return null;
}

/** Pointer handling on the paper itself: the ghost follows the pointer while
 *  placing and a click commits it; a selected artwork follows a drag; a
 *  click on bare paper clears the selection. An orbit drag is told apart
 *  from a click the way the body does it: by distance and time. */
function useFixturePointer(grab: Grab): PointerHandlers {
  const press = useRef<{ x: number; y: number; t: number } | null>(null);
  return {
    onPointerMove: (e) => {
      const state = useStore.getState();
      const hit = rayToChart(e.ray, state.chart);
      if (!hit) return;
      if (state.chartInteraction === "move" && state.chart.selected) {
        e.stopPropagation();
        const current = state.chart.items.find((item) => item.id === state.chart.selected);
        if (!current) return;
        const uv: [number, number] = [tenth(hit[0] - grab.current[0]), tenth(hit[1] - grab.current[1])];
        // A drag stops at the edge of the drawable area rather than shouting on every frame.
        if (!chartItemRefusal(state.chart, { ...current, uv })) state.chartUpdate(current.id, { uv });
        return;
      }
      if (!state.chartPlacing) return;
      e.stopPropagation();
      state.chartSetHover([tenth(hit[0]), tenth(hit[1])]);
    },
    onPointerDown: (e) => { press.current = { x: e.clientX, y: e.clientY, t: performance.now() }; },
    onPointerUp: (e) => {
      const state = useStore.getState();
      if (state.chartInteraction) { state.chartSetInteraction(null); press.current = null; e.stopPropagation(); return; }
      const p = press.current;
      press.current = null;
      if (!p) return;
      if (Math.hypot(e.clientX - p.x, e.clientY - p.y) > 8 || performance.now() - p.t > 800) return; // that was an orbit drag, not a click
      const hit = rayToChart(e.ray, state.chart);
      if (!hit) return;
      e.stopPropagation();
      if (state.chartPlacing) void state.chartCommit([tenth(hit[0]), tenth(hit[1])]);
      else state.chartSelect(null);
    },
    onPointerOut: () => { useStore.getState().chartSetHover(null); press.current = null; },
  };
}

function Ghost({ draft, grab }: { draft: ChartDraft; grab: Grab }) {
  const placing = useStore((s) => s.chartPlacing);
  const hover = useStore((s) => s.chartHover);
  const chartDraft = useStore((s) => s.chartDraft);
  const design = useStore((s) => s.designs.find((d) => d.id === s.chartPlacing));
  if (!placing || !hover || !design) return null;
  const item: Footprint = { uv: hover, size: chartDraft.size, rotation_rad: chartDraft.rotation_rad, mirror: false };
  // The ghost is judged by the same rule commit applies, so a spot that looks
  // placeable cannot then be refused for running off the paper.
  return <Artwork item={item} id={null} source={design} draft={draft} ghost invalid={Boolean(chartItemRefusal(draft, item))} grab={grab} />;
}

export function ChartScene() {
  const draft = useStore((s) => s.chart);
  const interaction = useStore((s) => s.chartInteraction);
  const grab = useRef<[number, number]>([0, 0]);
  const pointer = useFixturePointer(grab);
  // A freshly mounted canvas replacing the body's is deaf to its first
  // on-demand frame requests (the old root is torn down on a timer), so run
  // freely while that settles, then render only on change.
  const [frameloop, setFrameloop] = useState<"always" | "demand">("always");
  useEffect(() => { const settle = setTimeout(() => setFrameloop("demand"), 2000); return () => clearTimeout(settle); }, []);
  return (
    <div className="chart-stage">
      <Canvas
        camera={CAMERA}
        onCreated={({ camera }) => camera.up.set(0, 0, 1)}
        gl={GL}
        dpr={DPR}
        frameloop={frameloop}
        onPointerMissed={() => { const s = useStore.getState(); if (!s.chartInteraction && !s.chartPlacing) s.chartSelect(null); }}
      >
        <color attach="background" args={[BACKGROUND]} />
        <hemisphereLight args={["#ffffff", "#6a707c", 1.6]} />
        <directionalLight position={[0.3, -0.5, 0.9]} intensity={1.6} />
        <directionalLight position={[-0.6, 0.4, 0.5]} intensity={0.6} />
        {draft.kind === "plane" ? <Pad draft={draft} pointer={pointer} /> : <Tube draft={draft} pointer={pointer} />}
        <ChartBounds draft={draft} />
        {draft.items.map((item) => <Artwork key={item.id} item={item} id={item.id} source={item.artwork} draft={draft} selected={item.id === draft.selected} grab={grab} />)}
        <Ghost draft={draft} grab={grab} />
        <ChartSelectionControls />
        <ChartCamera draft={draft} />
        <OrbitControls makeDefault enabled={!interaction} minDistance={0.03} maxDistance={4} />
      </Canvas>
    </div>
  );
}
