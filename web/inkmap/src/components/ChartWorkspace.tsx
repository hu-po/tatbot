import { artworkNeedsRegeneration } from "../core/artwork-record.ts";
import { renderTattooProgramSvg } from "../core/human-representation/program-svg.ts";
import { useStore } from "../store.ts";
import type { ChartDraft } from "../core/chart-draft.ts";
import { bodyMode, useUi } from "../ui.ts";
import { exportArtworkSvg, exportChartArtworkSvg, exportChartDesign } from "../exports.ts";
import { atFixtureDimensions, fixtureCaption, fixtureFor, INCH_MM, PAPER_CYLINDER, PAPER_PAD } from "../core/fixtures.ts";
import { ArtworkChooser } from "./ArtworkChooser.tsx";
import { ChartAdjustBar } from "./AdjustBar.tsx";
import { ChartScene } from "./ChartScene.tsx";

/** The paper pad or paper cylinder in 3D at its measured size, with the
 *  placements bent onto it. Interaction is the body's: click to place the
 *  chosen artwork, drag to move it, ↻ ↔ to rotate and resize. */
export function ChartCanvas() {
  const draft = useStore((s) => s.chart);
  const fixture = fixtureFor(draft.kind);
  const surface = draft.kind === "cylinder"
    ? `${draft.width} mm along the axis · ${draft.height} mm of arc · ⌀${2 * draft.radius} mm`
    : `${draft.width} × ${draft.height} mm`;
  return (
    <figure className="chart-figure" aria-label="Surface placement preview">
      <ChartScene />
      <figcaption>
        {atFixtureDimensions(draft) ? fixtureCaption(draft.kind) : `${fixture.label} · ${surface} · ¼ in grid`}
      </figcaption>
    </figure>
  );
}

/** The fixture itself: which paper, and its dimensions when they differ from
 *  the measured object. A chart-only concern, so it lives here and not in
 *  the shared toolbar. */
function SurfaceDetails({ open }: { open: boolean }) {
  const draft = useStore((s) => s.chart);
  const patch = (value: Partial<ChartDraft>) => useStore.getState().setChart(value);
  const number = (label: string, value: number, change: (value: number) => void, min: number, max: number) =>
    <label>{label}<input aria-label={label} type="number" value={value} min={min} max={max} step="any" onKeyDown={(e) => { if (e.key !== "Escape") e.stopPropagation(); }} onChange={(event) => {
      const next = event.target.valueAsNumber; if (Number.isFinite(next) && next >= min && next <= max) change(next);
    }} /></label>;
  return (
    <details className="surface-details" open={open}>
      <summary>Surface</summary>
      <div className="controls chart-controls">
        <label>Design name<input aria-label="Design name" value={draft.name} maxLength={200} onKeyDown={(e) => { if (e.key !== "Escape") e.stopPropagation(); }} onChange={(event) => patch({ name: event.target.value })} /></label>
        <label>Surface<select aria-label="Surface" value={draft.kind} onChange={(event) => useUi.getState().setWorkspace(event.target.value === "cylinder" ? "cylinder" : "paper")}>
          <option value="plane">Paper pad — {PAPER_PAD.canvas_mm[0] / INCH_MM} × {PAPER_PAD.canvas_mm[1] / INCH_MM} in, {PAPER_PAD.thickness_mm} mm thick</option>
          <option value="cylinder">Paper cylinder — ⌀{PAPER_CYLINDER.thickness_mm} mm × {PAPER_CYLINDER.canvas_mm[0] / INCH_MM} in</option>
        </select></label>
        {number(draft.kind === "cylinder" ? "Surface length along axis (mm)" : "Surface width (mm)", draft.width, (width) => patch({ width }), 1, 2000)}
        {number(draft.kind === "cylinder" ? "Drawable arc around the crest (mm)" : "Surface height (mm)", draft.height, (height) => patch({ height }), 1, 2000)}
        {draft.kind === "cylinder" && number("Cylinder radius (mm)", draft.radius, (radius) => patch({ radius }), 1, 2000)}
        {number("Edge margin (mm)", draft.margin, (margin) => patch({ margin }), 0, 1000)}
      </div>
    </details>
  );
}

/** The chart's contextual panel: choose → place → adjust → ready, read off
 *  the chart's placement machine exactly as the body dock reads the body's.
 *  Reopening a placed artwork is the same adjust state; Cancel restores it. */
export function ChartDock({ mobile }: { mobile: boolean }) {
  const draft = useStore((s) => s.chart);
  const placing = useStore((s) => s.chartPlacing);
  const designs = useStore((s) => s.designs);
  const startPlacing = useStore((s) => s.chartStartPlacing);
  const cancelPlacing = useStore((s) => s.chartCancelPlacing);
  const select = useStore((s) => s.chartSelect);
  const reorder = useStore((s) => s.chartReorder);
  const choosing = useUi((s) => s.choosing);
  const setChoosing = useUi((s) => s.setChoosing);
  const mode = bodyMode(placing, draft.selected, draft.items.length, choosing);
  const placingDesign = placing ? designs.find((d) => d.id === placing) : undefined;
  const item = draft.items.find((candidate) => candidate.id === draft.selected);
  const paper = draft.kind === "cylinder" ? "cylinder" : "pad";

  if (mode === "choose") return (
    <div className="dock-body choose" data-mode="choose">
      <div className="dock-head">
        <h2>Choose artwork</h2>
        {draft.items.length > 0 && <button type="button" className="link" onClick={() => setChoosing(false)}>back to placements</button>}
      </div>
      <ArtworkChooser onPick={(id) => { setChoosing(false); startPlacing(id); }} useLabel="Place this artwork" />
      <SurfaceDetails open={draft.items.length === 0} />
    </div>
  );

  if (mode === "place") return (
    <div className="dock-body place" data-mode="place">
      <div className="dock-head">
        <h2>Place it</h2>
        <button type="button" className="link" onClick={() => { cancelPlacing(); setChoosing(true); }}>choose different artwork</button>
      </div>
      <div className="placing-card">
        {placingDesign && <img src={placingDesign.path} alt="" />}
        <div>
          <strong>{placingDesign?.name ?? placing}</strong>
          <p className="muted small">{mobile ? "Tap" : "Click"} the {paper} where it should go.</p>
        </div>
      </div>
      {placingDesign && <button type="button" className="ghost small" onClick={() => void exportArtworkSvg(placingDesign)}>Download artwork SVG</button>}
    </div>
  );

  if (mode === "adjust" && item) return (
    <div className="dock-body adjust" data-mode="adjust">
      <div className="dock-head"><h2>Adjust</h2></div>
      {mobile && <ChartAdjustBar />}
      {artworkNeedsRegeneration(item.artwork, item.size) && <p className="muted small">Resized preview: regenerate at {item.size[0].toFixed(1)} × {item.size[1].toFixed(1)} mm before drawing. Pen width stays fixed.</p>}
      <div className="io">
        <button type="button" disabled={draft.items[0].id === item.id} onClick={() => reorder(item.id, draft.items.findIndex((x) => x.id === item.id) - 1)}>Draw earlier</button>
        <button type="button" disabled={draft.items[draft.items.length - 1].id === item.id} onClick={() => reorder(item.id, draft.items.findIndex((x) => x.id === item.id) + 1)}>Draw later</button>
      </div>
      <details className="provenance">
        <summary>Technical details</summary>
        <dl>
          <dt>offset</dt><dd className="mono">u {item.uv[0].toFixed(1)} mm · v {item.uv[1].toFixed(1)} mm{draft.kind === "cylinder" ? " (along the axis · arc from the crest)" : ""}</dd>
          <dt>size</dt><dd>{item.size[0].toFixed(1)} × {item.size[1].toFixed(1)} mm · {(item.rotation_rad * 180 / Math.PI).toFixed(1)}°{item.mirror ? " · mirrored" : ""}</dd>
          <dt>surface</dt><dd>{fixtureCaption(draft.kind)}</dd>
        </dl>
      </details>
      <button type="button" className="ghost small" onClick={() => void exportChartArtworkSvg()}>Download artwork SVG — {item.artwork.name}</button>
    </div>
  );

  return (
    <div className="dock-body ready" data-mode="ready">
      <div className="dock-head"><h2>Placements <span className="muted">({draft.items.length})</span></h2></div>
      <ul className="layers">{draft.items.map((candidate) => <li key={candidate.id}><button type="button" onClick={() => select(candidate.id)}>
        <img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(renderTattooProgramSvg(candidate.artwork.program))}`} alt="" /><span>{candidate.artwork.name}<small>{candidate.size[0].toFixed(0)} mm · at {candidate.uv[0].toFixed(0)}, {candidate.uv[1].toFixed(0)} mm{artworkNeedsRegeneration(candidate.artwork, candidate.size) ? " · regenerate for this size" : ""}</small></span>
      </button></li>)}</ul>
      <div className="io">
        <button type="button" className="primary" onClick={() => setChoosing(true)}>Add artwork</button>
        <button type="button" disabled={!draft.items.length} onClick={() => void exportChartDesign()}>Export design</button>
      </div>
      <SurfaceDetails open={false} />
    </div>
  );
}
