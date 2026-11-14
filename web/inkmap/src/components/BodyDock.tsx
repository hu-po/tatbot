import { useMemo } from "react";
import type { ArtRecord } from "../core/art.ts";
import { overlaps } from "../core/overlap.ts";
import { turnFromUpright, useStore } from "../store.ts";
import { bodyMode, useUi } from "../ui.ts";
import { exportArtImage, exportBrief, exportCard } from "../exports.ts";
import { ArtworkChooser } from "./ArtworkChooser.tsx";
import { SentenceBar } from "./SentenceBar.tsx";
import { AdjustBar } from "./AdjustBar.tsx";
import { BodyStrip } from "./BodyControls.tsx";
import { FitPanel } from "./FitPanel.tsx";

/** Where the art came from and on what terms, as its record says. */
function ArtSource({ record }: { record: ArtRecord }) {
  const { source, palette } = record;
  return <>
    <dt>art</dt><dd>{source.kind}{source.identifier ? ` · ${source.identifier}` : ""} · {record.image.width_px} × {record.image.height_px} px</dd>
    <dt>licence</dt><dd>{source.license ?? "none recorded"}{source.attribution ? ` · ${source.attribution}` : ""}</dd>
    {source.generation && <><dt>generated</dt><dd>{[source.generation.generator, source.generation.model].filter(Boolean).join(" · ") || "unknown generator"}</dd></>}
    {palette && <><dt>inks</dt><dd>{palette.inks.join(", ")}{palette.style ? ` · ${palette.style}` : ""}</dd></>}
  </>;
}

/** The body editor's contextual panel: choose → place → adjust → ready, read
 *  off the placement store. Reopening an accepted tattoo is the same adjust
 *  state; the store's edit snapshot and undo entry are untouched by any of this. */
export function BodyDock({ mobile }: { mobile: boolean }) {
  const placing = useStore((s) => s.placing);
  const selected = useStore((s) => s.selected);
  const placements = useStore((s) => s.placements);
  const designs = useStore((s) => s.designs);
  const pending = useStore((s) => s.pending);
  const placeAt = useStore((s) => s.placeAt);
  const startPlacing = useStore((s) => s.startPlacing);
  const cancelPlacing = useStore((s) => s.cancelPlacing);
  const select = useStore((s) => s.select);
  const choosing = useUi((s) => s.choosing);
  const setChoosing = useUi((s) => s.setChoosing);
  const mode = bodyMode(placing, selected, placements.length, choosing);
  const pick = (designId: string) => {
    setChoosing(false);
    if (pending?.resolution.status === "resolved" && pending.resolution.anchor) { placeAt(designId, pending.resolution.anchor, pending); return; }
    startPlacing(designId);
  };
  const placingDesign = placing ? designs.find((d) => d.id === placing) : undefined;
  const body = useStore((s) => s.body);
  const crossing = useMemo(() => (mode === "ready" && body ? overlaps(body.restSkin.geometry, placements) : []), [mode, body, placements]);
  const nameOf = (p: { design_id: string }) => designs.find((d) => d.id === p.design_id)?.name ?? p.design_id;
  const placement = placements.find((p) => p.id === selected);
  const design = placement && designs.find((d) => d.id === placement.design_id);
  const at = placement ? placements.findIndex((p) => p.id === placement.id) : -1;

  if (mode === "choose") return (
    <div className="dock-body choose" data-mode="choose">
      <div className="dock-head">
        <h2>Choose artwork</h2>
        {placements.length > 0 && <button type="button" className="link" onClick={() => setChoosing(false)}>back to placements</button>}
      </div>
      <ArtworkChooser onPick={pick} useLabel="Place this artwork" />
      <SentenceBar compact />
      <BodyStrip />
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
          <p className="muted small">{mobile ? "Tap the body" : "Click the body"} where it should go, or describe a location.</p>
        </div>
      </div>
      <SentenceBar compact />
    </div>
  );

  if (mode === "adjust" && placement) return (
    <div className="dock-body adjust" data-mode="adjust">
      <div className="dock-head"><h2>Adjust</h2></div>
      {mobile && <AdjustBar />}
      <p className="muted small">Drawn at {placement.size_mm[0].toFixed(1)} × {placement.size_mm[1].toFixed(1)} mm: the art pipeline separates it at this width.</p>
      <FitPanel placement={placement} />
      {placement.language && <p className="muted caption">“{placement.language.sentence}”</p>}
      {placement.site && !placement.language && <p className="muted caption">{[placement.site.laterality && `patient's ${placement.site.laterality}`, placement.site.level, placement.site.aspect, placement.site.id.replace(/_/g, " ")].filter(Boolean).join(" · ")}</p>}
      <div className="io">
        <button type="button" disabled={at <= 0} onClick={() => useStore.getState().reorder(placement.id, at - 1)}>Draw earlier</button>
        <button type="button" disabled={at === placements.length - 1} onClick={() => useStore.getState().reorder(placement.id, at + 1)}>Draw later</button>
      </div>
      <details className="provenance">
        <summary>Technical details</summary>
        <dl>
          <dt>anchor</dt><dd className="mono">face {placement.anchor.face} · bary {placement.anchor.barycentric.map((w) => w.toFixed(3)).join(" ")}</dd>
          <dt>size</dt><dd>{placement.size_mm[0].toFixed(1)} × {placement.size_mm[1].toFixed(1)} mm · {(turnFromUpright(placement) * 180 / Math.PI).toFixed(1)}° from upright{placement.mirror ? " · mirrored" : ""}</dd>
          {placement.site && <><dt>site</dt><dd>{placement.site.id} · {placement.site.laterality ?? "no laterality"}{placement.site.uv ? ` · uv ${placement.site.uv.map((v) => v.toFixed(3)).join(", ")}` : ""}</dd></>}
          {placement.language?.resolution && <>
            <dt>request</dt><dd>{placement.language.resolution.intent.description}</dd>
            <dt>normalized</dt><dd>{placement.language.resolution.choice?.label ?? placement.language.resolution.intent.canonical_phrase}</dd>
            <dt>resolver</dt><dd>{placement.language.resolution.resolver.name} v{placement.language.resolution.resolver.version} · {placement.language.resolution.resolver.policy.id}</dd>
            <dt>surface</dt><dd className="mono">{placement.language.resolution.body.rest_surface_sha256?.slice(0, 12)}…</dd>
          </>}
          {design?.embedded && <ArtSource record={design.embedded} />}
        </dl>
      </details>
      {design && <button type="button" className="ghost small" onClick={() => exportArtImage(design)}>Download source image</button>}
    </div>
  );

  return (
    <div className="dock-body ready" data-mode="ready">
      <div className="dock-head">
        <h2>Placements <span className="muted">({placements.length})</span></h2>
      </div>
      <ul className="layers">
        {placements.map((x) => {
          const d = designs.find((item) => item.id === x.design_id);
          return (
            <li key={x.id}>
              <button type="button" onClick={() => select(x.id)}>
                {d && <img src={d.path} alt="" />}
                <span>{d?.name ?? x.design_id}<small>{x.size_mm[0].toFixed(1)} mm{x.site ? ` · ${[x.site.laterality, x.site.id.replace(/_/g, " ")].filter(Boolean).join(" ")}` : ""}</small></span>
              </button>
            </li>
          );
        })}
      </ul>
      {crossing.map(({ a, b, share }) => <p key={`${a.id}:${b.id}`} className="small overlap-note" role="status">{nameOf(a)} overlaps {nameOf(b)} ({Math.round(100 * share)}%)</p>)}
      <div className="io">
        <button type="button" className="primary" onClick={() => setChoosing(true)}>Add artwork</button>
        <button type="button" onClick={() => void exportBrief()}>Export brief</button>
        <button type="button" onClick={() => void exportCard()}>Placement card</button>
      </div>
      <SentenceBar compact />
      <BodyStrip />
    </div>
  );
}
