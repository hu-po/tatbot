import { artworkNeedsRegeneration } from "../core/artwork-record.ts";
import { useStore } from "../store.ts";
import { bodyMode, useUi } from "../ui.ts";
import { exportArtworkSvg, exportBodyDesign } from "../exports.ts";
import { ArtworkChooser } from "./ArtworkChooser.tsx";
import { SentenceBar } from "./SentenceBar.tsx";
import { AdjustBar } from "./AdjustBar.tsx";
import { BodyStrip } from "./BodyControls.tsx";

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
  const placement = placements.find((p) => p.id === selected);
  const design = placement && designs.find((d) => d.id === placement.design_id);

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
      {placingDesign && <button type="button" className="ghost small" onClick={() => void exportArtworkSvg(placingDesign)}>Download artwork SVG</button>}
    </div>
  );

  if (mode === "adjust" && placement) return (
    <div className="dock-body adjust" data-mode="adjust">
      <div className="dock-head"><h2>Adjust</h2></div>
      {mobile && <AdjustBar />}
      {design?.embedded && artworkNeedsRegeneration(design?.embedded, placement.size_mm) && <p className="muted small">Resized preview: regenerate at {placement.size_mm[0].toFixed(1)} × {placement.size_mm[1].toFixed(1)} mm before drawing. Pen width stays fixed.</p>}
      {placement.language && <p className="muted caption">“{placement.language.sentence}”</p>}
      {placement.site && !placement.language && <p className="muted caption">{[placement.site.laterality && `patient's ${placement.site.laterality}`, placement.site.level, placement.site.aspect, placement.site.id.replace(/_/g, " ")].filter(Boolean).join(" · ")}</p>}
      <div className="io">
        <button type="button" disabled={placements[0].id === placement.id} onClick={() => useStore.getState().reorder(placement.id, placements.findIndex((x) => x.id === placement.id) - 1)}>Draw earlier</button>
        <button type="button" disabled={placements[placements.length - 1].id === placement.id} onClick={() => useStore.getState().reorder(placement.id, placements.findIndex((x) => x.id === placement.id) + 1)}>Draw later</button>
      </div>
      <details className="provenance">
        <summary>Technical details</summary>
        <dl>
          <dt>anchor</dt><dd className="mono">face {placement.anchor.face} · bary {placement.anchor.barycentric.map((w) => w.toFixed(3)).join(" ")}</dd>
          <dt>size</dt><dd>{placement.size_mm[0].toFixed(1)} × {placement.size_mm[1].toFixed(1)} mm · {(placement.rotation_rad * 180 / Math.PI).toFixed(1)}°{placement.mirror ? " · mirrored" : ""}</dd>
          {placement.site && <><dt>site</dt><dd>{placement.site.id} · {placement.site.laterality ?? "no laterality"}{placement.site.uv ? ` · uv ${placement.site.uv.map((v) => v.toFixed(3)).join(", ")}` : ""}</dd></>}
          {placement.language?.resolution && <>
            <dt>request</dt><dd>{placement.language.resolution.intent.description}</dd>
            <dt>normalized</dt><dd>{placement.language.resolution.choice?.label ?? placement.language.resolution.intent.canonical_phrase}</dd>
            <dt>resolver</dt><dd>{placement.language.resolution.resolver.name} v{placement.language.resolution.resolver.version} · {placement.language.resolution.resolver.policy.id}</dd>
            <dt>surface</dt><dd className="mono">{placement.language.resolution.body.rest_surface_sha256?.slice(0, 12)}…</dd>
          </>}
          {design?.embedded?.source?.generation && <>
            <dt>generated</dt><dd>{design.embedded.source.generation!.model}{design.embedded.source.generation!.model_revision ? ` @ ${design.embedded.source.generation!.model_revision.slice(0, 12)}` : ""} · seed {design.embedded.source.generation!.seed}</dd>
          </>}
        </dl>
      </details>
      {design && <button type="button" className="ghost small" onClick={() => void exportArtworkSvg(design)}>Download artwork SVG</button>}
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
                <span>{d?.name ?? x.design_id}<small>{x.size_mm[0].toFixed(0)} mm{x.site ? ` · ${[x.site.laterality, x.site.id.replace(/_/g, " ")].filter(Boolean).join(" ")}` : ""}{d?.embedded && artworkNeedsRegeneration(d.embedded, x.size_mm) ? " · regenerate for this size" : ""}</small></span>
              </button>
            </li>
          );
        })}
      </ul>
      <div className="io">
        <button type="button" className="primary" onClick={() => setChoosing(true)}>Add artwork</button>
        <button type="button" onClick={() => void exportBodyDesign()}>Export design</button>
      </div>
      <SentenceBar compact />
      <BodyStrip />
    </div>
  );
}
