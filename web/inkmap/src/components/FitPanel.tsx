import { useEffect, useMemo, useState } from "react";
import { artProfile, knownInks } from "../core/art-image.ts";
import { thinInks, type ArtProfile } from "../core/art-profile.ts";
import type { ArtRecord } from "../core/art.ts";
import { buildDecal } from "../core/decal.ts";
import { estimateCost, formatMinutes } from "../core/estimate.ts";
import { inkById, INKS, MAX_INKS } from "../core/inks.ts";
import { OVERLAP_NOTICE_SHARE, overlapShare } from "../core/overlap.ts";
import { renderAtOtherWidth } from "../core/render-record.ts";
import type { Placement } from "../core/schema.ts";
import { fitWidthMm, overflowMm, STENCIL_TILES, type StencilTile } from "../core/tiles.ts";
import { importRenders } from "../exports.ts";
import { useSettled } from "../settled.ts";
import { renderFor, useStore } from "../store.ts";
import { useUi } from "../ui.ts";

function verdict(tile: StencilTile, size: readonly [number, number]): string | null {
  const [wide, tall] = overflowMm(tile, size);
  const parts = [wide > 0 && `${wide.toFixed(1)} mm too wide`, tall > 0 && `${tall.toFixed(1)} mm too tall`].filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}

function useArtProfile(record: ArtRecord | undefined): ArtProfile | null {
  const [profile, setProfile] = useState<ArtProfile | null>(null);
  useEffect(() => {
    let live = true;
    setProfile(null);
    if (record) artProfile(record).then((value) => { if (live) setProfile(value); }).catch(console.error);
    return () => { live = false; };
  }, [record]);
  return profile;
}

/** The inks the art pipeline will load for this design. */
function InkChips({ designId, inks }: { designId: string; inks: string[] }) {
  const setArtInks = useStore((s) => s.setArtInks);
  const rest = INKS.filter((ink) => !inks.includes(ink.id));
  return (
    <div className="ink-chips" role="group" aria-label="Inks">
      {inks.map((id) => {
        const ink = inkById.get(id);
        return (
          <span key={id} className="ink-chip">
            <i style={{ background: ink ? `rgb(${ink.rgb.join(" ")})` : "transparent" }} />{ink?.name ?? id}
            {inks.length > 1 && <button type="button" aria-label={`Remove ${ink?.name ?? id}`} onClick={() => void setArtInks(designId, inks.filter((item) => item !== id))}>×</button>}
          </span>
        );
      })}
      {inks.length < MAX_INKS && rest.length > 0 && (
        <select aria-label="Add an ink" value="" onChange={(event) => void setArtInks(designId, [...inks, event.target.value])}>
          <option value="" disabled>Add ink</option>
          {rest.map((ink) => <option key={ink.id} value={ink.id}>{ink.name}</option>)}
        </select>
      )}
    </div>
  );
}

/** What the chosen size means downstream: which stencil it fits, how far it wraps, roughly what it costs and which
 *  colours it may lose. All of it is advice; the compile is what refuses a design its page cannot hold. */
export function FitPanel({ placement }: { placement: Placement }) {
  const body = useStore((s) => s.body);
  const update = useStore((s) => s.update);
  const record = useStore((s) => s.designs.find((d) => d.id === placement.design_id)?.embedded);
  const tileId = useUi((s) => s.tile);
  const setTile = useUi((s) => s.setTile);
  const tile = STENCIL_TILES.find((item) => item.id === tileId)!;
  const size = placement.size_mm;
  const profile = useArtProfile(record);
  const settled = useSettled(placement);
  const wrapDeg = useMemo(() => {
    if (!body) return null;
    try {
      const built = buildDecal(body.restSkin.geometry, body.skin.geometry,
        { anchor: settled.anchor, rotationRad: settled.rotation_rad, sizeMm: settled.size_mm });
      built.geometry.dispose();
      return built.quality.wrapDeg;
    } catch { return null; }
  }, [body, settled]);
  const renders = useStore((s) => s.renders);
  const stockRenders = useStore((s) => s.stockRenders);
  const showRender = useStore((s) => s.showRender);
  const toggleRender = useStore((s) => s.toggleRender);
  const placements = useStore((s) => s.placements);
  const designs = useStore((s) => s.designs);
  const overlapping = useMemo(() => {
    if (!body) return [];
    return placements.filter((other) => other.id !== settled.id && overlapShare(body.restSkin.geometry, settled, other) >= OVERLAP_NOTICE_SHARE)
      .map((other) => designs.find((d) => d.id === other.design_id)?.name ?? other.design_id);
  }, [body, settled, placements, designs]);
  const inks = record?.palette?.inks ?? [];
  const cost = profile && estimateCost(profile.coverage, size, inks.length);
  const thin = profile && record ? thinInks(profile, knownInks(record), size[0]) : [];
  const render = renderFor({ renders, stockRenders }, record?.content_sha256, size[0], placement.mirror);
  const elsewhere = render ? null : renderAtOtherWidth([...renders, ...stockRenders], record?.content_sha256, size[0], placement.mirror);
  const useWidth = (width: number) => update(placement.id, { size_mm: [width, Number((width * size[1] / size[0]).toFixed(3))] });
  return (
    <section className="fit" aria-label="Fit and cost">
      <h3>Stencil tile</h3>
      <fieldset className="fit-tiles" aria-label="Stencil tile">
        {STENCIL_TILES.map((item) => {
          const over = verdict(item, size);
          return (
            <label key={item.id}>
              <input type="radio" name="stencil-tile" checked={item.id === tile.id} onChange={() => setTile(item.id)} />
              <span>{item.name}</span>
              <span className="size">{item.size_mm[0]} × {item.size_mm[1]}</span>
              <span className={over ? "verdict over" : "verdict"}>{over ?? "fits"}</span>
            </label>
          );
        })}
      </fieldset>
      {verdict(tile, size) && <button type="button" className="fit-action" onClick={() => useWidth(fitWidthMm(tile, size))}>Fit to {tile.name} ({fitWidthMm(tile, size).toFixed(1)} mm)</button>}
      {render ? <h3>Separated <span>the art pipeline's own figures</span></h3> : <h3>Estimate <span>from the art, before any strokes</span></h3>}
      <dl className="fit-row">
        {render ? <><dt>Time</dt><dd>{formatMinutes(render.minutes)}, {render.dips} dips, {render.rinses} rinses</dd></>
          : cost && <><dt>Time</dt><dd>about {formatMinutes(cost.minutes)}, {Math.round(cost.dips / 5) * 5} dips</dd></>}
        {wrapDeg !== null && <><dt>Wraps</dt><dd>{Math.round(wrapDeg)}° of the skin's curve</dd></>}
        {render ? render.lost_inks.length > 0 && <><dt>Detail</dt><dd className="warn">{render.lost_inks.map((id) => <div key={id}>{inkById.get(id)?.name ?? id} drew nothing at this size</div>)}</dd></>
          : thin.length > 0 && <><dt>Detail</dt><dd className="warn">{thin.map(({ ink, minWidthMm }) => <div key={ink.id}>{ink.name} may be lost below about {Math.ceil(minWidthMm)} mm</div>)}</dd></>}
        {overlapping.length > 0 && <><dt>Overlaps</dt><dd className="warn">{overlapping.join(", ")}</dd></>}
      </dl>
      {record && <><h3>Inks</h3><InkChips designId={placement.design_id} inks={inks} /></>}
      <h3>Robot render</h3>
      {render && <label className="fit-check"><input type="checkbox" checked={showRender} onChange={toggleRender} />Show it on the body</label>}
      {elsewhere && <div className="fit-note">The render is for {elsewhere.width_mm.toFixed(1)} mm. <button type="button" className="link" onClick={() => useWidth(elsewhere.width_mm)}>Use {elsewhere.width_mm.toFixed(1)} mm</button></div>}
      {!render && !elsewhere && <p className="muted small">Export the brief, run <code>tatbot art separate --brief</code>, then load its render here.</p>}
      <label className="file fit-action">Load render<input type="file" multiple accept=".json,application/json" aria-label="Load robot render"
        onChange={(event) => { void importRenders([...(event.target.files ?? [])]); event.target.value = ""; }} /></label>
    </section>
  );
}
