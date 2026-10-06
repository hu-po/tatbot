import { MAX_WIDTH_MM, MIN_WIDTH_MM, useStore } from "../store.ts";

/** The one stable toolbar for a selected artwork: width in mm, rotation in
 *  degrees, mirror, Accept and Cancel. Direct manipulation on the surface
 *  (drag, the ↻ and ↔ handles) edits the same placement; this is the numeric
 *  edge. The body and the paper/cylinder chart each bind it to their own
 *  store actions; the markup, labels and keys are the same on both. */
export function AdjustToolbar({ name, widthMm, aspect, rotationRad, mirror, onWidth, onRotation, onMirror, onFocus, onDelete, onCancel, onAccept, label = "Adjust tattoo" }: {
  name: string; widthMm: number; aspect: number; rotationRad: number; mirror: boolean;
  onWidth: (widthMm: number, heightMm: number) => void; onRotation: (rad: number) => void; onMirror: (mirror: boolean) => void;
  onFocus?: () => void; onDelete: () => void; onCancel: () => void; onAccept: () => void; label?: string;
}) {
  const stop = (e: React.KeyboardEvent) => { if (e.key !== "Escape") e.stopPropagation(); };
  return (
    <div className="adjustbar" role="toolbar" aria-label={label}>
      <span className="name" title={name}>{name}</span>
      <label>Width<input type="number" aria-label="Width in millimeters" min={MIN_WIDTH_MM} max={MAX_WIDTH_MM} step={1} value={Number(widthMm.toFixed(1))} onKeyDown={stop}
        onChange={(e) => { const w = e.target.valueAsNumber; if (Number.isFinite(w) && w >= MIN_WIDTH_MM && w <= MAX_WIDTH_MM) onWidth(w, w / aspect); }} /><span className="unit">mm</span></label>
      <label>Rotation<input type="number" aria-label="Rotation in degrees" min={-180} max={180} step={1} value={Number((rotationRad * 180 / Math.PI).toFixed(1))} onKeyDown={stop}
        onChange={(e) => { const d = e.target.valueAsNumber; if (Number.isFinite(d) && d >= -180 && d <= 180) onRotation(d * Math.PI / 180); }} /><span className="unit">°</span></label>
      <label className="check"><input type="checkbox" checked={mirror} onChange={(e) => onMirror(e.target.checked)} />Mirror</label>
      {onFocus && <button type="button" className="ghost" onClick={onFocus} title="Bring the camera to this tattoo">Focus</button>}
      <button type="button" className="ghost danger" onClick={onDelete} title="Delete this artwork">Delete</button>
      <span className="grow" />
      <button type="button" className="discard" onClick={onCancel} title="Escape">✕ Cancel</button>
      <button type="button" className="accept" onClick={onAccept} title="Enter">✓ Accept</button>
    </div>
  );
}

/** The body's binding: the selected tattoo, edited through the body store. */
export function AdjustBar() {
  const selected = useStore((s) => s.selected);
  const placing = useStore((s) => s.placing);
  const placement = useStore((s) => s.placements.find((x) => x.id === s.selected));
  const design = useStore((s) => s.designs.find((d) => d.id === placement?.design_id));
  const accept = useStore((s) => s.accept);
  const discard = useStore((s) => s.discard);
  const update = useStore((s) => s.update);
  const remove = useStore((s) => s.remove);
  const requestCamera = useStore((s) => s.requestCamera);
  if (!selected || placing || !placement) return null;
  const p = placement;
  const aspect = design ? design.default_size_mm[0] / design.default_size_mm[1] : p.size_mm[0] / p.size_mm[1];
  return (
    <AdjustToolbar name={design?.name ?? p.design_id} widthMm={p.size_mm[0]} aspect={aspect} rotationRad={p.rotation_rad} mirror={p.mirror}
      onWidth={(w, h) => update(p.id, { size_mm: [w, h] })} onRotation={(rad) => update(p.id, { rotation_rad: rad })} onMirror={(mirror) => update(p.id, { mirror })}
      onFocus={() => requestCamera("selection")} onDelete={() => remove(p.id)} onCancel={discard} onAccept={accept} />
  );
}

/** The chart's binding: the selected artwork on the paper pad or cylinder. */
export function ChartAdjustBar() {
  const item = useStore((s) => s.chart.items.find((x) => x.id === s.chart.selected));
  const placing = useStore((s) => s.chartPlacing);
  const accept = useStore((s) => s.chartAccept);
  const discard = useStore((s) => s.chartDiscard);
  const update = useStore((s) => s.chartUpdate);
  const remove = useStore((s) => s.chartRemove);
  if (!item || placing) return null;
  return (
    <AdjustToolbar label="Adjust artwork" name={item.artwork.name} widthMm={item.size[0]} aspect={item.size[0] / item.size[1]} rotationRad={item.rotation_rad} mirror={item.mirror}
      onWidth={(w, h) => update(item.id, { size: [w, h] })} onRotation={(rad) => update(item.id, { rotation_rad: rad })} onMirror={(mirror) => update(item.id, { mirror })}
      onDelete={() => remove(item.id)} onCancel={discard} onAccept={accept} />
  );
}
