import { useState } from "react";
import { maxWidthMm, MIN_WIDTH_MM } from "../core/art.ts";
import { turnFromUpright, uprightAt, useStore } from "../store.ts";

/** A number being typed. A person's first digit is rarely the value, so the field holds what they have typed and only
 *  a value in range reaches the placement; leaving the field shows the placement's own value again. */
function NumberField({ label, value, min, max, step, onCommit }: {
  label: string; value: number; min: number; max: number; step: number; onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const valid = (text: string) => text.trim() !== "" && Number(text) >= min && Number(text) <= max;
  return (
    <input type="number" aria-label={label} aria-invalid={draft !== null && !valid(draft)} min={min} max={max} step={step} value={draft ?? value}
      onChange={(e) => { setDraft(e.target.value); if (valid(e.target.value)) onCommit(Number(e.target.value)); }}
      onBlur={() => setDraft(null)}
      onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key !== "Escape") e.stopPropagation(); }} />
  );
}

/** The one stable toolbar for a selected tattoo: width in mm, rotation in
 *  degrees, mirror, Accept and Cancel. Direct manipulation on the body (drag,
 *  the ↻ and ↔ handles) edits the same placement; this is the numeric edge. */
function AdjustToolbar({ name, widthMm, aspect, rotationRad, mirror, onWidth, onRotation, onMirror, onFocus, onDelete, onCancel, onAccept }: {
  name: string; widthMm: number; aspect: number; rotationRad: number; mirror: boolean;
  onWidth: (widthMm: number, heightMm: number) => void; onRotation: (rad: number) => void; onMirror: (mirror: boolean) => void;
  onFocus: () => void; onDelete: () => void; onCancel: () => void; onAccept: () => void;
}) {
  return (
    <div className="adjustbar" role="toolbar" aria-label="Adjust tattoo">
      <span className="name" title={name}>{name}</span>
      <label>Width<NumberField label="Width in millimeters" value={Number(widthMm.toFixed(1))} min={MIN_WIDTH_MM} max={maxWidthMm([aspect, 1])} step={0.1}
        onCommit={(w) => { const snapped = Math.round(w * 10) / 10; onWidth(snapped, Number((snapped / aspect).toFixed(3))); }} /><span className="unit">mm</span></label>
      <label>Rotation<NumberField label="Rotation in degrees" value={Number((rotationRad * 180 / Math.PI).toFixed(1))} min={-180} max={180} step={1}
        onCommit={(degrees) => onRotation(degrees * Math.PI / 180)} /><span className="unit">°</span></label>
      <label className="check"><input type="checkbox" checked={mirror} onChange={(e) => onMirror(e.target.checked)} />Mirror</label>
      <button type="button" className="ghost" onClick={onFocus} title="Bring the camera to this tattoo">Focus</button>
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
    <AdjustToolbar name={design?.name ?? p.design_id} widthMm={p.size_mm[0]} aspect={aspect} rotationRad={turnFromUpright(p)} mirror={p.mirror}
      onWidth={(w, h) => update(p.id, { size_mm: [w, h] })} onRotation={(rad) => update(p.id, { rotation_rad: rad + uprightAt(p.anchor) })} onMirror={(mirror) => update(p.id, { mirror })}
      onFocus={() => requestCamera("selection")} onDelete={() => remove(p.id)} onCancel={discard} onAccept={accept} />
  );
}
