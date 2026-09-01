import { useStore } from "../store.ts";

/** Floats over the viewport while a tattoo is selected: lock it in, or throw it away. */
export function AcceptBar() {
  const selected = useStore((s) => s.selected);
  const placing = useStore((s) => s.placing);
  const accept = useStore((s) => s.accept);
  const discard = useStore((s) => s.discard);
  const placements = useStore((s) => s.placements);
  const designs = useStore((s) => s.designs);
  if (!selected || placing) return null;
  const p = placements.find((x) => x.id === selected);
  const name = p ? designs.find((d) => d.id === p.design_id)?.name ?? p.design_id : "";
  return (
    <div className="acceptbar">
      <span className="muted">{name} — adjust with <kbd>A</kbd><kbd>D</kbd><kbd>W</kbd><kbd>S</kbd>, then</span>
      <button type="button" className="accept" onClick={accept} title="Enter">✓ Accept</button>
      <button type="button" className="discard" onClick={discard} title="Delete">✕ Discard</button>
    </div>
  );
}
