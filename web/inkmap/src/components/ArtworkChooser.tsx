import { useMemo, useState } from "react";
import type { DesignMeta } from "../core/schema.ts";
import { useStore } from "../store.ts";
import { useUi } from "../ui.ts";
import { importArt } from "../exports.ts";
import { SHORT_LIBRARY, shortLibrary, visibleLibrary } from "../library.ts";

/** The art library and Import art. `onPick` is what choosing means: the body places the design. */
export function ArtworkChooser({ onPick, active = null }: { onPick: (designId: string) => void; active?: string | null; useLabel?: string }) {
  const loaded = useStore((s) => s.designs);
  // Filtered once per library change, never inside the selector: a selector
  // that returns a fresh array every call re-renders forever.
  const designs = useMemo(() => visibleLibrary(loaded), [loaded]);
  const browsing = useUi((s) => s.browsingAll);
  const setBrowsing = useUi((s) => s.setBrowsingAll);
  const [filter, setFilter] = useState("");
  const shown = useMemo(() => {
    if (!browsing) return shortLibrary(designs);
    const q = filter.trim().toLowerCase();
    return q ? designs.filter((d) => d.name.toLowerCase().includes(q)) : designs;
  }, [designs, browsing, filter]);
  const tile = (d: DesignMeta) => (
    <button key={d.id} type="button" className={active === d.id ? "design active" : "design"} aria-pressed={active === d.id}
      onClick={() => onPick(d.id)} aria-label={d.name} title={d.name}>
      <img src={d.path} alt="" />
      <span>{d.name}</span>
    </button>
  );
  return (
    <section className="chooser" aria-label="Choose art">
      {browsing && <input type="search" className="filter" aria-label="Filter art" placeholder="filter by name" value={filter} onChange={(e) => setFilter(e.target.value)} onKeyDown={(e) => { if (e.key !== "Escape") e.stopPropagation(); }} />}
      <div className={browsing ? "picker all" : "picker"}>{shown.map(tile)}</div>
      {!designs.length && <p className="muted small">Import a PNG or JPEG of your design.</p>}
      <div className="chooser-actions">
        {designs.length > SHORT_LIBRARY && <button type="button" onClick={() => setBrowsing(!browsing)}>{browsing ? "Show fewer" : `Browse all ${designs.length}`}</button>}
        <label className="file" title="A PNG or JPEG of the design; select its generation sidecar (.json) too to keep its prompt">Import art<input type="file" multiple accept="image/png,image/jpeg,.png,.jpg,.jpeg,.json,application/json" onChange={(e) => { const files = [...(e.target.files ?? [])]; e.target.value = ""; void importArt(files).then((id) => { if (id) onPick(id); }); }} /></label>
      </div>
    </section>
  );
}
