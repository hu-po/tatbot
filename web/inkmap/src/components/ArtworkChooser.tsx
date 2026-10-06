import { useMemo, useState } from "react";
import type { DesignMeta } from "../core/schema.ts";
import { useStore } from "../store.ts";
import { useUi } from "../ui.ts";
import { importArtwork } from "../exports.ts";
import { SHORT_LIBRARY, shortLibrary, visibleLibrary } from "../library.ts";
import { Generate } from "./Generate.tsx";

/** Library / Generate, shared by the body and the paper/cylinder editors.
 *  `onPick` is what choosing means where it is mounted: the body places, the chart adds. */
export function ArtworkChooser({ onPick, active = null, useLabel }: { onPick: (designId: string) => void; active?: string | null; useLabel?: string }) {
  const loaded = useStore((s) => s.designs);
  // Filtered once per library change, never inside the selector: a selector
  // that returns a fresh array every call re-renders forever.
  const designs = useMemo(() => visibleLibrary(loaded), [loaded]);
  const tab = useUi((s) => s.chooserTab);
  const setTab = useUi((s) => s.setChooserTab);
  const browsing = useUi((s) => s.browsingAll);
  const setBrowsing = useUi((s) => s.setBrowsingAll);
  const [filter, setFilter] = useState("");
  const shown = useMemo(() => {
    if (!browsing) return shortLibrary(designs);
    const q = filter.trim().toLowerCase();
    return q ? designs.filter((d) => d.name.toLowerCase().includes(q) || d.family?.includes(q)) : designs;
  }, [designs, browsing, filter]);
  const tile = (d: DesignMeta) => (
    <button key={d.id} type="button" className={active === d.id ? "design active" : "design"} aria-pressed={active === d.id}
      onClick={() => onPick(d.id)} aria-label={d.name} title={`${d.name} — ${d.default_size_mm[0]}×${d.default_size_mm[1]} mm`}>
      <img src={d.path} alt="" />
      <span>{d.name}</span>
    </button>
  );
  return (
    <section className="chooser" aria-label="Choose artwork">
      <div className="tabs" role="tablist" aria-label="Artwork source">
        <button type="button" role="tab" aria-selected={tab === "library"} onClick={() => setTab("library")}>Library</button>
        <button type="button" role="tab" aria-selected={tab === "generate"} onClick={() => setTab("generate")}>Generate</button>
      </div>
      {tab === "library" ? (
        <div role="tabpanel" aria-label="Library">
          {browsing && <input type="search" className="filter" aria-label="Filter artwork" placeholder="filter by name" value={filter} onChange={(e) => setFilter(e.target.value)} onKeyDown={(e) => { if (e.key !== "Escape") e.stopPropagation(); }} />}
          <div className={browsing ? "picker all" : "picker"}>{shown.map(tile)}</div>
          {!designs.length && <p className="muted small">Generate with DrawingBot V3, then import artwork.json.</p>}
          <div className="chooser-actions">
            {designs.length > SHORT_LIBRARY && <button type="button" onClick={() => setBrowsing(!browsing)}>{browsing ? "Show fewer" : `Browse all ${designs.length}`}</button>}
            <label className="file">Import artwork<input type="file" accept=".json,application/json" onChange={(e) => { void importArtwork(e.target.files?.[0]).then((id) => { if (id) onPick(id); }); e.target.value = ""; }} /></label>
          </div>
        </div>
      ) : (
        <div role="tabpanel" aria-label="Generate"><Generate onUse={onPick} useLabel={useLabel} /></div>
      )}
    </section>
  );
}
