import { useState } from "react";
import { useStore } from "../store.ts";
import { useUi } from "../ui.ts";
import { artworkSources } from "../core/design-assets.ts";
import { downloadJson } from "../core/download.ts";

export function SimulationExport() {
  const [busy, setBusy] = useState(false);
  const [previewName, setPreviewName] = useState<string | null>(null);
  // The tool and seed outlive this panel: it lives in a menu that unmounts on close.
  const seed = useUi((s) => s.simSeed);
  const tool = useUi((s) => s.simTool);
  const setSeed = (simSeed: number) => useUi.setState({ simSeed });
  const setTool = (simTool: string) => useUi.setState({ simTool });
  const exportBundle = async () => {
    const state = useStore.getState();
    const file = state.toFile();
    if (!file || !state.atlas) return;
    setBusy(true);
    try {
      const { makeSimulationBundle, simulationRequest } = await import("../core/sim-bundle.ts");
      const bundle = await makeSimulationBundle(file, state.atlas.atlas,
        simulationRequest(state.poseId, state.skinTone, state.cameraSnapshot, tool, seed), artworkSources(state.designs));
      downloadJson(bundle, "inkmap-simulation.bundle.json");
      state.setError(null);
    } catch (error) { state.setError(String(error)); }
    finally { setBusy(false); }
  };
  const openCompiledPreview = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    const state = useStore.getState();
    setBusy(true);
    try {
      const { parseJsonStrict } = await import("../core/human-representation/schema.ts");
      const value = parseJsonStrict(await file.text()) as { schema_version?: unknown };
      if (value.schema_version !== 3) throw new Error("compiled preview requires a typed tattoo scenario version 3");
      await state.loadShowcaseScenario(value);
      setPreviewName(file.name);
      state.setToast("Compiled preview loaded");
      state.setError(null);
    } catch (error) { state.setError(String(error)); }
    finally { setBusy(false); input.value = ""; }
  };
  return <details className="simulation-export">
    <summary>Simulation export</summary>
    <p className="muted caption">Frozen artwork and placements for offline compilation. Body assets resolve from the pinned local cache. No robot motion or release approval.</p>
    <label>Simulation tool<input aria-label="Simulation tool" value={tool} onKeyDown={e => { if (e.key !== "Escape") e.stopPropagation(); }} onChange={e => setTool(e.target.value)} /></label>
    <label>Simulation seed<input type="number" aria-label="Simulation seed" min={0} step={1} value={seed} onKeyDown={e => { if (e.key !== "Escape") e.stopPropagation(); }}
      onChange={e => { if (Number.isSafeInteger(e.target.valueAsNumber) && e.target.valueAsNumber >= 0) setSeed(e.target.valueAsNumber); }} /></label>
    <button type="button" disabled={busy} onClick={exportBundle}>{busy ? "Preparing bundle…" : "Export simulation bundle"}</button>
    <p className="muted caption">Compile the download with <code>tatbot sim compile FILE -- --output SCENARIO</code>.</p>
    <label className="file-action">Open compiled preview
      <input type="file" accept="application/json,.json" disabled={busy} onChange={openCompiledPreview} />
    </label>
    {previewName && <p className="muted caption" role="status">Loaded {previewName}. Preview only; this is not execution validation or motion approval.</p>}
  </details>;
}
