import { useState } from "react";
import { PosePicker, SkinTonePicker } from "./BodyControls.tsx";
import { downloadText } from "../core/download.ts";
import { SCHEMA_VERSION } from "../core/schema.ts";
import { useStore } from "../store.ts";
import { useUi, workspaceTab, type WorkspaceTab } from "../ui.ts";
import { exportArtworkSvg, exportBodyDesign, exportChartArtworkSvg, exportChartDesign, exportPlacementFile, exportProjectBackup, loadPlacementFile, openDesign, openProject, type UnsupportedFile } from "../exports.ts";
import { Menu } from "./Menu.tsx";
import { SimulationExport } from "./SimulationExport.tsx";

const TABS: { id: WorkspaceTab; label: string }[] = [{ id: "body", label: "Body" }, { id: "paper", label: "Paper" }, { id: "cylinder", label: "Cylinder" }];

export function TopBar({ mobile, onReview }: { mobile: boolean; onReview: () => void }) {
  const workspace = useUi((s) => s.workspace);
  const chartKind = useStore((s) => s.chart.kind);
  const setWorkspace = useUi((s) => s.setWorkspace);
  const ready = useStore((s) => s.projectReady);
  const canUndo = useStore((s) => workspace === "body" ? Boolean(s.editBefore || s.past.length) : Boolean(s.chartEditBefore || s.chartPast.length));
  const canRedo = useStore((s) => workspace === "body" ? !s.editBefore && Boolean(s.future.length) : !s.chartEditBefore && Boolean(s.chartFuture.length));
  const undo = () => (workspace === "body" ? useStore.getState().undo() : useStore.getState().chartUndo());
  const redo = () => (workspace === "body" ? useStore.getState().redo() : useStore.getState().chartRedo());
  const selected = useStore((s) => s.selected);
  const requestCamera = useStore((s) => s.requestCamera);
  const status = useStore((s) => s.saveStatus);
  const tab = workspaceTab(workspace, chartKind);
  return (
    <header className="topbar">
      <h1>Inkmap</h1>
      <nav className="workspaces" role="tablist" aria-label="Workspace">
        {TABS.map((t) => <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} disabled={!ready} onClick={() => setWorkspace(t.id)}>{t.label}</button>)}
      </nav>
      <FileMenu onReview={onReview} />
      <div className="history" role="group" aria-label="History">
        <button type="button" disabled={!ready || !canUndo} onClick={undo} aria-label="Undo" title="Undo (Ctrl+Z)">{mobile ? "↶" : "Undo"}</button>
        <button type="button" disabled={!ready || !canRedo} onClick={redo} aria-label="Redo" title="Redo (Ctrl+Shift+Z)">{mobile ? "↷" : "Redo"}</button>
      </div>
      {workspace === "body" && <>
        <div className="camera" role="group" aria-label="Camera">
          <button type="button" onClick={() => requestCamera("reset")} aria-label="Reset camera" title="Fit the whole body">{mobile ? "Fit" : "Fit body"}</button>
          {!mobile && <button type="button" disabled={!selected} onClick={() => requestCamera("selection")} title="Bring the camera to the selected tattoo">Focus</button>}
        </div>
        <ViewMenu />
      </>}
      <span className="grow" />
      <span role="status" className={`save-status ${status}`} data-testid="save-status">
        {status === "saved" ? "Saved locally" : status === "saving" ? "Saving…" : status === "loading" ? "Loading…" : "Local save failed"}
      </span>
    </header>
  );
}

function FileMenu({ onReview }: { onReview: () => void }) {
  const [unsupported, setUnsupported] = useState<UnsupportedFile | null>(null);
  const ready = useStore((s) => s.projectReady);
  const name = useStore((s) => s.projectName);
  const status = useStore((s) => s.saveStatus);
  const saveError = useStore((s) => s.saveError);
  const workspace = useUi((s) => s.workspace);
  const notice = useUi((s) => s.notice);
  const hasBody = useStore((s) => s.placements.length > 0);
  const hasChart = useStore((s) => s.chart.items.length > 0);
  const artwork = useStore((s) => {
    const id = s.placing ?? s.placements.find((p) => p.id === s.selected)?.design_id ?? null;
    return id ? s.designs.find((d) => d.id === id) ?? null : null;
  });
  const chartSelected = useStore((s) => s.chart.items.find((item) => item.id === s.chart.selected) ?? null);
  const file = (label: string, accept: string, onFile: (file: File | undefined) => void, disabled = false) => (
    <label className="menu-item file"><span>{label}</span><input type="file" accept={accept} disabled={disabled} onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} /></label>
  );
  const artworkExport = workspace === "body"
    ? { label: artwork ? `Artwork SVG — ${artwork.name}` : "Artwork SVG (choose or select artwork first)", enabled: Boolean(artwork), run: () => { if (artwork) void exportArtworkSvg(artwork); } }
    : { label: chartSelected ? `Artwork SVG — ${chartSelected.artwork.name}` : "Artwork SVG (select a placement first)", enabled: Boolean(chartSelected), run: () => void exportChartArtworkSvg() };
  return (
    <Menu name="file" label="File">
      <label className="menu-field">Project name<input aria-label="Project name" value={name} maxLength={200} disabled={!ready} onKeyDown={(e) => { if (e.key !== "Escape") e.stopPropagation(); }}
        onChange={(event) => useStore.setState({ projectName: event.target.value })} /></label>
      <p className="small muted">{status === "saved" ? "Saved in this browser." : status === "saving" ? "Saving in this browser…" : status === "loading" ? "Loading the local project…" : "Local save failed — download a project backup to keep your work."}</p>
      {saveError && <p role="alert" className="error small">{saveError}</p>}
      {status === "failed" && <div className="menu-row">
        <button type="button" className="menu-item" onClick={() => window.dispatchEvent(new Event("inkmap-save-retry"))}>Retry local save</button>
        <button type="button" className="menu-item" onClick={() => window.location.reload()}>Reload saved project</button>
      </div>}
      <hr />
      <button type="button" className="menu-item" data-closes disabled={!ready} onClick={() => {
        const s = useStore.getState();
        const holds = s.placements.length + s.chart.items.length;
        if (holds && !window.confirm(`Start a new project? The current one (${holds} placement${holds === 1 ? "" : "s"}) is replaced; download a project backup first to keep it.`)) return;
        s.newProject();
      }}>New project</button>
      {file("Open project…", "application/json,.json", (f) => void openProject(f), !ready)}
      {file("Open design…", "application/json,.json", (f) => void openDesign(f).then(setUnsupported), !ready)}
      <button type="button" className="menu-item" data-closes onClick={onReview}>Review study…</button>
      {unsupported && <button type="button" className="menu-item" onClick={() => downloadText(unsupported.text, unsupported.name, "application/json")}>Download {unsupported.name} unchanged</button>}
      <hr />
      <button type="button" className="menu-item" data-closes disabled={!ready || (workspace === "body" ? !hasBody : !hasChart)} onClick={() => void (workspace === "body" ? exportBodyDesign() : exportChartDesign())}>
        Portable design (JSON){workspace === "body" ? " — accepted body placements" : ` — ${useStore.getState().chart.kind === "cylinder" ? "cylinder" : "paper"} draft`}
      </button>
      <button type="button" className="menu-item" data-closes disabled={!artworkExport.enabled} onClick={artworkExport.run}>{artworkExport.label}</button>
      <button type="button" className="menu-item" data-closes disabled={!ready} onClick={() => void exportProjectBackup()}>Project backup (JSON) — body, chart, history</button>
      {notice && <p role="alert" className="error small">{notice.action}: {notice.message}</p>}
      {workspace === "body" && <details className="menu-advanced">
        <summary>Advanced</summary>
        <button type="button" className="menu-item" disabled={!ready || !hasBody} onClick={exportPlacementFile}>Placement file (v{SCHEMA_VERSION} JSON) — robot pipeline contract</button>
        {file("Load placement file…", "application/json,.json", (f) => void loadPlacementFile(f), !ready)}
        <SimulationExport />
      </details>}
    </Menu>
  );
}

function ViewMenu() {
  const atlas = useStore((state) => state.atlas);
  const showAtlas = useStore((state) => state.showAtlas);
  const toggleAtlas = useStore((state) => state.toggleAtlas);
  const requestCamera = useStore((state) => state.requestCamera);
  const showQuality = useStore((state) => state.showQuality);
  const toggleQuality = useStore((state) => state.toggleQuality);
  const neutralLight = useStore((state) => state.neutralLight);
  const toggleNeutralLight = useStore((state) => state.toggleNeutralLight);
  return (
    <Menu name="view" label="View" align="end">
      <div className="menu-section">
        <span className="menu-label">Camera</span>
        <div className="camera-presets" role="group" aria-label="Camera views">
          {(["front", "back", "left", "right"] as const).map((preset) => <button key={preset} type="button" onClick={() => requestCamera(preset)}>{preset[0].toUpperCase() + preset.slice(1)}</button>)}
          <button type="button" onClick={() => requestCamera("reset")}>Reset</button>
        </div>
      </div>
      <div className="menu-section">
        <PosePicker />
      </div>
      <div className="menu-section">
        <span className="menu-label">Skin tone</span>
        <SkinTonePicker />
      </div>
      <div className="menu-section view-toggles" role="group" aria-label="Overlays">
        <button type="button" aria-pressed={showAtlas} disabled={!atlas} onClick={toggleAtlas} aria-label="Toggle body-site atlas">Atlas</button>
        <button type="button" aria-pressed={neutralLight} onClick={toggleNeutralLight}>Neutral light</button>
        <button type="button" aria-pressed={showQuality} onClick={toggleQuality}>Quality</button>
      </div>
    </Menu>
  );
}
