import { useMemo } from "react";
import { PosePicker, SkinTonePicker } from "./BodyControls.tsx";
import { SCHEMA_VERSION } from "../core/schema.ts";
import { useStore } from "../store.ts";
import { useUi } from "../ui.ts";
import { exportArtImage, exportBrief, exportCard, exportMockup, exportPlacementFile, exportProjectBackup, importRenders, loadPlacementFile, openProject } from "../exports.ts";
import { Menu } from "./Menu.tsx";

export function TopBar({ mobile }: { mobile: boolean }) {
  const ready = useStore((s) => s.projectReady);
  const canUndo = useStore((s) => Boolean(s.editBefore || s.past.length));
  const canRedo = useStore((s) => !s.editBefore && Boolean(s.future.length));
  const selected = useStore((s) => s.selected);
  const requestCamera = useStore((s) => s.requestCamera);
  const status = useStore((s) => s.saveStatus);
  return (
    <header className="topbar">
      <h1>Inkmap</h1>
      <FileMenu />
      <div className="history" role="group" aria-label="History">
        <button type="button" disabled={!ready || !canUndo} onClick={() => useStore.getState().undo()} aria-label="Undo" title="Undo (Ctrl+Z)">{mobile ? "↶" : "Undo"}</button>
        <button type="button" disabled={!ready || !canRedo} onClick={() => useStore.getState().redo()} aria-label="Redo" title="Redo (Ctrl+Shift+Z)">{mobile ? "↷" : "Redo"}</button>
      </div>
      <div className="camera" role="group" aria-label="Camera">
        <button type="button" onClick={() => requestCamera("reset")} aria-label="Reset camera" title="Fit the whole body">{mobile ? "Fit" : "Fit body"}</button>
        {!mobile && <button type="button" disabled={!selected} onClick={() => requestCamera("selection")} title="Bring the camera to the selected tattoo">Focus</button>}
      </div>
      <ViewMenu />
      <span className="grow" />
      <span role="status" className={`save-status ${status}`} data-testid="save-status">
        {status === "saved" ? "Saved locally" : status === "saving" ? "Saving…" : status === "loading" ? "Loading…" : "Local save failed"}
      </span>
    </header>
  );
}

function FileMenu() {
  const ready = useStore((s) => s.projectReady);
  const name = useStore((s) => s.projectName);
  const status = useStore((s) => s.saveStatus);
  const saveError = useStore((s) => s.saveError);
  const notice = useUi((s) => s.notice);
  const hasBody = useStore((s) => s.placements.length > 0);
  const placing = useStore((s) => s.placing);
  const selected = useStore((s) => s.selected);
  const placements = useStore((s) => s.placements);
  const designs = useStore((s) => s.designs);
  // The art being placed, or the selected tattoo's.
  const artwork = useMemo(() => {
    const id = placing ?? placements.find((p) => p.id === selected)?.design_id ?? null;
    return id ? designs.find((d) => d.id === id) ?? null : null;
  }, [placing, selected, placements, designs]);
  const file = (label: string, accept: string, onFile: (file: File | undefined) => void, disabled = false) => (
    <label className="menu-item file"><span>{label}</span><input type="file" accept={accept} disabled={disabled} onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} /></label>
  );
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
        const holds = s.placements.length;
        if (holds && !window.confirm(`Start a new project? The current one (${holds} placement${holds === 1 ? "" : "s"}) is replaced; download a project backup first to keep it.`)) return;
        s.newProject();
      }}>New project</button>
      {file("Open project…", "application/json,.json", (f) => void openProject(f), !ready)}
      {file("Load robot render…", "application/json,.json", (f) => void importRenders(f ? [f] : []), !ready)}
      <hr />
      <button type="button" className="menu-item" data-closes disabled={!ready || !hasBody} onClick={() => void exportBrief()}>Tattoo brief (JSON) — each design's art and size, for the art pipeline</button>
      <button type="button" className="menu-item" data-closes disabled={!ready || !hasBody} onClick={() => void exportCard()}>Placement card (HTML) — one page per design, to print and lay the stencil by</button>
      <button type="button" className="menu-item" data-closes disabled={!ready} onClick={exportMockup}>Mockup image (JPEG) — this view of the body</button>
      <button type="button" className="menu-item" data-closes disabled={!artwork} onClick={() => { if (artwork) exportArtImage(artwork); }}>{artwork ? `Source image — ${artwork.name}` : "Source image (choose or select art first)"}</button>
      <button type="button" className="menu-item" data-closes disabled={!ready} onClick={() => void exportProjectBackup()}>Project backup (JSON) — placements and history</button>
      {notice && <p role="alert" className="error small">{notice.action}: {notice.message}</p>}
      <details className="menu-advanced">
        <summary>Advanced</summary>
        <button type="button" className="menu-item" disabled={!ready || !hasBody} onClick={exportPlacementFile}>Placement file (v{SCHEMA_VERSION} JSON) — placements with their art</button>
        {file("Load placement file…", "application/json,.json", (f) => void loadPlacementFile(f), !ready)}
      </details>
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
