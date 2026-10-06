import { useEffect, useRef, useState } from "react";
import { Scene } from "./components/Scene.tsx";
import { ShowcaseHud, ShowcasePanel } from "./components/Showcase.tsx";
import { StudyReview } from "./components/StudyReview.tsx";
import { TopBar } from "./components/TopBar.tsx";
import { BodyDock } from "./components/BodyDock.tsx";
import { AdjustBar, ChartAdjustBar } from "./components/AdjustBar.tsx";
import { ChartCanvas, ChartDock } from "./components/ChartWorkspace.tsx";
import { MappingQuality } from "./components/MappingQuality.tsx";
import { useStore } from "./store.ts";
import { bodyMode, useUi } from "./ui.ts";
import type { DesignMeta } from "./core/schema.ts";
import { freezeDesign } from "./core/design-assets.ts";
import { chartItemRefusal } from "./core/chart-draft.ts";
import { startProjectSession } from "./project-session.ts";
import { exportProjectBackup } from "./exports.ts";

const PHONE_QUERY = "(max-width: 768px) and (pointer: coarse)";

export function App() {
  // The phone layout (a tray under the stage) is for touch screens; a desktop
  // window dragged narrow keeps its dock and toolbar where they are.
  const [mobile, setMobile] = useState(() => window.matchMedia(PHONE_QUERY).matches);
  const [reviewOpen, setReviewOpen] = useState(false);
  const showcase = new URLSearchParams(window.location.search).get("showcase") === "1";
  const setDesigns = useStore((s) => s.setDesigns);
  const setError = useStore((s) => s.setError);
  const error = useStore((s) => s.error);
  const cancelPlacing = useStore((s) => s.cancelPlacing);
  const nudgeRotation = useStore((s) => s.nudgeRotation);
  const nudgeSize = useStore((s) => s.nudgeSize);
  const placing = useStore((s) => s.placing);
  const selected = useStore((s) => s.selected);
  const accept = useStore((s) => s.accept);
  const toast = useStore((s) => s.toast);
  const setToast = useStore((s) => s.setToast);
  const body = useStore((s) => s.body);
  const projectReady = useStore((s) => s.projectReady);
  const saveStatus = useStore((s) => s.saveStatus);
  const saveError = useStore((s) => s.saveError);
  const placementCount = useStore((s) => s.placements.length);
  const hover = useStore((s) => s.hover);
  const atlas = useStore((s) => s.atlas);
  const placingDesign = useStore((s) => s.designs.find((d) => d.id === s.placing));
  const workspace = useUi((s) => s.workspace);
  const chartPlacing = useStore((s) => s.chartPlacing);
  const chartSelected = useStore((s) => s.chart.selected);
  const chartCount = useStore((s) => s.chart.items.length);
  const chartKind = useStore((s) => s.chart.kind);
  const chartHoverRefused = useStore((s) => Boolean(s.chartPlacing && s.chartHover && chartItemRefusal(s.chart, { uv: s.chartHover, size: s.chartDraft.size, rotation_rad: s.chartDraft.rotation_rad })));
  const chartPlacingDesign = useStore((s) => s.designs.find((d) => d.id === s.chartPlacing));
  const choosing = useUi((s) => s.choosing);
  const tray = useUi((s) => s.tray);
  const setTray = useUi((s) => s.setTray);
  const popover = useUi((s) => s.popover);
  const setPopover = useUi((s) => s.setPopover);
  const notice = useUi((s) => s.notice);
  const setNotice = useUi((s) => s.setNotice);
  const dock = useRef<HTMLElement>(null);
  // One mode per editor, both read the same way: the chart's machine mirrors the body's.
  const mode = workspace === "body" ? bodyMode(placing, selected, placementCount, choosing) : bodyMode(chartPlacing, chartSelected, chartCount, choosing);

  useEffect(() => { if (!showcase) return startProjectSession(); }, [showcase]);
  useEffect(() => {
    const query = window.matchMedia(PHONE_QUERY);
    const change = () => setMobile(query.matches); query.addEventListener("change", change);
    return () => query.removeEventListener("change", change);
  }, []);
  // The tray follows the work: each new step opens it to its working size, so
  // the Accept toolbar or the placing card is never hidden behind a collapsed
  // handle; the person may still collapse or expand it afterwards.
  useEffect(() => { if (mobile) setTray("working"); }, [mobile, mode, workspace, setTray]);
  useEffect(() => { if (mobile && tray !== "collapsed" && mode === "choose") dock.current?.querySelector<HTMLElement>("[role=tab][aria-selected=true]")?.focus({ preventScroll: true }); }, [mobile, tray, mode]);

  useEffect(() => {
    document.title = showcase ? "Tatbot — procedural body showcase" : "Inkmap — tattoo preview";
  }, [showcase]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("designs/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`designs/manifest.json: HTTP ${r.status}`))))
      .then((m: { designs: DesignMeta[] }) => Promise.all(m.designs.map(design => freezeDesign(design, controller.signal))))
      .then((designs) => { if (!controller.signal.aborted) setDesigns(designs); })
      .catch((e: Error) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [setDesigns, setError]);

  useEffect(() => {
    if (showcase) return;
    const onKey = (e: KeyboardEvent) => {
      if (reviewOpen) return;
      // Escape closes what is on top first: a menu, then an expanded tray,
      // then the edit itself. An edit is never cancelled by closing a menu.
      if (e.key === "Escape") {
        if (useUi.getState().popover) { setPopover(null); e.preventDefault(); return; }
        if (mobile && useUi.getState().tray === "expanded") { setTray("working"); e.preventDefault(); return; }
        if (useUi.getState().workspace === "chart") { useStore.getState().chartCancelPlacing(); return; }
        cancelPlacing(); return;
      }
      // Keys never fire while typing in a field.
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      // The same keys drive whichever editor is showing; each has its own actions.
      const chart = useUi.getState().workspace === "chart";
      const store = useStore.getState();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (chart) { if (e.shiftKey) store.chartRedo(); else store.chartUndo(); } else if (e.shiftKey) store.redo(); else store.undo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") { if (t?.tagName === "BUTTON" || t?.tagName === "A" || t?.tagName === "SUMMARY" || t?.tagName === "LABEL") return; if (chart) store.chartAccept(); else accept(); e.preventDefault(); return; }
      if (e.key === "Delete" || e.key === "Backspace") { if (chart) { if (store.chart.selected) store.chartRemove(store.chart.selected); } else if (store.selected) store.remove(store.selected); e.preventDefault(); return; }
      // WASD nudges the ghost or the selected artwork.
      const step = e.shiftKey ? 3 : 1; // shift = coarser
      const rotate = chart ? store.chartNudgeRotation : nudgeRotation;
      const size = chart ? store.chartNudgeSize : nudgeSize;
      switch (e.key.toLowerCase()) {
        case "a": rotate((-5 * step * Math.PI) / 180); break;
        case "d": rotate((5 * step * Math.PI) / 180); break;
        case "w": size(1 + 0.05 * step); break;
        case "s": size(1 / (1 + 0.05 * step)); break;
        default: return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showcase, reviewOpen, cancelPlacing, nudgeRotation, nudgeSize, accept, mobile, setPopover, setTray]);

  // Toasts fade out on their own.
  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 2200);
    return () => window.clearTimeout(id);
  }, [toast, setToast]);

  if (showcase) return (
    <div className="app showcase">
      <aside className="sidebar" aria-label="Editor controls"><ShowcasePanel />{error && <p className="error">{error}</p>}</aside>
      <main className="viewport"><Scene /><ShowcaseHud />{toast && <div className="toast" role="status">{toast}</div>}</main>
    </div>
  );

  const hoverInvalid = Boolean(placing && hover && atlas && !atlas.isValidAnchor(hover));
  const paper = chartKind === "cylinder" ? "cylinder" : "pad";
  const hint = !body ? "loading body…"
    : workspace !== "body" ? (mode === "place"
      ? (chartHoverRefused ? `Off the paper — move onto the ${paper}'s drawable area` : `${mobile ? "Tap" : "Click"} the ${paper} to place ${chartPlacingDesign?.name ?? "the artwork"}${mobile ? "" : " · A/D rotate · W/S size · Esc cancels"}`)
      : null)
    : mode === "place" ? (hoverInvalid ? "Unsupported here — move to a supported area of the body" : `${mobile ? "Tap" : "Click"} the body to place ${placingDesign?.name ?? "the artwork"}${mobile ? "" : " · A/D rotate · W/S size · Esc cancels"}`)
    : null;

  return (
    <div className={`app editor ${mobile ? "mobile" : "desktop"} tray-${tray} workspace-${workspace} mode-${mode}${popover ? " popover-open" : ""}`}>
      <TopBar mobile={mobile} onReview={() => setReviewOpen(true)} />
      <StudyReview open={reviewOpen} onClose={() => setReviewOpen(false)} />
      <div className="workarea">
        <main className="stage" aria-label={workspace === "body" ? "Body" : "Chart"}>
          {workspace === "body" ? <>
            <Scene />
            <MappingQuality />
            {hint && <div className="hud" role="status">{hint}</div>}
            {!mobile && <AdjustBar />}
          </> : <>
            <ChartCanvas />
            {hint && <div className="hud" role="status">{hint}</div>}
            {!mobile && <ChartAdjustBar />}
          </>}
          <div className="messages" aria-live="polite">
            {toast && <div className="toast" role="status">{toast}</div>}
            {error && <p className="message error" role="alert">{error}<button type="button" className="link" onClick={() => setError(null)}>dismiss</button></p>}
            {notice && <p className="message error" role="alert">{notice.action}: {notice.message}<button type="button" className="link" onClick={() => setNotice(null)}>dismiss</button></p>}
            {saveStatus === "failed" && <p className="message warn" role="alert">Local save failed{saveError ? ` — ${saveError}` : ""}. Your work is still here; download a project backup to keep it.
              <button type="button" onClick={() => void exportProjectBackup()}>Download project backup</button>
              <button type="button" onClick={() => window.dispatchEvent(new Event("inkmap-save-retry"))}>Retry local save</button></p>}
          </div>
        </main>
        <aside ref={dock} className="dock" aria-label={workspace === "body" ? "Body editor panel" : "Chart editor panel"}>
          {mobile && <div className="tray-handle">
            <button type="button" className="tray-toggle" aria-expanded={tray !== "collapsed"} onClick={() => setTray(tray === "collapsed" ? "working" : tray === "working" ? "expanded" : "collapsed")}>
              {tray === "collapsed" ? `▲ ${mode === "place" ? "Place it" : mode === "adjust" ? "Adjust" : mode === "choose" ? "Choose artwork" : "Placements"}` : tray === "working" ? "▲ more" : "▼ less"}
            </button>
          </div>}
          <div className="dock-scroll" hidden={mobile && tray === "collapsed"}>
            <fieldset disabled={!projectReady} className="editor-fields">
              {workspace === "body" ? <BodyDock mobile={mobile} /> : <ChartDock mobile={mobile} />}
            </fieldset>
          </div>
        </aside>
      </div>
    </div>
  );
}
