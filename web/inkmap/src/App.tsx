import { useEffect } from "react";
import { Scene } from "./components/Scene.tsx";
import { SentenceBar } from "./components/SentenceBar.tsx";
import { Picker } from "./components/Picker.tsx";
import { Inspector } from "./components/Inspector.tsx";
import { BodyBar } from "./components/BodyToggle.tsx";
import { AcceptBar } from "./components/AcceptBar.tsx";
import { Generate } from "./components/Generate.tsx";
import { useStore } from "./store.ts";
import type { DesignMeta } from "./core/schema.ts";

export function App() {
  const setDesigns = useStore((s) => s.setDesigns);
  const setError = useStore((s) => s.setError);
  const error = useStore((s) => s.error);
  const cancelPlacing = useStore((s) => s.cancelPlacing);
  const nudgeRotation = useStore((s) => s.nudgeRotation);
  const nudgeSize = useStore((s) => s.nudgeSize);
  const placing = useStore((s) => s.placing);
  const selected = useStore((s) => s.selected);
  const accept = useStore((s) => s.accept);
  const discard = useStore((s) => s.discard);
  const toast = useStore((s) => s.toast);
  const setToast = useStore((s) => s.setToast);
  const accepted = useStore((s) => s.accepted);
  const body = useStore((s) => s.body);

  useEffect(() => {
    fetch("designs/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`designs/manifest.json: HTTP ${r.status}`))))
      .then((m: { designs: DesignMeta[] }) => setDesigns(m.designs))
      .catch((e: Error) => setError(e.message));
  }, [setDesigns, setError]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { cancelPlacing(); return; }
      // Keys never fire while typing in a field.
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") { accept(); e.preventDefault(); return; }
      if (e.key === "Delete" || e.key === "Backspace") { discard(); e.preventDefault(); return; }
      // WASD nudges the ghost or the selected tattoo.
      const step = e.shiftKey ? 3 : 1; // shift = coarser
      switch (e.key.toLowerCase()) {
        case "a": nudgeRotation((-5 * step * Math.PI) / 180); break;
        case "d": nudgeRotation((5 * step * Math.PI) / 180); break;
        case "w": nudgeSize(1 + 0.05 * step); break;
        case "s": nudgeSize(1 / (1 + 0.05 * step)); break;
        default: return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancelPlacing, nudgeRotation, nudgeSize, accept, discard]);

  // Toasts fade out on their own.
  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 2200);
    return () => window.clearTimeout(id);
  }, [toast, setToast]);

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>Inkmap</h1>
          <p className="muted">Pick a design, hover the body, click to place.</p>
        </header>
        <SentenceBar />
        <Picker key={accepted} pulse={accepted > 0} />
        <Generate />
        <Inspector />
        {error && <p className="error">{error}</p>}
      </aside>
      <main className="viewport">
        <Scene />
        <BodyBar />
        <AcceptBar />
        {toast && <div className="toast" role="status">{toast}</div>}
        <div className="hud">
          {!body && !error && <span>loading body…</span>}
          {placing && <span>placing <b>{placing}</b> — click on the body · <kbd>A</kbd>/<kbd>D</kbd> rotate · <kbd>W</kbd>/<kbd>S</kbd> size · Esc cancels</span>}
          {!placing && selected && body && <span><kbd>A</kbd>/<kbd>D</kbd> rotate · <kbd>W</kbd>/<kbd>S</kbd> size · <kbd>Enter</kbd> accept · <kbd>Del</kbd> discard</span>}
        </div>
      </main>
    </div>
  );
}
