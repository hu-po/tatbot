import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import "./styles.css";
import { useStore } from "./store.ts";
import { useUi } from "./ui.ts";

// Dev/debug handle: inspect the live state from the console (window.__inkmap.getState()).
// The generation and presentation stores are exposed the same way for the browser suite.
Object.assign(window as unknown as Record<string, unknown>, { __inkmap: useStore, __inkmapUi: useUi });
if (import.meta.env.DEV) {
  const logs: string[] = ((window as unknown as { __logs: string[] }).__logs = []);
  for (const k of ["info", "warn", "error"] as const) {
    const orig = console[k].bind(console);
    console[k] = (...a: unknown[]) => { logs.push(`${k}: ${a.map((x) => (x instanceof Error ? x.stack ?? x.message : String(x))).join(" ")}`); orig(...a); };
  }
  window.addEventListener("error", (e) => logs.push(`uncaught: ${e.message}`));
  window.addEventListener("unhandledrejection", (e) => logs.push(`unhandled: ${String((e as PromiseRejectionEvent).reason)}`));
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
