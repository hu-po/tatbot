// Where the design generator lives. Configuration, never code: the Space, a
// laptop and a garage node run the same app and differ only here.
//   1. ?api=<url> on the page URL (also remembered), 2. VITE_INKMAP_API at build/dev time
//   (`tatbot inkmap dev --api …`), 3. localStorage "inkmap.api", 4. config.json next to
//   the app (written by the deploy; public/config.json in dev), 5. the hosted default.
export const DEFAULT_API = "https://hu-po-inkgen.hf.space";

export interface AppConfig {
  api: string;
  /** Cloudflare Turnstile site key; present only when the generator verifies tokens. */
  turnstileSiteKey?: string;
}

let loaded: Promise<AppConfig> | null = null;

export function loadConfig(): Promise<AppConfig> {
  if (!loaded) loaded = resolve();
  return loaded;
}

async function resolve(): Promise<AppConfig> {
  let file: Partial<AppConfig> = {};
  try {
    const r = await fetch("config.json", { cache: "no-store" });
    if (r.ok) file = (await r.json()) as Partial<AppConfig>;
  } catch { /* no config.json: fine */ }
  const fromEnv = (import.meta.env.VITE_INKMAP_API as string | undefined) || "";
  let api = file.api || DEFAULT_API;
  try {
    const q = new URLSearchParams(location.search).get("api");
    if (q) { api = q; localStorage.setItem("inkmap.api", q); }
    else if (fromEnv) api = fromEnv;
    else { const s = localStorage.getItem("inkmap.api"); if (s) api = s; }
  } catch { if (fromEnv) api = fromEnv; }
  return { ...file, api: api.replace(/\/+$/, "") };
}

export function apiSource(cfg: AppConfig): "url" | "env" | "saved" | "file" | "default" {
  try {
    if (new URLSearchParams(location.search).get("api")) return "url";
    if (import.meta.env.VITE_INKMAP_API) return "env";
    if (localStorage.getItem("inkmap.api")) return "saved";
  } catch { /* ignore */ }
  return cfg.api === DEFAULT_API ? "default" : "file";
}
