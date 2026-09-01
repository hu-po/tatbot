// Design generation: the inkgen service (web/inkgen) draws, the browser
// traces. The service holds the prompt blurb; this file only knows the API.
import { binarise, cropToInk, inkCoverage, type Pixels, type TraceOptions } from "./trace.ts";
import type { TraceReply, TraceRequest } from "./trace.worker.ts";
import { loadConfig } from "./config.ts";

/** Attempts in order; each runs in a fresh worker because a vtracer panic poisons its wasm. */
const TRACE_LADDER: TraceOptions[] = [{}, { lengthThreshold: 8 }, { mode: "polygon" }, { mode: "polygon", filterSpeckle: 16 }];

function traceOnce(bin: Pixels, opts: TraceOptions): Promise<string> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./trace.worker.ts", import.meta.url), { type: "module" });
    const done = (f: () => void) => { w.terminate(); f(); };
    w.onmessage = (e: MessageEvent<TraceReply>) => done(() => (e.data.ok ? resolve(e.data.svg) : reject(new Error(e.data.error))));
    w.onerror = (e) => done(() => reject(new Error(e.message || "trace worker crashed")));
    w.postMessage({ bin, opts } satisfies TraceRequest);
  });
}

/** Trace with fallbacks: spline, coarser spline, then polygons. */
export async function traceRobust(bin: Pixels): Promise<string> {
  let last: Error | null = null;
  for (const opts of TRACE_LADDER) {
    try { return await traceOnce(bin, opts); } catch (e) { last = e as Error; }
  }
  throw new Error(`could not trace the drawing: ${last?.message ?? "unknown"}`);
}

export interface Health {
  ok: boolean; model: string; device: string; zero_gpu: boolean; steps: number; size: number;
  turnstile: boolean; budget_s: number; budget_spent_s: number; uptime_s: number;
}

export interface Generated {
  prompt: string;
  seed: number;
  /** The raster the model drew, for showing what came back. */
  pngUrl: string;
  /** The traced design. */
  svg: string;
  width: number;
  height: number;
  coverage: number;
  seconds: number;
}

export async function health(): Promise<Health> {
  const { api } = await loadConfig();
  const r = await fetch(`${api}/api/health`, { signal: AbortSignal.timeout(20_000) });
  if (!r.ok) throw new Error(`generator ${api}: HTTP ${r.status}`);
  return (await r.json()) as Health;
}

async function blobToPixels(blob: Blob): Promise<Pixels> {
  const bmp = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width; canvas.height = bmp.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
  return { data: img.data, width: img.width, height: img.height };
}

/** Ask the service for one raster, then trace it to a design. `style` is an
 *  optional prompt fragment (from the inklang style slots) that replaces the
 *  generator's default look. */
export async function generateDesign(subject: string, seed: number | undefined, turnstile: string | undefined, signal?: AbortSignal, style?: string): Promise<Generated> {
  const { api } = await loadConfig();
  const r = await fetch(`${api}/api/generate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject, seed, turnstile, style }), signal,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = ((await r.json()) as { error?: string }).error ?? msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  const body = (await r.json()) as { png_base64: string; seed: number; prompt: string; seconds: number };
  const blob = new Blob([Uint8Array.from(atob(body.png_base64), (c) => c.charCodeAt(0))], { type: "image/png" });
  const usedSeed = body.seed;
  const prompt = body.prompt;
  const seconds = body.seconds;
  const px = await blobToPixels(blob);
  const bin = binarise(px);
  const coverage = inkCoverage(bin);
  if (coverage < 0.005 || coverage > 0.9) throw new Error("the model drew nothing usable — try another seed or subject");
  const { svg, width, height } = cropToInk(await traceRobust(bin), bin);
  return { prompt, seed: usedSeed, pngUrl: URL.createObjectURL(blob), svg, width, height, coverage, seconds };
}

export function svgDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}
