// One trace per worker: vtracer can panic (wasm `unreachable`) on some shapes,
// and a panic poisons the module, so the main thread spawns a fresh worker per
// attempt and terminates it afterwards. Nothing here touches the DOM.
import { initTracer, traceSvg, type Pixels, type TraceOptions } from "./trace.ts";
import wasmUrl from "../vendor/vectortracer/vectortracer_bg.wasm?url";

export interface TraceRequest { bin: Pixels; opts: TraceOptions }
export type TraceReply = { ok: true; svg: string } | { ok: false; error: string };

self.onmessage = async (e: MessageEvent<TraceRequest>) => {
  try {
    // wasmUrl is a data: URL (see vite.config.ts); read it to bytes rather than streaming,
    // so no network request and no MIME requirement.
    await initTracer(async () => {
      const r = await fetch(wasmUrl);
      if (!r.ok) throw new Error(`tracer wasm: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    const svg = traceSvg(e.data.bin, e.data.opts);
    (self as unknown as Worker).postMessage({ ok: true, svg } satisfies TraceReply);
  } catch (err) {
    (self as unknown as Worker).postMessage({ ok: false, error: (err as Error).message ?? String(err) } satisfies TraceReply);
  }
};
