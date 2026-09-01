// Raster → tattoo SVG. A generated picture becomes a design only after it has
// been forced to black ink on white and traced to closed vector paths: that
// is what the placement file stores, what the picker shows, and what a robot
// could draw. No DOM here; the caller supplies pixels and the wasm bytes.
import * as bg from "../vendor/vectortracer/vectortracer_bg.js";

export interface Pixels { data: Uint8ClampedArray; width: number; height: number }

/** Otsu's threshold over 8-bit luminance: the split that minimises intra-class variance. */
export function otsu(gray: Uint8ClampedArray | Uint8Array): number {
  const hist = new Float64Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const total = gray.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, best = 0, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; threshold = t; }
  }
  return threshold;
}

export function luminance(px: Pixels): Uint8ClampedArray {
  const out = new Uint8ClampedArray(px.width * px.height);
  const d = px.data;
  for (let i = 0, j = 0; i < d.length; i += 4, j++) out[j] = (d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000;
  return out;
}

/** Ink = at or below the threshold (Otsu puts the lower class at <= t). Returns RGBA: black ink on opaque white. */
export function binarise(px: Pixels, threshold = otsu(luminance(px))): Pixels {
  const lum = luminance(px);
  const data = new Uint8ClampedArray(px.width * px.height * 4);
  for (let j = 0; j < lum.length; j++) {
    const v = lum[j] <= threshold ? 0 : 255;
    data[4 * j] = data[4 * j + 1] = data[4 * j + 2] = v;
    data[4 * j + 3] = 255;
  }
  return { data, width: px.width, height: px.height };
}

/** Fraction of ink pixels; a sanity gate (blank or solid black outputs are not designs). */
export function inkCoverage(bin: Pixels): number {
  let ink = 0;
  for (let i = 0; i < bin.data.length; i += 4) if (bin.data[i] === 0) ink++;
  return ink / (bin.width * bin.height);
}

export interface TraceOptions {
  /** Curve fitting; "polygon" is the robust fallback when spline fitting panics on a shape. */
  mode?: "spline" | "polygon";
  /** Drop connected blobs smaller than this many pixels (dust from the raster). */
  filterSpeckle?: number;
  cornerThreshold?: number;
  lengthThreshold?: number;
  spliceThreshold?: number;
  pathPrecision?: number;
  fill?: string;
}

let ready: Promise<void> | null = null;

/**
 * Instantiate the vtracer wasm once. `source` is whatever `WebAssembly.instantiate`
 * accepts: a Response (browser, streaming) or bytes (node tests).
 */
export function initTracer(source: () => Promise<Response | BufferSource>): Promise<void> {
  if (!ready) {
    ready = (async () => {
      const src = await source();
      const imports = { "./vectortracer_bg.js": bg } as WebAssembly.Imports;
      const result = typeof Response !== "undefined" && src instanceof Response
        ? await WebAssembly.instantiateStreaming(src, imports)
        : await WebAssembly.instantiate(src as BufferSource, imports);
      const instance = "instance" in result ? result.instance : result;
      bg.__wbg_set_wasm(instance.exports);
      (instance.exports as { __wbindgen_start?: () => void }).__wbindgen_start?.();
    })().catch((e) => { ready = null; throw e; });
  }
  return ready;
}

/** Trace a binarised image (black ink on white) to an SVG string. Call initTracer first. */
export function traceSvg(bin: Pixels, o: TraceOptions = {}): string {
  const conv = new bg.BinaryImageConverter(
    bin as unknown as ImageData,
    {
      debug: false, mode: o.mode ?? "spline",
      cornerThreshold: o.cornerThreshold ?? 60, lengthThreshold: o.lengthThreshold ?? 4,
      maxIterations: 10, spliceThreshold: o.spliceThreshold ?? 45,
      filterSpeckle: o.filterSpeckle ?? 6, pathPrecision: o.pathPrecision ?? 2,
    },
    { invert: false, pathFill: o.fill ?? "#111111", backgroundColor: undefined, attributes: undefined, scale: 1 },
  );
  try {
    conv.init();
    while (!conv.tick()) { /* synchronous; images are ≤ 768² */ }
    return conv.getResult();
  } catch (e) {
    // A panic inside the wasm leaves the object borrowed; free() would then throw a
    // misleading "recursive use of an object" and hide the real error.
    poisoned = true;
    throw new Error(`vtracer failed (${(e as Error).message || e}) — mode ${o.mode ?? "spline"}`);
  } finally {
    if (!poisoned) conv.free();
  }
}

/** True after a wasm panic: the module must be re-instantiated before the next trace. */
let poisoned = false;
export function tracerPoisoned(): boolean { return poisoned; }
export function resetTracer(): void { poisoned = false; ready = null; }

/** Crop the SVG's viewBox to the ink (plus margin) so a design's aspect is the drawing's, not the canvas's. */
export function cropToInk(svg: string, bin: Pixels, marginPx = 8): { svg: string; width: number; height: number } {
  let x0 = bin.width, y0 = bin.height, x1 = -1, y1 = -1;
  for (let y = 0; y < bin.height; y++) for (let x = 0; x < bin.width; x++) {
    if (bin.data[4 * (y * bin.width + x)] === 0) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return { svg, width: bin.width, height: bin.height };
  x0 = Math.max(0, x0 - marginPx); y0 = Math.max(0, y0 - marginPx);
  x1 = Math.min(bin.width - 1, x1 + marginPx); y1 = Math.min(bin.height - 1, y1 + marginPx);
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  // vtracer paints a white page background into the tag; a design must be ink on nothing.
  const out = svg.replace(/<svg\b[^>]*>/, (tag) =>
    tag.replace(/\s(width|height|viewBox|style)="[^"]*"/g, "").replace(/\s*>$/, ` width="${w}" height="${h}" viewBox="${x0} ${y0} ${w} ${h}">`));
  return { svg: out, width: w, height: h };
}
