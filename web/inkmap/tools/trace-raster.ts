/** Batch entrypoint into the browser's exact raster-to-painted-SVG converter. */
import { readFileSync } from "node:fs";
import { binarise, cropToInk, inkCoverage, initTracer, luminance, otsu, resetTracer,
  traceSvg, TRACE_ALGORITHM, TRACE_LADDER } from "../src/core/trace.ts";

export async function traceRaster(request: { width: number; height: number; rgba_base64: string }) {
  const { width, height } = request;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1
      || width * height > 4_194_304) throw new Error("raster exceeds pixel budget");
  const data = new Uint8ClampedArray(Buffer.from(request.rgba_base64, "base64"));
  if (data.length !== width * height * 4) throw new Error("RGBA byte count differs");
  const pixels = { data, width, height };
  const threshold = otsu(luminance(pixels));
  const bin = binarise(pixels, threshold);
  const coverage = inkCoverage(bin);
  if (coverage < .005 || coverage > .9) throw new Error("ink coverage outside usable 0.5%-90% range");
  let last: unknown;
  for (const options of TRACE_LADDER) {
    resetTracer();
    try {
      await initTracer(async () => readFileSync(new URL("../src/vendor/vectortracer/vectortracer_bg.wasm", import.meta.url)));
      const cropped = cropToInk(traceSvg(bin, options), bin);
      return { ...cropped, trace: { algorithm: TRACE_ALGORITHM, threshold, coverage,
        source_px: [width, height], size_px: [cropped.width, cropped.height], options } };
    } catch (error) { last = error; }
  }
  throw new Error(`raster tracing refused: ${String(last)}`);
}
