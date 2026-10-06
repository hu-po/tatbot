/** Hermetic stdin/stdout adapter; the browser imports the exact same module. */
import { readFileSync } from "node:fs";
import { SvgDOMParser } from "./svg-dom.ts";
import { tattooProgramFromSvg } from "../src/core/human-representation/svg-program.ts";
import { canonicalJson, parseJsonStrict } from "../src/core/human-representation/schema.ts";
import { tattooProgramToSvg } from "../src/core/human-representation/program-svg.ts";
import { validateArtworkRecord } from "../src/core/artwork-record.ts";
import { makeSimulationBundle, validateSimulationBundle } from "../src/core/sim-bundle.ts";
import { artworkFromSvg } from "./artwork-import.ts";
import type { AtlasData } from "../src/core/atlas.ts";
import { traceRaster } from "./trace-raster.ts";
import { makeDesign, validateDesign } from "../src/core/design.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
try {
  const input = readFileSync(0, "utf8");
  if (Buffer.byteLength(input) > 30_000_000) throw new Error("artwork request exceeds 30 MB");
  const request = parseJsonStrict(input) as Record<string, unknown>;
  if (request.operation !== undefined && !["trace_raster", "materialize_record", "validate_record", "materialize_bundle", "validate_bundle", "materialize_design", "validate_design"].includes(String(request.operation))) throw new Error("unknown artwork operation");
  // The caller cannot supply an alternate atlas path, URL, or body provider.
  const atlas = () => parseJsonStrict(readFileSync(new URL("../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8")) as unknown as AtlasData;
  const result = request.operation === "materialize_design"
    ? await makeDesign(request.input as Parameters<typeof makeDesign>[0])
    : request.operation === "validate_design"
    ? await validateDesign(request.design)
    : request.operation === "trace_raster"
    ? await traceRaster(request as unknown as Parameters<typeof traceRaster>[0])
    : request.operation === "materialize_bundle"
    ? await makeSimulationBundle(request.file as Parameters<typeof makeSimulationBundle>[0], atlas(), request.request as Parameters<typeof makeSimulationBundle>[2], request.sources as Parameters<typeof makeSimulationBundle>[3])
    : request.operation === "validate_bundle"
    ? await validateSimulationBundle(request.bundle, atlas())
    : request.operation === "materialize_record"
    ? await artworkFromSvg(request.input as Parameters<typeof artworkFromSvg>[0])
    : request.operation === "validate_record"
    ? await validateArtworkRecord(request.record)
    : request.program
    ? { svg: await tattooProgramToSvg(request.program as Parameters<typeof tattooProgramToSvg>[0]) }
    : await tattooProgramFromSvg(request.svg as string, request.options as Parameters<typeof tattooProgramFromSvg>[1]);
  process.stdout.write(canonicalJson(result) + "\n");
} catch (error) {
  const failure = error as Error & { code?: string; path?: string };
  process.stderr.write(JSON.stringify({ code: failure.code ?? "tattoo_program_unsupported", path: failure.path ?? "$.svg", detail: failure.message }) + "\n");
  process.exitCode = 2;
}
