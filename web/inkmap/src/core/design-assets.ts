import type { DesignMeta } from "./schema.ts";
import { sha256Hex } from "./sha256.ts";
import { parseJsonStrict } from "./human-representation/schema.ts";
import { artworkSizeM, requireAcquiredArtwork, validateArtworkRecord, type ArtworkSource } from "./artwork-record.ts";
import { renderTattooProgramSvg } from "./human-representation/program-svg.ts";

export function artworkSources(designs: DesignMeta[]): Record<string, ArtworkSource> {
  return Object.fromEntries(designs.filter(design => design.embedded).map(design => [design.id, design.embedded!.source]));
}

/** Load immutable acquired JSON. Preview SVG is derived only from the frozen program. */
export async function freezeDesign(design: DesignMeta, signal?: AbortSignal): Promise<DesignMeta> {
  let record = design.embedded;
  if (record) record = await validateArtworkRecord(record);
  else {
    if (!/^designs\/dbv3-[a-z0-9-]+\/artwork\.json$/.test(design.path)) throw new Error(`design_asset_unsupported: ${design.id}; import DBV3 artwork.json`);
    const response = await fetch(design.path, { signal });
    if (!response.ok) throw new Error(`design_asset_unavailable: ${design.id} (HTTP ${response.status})`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > 20_000_000) throw new Error(`design_asset_over_budget: ${design.id}`);
    const digest = await sha256Hex(bytes);
    if (!design.sha256 || design.sha256 !== digest) throw new Error(`design_asset_digest_mismatch: ${design.id}`);
    record = await validateArtworkRecord(parseJsonStrict(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
  }
  requireAcquiredArtwork(record);
  return { ...design, name: record.name, default_size_mm: artworkSizeM(record).map(m => m * 1000) as [number, number],
    sourcePath: design.sourcePath ?? design.path, sourceSha256: record.source_sha256,
    path: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(renderTattooProgramSvg(record.program))}`, embedded: record };
}
