/** The stock art collection (public/designs/manifest.json, `tatbot.inkmap-art-collection/1`): each entry an
 *  image whose bytes the manifest pins, with its art rectangle as the art pipeline measures it. */
import { bytesToBase64, makeArtRecord, sha256Bytes, validateArtRecord, type ArtPalette, type ArtRecord, type ArtSource } from "./art.ts";
import { artSizeMm } from "./art.ts";
import { artThumbnail } from "./art-image.ts";
import { validateRender, type InkmapRender } from "./render-record.ts";
import type { DesignMeta } from "./schema.ts";

export interface StockArt {
  id: string;
  name: string;
  image: string;
  sha256: string;
  width_px: number;
  height_px: number;
  art_px: [number, number, number, number];
  palette: ArtPalette | null;
  source: ArtSource;
  /** Robot renders of the design, one per width, fetched once it is placed. */
  renders?: string[];
}

export const COLLECTION_SCHEMA = "tatbot.inkmap-art-collection/1";

/** A library entry for a record: its thumbnail and its size at the default width. */
export async function libraryEntry(id: string, record: ArtRecord, extra: Partial<DesignMeta> = {}): Promise<DesignMeta> {
  return { id, name: record.name, path: await artThumbnail(record), default_size_mm: artSizeMm(record), embedded: record, ...extra };
}

/** A stock entry's record from its image bytes, which must be the bytes the manifest pins. */
export async function stockRecord(entry: StockArt, bytes: Uint8Array): Promise<ArtRecord> {
  if (!/^designs\/[a-z0-9-]+\.(jpg|png)$/.test(entry.image)) throw new Error(`design_asset_unsupported: ${entry.id}`);
  if (await sha256Bytes(bytes) !== entry.sha256) throw new Error(`design_asset_digest_mismatch: ${entry.id}`);
  return makeArtRecord({
    name: entry.name, art_px: entry.art_px, source: entry.source, palette: entry.palette,
    image: { media_type: entry.image.endsWith(".png") ? "image/png" : "image/jpeg", sha256: entry.sha256,
      width_px: entry.width_px, height_px: entry.height_px, data_base64: bytesToBase64(bytes) },
  });
}

/** One stock entry: its image fetched and checked against the manifest, then frozen as a record. */
export async function loadStockArt(entry: StockArt, signal?: AbortSignal): Promise<DesignMeta> {
  if (!/^designs\/[a-z0-9-]+\.(jpg|png)$/.test(entry.image)) throw new Error(`design_asset_unsupported: ${entry.id}`);
  const response = await fetch(entry.image, { signal });
  if (!response.ok) throw new Error(`design_asset_unavailable: ${entry.id} (HTTP ${response.status})`);
  return libraryEntry(entry.id, await stockRecord(entry, new Uint8Array(await response.arrayBuffer())), { sourcePath: entry.image, renderPaths: entry.renders });
}

/** A render the collection ships. */
export async function loadStockRender(path: string): Promise<InkmapRender> {
  if (!/^designs\/renders\/[a-z0-9.-]+\.json$/.test(path)) throw new Error(`design_asset_unsupported: ${path}`);
  const response = await fetch(path);
  if (!response.ok) throw new Error(`design_asset_unavailable: ${path} (HTTP ${response.status})`);
  return validateRender(await response.json());
}

/** Re-admit a held entry (a restored project's art): validate its record, rebuild its thumbnail. */
export async function refreezeDesign(design: DesignMeta): Promise<DesignMeta> {
  if (!design.embedded) throw new Error(`design_asset_missing: ${design.id}`);
  const record = await validateArtRecord(design.embedded);
  return { ...design, name: record.name, path: await artThumbnail(record), embedded: record };
}
