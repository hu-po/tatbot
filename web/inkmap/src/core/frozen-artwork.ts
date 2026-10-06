/** Admit the one shared artwork record for every placement target. */
import { validateArtworkRecord, type ArtworkRecord, type ArtworkSource } from "./artwork-record.ts";
import { canonicalJson } from "./human-representation/schema.ts";

export async function frozenArtwork(id: string, artwork: ArtworkRecord, source?: ArtworkSource): Promise<ArtworkRecord> {
  const record = await validateArtworkRecord(artwork);
  if (source && canonicalJson(source) !== canonicalJson(record.source)) {
    throw new Error(`artwork_source_mismatch: ${id}`);
  }
  return record;
}
