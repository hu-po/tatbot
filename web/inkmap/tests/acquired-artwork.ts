import { readFileSync } from "node:fs";
import { validateArtworkRecord } from "../src/core/artwork-record.ts";

/** Actual native acquisitions, used unchanged by editor acceptance tests. */
export async function acquiredArtwork(id = "dbv3-orbit") {
  return validateArtworkRecord(JSON.parse(readFileSync(new URL(`../public/designs/${id}/artwork.json`, import.meta.url), "utf8")));
}
