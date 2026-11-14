import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateArtRecord } from "../src/core/art.ts";
import { stockRecord } from "../src/core/design-assets.ts";

const root = new URL("../public/designs/", import.meta.url);
const entry = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8")).designs[0];
const bytes = new Uint8Array(readFileSync(new URL(entry.image.replace(/^designs\//, ""), root)));

test("a stock image freezes as a source-art record with the collection's art box and inks", async () => {
  const record = await stockRecord(entry, bytes);
  assert.deepEqual(await validateArtRecord(record), record);
  assert.deepEqual([record.name, record.art_px, record.palette, record.source], [entry.name, entry.art_px, entry.palette, entry.source]);
  assert.equal(record.image.media_type, "image/jpeg");
});

test("changed bytes and unsupported paths fail without a fallback", async () => {
  const changed = bytes.slice(); changed[changed.length - 3] ^= 1;
  await assert.rejects(stockRecord(entry, changed), /digest_mismatch/);
  await assert.rejects(stockRecord({ ...entry, image: "designs/coffee-cup.svg" }, bytes), /design_asset_unsupported/);
  await assert.rejects(stockRecord({ ...entry, image: "../secret.jpg" }, bytes), /design_asset_unsupported/);
});
