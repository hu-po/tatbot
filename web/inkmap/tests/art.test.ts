import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { artBox, artSizeMm, imageDimensions, inkOnly, isPaper, jpegOrientation, makeArtRecord, maxWidthMm, validateArtRecord, type ArtRecord } from "../src/core/art.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";

const example = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v7.json", import.meta.url), "utf8"));
const ring = example.designs.ring as ArtRecord;
const collection = JSON.parse(readFileSync(new URL("../public/designs/manifest.json", import.meta.url), "utf8"));

/** An RGBA image of white with `ink` pixels set to a colour. */
function image(width: number, height: number, ink: [number, number, [number, number, number]][]): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4).fill(255);
  for (const [x, y, [r, g, b]] of ink) rgba.set([r, g, b, 255], 4 * (y * width + x));
  return rgba;
}

test("a record's image bytes, size and digest hold together", async () => {
  assert.deepEqual(await validateArtRecord(ring), ring);
  for (const mutate of [
    (r: any) => { r.name = "Changed"; },
    (r: any) => { r.image.sha256 = "0".repeat(64); r.content_sha256 = ""; },
    (r: any) => { r.image.width_px = 65; },
    (r: any) => { r.art_px = [0, 0, 65, 10]; },
    (r: any) => { r.schema = "tatbot.inkmap-artwork/2"; },
    (r: any) => { r.program = {}; },
    (r: any) => { r.palette = { inks: ["Bright Red"], style: null }; },
  ]) {
    const changed = structuredClone(ring) as any; mutate(changed);
    if (changed.content_sha256 === "") changed.content_sha256 = await canonicalDigest(changed);
    await assert.rejects(validateArtRecord(changed));
  }
});

test("pixel sizes come from the PNG and JPEG headers", () => {
  assert.deepEqual(imageDimensions(new Uint8Array(readFileSync(new URL("./fixtures/art-ring.png", import.meta.url))), "image/png"), [64, 64]);
  for (const entry of collection.designs) {
    const bytes = new Uint8Array(readFileSync(new URL(`../public/${entry.image}`, import.meta.url)));
    assert.deepEqual(imageDimensions(bytes, "image/jpeg"), [entry.width_px, entry.height_px], entry.id);
  }
  assert.throws(() => imageDimensions(new Uint8Array([1, 2, 3]), "image/png"), /not a PNG/);
});

test("paper is light and neutral; the art is the ink's box padded by 3% of the longer side, halves to even", () => {
  assert.ok(isPaper(255, 255, 255) && isPaper(240, 238, 232));
  assert.ok(!isPaper(250, 210, 80), "a light yellow is ink");
  assert.ok(!isPaper(30, 30, 30));
  // 50 px: a 1.5 px margin rounds to 2 (even); paper-coloured and near-white pixels are not ink.
  const rgba = image(50, 40, [[10, 12, [200, 30, 30]], [30, 20, [20, 20, 20]], [45, 35, [240, 238, 232]], [5, 5, [252, 252, 252]]]);
  assert.deepEqual(artBox(rgba, 50, 40), [8, 10, 25, 13]);
  assert.deepEqual(artBox(image(100, 10, [[50, 5, [0, 0, 0]]]), 100, 10), [47, 2, 7, 7], "a 3 px margin, clamped to the image");
  assert.throws(() => artBox(image(8, 8, []), 8, 8), /every pixel is paper/);
  const shown = inkOnly(rgba);
  assert.equal(shown[4 * (12 * 50 + 10) + 3], 255);
  assert.equal(shown[4 * (35 * 50 + 45) + 3], 0, "paper shows the skin through");
});

test("a placed size keeps the art's aspect and snaps the width to 0.1 mm", async () => {
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 865, 888] }, 60), [60, 61.595]);
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 100, 50] }, 42.04), [42, 21]);
  const made = await makeArtRecord({ ...structuredClone(ring), name: "Ring again" } as Omit<ArtRecord, "schema" | "content_sha256">);
  assert.equal(made.content_sha256, await canonicalDigest(made));
  assert.notEqual(made.content_sha256, ring.content_sha256);
});

test("a design starts at 60 mm wide, and a tall one starts short enough to stay on a limb", () => {
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 100, 100] }), [60, 60]);
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 800, 100] }), [60, 7.5]);
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 100, 800] }), [15, 120]);
  assert.deepEqual(artSizeMm({ art_px: [0, 0, 10, 1000] }), [10, 1000], "the smallest width holds, however long that makes it");
});

test("the size limit holds on the longer side, either way round", () => {
  assert.equal(maxWidthMm([100, 100]), 300);
  assert.equal(maxWidthMm([800, 100]), 300);
  assert.equal(maxWidthMm([100, 800]), 37.5);
  assert.equal(maxWidthMm([1, 1000]), 10, "never under the smallest width");
});

test("a JPEG's EXIF turn is read, and a file without one asks for none", () => {
  const read = (name: string) => new Uint8Array(readFileSync(new URL(name, import.meta.url)));
  assert.equal(jpegOrientation(read("fixtures/art-exif6.jpg")), 6);
  assert.equal(jpegOrientation(read("../public/designs/swallow-poppies.jpg")), 1);
  assert.equal(jpegOrientation(read("fixtures/art-exif6.jpg").subarray(0, 30)), 1, "a truncated header asks for nothing");
});
