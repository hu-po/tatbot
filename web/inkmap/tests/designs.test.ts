import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { artSizeMm, imageDimensions } from "../src/core/art.ts";
import { COLLECTION_SCHEMA, stockRecord } from "../src/core/design-assets.ts";
import { validateRender } from "../src/core/render-record.ts";
import { STENCIL_TILES, fitWidthMm } from "../src/core/tiles.ts";

const root = new URL("../public/designs/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");

test("the collection is three source images, each with its art box and inks", async () => {
  assert.equal(manifest.schema, COLLECTION_SCHEMA);
  assert.deepEqual(manifest.designs.map((d: any) => d.id), ["swallow-poppies", "sun-and-moon", "snake-and-dagger"]);
  for (const entry of manifest.designs) {
    const bytes = readFileSync(new URL(entry.image.replace(/^designs\//, ""), root));
    assert.equal(entry.sha256, hash(bytes), entry.id);
    assert.deepEqual(imageDimensions(new Uint8Array(bytes), "image/jpeg"), [entry.width_px, entry.height_px]);
    const record = await stockRecord(entry, new Uint8Array(bytes));
    assert.deepEqual(record.art_px, entry.art_px);
    assert.ok(entry.palette.inks.length >= 3, `${entry.id} is multicolour`);
    assert.equal(entry.source.license, null);
  }
});

test("each design ships a render at its default width and at the skin tile's, for the art a visitor places", async () => {
  const tile = STENCIL_TILES.find((item) => item.id === "forearm")!;
  for (const entry of manifest.designs) {
    const record = await stockRecord(entry, new Uint8Array(readFileSync(new URL(entry.image.replace(/^designs\//, ""), root))));
    const renders = await Promise.all(entry.renders.map((path: string) => validateRender(JSON.parse(readFileSync(new URL(path.replace(/^designs\//, ""), root), "utf8")))));
    for (const render of renders) {
      assert.equal(render.art_sha256, record.content_sha256, `${entry.id}: run \`tatbot art stock-renders\`, this render is of other art`);
      assert.equal(render.mirror, false);
    }
    const size = artSizeMm(record);
    assert.deepEqual(renders.map((render) => render.width_mm).sort((a, b) => a - b), [fitWidthMm(tile, size), size[0]], entry.id);
  }
});

test("the public collection holds images, pictures of them and their notice: no stroke artwork ships", () => {
  const names = (readdirSync(root, { recursive: true }) as string[]).filter((name) => name !== "renders");
  const others = names.filter((name) => !/^[a-z0-9-]+\.jpg$/.test(name) && !/^renders\/[a-z0-9.-]+\.json$/.test(name));
  assert.deepEqual(others.sort(), ["LICENSE.txt", "manifest.json"]);
  for (const name of names.filter((item) => item.startsWith("renders/"))) {
    const { image: _image, ...fields } = JSON.parse(readFileSync(new URL(name, root), "utf8"));
    assert.doesNotMatch(JSON.stringify(fields), /points|program|strokes/, `${name} is a picture and its figures, never paths`);
  }
  const license = readFileSync(new URL("LICENSE.txt", root), "utf8");
  assert.match(license, /not dedicated to the\s+public domain/);
  assert.doesNotMatch(JSON.stringify(manifest), /dbv3|tattoo-program|recipe/i);
});
