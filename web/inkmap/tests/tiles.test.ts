import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DEFAULT_TILE_ID, fitWidthMm, overflowMm, STENCIL_TILES } from "../src/core/tiles.ts";

const tile = (id: string) => STENCIL_TILES.find((item) => item.id === id)!;

test("the tiles are the generator's layouts, and the default is a skin tile", () => {
  const generator = readFileSync(new URL("../../../scripts/lib/stencil_tile.py", import.meta.url), "utf8");
  assert.match(generator, /config\/stencil-tiles\.json/, "the tile generator reads the same file");
  assert.deepEqual(STENCIL_TILES.map((item) => item.id), ["forearm", "forearm18", "silicone"]);
  assert.deepEqual(tile("forearm").size_mm, [45, 63]);
  assert.equal(tile(DEFAULT_TILE_ID).for, "skin");
});

test("a design fits a tile by its own width and height", () => {
  assert.deepEqual(overflowMm(tile("forearm"), [45, 63]), [0, 0]);
  assert.deepEqual(overflowMm(tile("forearm"), [60, 61.595]), [15, 0]);
  assert.deepEqual(overflowMm(tile("forearm18"), [60, 70]), [20, 8]);
});

test("the width that fits keeps the design's aspect and rounds down to 0.1 mm", () => {
  assert.equal(fitWidthMm(tile("forearm"), [60, 61.595]), 45);
  assert.equal(fitWidthMm(tile("forearm"), [60, 120]), 31.5, "a tall design is held by the tile's height");
  const [width, height] = [fitWidthMm(tile("forearm18"), [37, 100]), 100 / 37];
  assert.ok(width * height <= 62, "rounding down never leaves the tile");
});
