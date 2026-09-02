import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const root = new URL("../public/designs/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8")) as {
  designs: Array<{ id: string; name: string; path: string; default_size_mm: [number, number] }>;
};

test("the bundled picker contains the six reviewed vector designs", () => {
  assert.deepEqual(manifest.designs.map((design) => design.id), [
    "octopus",
    "lunar-moth",
    "peony",
    "koi",
    "swallow",
    "tiger",
  ]);

  for (const design of manifest.designs) {
    assert.match(design.path, new RegExp(`^designs/${design.id}\\.svg$`));
    assert.ok(design.name.length > 0);
    assert.ok(design.default_size_mm.every((value) => value > 0));

    const svg = readFileSync(new URL(`${design.id}.svg`, root), "utf8");
    assert.match(svg, /^<svg\b/);
    assert.match(svg, /<path\b/);
    assert.doesNotMatch(svg, /<(?:image|script|foreignObject)\b/i);

    const viewBox = svg.match(/\bviewBox="[-.\d]+\s+[-.\d]+\s+([.\d]+)\s+([.\d]+)"/);
    assert.ok(viewBox, `${design.id}: SVG needs a numeric viewBox`);
    const svgAspect = Number(viewBox[1]) / Number(viewBox[2]);
    const sizeAspect = design.default_size_mm[0] / design.default_size_mm[1];
    assert.ok(Math.abs(svgAspect - sizeAspect) < 0.02, `${design.id}: default size must preserve aspect`);
  }
});
