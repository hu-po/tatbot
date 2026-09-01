import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { binarise, cropToInk, initTracer, inkCoverage, otsu, traceSvg, type Pixels } from "../src/core/trace.ts";

function disc(size = 128, r = 40, ink = 20, paper = 235): Pixels {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const inside = (x - size / 2) ** 2 + (y - size / 2) ** 2 < r * r;
    const v = inside ? ink : paper;
    const i = 4 * (y * size + x);
    data[i] = data[i + 1] = data[i + 2] = v; data[i + 3] = 255;
  }
  return { data, width: size, height: size };
}

test("otsu splits a bimodal histogram between the modes", () => {
  const t = otsu(new Uint8ClampedArray([...Array(500).fill(20), ...Array(500).fill(235)]));
  assert.ok(t >= 20 && t < 235, `threshold ${t}`);
});

test("binarise yields pure black/white with the disc's coverage", () => {
  const bin = binarise(disc());
  const vals = new Set(Array.from(bin.data.filter((_, i) => i % 4 === 0)));
  assert.deepEqual([...vals].sort(), [0, 255]);
  const cov = inkCoverage(bin);
  assert.ok(Math.abs(cov - Math.PI * 40 * 40 / (128 * 128)) < 0.01, `coverage ${cov}`);
});

test("vtracer wasm traces the disc to one closed path and crop fits it", async () => {
  await initTracer(async () => readFileSync(new URL("../src/vendor/vectortracer/vectortracer_bg.wasm", import.meta.url)));
  const bin = binarise(disc());
  const svg = traceSvg(bin);
  assert.match(svg, /<svg/);
  assert.equal((svg.match(/<path/g) ?? []).length, 1);
  assert.match(svg, /fill="#111111"/);
  const { svg: cropped, width, height } = cropToInk(svg, bin);
  assert.ok(width >= 80 && width <= 100 && height >= 80 && height <= 100, `${width}x${height}`);
  // The attributes must land inside the <svg> tag (vtracer writes "<svg ... >" with a space before ">").
  assert.match(cropped, /<svg[^>]*\sviewBox="\d+ \d+ \d+ \d+"[^>]*>/);
  assert.doesNotMatch(cropped, />\s*width=/);
  assert.doesNotMatch(cropped, /background/);
});
