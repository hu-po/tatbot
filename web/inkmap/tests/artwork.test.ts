import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { tattooProgramFromSvg } from "../src/core/human-representation/svg-program.ts";
import { tattooProgramToSvg } from "../src/core/human-representation/program-svg.ts";
import type { TattooProgram } from "../src/core/human-representation/tattoo-program.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const fixture = (name: string) => readFileSync(new URL(`./fixtures/artwork/${name}.svg`, import.meta.url), "utf8");
const importSvg = (svg: string) => {
  const box = svg.match(/viewBox="([^"]+)"/)![1].split(/\s+/).map(Number);
  return tattooProgramFromSvg(svg, { canvas_m: [box[2] / 1000, box[3] / 1000], semantic_intent: "acceptance fixture" });
};

function colorAt(program: TattooProgram, x: number, y: number): number[] | null {
  let result: number[] | null = null;
  for (const layer of program.layers) {
    for (const element of layer.elements) {
      const polygon = element.points_m!;
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const a = polygon[i], b = polygon[j];
        if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
      }
      if (inside) result = program.inks.find(ink => ink.id === layer.ink_id)!.color_srgb;
    }
  }
  return result;
}

test("filled interiors, transparent holes, dots, and overlapping inks retain their visual meaning", async () => {
  const solid = await importSvg(fixture("blackwork"));
  assert.ok(colorAt(solid, 0.03, 0.03));
  const ring = await importSvg(fixture("negative-space"));
  assert.equal(colorAt(ring, 0.03, 0.03), null);
  assert.ok(colorAt(ring, 0.05, 0.03));
  assert.equal(colorAt(ring, 0.001, 0.001), null);
  const dots = await importSvg(fixture("stipple"));
  assert.ok(colorAt(dots, 0.01, 0.03));
  assert.equal(colorAt(dots, 0.015, 0.03), null);
  const colors = await importSvg(fixture("color-layers"));
  assert.deepEqual(colorAt(colors, 0.01, 0.02), [201 / 255, 40 / 255, 40 / 255]);
  assert.deepEqual(colorAt(colors, 0.04, 0.02), [36 / 255, 89 / 255, 201 / 255]);
});

test("all five fixed fixtures produce deterministic metric programmes and review SVG", async () => {
  for (const name of ["linework", "blackwork", "negative-space", "stipple", "color-layers"]) {
    const left = await importSvg(fixture(name));
    assert.deepEqual(left, await importSvg(fixture(name)));
    const svg = await tattooProgramToSvg(left);
    assert.match(svg, /fill-rule="nonzero"/);
    assert.ok(left.layers.every(layer => layer.elements.every(element => element.fill)));
  }
});

test("invalid and unsupported SVG operations refuse instead of silently changing the artwork", async () => {
  const cases = [
    '<image href="https://example.invalid/x.png"/>', '<text x="2" y="3">ink</text>',
    '<rect width="5" height="5" filter="url(#blur)"/>', '<path d="M1 1 L5 5" stroke="url(#paint)"/>',
    '<rect width="5" height="5" style="mix-blend-mode:multiply"/>',
    '<rect width="5" height="5" opacity="0.5"/>', '<use href="#x"/>',
    '<g transform="scale(2 1)"><path d="M1 1 L3 4" fill="none" stroke="black"/></g>',
    '<rect width="5" height="5" fill="not-a-color"/>', '<path d="M1 1 L3 4" fill="none"/>',
    '<rect width="5" height="5"></g>',
    '<g transform="unknown(2)"><rect width="5" height="5"/></g>',
    '<path d="M1 1 L2"/>', '<path d="M1 1 X2 3"/>', '<circle cx="5" cy="5" r="-2"/>',
  ];
  for (const inner of cases) {
    await assert.rejects(importSvg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">${inner}</svg>`), /tattoo_program_unsupported/);
  }
  await assert.rejects(tattooProgramFromSvg(fixture("blackwork"), { canvas_m: [0.06, 0.06], semantic_intent: "x",
    provenance: { source_sha256: "0".repeat(64) } }), /wrong_hash/);
});

test("global negative-space masks render transparency, not white ink", async () => {
  const program = JSON.parse(readFileSync(new URL("../../../config/human-representation/examples/blackwork/program.json", import.meta.url), "utf8"));
  const svg = await tattooProgramToSvg(program);
  assert.match(svg, /mask="url\(#negative-space\)"/);
  assert.doesNotMatch(svg, /data-mask=/);
});

test("centerline mode keeps a stroked path as one open line per subpath, drawn once at the planning width", async () => {
  // A 10 mm horizontal line, a closed 4 mm square, and a skewed line that the
  // outline mode refuses; the artist's 2-unit stroke width must not survive.
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="#111" stroke-width="2">'
    + '<path d="M2 10 L12 10 M14 4 L18 4 L18 8 L14 8 Z"/>'
    + '<g transform="scale(2 1)"><path d="M1 15 L5 18"/></g></svg>';
  const options = { canvas_m: [0.02, 0.02] as [number, number], semantic_intent: "lines", width_m: 0.0005 };
  const program = await tattooProgramFromSvg(svg, { ...options, strokes: "centerline" });
  const elements = program.layers.flatMap(layer => layer.elements);
  assert.deepEqual(elements.map(e => [e.kind, e.fill, e.closed, e.width_m]),
    [["path", false, false, 0.0005], ["path", false, true, 0.0005], ["path", false, false, 0.0005]]);
  assert.deepEqual(elements[0].points_m, [[0.002, 0.01], [0.012, 0.01]]);
  assert.equal(elements[1].points_m!.length, 4);
  assert.deepEqual(elements[2].points_m, [[0.002, 0.005], [0.01, 0.002]]);
  assert.match(await tattooProgramToSvg(program), /<polyline /);
  const outline = await tattooProgramFromSvg(svg.replace('<g transform="scale(2 1)"><path d="M1 15 L5 18"/></g>', ""), options);
  assert.ok(outline.layers.every(layer => layer.elements.every(element => element.kind === "region" && element.fill)));
  await assert.rejects(tattooProgramFromSvg(svg, { ...options, strokes: "bogus" as never }), /unsupported stroke mode/);
});


test("metric preview scales dot centres without growing dots or altering canonical artwork", async () => {
  const program = JSON.parse(readFileSync(new URL("../../../config/human-representation/examples/linework/program.json", import.meta.url), "utf8"));
  const line = program.layers[0].elements[0];
  program.layers[0].elements = [{ ...line, id: "dot", kind: "dots", points_m: [[.03, .02]], width_m: .0005 }];
  program.content_sha256 = await canonicalDigest(program);
  const canonical = await tattooProgramToSvg(program);
  for (const scale of [.4, 1, 2]) {
    const doc = new SvgDOMParser().parseFromString(await tattooProgramToSvg(program, [.06 * scale, .06 * scale]), "image/svg+xml");
    const circle = doc.getElementsByTagName("circle")[0];
    assert.equal(Number(circle.getAttribute("r")), .00025);
    assert.ok(Math.abs(Number(circle.getAttribute("cx")) - .03 * scale) < 1e-12);
    assert.ok(Math.abs(Number(circle.getAttribute("cy")) - .04 * scale) < 1e-12);
  }
  assert.equal(await tattooProgramToSvg(program), canonical);
  for (const width of [0, -1, Infinity, NaN]) await assert.rejects(tattooProgramToSvg(program, [width, .06]), /positive finite metres/);
});
