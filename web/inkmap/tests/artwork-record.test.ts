import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { artworkNeedsRegeneration, artworkSizeM, validateArtworkRecord, type ArtworkRecord } from "../src/core/artwork-record.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";

import { artworkFromSvg } from "../tools/artwork-import.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const names = ["linework", "blackwork", "negative-space", "stipple", "color-layers"];
async function fixture(name = "blackwork") {
  const svg = readFileSync(new URL(`./fixtures/artwork/${name}.svg`, import.meta.url), "utf8");
  const box = svg.match(/viewBox="([^"]+)"/)![1].split(/\s+/).map(Number);
  return artworkFromSvg({ name, original_svg: svg,
    source: { kind: "fixture", identifier: name, license: "CC0-1.0", attribution: "Tatbot", generation: null },
    conversion: { adapter: "tatbot-svg-paint/1", canvas_m: [box[2] / 1000, box[3] / 1000], semantic_intent: name,
      width_m: 0.0008, deposition: 0.7, chord_error_m: 0.0001 } });
}

test("all five portable artwork records round-trip, freeze their inputs, and reproduce", async () => {
  for (const name of names) {
    const record = await fixture(name);
    assert.deepEqual(await validateArtworkRecord(JSON.parse(JSON.stringify(record))), record);
    assert.deepEqual(await fixture(name), record);
    const clone = await validateArtworkRecord(record);
    clone.source.license = null;
    assert.equal(record.source.license, "CC0-1.0");
    assert.equal(record.source_sha256, record.program.preview_sha256);
  }
});

test("rehashed outer records still verify the program digest and source binding", async () => {
  for (const mutate of [
    (r: ArtworkRecord) => { r.source_sha256 = "0".repeat(64); },
    (r: ArtworkRecord) => { r.program.layers[0].elements[0].width_m = .001; },
  ]) {
    const record = await fixture(); mutate(record);
    record.content_sha256 = await canonicalDigest(record);
    await assert.rejects(validateArtworkRecord(record), /wrong_hash/);
  }
});

test("a newly acquired program validates without reconstructing source SVG", async () => {
  const record = await fixture();
  record.program.layers[0].elements[0].width_m = .001;
  record.program.content_sha256 = await canonicalDigest(record.program);
  record.conversion = { adapter: "dbv3-batik-paths/1", recipe_sha256: "a".repeat(64), chord_error_m: .000005 };
  record.content_sha256 = await canonicalDigest(record);
  assert.deepEqual(await validateArtworkRecord(record), record);
  assert.ok(!("original_svg" in record) && !("preview_svg" in record));
});

test("unknown fields, unsupported versions, incomplete generation provenance, and resource limits refuse", async () => {
  const initial = await fixture();
  for (const change of [
    (r: Record<string, any>) => { r.extra = true; },
    (r: Record<string, any>) => { r.source.license_url = "https://example.invalid"; },
    (r: Record<string, any>) => { r.schema = "tatbot.inkmap-artwork/99"; },
    (r: Record<string, any>) => { r.source.kind = "generated"; },
    (r: Record<string, any>) => { r.conversion.chord_error_m = 0.01; },
    (r: Record<string, any>) => { r.original_svg = "obsolete source"; },
  ]) {
    const broken = structuredClone(initial); change(broken);
    broken.content_sha256 = await canonicalDigest(broken);
    await assert.rejects(validateArtworkRecord(broken), /artwork_record_invalid/);
  }
});

test("public stroke mode is resolved at import and survives as frozen geometry", async () => {
  const record = await fixture("linework");
  const svg = readFileSync(new URL("./fixtures/artwork/linework.svg", import.meta.url), "utf8");
  const lines = await artworkFromSvg({ name: "linework", original_svg: svg, source: record.source,
    conversion: { canvas_m: [.03, .03], semantic_intent: "linework", width_m: .0005, strokes: "centerline" } });
  assert.notEqual(lines.content_sha256, record.content_sha256);
  assert.ok(lines.program.layers.some(layer => layer.elements.some(element => element.kind === "path" && !element.fill)));
  assert.deepEqual(await validateArtworkRecord(lines), lines);
});


test("physical recipe size changes require regeneration and undo restores the acquired size", async () => {
  const record = await fixture("linework");
  record.conversion.recipe_sha256 = "a".repeat(64);
  const size = artworkSizeM(record).map(m => m * 1000) as [number, number];
  assert.equal(artworkNeedsRegeneration(record, size), false);
  assert.equal(artworkNeedsRegeneration(record, [size[0] * 2, size[1]]), true);
  assert.equal(artworkNeedsRegeneration(record, [size[0], size[1] * 2]), true);
  assert.equal(artworkNeedsRegeneration(record, size), false);
});
