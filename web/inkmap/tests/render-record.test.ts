import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020.js";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import { findRender, renderAtOtherWidth, validateRender, type InkmapRender } from "../src/core/render-record.ts";

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
// Written by tatbot_contracts.art.make_render, which scripts/tests/test_art_brief.py reads back: the two readers agree.
const fixture = read("fixtures/render.json") as InkmapRender;
const resealed = async (change: (render: InkmapRender) => void) => {
  const render = structuredClone(fixture);
  change(render);
  render.content_sha256 = await canonicalDigest(render);
  return render;
};

test("a render from the pipeline holds together, against its schema too", async () => {
  assert.deepEqual(await validateRender(fixture), fixture);
  assert.ok(new Ajv2020({ strict: false }).compile(read("../../../config/inkmap/render.schema.json"))(fixture));
});

test("a changed, short or foreign record is refused", async () => {
  await assert.rejects(validateRender({ ...fixture, width_mm: 7 }), /content digest differs/);
  await assert.rejects(validateRender({ schema: "tatbot.inkmap-brief/1" }), /not a robot render/);
  await assert.rejects(validateRender(await resealed((r) => { r.image.sha256 = "0".repeat(64); })), /differ from their digest/);
  await assert.rejects(validateRender(await resealed((r) => { r.image.width_px = 61; })), /size differs/);
  await assert.rejects(validateRender(await resealed((r) => { (r as unknown as { strokes: [] }).strokes = []; })), /unknown fields/);
});

test("a render shows only for the art, width and mirror it was made for", async () => {
  const art = fixture.art_sha256, renders = [fixture];
  assert.equal(findRender(renders, art, 6, false), fixture);
  assert.equal(findRender(renders, art, 6.04, false), fixture, "within the 0.1 mm a width is kept to");
  assert.equal(findRender(renders, art, 6.1, false), null);
  assert.equal(findRender(renders, art, 6, true), null, "a mirrored design was separated mirrored");
  assert.equal(findRender(renders, "b".repeat(64), 6, false), null, "new inks are new art");
  assert.equal(renderAtOtherWidth(renders, art, 7, false), fixture);
  assert.equal(renderAtOtherWidth(renders, art, 6, false), null);
  const wider = { ...fixture, width_mm: 60 };
  assert.equal(renderAtOtherWidth([fixture, wider], art, 50, false), wider, "the nearest other width");
});
