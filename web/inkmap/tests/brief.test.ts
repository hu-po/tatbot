import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020.js";
import { makeBrief, validateBrief } from "../src/core/brief.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import type { PlacementFile } from "../src/core/schema.ts";

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
const file = read("../../../config/inkmap/examples/forearm-placement-v7.json") as PlacementFile;
const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
for (const name of ["art", "placement"]) ajv.addSchema(read(`../../../config/inkmap/${name}.schema.json`));
const schema = ajv.compile(read("../../../config/inkmap/brief.schema.json"));

test("a brief carries each placement's art and the width to separate it at", async () => {
  const placed = structuredClone(file);
  placed.placements[0].size_mm = [41.27, 41.27];
  const brief = await makeBrief("Forearm ring", placed);
  assert.equal(brief.schema, "tatbot.inkmap-brief/1");
  assert.deepEqual(Object.keys(brief.art), ["ring"]);
  assert.equal(brief.placements[0].width_mm, 41.3, "the width to 0.1 mm, the pipeline's 10 px/mm");
  assert.equal(brief.placements[0].art_id, "ring");
  assert.deepEqual(brief.placements[0].anchor, placed.placements[0].anchor);
  assert.equal(schema(brief), true, ajv.errorsText(schema.errors));
  assert.deepEqual(await validateBrief(JSON.parse(JSON.stringify(brief))), brief);
});

test("a tampered or incomplete brief is refused", async () => {
  const brief = await makeBrief("Forearm ring", structuredClone(file));
  for (const [mutate, pattern] of [
    [(b: any) => { b.name = "Changed"; }, /content digest/],
    [(b: any) => { b.placements[0].width_mm = 41.27; b.content_sha256 = ""; }, /to 0.1 mm/],
    [(b: any) => { b.art.extra = b.art.ring; b.content_sha256 = ""; }, /exactly the placed art/],
    [(b: any) => { b.placements = []; b.content_sha256 = ""; }, /at least one placement/],
  ] as const) {
    const changed = structuredClone(brief) as any; mutate(changed);
    if (changed.content_sha256 === "") changed.content_sha256 = await canonicalDigest(changed);
    await assert.rejects(validateBrief(changed), pattern);
  }
});
