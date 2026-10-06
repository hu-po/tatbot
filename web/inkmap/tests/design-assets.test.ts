import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { renderTattooProgramSvg } from "../src/core/human-representation/program-svg.ts";
import { freezeDesign } from "../src/core/design-assets.ts";

const root = new URL("../public/designs/", import.meta.url);
const input = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8")).designs[0];
const bytes = readFileSync(new URL("dbv3-orbit/artwork.json", root));
test("acquired artwork freezes identical display/export geometry and provenance", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(bytes);
  try {
    const design = await freezeDesign(input);
    assert.equal(decodeURIComponent(design.path.split(",")[1]), renderTattooProgramSvg(design.embedded!.program));
    assert.equal(design.embedded!.conversion.adapter, "dbv3-batik-paths/1");
    globalThis.fetch = async () => { throw new Error("offline"); };
    assert.deepEqual(await freezeDesign(design), design);
  } finally { globalThis.fetch = original; }
});
test("missing, changed and legacy assets fail without a tracer fallback", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response("missing", { status: 404 });
    await assert.rejects(freezeDesign(input), /design_asset_unavailable/);
    await assert.rejects(freezeDesign({ ...input, path: "designs/coffee-cup.svg" }), /design_asset_unsupported/);
    globalThis.fetch = async () => new Response(bytes);
    await assert.rejects(freezeDesign({ ...input, sha256: "0".repeat(64) }), /digest_mismatch/);
  } finally { globalThis.fetch = original; }
});
