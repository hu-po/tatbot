import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { requireAcquiredArtwork, validateArtworkRecord } from "../src/core/artwork-record.ts";

const root = new URL("../public/designs/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
test("every catalogue ID is a native acquired example with portable recipe and pen evidence", async () => {
  assert.deepEqual(manifest.designs.map((d: any) => d.id), ["dbv3-orbit", "dbv3-sprout", "dbv3-ridges"]);
  assert.ok(!readdirSync(root).some(name => name.endsWith(".svg")), "retired finished assets remain");
  for (const design of manifest.designs) {
    assert.equal(design.library, undefined);
    const dir = new URL(`${design.id}/`, root);
    const bytes = readFileSync(new URL("artwork.json", dir));
    assert.equal(design.sha256, hash(bytes));
    const artwork = await validateArtworkRecord(JSON.parse(bytes.toString()));
    requireAcquiredArtwork(artwork);
    const recipeBytes = readFileSync(new URL("recipe/recipe.json", dir));
    const recipe = JSON.parse(recipeBytes.toString());
    assert.equal(artwork.conversion.recipe_sha256, hash(recipeBytes));
    assert.equal(recipe.schema, "tatbot.dbv3-recipe/3");
    assert.equal(recipe.variant.schema, "tatbot.dbv3-job/3");
    assert.deepEqual(recipe.effective.drawing.size_mm, design.default_size_mm);
    assert.equal(recipe.variant.source.sha256, artwork.source_sha256);
    for (const [file, digest] of Object.entries(recipe.files)) assert.equal(hash(readFileSync(new URL(`recipe/${file}`, dir))), digest);
    assert.ok(recipe.effective.drawing_set.pens.length);
    const receipt = JSON.parse(readFileSync(new URL("result.json", dir), "utf8"));
    for (const [file, digest] of Object.entries(receipt.outputs)) assert.equal(hash(readFileSync(new URL(file, dir))), digest);
  }
});
