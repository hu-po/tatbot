import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { REST_ASSET_SHA256 } from "../src/core/body.ts";
import { POSE_CATALOG } from "../src/core/pose.ts";

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
const manifest = read("../public/showcase/manifest.json");
const collection = read("../public/designs/manifest.json");

test("the showcase places collection artwork on one SOMA body in all five tattoo-session poses", () => {
  assert.equal(manifest.schema_version, 3);
  assert.equal(manifest.validation.placements, manifest.slides.length);
  assert.equal(manifest.validation.body_asset_sha256, REST_ASSET_SHA256);
  assert.equal(manifest.validation.pose_asset_sha256, POSE_CATALOG.pose_asset.sha256);
  assert.equal(manifest.validation.visual_review, "pending");
  assert.deepEqual(manifest.coverage, { bodies: 1, poses: 5, sites: 4, designs: 3 });

  const poses = new Set<string>();
  const designs = new Set<string>();
  for (const slide of manifest.slides) {
    assert.ok(POSE_CATALOG.pose_ids.includes(slide.pose_id), slide.id);
    assert.equal(slide.placement.design_id, slide.artwork_id, slide.id);
    const entry = collection.designs.find((design: { id: string }) => design.id === slide.artwork_id);
    assert.ok(entry, `${slide.id}: ${slide.artwork_id} is in the artwork collection`);
    const artwork = read(`../public/${entry.path}`);
    assert.equal(artwork.source_sha256, slide.artwork_sha256, `${slide.id}: the slide names the collection's bytes`);
    assert.equal(slide.placement.anchor.barycentric.length, 3);
    assert.ok(Number.isInteger(slide.placement.anchor.face));
    poses.add(slide.pose_id);
    designs.add(slide.artwork_id);
  }
  assert.equal(designs.size, manifest.coverage.designs);
  assert.deepEqual([...poses].sort(), [
    "prone",
    "reclined-left-arm-supported",
    "reclined-right-arm-supported",
    "reclined-seated",
    "supine",
  ]);
});
