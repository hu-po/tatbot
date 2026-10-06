import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DEFAULT_POSE_ID, DEFAULT_SKIN_TONE, SKIN_TONES } from "../src/core/defaults.ts";

const catalog = JSON.parse(
  readFileSync(new URL("../../../config/inkmap/body-poses.json", import.meta.url), "utf8"),
) as { pose_ids: string[]; poses: Record<string, { label: string }> };

test("editor defaults to the neutral SOMA pose and middle natural tone", () => {
  assert.equal(DEFAULT_POSE_ID, "standing-neutral");
  assert.equal(DEFAULT_SKIN_TONE, SKIN_TONES[Math.floor(SKIN_TONES.length / 2)]);
});

test("pose labels are short operator-facing descriptions", () => {
  for (const poseId of catalog.pose_ids) {
    assert.ok(catalog.poses[poseId].label.split(/\s+/).length <= 5, poseId);
  }
});
