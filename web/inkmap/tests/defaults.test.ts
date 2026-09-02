import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DEFAULT_BODY_ID, DEFAULT_POSE_ID, DEFAULT_SKIN_TONE, SKIN_TONES } from "../src/core/defaults.ts";

const catalog = JSON.parse(
  readFileSync(new URL("../../../config/inkmap/body-poses.json", import.meta.url), "utf8"),
) as { pose_ids: string[]; bodies: Record<string, { poses: Record<string, { label: string }> }> };

test("editor defaults to the female body, neutral standing pose, and middle natural tone", () => {
  assert.equal(DEFAULT_BODY_ID, "hbm-female-stylized");
  assert.equal(DEFAULT_POSE_ID, "standing-neutral");
  assert.equal(DEFAULT_SKIN_TONE, SKIN_TONES[Math.floor(SKIN_TONES.length / 2)]);
});

test("pose labels stay within two words", () => {
  for (const body of Object.values(catalog.bodies)) {
    for (const poseId of catalog.pose_ids) {
      assert.match(body.poses[poseId].label, /^\S+(?:\s+\S+)?$/);
    }
  }
});
