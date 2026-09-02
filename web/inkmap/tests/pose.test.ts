import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as cloneSkeleton } from "three/examples/jsm/utils/SkeletonUtils.js";
import { BODIES, buildPosedSkin } from "../src/core/body.ts";

if (typeof ProgressEvent === "undefined") {
  Object.defineProperty(globalThis, "ProgressEvent", {
    value: class ProgressEvent {
      type: string;
      constructor(type: string, init: object = {}) { this.type = type; Object.assign(this, init); }
    },
  });
}

const catalog = JSON.parse(readFileSync(new URL("../../../config/inkmap/body-poses.json", import.meta.url), "utf8"));

async function loadGlb(path: URL) {
  const bytes = readFileSync(path);
  const data = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  return await new Promise<any>((resolve, reject) => new GLTFLoader().parse(data, "", resolve, reject));
}

test("Three.js reproduces Blender-authored pose samples within 0.1 mm", async () => {
  for (const spec of BODIES) {
    const gltf = await loadGlb(new URL(`../public/${spec.rigPath}`, import.meta.url));
    const body = catalog.bodies[spec.id];
    assert.equal(body.rigged_path, spec.rigPath);
    for (const poseId of catalog.pose_ids) {
      const pose = body.poses[poseId];
      const skin = buildPosedSkin(cloneSkeleton(gltf.scene), spec, pose.joint_rotations, pose.body_rotation_xyzw);
      const positions = skin.geometry.getAttribute("position");
      let maxError = 0;
      body.validation_vertex_indices.forEach((vertex: number, sample: number) => {
        const expected = pose.validation_vertices[sample];
        const error = Math.hypot(
          positions.getX(vertex) - expected[0],
          positions.getY(vertex) - expected[1],
          positions.getZ(vertex) - expected[2],
        );
        maxError = Math.max(maxError, error);
      });
      assert.ok(maxError <= 1e-4, `${spec.id}/${poseId}: max error ${maxError * 1000} mm`);
      if (poseId.startsWith("reclined")) {
        assert.ok(pose.anatomy["bend_offset_m:hip.L-knee.L-ankle.L"] >= 0.045);
        assert.ok(pose.anatomy["bend_offset_m:hip.R-knee.R-ankle.R"] >= 0.045);
        assert.ok(pose.anatomy["bend_off_axis_m:hip.L-knee.L-ankle.L"] <= 0.01);
        assert.ok(pose.anatomy["bend_off_axis_m:hip.R-knee.R-ankle.R"] <= 0.01);
      }
      if (poseId.endsWith("left-arm-supported")) {
        assert.ok(pose.anatomy["angle_deg:shoulder.L-elbow.L-wrist.L"] >= 110);
        assert.ok(pose.anatomy["angle_deg:shoulder.L-elbow.L-wrist.L"] <= 145);
        assert.ok(pose.anatomy["angle_deg:elbow.L-wrist.L-hand_tip.L"] >= 170);
      }
      if (poseId.endsWith("right-arm-supported")) {
        assert.ok(pose.anatomy["angle_deg:shoulder.R-elbow.R-wrist.R"] >= 110);
        assert.ok(pose.anatomy["angle_deg:shoulder.R-elbow.R-wrist.R"] <= 145);
        assert.ok(pose.anatomy["angle_deg:elbow.R-wrist.R-hand_tip.R"] >= 170);
      }
      skin.geometry.dispose();
    }
  }
});
