import { before, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { anchorToPoint, faceCentroids, frameAt, type Anchor } from "../src/core/anchor.ts";
import { AtlasIndex, parseAtlas } from "../src/core/atlas.ts";
import { BODY_SPEC, MODEL_SPEC_ID, buildSkin } from "../src/core/body.ts";
import { uprightRotation } from "../src/core/landmarks.ts";

// The rest surface is a T-pose, palms down: arms lie along x, so left and right, inner and outer, and up all differ
// from a body standing with its arms down. Each claim here is about where the skin is, not what it is called.
let index: AtlasIndex;
let rest: THREE.BufferGeometry;
let centroids: Float32Array;

before(async () => {
  const bytes = readFileSync(new URL(`../public/${BODY_SPEC.path}`, import.meta.url));
  const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
  const skin = buildSkin(gltf.scene);
  rest = skin.geometry;
  centroids = faceCentroids(rest);
  const atlas = parseAtlas(JSON.parse(readFileSync(new URL(`../public/bodies/${MODEL_SPEC_ID}.regions.json`, import.meta.url), "utf8")), centroids.length / 3);
  index = new AtlasIndex(atlas, skin.geometry, skin.centroids);
});

const centre = (face: number) => new THREE.Vector3().fromArray(centroids, 3 * face);
const SIDES = [["left", 1], ["right", -1]] as const;
const site = (id: string, laterality: "left" | "right", aspect: string | null = null, level: "upper" | "lower" | null = null) =>
  ({ id, laterality, aspect, level });
const placed = (anchor: Anchor) => anchorToPoint(rest, anchor);
const centreOf = (face: number): Anchor => ({ face, barycentric: [1 / 3, 1 / 3, 1 / 3] });

for (const [side, sign] of SIDES) {
  test(`${side} arm: outer is the lateral face of the upper arm and the back of the forearm, inner the other side`, () => {
    for (const id of ["bicep", "tricep", "forearm"]) {
      const outer = placed(index.anchorFor(site(id, side, "outer")));
      const inner = placed(index.anchorFor(site(id, side, "inner")));
      assert.ok(outer.n.z > 0.25, `outer ${id} faces out, up on the rest surface: ${outer.n.z}`);
      assert.ok(inner.n.z < -0.25, `inner ${id} faces the body, down on the rest surface: ${inner.n.z}`);
      assert.ok(outer.p.z > inner.p.z);
    }
  });

  test(`${side} arm: the outer bicep is the middle of the upper arm, nowhere near the armpit`, () => {
    const { p } = placed(index.anchorFor(site("bicep", side, "outer")));
    assert.ok(Math.abs(p.x) > 0.27 && Math.abs(p.x) < 0.37, `along the arm: ${p.x}`);
    assert.ok(p.x * sign > 0);
  });

  test(`${side} arm: upper is toward the shoulder on every long arm site`, () => {
    for (const id of ["bicep", "tricep", "forearm"]) {
      const upper = placed(index.anchorFor(site(id, side, null, "upper"))).p;
      const lower = placed(index.anchorFor(site(id, side, null, "lower"))).p;
      assert.ok(Math.abs(upper.x) < Math.abs(lower.x) - 0.02, `${id}: upper ${upper.x}, lower ${lower.x}`);
    }
    assert.equal(index.describe(index.anchorFor(site("bicep", side, "outer", "upper")))?.level, "upper");
  });

  test(`${side} arm: above and below follow the limb, as they do on a hanging arm`, () => {
    const elbow = placed(index.anchorFor(site("elbow", side))).p;
    const walk = (kind: "above" | "below") => index.anchorForPhraseDetailed({ ...site("elbow", side), relation: { kind, offset_m: 0.0508, render: "two inches" } });
    const above = walk("above"), below = walk("below");
    assert.ok(Math.abs(placed(above.anchor).p.x) < Math.abs(elbow.x) - 0.03, "above is toward the shoulder");
    assert.ok(Math.abs(placed(below.anchor).p.x) > Math.abs(elbow.x) + 0.03, "below is toward the wrist");
    assert.ok(Math.abs(above.relative!.achieved_m - 0.0508) < 0.012, `walked ${above.relative!.achieved_m} of 50.8 mm`);
  });

  test(`${side} arm: upright is along the limb all the way round it, with no jump at the outer line`, () => {
    const shoulderward = new THREE.Vector3(-sign, 0, 0);
    for (let face = 0; face < centroids.length / 3; face++) {
      const anchor = centreOf(face), ref = index.regionOf(face), x = centre(face).x * sign;
      if (!ref || ref.laterality !== side || !["bicep", "tricep", "forearm"].includes(ref.id) || x < 0.28 || !index.isValidAnchor(anchor)) continue;
      const { n, v: top } = frameAt(rest, anchor, uprightRotation(index, rest, anchor));
      // Where a limb flares its skin leans off the axis, so the top can only point as far along it as the skin allows.
      const along = shoulderward.clone().addScaledVector(n, -shoulderward.dot(n)).normalize();
      assert.ok(top.dot(along) > 0.99, `${ref.id} face ${face}: its top is ${top.dot(along).toFixed(3)} of the way along the skin to the shoulder`);
    }
  });
}

test("everywhere off the arm, up is the body's and upright needs no turn, exactly", () => {
  for (let face = 0; face < centroids.length / 3; face += 7) {
    const ref = index.regionOf(face);
    if (!ref || ["shoulder_cap", "bicep", "tricep", "elbow", "elbow_ditch", "forearm", "wrist", "hand", "palm", "fingers", "knuckles", "thumb", "shoulder_blade"].includes(ref.id)) continue;
    const up = index.upAt(face);
    assert.ok(up.x === 0 && up.y === 0 && up.z === 1, `${ref.id} face ${face}`);
    assert.equal(uprightRotation(index, rest, centreOf(face)), 0);
  }
});

test("up turns smoothly over the shoulder rather than switching", () => {
  const up = (x: number) => { let best = 0, gap = Infinity; for (let f = 0; f < centroids.length / 3; f++) { const ref = index.regionOf(f); if (ref?.laterality === "left" && ["shoulder_cap", "bicep"].includes(ref.id) && Math.abs(centre(f).x - x) < gap) { gap = Math.abs(centre(f).x - x); best = f; } } return index.upAt(best); };
  const steps = [0.2, 0.215, 0.23, 0.245, 0.26, 0.275].map(up);
  for (let i = 1; i < steps.length; i++) assert.ok(steps[i].angleTo(steps[i - 1]) < 0.6, `${i}: ${steps[i].angleTo(steps[i - 1])} rad between neighbours`);
  assert.ok(up(0.19).z > 0.99 && up(0.3).x < -0.99);
});

test("legs keep their inner and outer: the thigh's outer side still faces away from the midline", () => {
  for (const [side, sign] of SIDES) {
    const outer = placed(index.anchorFor(site("thigh", side, "outer"))), inner = placed(index.anchorFor(site("thigh", side, "inner")));
    assert.ok(outer.n.x * sign > 0.35 && inner.n.x * sign < -0.35);
    assert.ok(placed(index.anchorFor(site("shin", side, "outer"))).n.x * sign > 0.35);
  }
});
