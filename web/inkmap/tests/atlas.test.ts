import { before, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { AtlasIndex, parseAtlas, validateExclusionMask, type AtlasData } from "../src/core/atlas.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  MODEL_SPEC_SHA256,
  REFERENCE_IDENTITY_SHA256,
  REST_SURFACE_SHA256,
  TOPOLOGY_SHA256,
  buildSkin,
} from "../src/core/body.ts";
import { POSE_CATALOG } from "../src/core/pose.ts";
import { SITES } from "../src/core/inklang/index.ts";

let atlas: AtlasData;
let index: AtlasIndex;

before(async () => {
  const bytes = readFileSync(new URL(`../public/${BODY_SPEC.path}`, import.meta.url));
  const gltf = await new GLTFLoader().parseAsync(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    "",
  );
  const skin = buildSkin(gltf.scene);
  atlas = parseAtlas(
    JSON.parse(readFileSync(new URL(`../public/bodies/${MODEL_SPEC_ID}.regions.json`, import.meta.url), "utf8")),
    skin.centroids.length / 3,
  );
  index = new AtlasIndex(atlas, skin.geometry, skin.centroids);
});

test("the atlas is bound to the one reviewed MHR/SOMA surface", () => {
  assert.deepEqual(atlas.body, {
    model_spec_id: MODEL_SPEC_ID,
    model_spec_sha256: MODEL_SPEC_SHA256,
    identity_sha256: REFERENCE_IDENTITY_SHA256,
    topology_sha256: TOPOLOGY_SHA256,
    rest_surface_sha256: REST_SURFACE_SHA256,
    asset_sha256: atlas.body.asset_sha256,
  });
  assert.equal(atlas.sites.length, 59);
  assert.equal(atlas.region_meaning_count, 99);
  assert.equal(Object.keys(atlas.site_status).length, 59);
  assert.equal(atlas.faces.length, 36_108);
  assert.equal(atlas.eligible_faces.length, 36_108);
});

test("every mapped skin site can produce placement anchors", () => {
  const supported = Object.entries(atlas.site_status)
    .filter(([, record]) => record.status === "mapped_supported")
    .map(([id]) => id)
    .sort();
  const mapped = Object.entries(atlas.site_status)
    .filter(([, record]) => record.face_count > 0)
    .map(([id]) => id)
    .sort();
  assert.deepEqual(supported, mapped);
  assert.equal(supported.length, 56);
  for (const id of ["forearm", "chest", "sternum", "scalp", "palm"]) assert.ok(supported.includes(id), id);

  for (const id of supported) {
    let resolved = 0;
    for (const laterality of SITES[id].laterality === "sided"
      ? (["left", "right"] as const)
      : ([null] as const)) {
      // A sided site may be mapped on one side only (behind_ear has a single face).
      if (laterality && index.facesOf(id, laterality).length === 0) continue;
      const anchor = index.anchorFor({ id, laterality });
      assert.ok(index.isValidAnchor(anchor), `${laterality ?? "center"} ${id}`);
      assert.ok(index.contains({ id, laterality }, anchor), `${laterality ?? "center"} ${id}`);
      resolved += 1;
    }
    assert.ok(resolved > 0, id);
  }

  for (const [id, status] of Object.entries(atlas.site_status)) {
    if (status.status === "mapped_supported") continue;
    const laterality = SITES[id].laterality === "sided" ? "left" as const : null;
    assert.throws(() => index.anchorFor({ id, laterality }), /no faces/);
  }
});

test("every labelled skin face is eligible; only the not-skin segments are denied", () => {
  let labeledUnsupported = 0;
  let eligible = 0;
  for (let face = 0; face < atlas.faces.length; face += 1) {
    if (atlas.eligible_faces[face] === 1) {
      eligible += 1;
      const described = index.describe({ face, barycentric: [1 / 3, 1 / 3, 1 / 3] });
      assert.ok(described, `eligible face ${face} has no semantic label`);
      assert.equal(atlas.site_status[described!.id].status, "mapped_supported");
      assert.ok(index.isValidAnchor({ face, barycentric: [1 / 3, 1 / 3, 1 / 3] }));
    } else if (index.describe({ face, barycentric: [1 / 3, 1 / 3, 1 / 3] })) {
      labeledUnsupported += 1;
    }
  }
  assert.equal(eligible, 34_606);
  assert.equal(eligible + atlas.upstream_exclusions.excluded_faces, atlas.faces.length);
  assert.equal(labeledUnsupported, 0);
});

test("laterality and proximal-to-distal levels are stable on eligible limbs", () => {
  const left = index.anchorFor({ id: "forearm", laterality: "left" });
  const right = index.anchorFor({ id: "forearm", laterality: "right" });
  assert.notEqual(left.face, right.face);
  assert.equal(index.describe(left)!.laterality, "left");
  assert.equal(index.describe(right)!.laterality, "right");

  const upper = index.anchorFor({ id: "forearm", laterality: "left", level: "upper" });
  const lower = index.anchorFor({ id: "forearm", laterality: "left", level: "lower" });
  assert.ok(index.uvOf(upper.face)![0] < index.uvOf(lower.face)![0]);
});

test("the upstream exclusion mask can never yield an eligible face", () => {
  assert.equal(atlas.upstream_exclusions.excluded_faces, 1_502);
  assert.match(atlas.upstream_exclusions.asset_sha256, /^[0-9a-f]{64}$/);
  for (let face = 0; face < atlas.faces.length; face += 1) {
    if (atlas.faces[face] < 0) assert.equal(atlas.eligible_faces[face], 0);
  }
  const bytes = readFileSync(new URL(`../public/${POSE_CATALOG.exclusion_asset.path}`, import.meta.url));
  validateExclusionMask(atlas, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const corrupted = Uint8Array.from(bytes);
  const firstExcluded = corrupted.findIndex((value) => value === 1);
  const faceCode = atlas.faces[firstExcluded];
  atlas.faces[firstExcluded] = 0;
  assert.throws(
    () => validateExclusionMask(atlas, corrupted.buffer),
    /remains labeled or eligible/,
  );
  atlas.faces[firstExcluded] = faceCode;
});
