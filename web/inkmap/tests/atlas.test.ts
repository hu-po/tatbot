import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { buildSkin, BODIES, type BodySpec } from "../src/core/body.ts";
import { parseAtlas, AtlasIndex } from "../src/core/atlas.ts";
import { SITES, ZONES, parseSentence, realize } from "../src/core/lang.ts";

// The real thing, both bodies: the shipped regions.json against the shipped GLB.
const loaded: { spec: BodySpec; index: AtlasIndex }[] = [];

before(async () => {
  for (const spec of BODIES) {
    const bytes = readFileSync(new URL(`../public/${spec.path}`, import.meta.url));
    const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
    const skin = buildSkin(gltf.scene, spec);
    const raw = JSON.parse(readFileSync(new URL(`../public/bodies/${spec.id}.regions.json`, import.meta.url), "utf8"));
    const nFaces = (skin.geometry.getAttribute("position").count) / 3;
    const atlas = parseAtlas(raw, nFaces);
    loaded.push({ spec, index: new AtlasIndex(atlas, skin.geometry, skin.centroids) });
  }
});

test("every leaf site resolves to a contained anchor on both bodies", () => {
  for (const { spec, index } of loaded) {
    assert.equal(index.presentSites().length, 59, spec.id);
    for (const id of Object.keys(SITES)) {
      for (const lat of SITES[id].laterality === "sided" ? (["left", "right"] as const) : ([null] as const)) {
        const a = index.anchorFor({ id, laterality: lat });
        assert.ok(index.contains({ id, laterality: lat }, a), `${spec.id}: anchor for ${lat ?? ""} ${id} not inside its own region`);
        const desc = index.describe(a);
        assert.ok(desc, `${spec.id}: ${id} anchor has no description`);
        assert.equal(desc.id, id, `${spec.id}: ${lat ?? ""} ${id} described as ${desc.id}`);
      }
    }
  }
});

test("zones resolve through their members", () => {
  for (const { spec, index } of loaded) {
    for (const id of Object.keys(ZONES)) {
      if (ZONES[id].members.length === 0) continue; // bodysuit
      const a = index.anchorFor({ id, laterality: ZONES[id].laterality === "sided" ? "left" : null });
      assert.ok(index.contains({ id, laterality: "left" }, a), `${spec.id}: zone ${id}`);
    }
  }
});

test("laterality is honored: a left anchor is on the +x side", () => {
  for (const { index } of loaded) {
    for (const id of ["forearm", "thigh", "knee_ditch", "shoulder_cap"]) {
      const l = index.anchorFor({ id, laterality: "left" });
      const r = index.anchorFor({ id, laterality: "right" });
      assert.notEqual(l.face, r.face);
      assert.equal(index.describe(l)!.laterality, "left");
      assert.equal(index.describe(r)!.laterality, "right");
    }
  }
});

test("levels move the anchor along the region", () => {
  for (const { spec, index } of loaded) {
    const upper = index.anchorFor({ id: "forearm", laterality: "left", level: "upper" });
    const lower = index.anchorFor({ id: "forearm", laterality: "left", level: "lower" });
    const uu = index.uvOf(upper.face)!, ul = index.uvOf(lower.face)!;
    assert.ok(uu[0] < ul[0], `${spec.id}: upper u=${uu[0].toFixed(2)} not above lower u=${ul[0].toFixed(2)}`);
  }
});

test("the founding sentence grounds end to end", () => {
  for (const { spec, index } of loaded) {
    const p = parseSentence("a fine line octopus on the left knee ditch");
    const a = index.anchorFor(p.site);
    assert.ok(index.contains(p.site, a), spec.id);
    const desc = index.describe(a)!;
    // The derived caption names the same place the request did.
    const caption = realize({ ...p, site: { id: desc.id, laterality: desc.laterality, aspect: null, level: null } });
    assert.equal(caption, "a fine line octopus on the left knee ditch", spec.id);
  }
});

test("an anchor placed by hand gets a truthful site description", () => {
  for (const { index } of loaded) {
    // Sample every 50th face; each described face must contain itself.
    let described = 0;
    for (let f = 0; f < 28200; f += 50) {
      const d = index.describe({ face: f, barycentric: [1 / 3, 1 / 3, 1 / 3] });
      if (!d) continue; // eyes
      described++;
      assert.ok(index.contains({ id: d.id, laterality: d.laterality }, { face: f, barycentric: [1 / 3, 1 / 3, 1 / 3] }));
      if (d.aspect) assert.ok((SITES[d.id].aspects ?? []).includes(d.aspect), `${d.id}: derived aspect ${d.aspect} not in the lexicon allowlist`);
    }
    assert.ok(described >= 490, `only ${described} of ~500 skin samples described`);
  }
});

test("joint goldens: limbs are ordered and the elbow is the olecranon", () => {
  for (const { spec, index } of loaded) {
    const z = (s: string) => {
      const a = index.anchorFor({ id: s, laterality: "left" });
      return index["centroids"][3 * a.face + 2];
    };
    const chainArm = ["shoulder_cap", "bicep", "elbow", "forearm", "wrist", "hand"].map(z);
    for (let i = 1; i < chainArm.length; i++) assert.ok(chainArm[i] < chainArm[i - 1], `${spec.id}: arm chain out of order at ${i}`);
    const chainLeg = ["thigh", "knee", "shin", "ankle", "foot_top"].map(z);
    for (let i = 1; i < chainLeg.length; i++) assert.ok(chainLeg[i] < chainLeg[i - 1], `${spec.id}: leg chain out of order at ${i}`);
    // The elbow and wrist are distinct joints, not neighbours.
    assert.ok(z("elbow") - z("wrist") > 0.10, `${spec.id}: elbow ${z("elbow").toFixed(3)} too close to wrist ${z("wrist").toFixed(3)}`);
    // Anchor hints: the elbow anchor faces backward (olecranon), the knee forward (kneecap).
    const e = index.anchorFor({ id: "elbow", laterality: "left" });
    const k = index.anchorFor({ id: "knee", laterality: "left" });
    assert.ok(index["centroids"][3 * e.face + 1] > 0, `${spec.id}: elbow anchor is not on the back of the arm`);
    assert.ok(index["centroids"][3 * k.face + 1] < 0, `${spec.id}: knee anchor is not on the front of the leg`);
  }
});

test("relative anchors and parent containment", () => {
  for (const { spec, index } of loaded) {
    const below = index.anchorForPhrase({ id: "collarbone", laterality: "left", aspect: null, level: null, rel: { kind: "below", offset_m: 0.05, render: "two inches" } });
    const base = index.anchorFor({ id: "collarbone", laterality: "left" });
    const dz = index["centroids"][3 * base.face + 2] - index["centroids"][3 * below.face + 2];
    assert.ok(dz > 0.02 && dz < 0.10, `${spec.id}: below-offset moved ${dz.toFixed(3)} m`);
    const mid = index.anchorForPhrase({ id: "shoulder_blade", laterality: "left", aspect: null, level: null, rel: { kind: "between", other: { id: "shoulder_blade", laterality: "right" } } });
    assert.ok(Math.abs(index["centroids"][3 * mid.face]) < 0.04, `${spec.id}: between-the-blades not on the midline`);
    // The navel is a child of the stomach: containment rolls up.
    const nav = index.anchorFor({ id: "navel", laterality: null });
    assert.ok(index.contains({ id: "stomach", laterality: null }, nav), `${spec.id}: navel not inside stomach`);
  }
});
