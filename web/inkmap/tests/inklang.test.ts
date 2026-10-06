import { before, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { AtlasIndex, parseAtlas } from "../src/core/atlas.ts";
import {
  BODY_SPEC,
  MODEL_SPEC_ID,
  REST_SURFACE_SHA256,
  buildSkin,
  canonicalSurfaceBytes,
} from "../src/core/body.ts";
import {
  SITES,
  acceptResolutionCandidate,
  intentFromSite,
  parsePlacement,
  realizePlacement,
  resolveIntent,
  type SitePhrase,
} from "../src/core/inklang/index.ts";
import { parseSentence } from "../src/core/lang.ts";
import { sha256Hex } from "../src/core/sha256.ts";
import { canonicalJson, resolvePrompt } from "../tools/resolve.ts";

let index: AtlasIndex;

before(async () => {
  const bytes = readFileSync(new URL(`../public/${BODY_SPEC.path}`, import.meta.url));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const gltf = await new GLTFLoader().parseAsync(buffer, "");
  const skin = buildSkin(gltf.scene);
  const raw = JSON.parse(readFileSync(
    new URL(`../public/bodies/${MODEL_SPEC_ID}.regions.json`, import.meta.url),
    "utf8",
  ));
  index = new AtlasIndex(parseAtlas(raw, skin.centroids.length / 3), skin.geometry, skin.centroids);
  assert.equal(await sha256Hex(canonicalSurfaceBytes(skin.geometry)), REST_SURFACE_SHA256);
});

test("InkLang parses placement descriptions independently of design wording", () => {
  const refined = parsePlacement("left upper inner forearm");
  assert.equal(refined.canonical_phrase, "on the left upper inner forearm");
  assert.deepEqual(refined.site, {
    id: "forearm", laterality: "left", aspect: "inner", level: "upper",
  });
  assert.deepEqual(
    parsePlacement("two inches below the left forearm").site?.relation,
    { kind: "below", offset_m: 0.0508, render: "two inches" },
  );
  assert.equal(parsePlacement("popliteal fossa").site?.id, "knee_ditch");
  assert.equal(parsePlacement("a fine line octopus on the left knee ditch").site, null);
  assert.equal(realizePlacement(parseSentence("a fine line octopus on the left knee ditch").site), "on the left knee ditch");
});

test("ordinary determiners work in direct, refined and relative placement phrases", async () => {
  for (const [prompt, canonical] of [
    ["on my left forearm", "on the left forearm"],
    ["on your right forearm", "on the right forearm"],
    ["on left forearm", "on the left forearm"],
    ["my left forearm", "on the left forearm"],
    ["on the inside of my left forearm", "on the left inner forearm"],
    ["on the outside of your right forearm", "on the right outer forearm"],
    ["two inches below my left collarbone", "two inches below the left collarbone"],
    ["behind my left ear", "behind the left ear"],
    ["between my shoulders", "between the shoulders"],
  ]) {
    const expected = await resolvePrompt({ prompt: canonical });
    const actual = await resolvePrompt({ prompt });
    assert.equal(actual.status, expected.status, prompt);
    assert.deepEqual(actual.intent.site, expected.intent.site, prompt);
    assert.deepEqual(actual.anchor, expected.anchor, prompt);
  }
  assert.equal(parseSentence("a koi on my left forearm").motif, "koi");
});

test("placement vetoes cannot become artwork through the compatibility adapter", async () => {
  for (const prompt of [
    "not on the left forearm", "not on my left forearm", "a rose not on my left forearm",
    "never on my left forearm", "don't put it on my left forearm", "don’t place it on my left forearm",
    "do not draw a rose on my left forearm", "avoid placing a rose on my left forearm",
    "no tattoo on my left forearm", "a rose not two inches below my left collarbone",
  ]) {
    assert.equal((await resolvePrompt({ prompt })).status, "rejected", prompt);
    assert.throws(() => parseSentence(prompt), /where the tattoo should not go/, prompt);
  }
});

test("negation inside artwork remains distinct from a placement veto", async () => {
  for (const [motif, prompt] of [
    ["forget me not", "a forget me not on my left forearm"],
    ["forget me not", "a forget-me-not on the left forearm"],
    ["flower without leaves", "a flower without leaves on my left forearm"],
    ["no regrets", "no regrets on my left forearm"],
  ]) {
    assert.equal(parseSentence(prompt).motif, motif);
    assert.equal((await resolvePrompt({ prompt })).status, "resolved", prompt);
  }
});

test("interactive resolution exposes eligible laterality choices", () => {
  const result = resolveIntent(parsePlacement("forearm"), index);
  assert.equal(result.status, "needs_choice");
  assert.deepEqual(result.candidates.map((candidate) => candidate.laterality).sort(), ["left", "right"]);
  const accepted = acceptResolutionCandidate(result, 1, index);
  assert.equal(accepted.status, "resolved");
  assert.equal(accepted.choice?.laterality, "right");
  assert.ok(index.isValidAnchor(accepted.anchor!));
});

test("seeded resolution is deterministic and records the full body binding", () => {
  const first = resolveIntent(parsePlacement("forearm"), index, { id: "seeded-v1", seed: 712 });
  const second = resolveIntent(parsePlacement("forearm"), index, { id: "seeded-v1", seed: 712 });
  assert.equal(first.status, "resolved");
  assert.equal(canonicalJson(first), canonicalJson(second));
  assert.equal(first.body.model_spec_id, MODEL_SPEC_ID);
  assert.equal(first.body.rest_surface_sha256, REST_SURFACE_SHA256);
  assert.match(first.body.identity_sha256!, /^[0-9a-f]{64}$/);
  assert.match(first.body.topology_sha256!, /^[0-9a-f]{64}$/);
  assert.match(first.body.asset_sha256!, /^[0-9a-f]{64}$/);
});

test("eligible sites resolve while named unsupported anatomy rejects explicitly", () => {
  // Every mapped skin site is eligible; only sites with no SOMA faces after the
  // not-skin exclusions (armpit, ear_lobe, shoulder_blade) reject.
  for (const [id, site] of Object.entries(SITES)) {
    const laterality = site.laterality === "sided" ? "left" as const : null;
    const phrase: SitePhrase = { id, laterality, aspect: null, level: null };
    const result = resolveIntent(intentFromSite(realizePlacement(phrase), phrase), index);
    if (index.facesOf(id, laterality).length > 0) {
      assert.equal(result.status, "resolved", id);
      assert.ok(index.isValidAnchor(result.anchor!), id);
    } else {
      assert.equal(result.status, "rejected", id);
      assert.equal(result.issues[0].code, "INKLANG_NO_REGION", id);
      assert.match(result.issues[0].message, /reviewed eligible surface|no faces/, id);
    }
  }
});

test("zones consider only eligible members and never select an unsupported face", () => {
  for (const prompt of ["quarter sleeve", "half sleeve", "full sleeve", "leg sleeve"]) {
    const interactive = resolveIntent(parsePlacement(prompt), index);
    assert.ok(["needs_choice", "resolved"].includes(interactive.status), prompt);
    for (const candidate of interactive.candidates) assert.ok(index.isValidAnchor(candidate.anchor), prompt);
    const seeded = resolveIntent(parsePlacement(prompt), index, { id: "seeded-v1", seed: 903 });
    assert.equal(seeded.status, "resolved", prompt);
    assert.ok(index.isValidAnchor(seeded.anchor!), prompt);
  }
  assert.equal(resolveIntent(parsePlacement("bodysuit"), index).status, "rejected");
});

test("levels, aspects, and relative walks remain within the eligible forearm", () => {
  const phrases: SitePhrase[] = [
    { id: "forearm", laterality: "left", aspect: "inner", level: null },
    { id: "forearm", laterality: "left", aspect: "outer", level: null },
    { id: "forearm", laterality: "left", aspect: null, level: "upper" },
    { id: "forearm", laterality: "left", aspect: null, level: "mid" },
    { id: "forearm", laterality: "left", aspect: null, level: "lower" },
    {
      id: "forearm", laterality: "left", aspect: null, level: null,
      relation: { kind: "below", offset_m: 0.02, render: "2 cm" },
    },
  ];
  for (const phrase of phrases) {
    const result = resolveIntent(intentFromSite(realizePlacement(phrase), phrase), index);
    assert.equal(result.status, "resolved", realizePlacement(phrase));
    assert.ok(index.isValidAnchor(result.anchor!), realizePlacement(phrase));
  }
  const relative = resolveIntent(intentFromSite(realizePlacement(phrases.at(-1)!), phrases.at(-1)!), index);
  assert.ok(relative.relative);
  assert.ok(relative.relative!.achieved_m > 0);
});

test("unknown anatomy, invalid combinations, and old body ids fail closed", async () => {
  const unknown = resolveIntent(parsePlacement("flux capacitor"), index);
  assert.equal(unknown.status, "rejected");
  assert.equal(unknown.issues[0].code, "INKLANG_UNKNOWN_SITE");
  const impossible = resolveIntent(parsePlacement("left sternum"), index);
  assert.equal(impossible.status, "rejected");
  assert.equal(impossible.issues[0].code, "INKLANG_INVALID_LATERALITY");
  const oldBody = await resolvePrompt({ body: "legacy-body", prompt: "left forearm" } as any);
  assert.equal(oldBody.status, "rejected");
  assert.equal(oldBody.issues[0].code, "INKLANG_UNKNOWN_BODY");
  assert.match(oldBody.issues[0].message, /unsupported schema\/model/);
});

test("the checked-in normative corpus is complete and byte-stable", async () => {
  const corpus = JSON.parse(readFileSync(
    new URL("../../../config/inkmap/examples/inklang/corpus-v1.json", import.meta.url),
    "utf8",
  )) as {
    corpus_schema_version: number;
    cases: { id: string; request: Parameters<typeof resolvePrompt>[0]; expected: unknown }[];
  };
  assert.equal(corpus.corpus_schema_version, 1);
  assert.ok(corpus.cases.length >= 100, `only ${corpus.cases.length} cases`);
  assert.ok(corpus.cases.every((item) => !("body" in item.request)));
  for (const siteId of Object.keys(SITES)) {
    assert.ok(corpus.cases.some((item) => item.id.startsWith(`${MODEL_SPEC_ID}:leaf:`) && item.id.endsWith(`:${siteId}`)), siteId);
  }
  for (const item of corpus.cases) {
    assert.equal(canonicalJson(await resolvePrompt(item.request)), canonicalJson(item.expected), item.id);
  }
});

test("structured consumers resolve through the same core", async () => {
  const request = {
    site: { id: "forearm", laterality: "left" as const, aspect: "inner", level: "upper" as const },
    description: "left upper inner forearm",
    policy: "seeded-v1" as const,
    seed: 41,
  };
  assert.equal(
    canonicalJson(await resolvePrompt(request)),
    canonicalJson(await resolvePrompt({
      prompt: "left upper inner forearm",
      policy: request.policy,
      seed: request.seed,
    })),
  );
});
