import assert from "node:assert/strict";
import test from "node:test";
import { fixtureReview } from "./review-fixture.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import { feedbackDocument, loadReview, mergeObservations, observe, readObservations, saveObservation, validateObservation } from "../src/core/study-review.ts";

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return { get length() { return values.size; }, key: n => [...values.keys()][n] ?? null,
    getItem: k => values.get(k) ?? null, setItem: (k, v) => { values.set(k, v); }, removeItem: k => { values.delete(k); }, clear: () => values.clear() };
}

test("shared renderer yields metric round strokes and feedback binds exact preparation", async () => {
  const review = await loadReview(JSON.stringify(await fixtureReview()));
  const preview = review.previews[review.bundle.entries[0].artwork_sha256];
  assert.match(preview.svg, /width="30mm" height="30mm"/);
  assert.match(preview.svg, /stroke-linecap="round" stroke-linejoin="round"/);
  const vote = observe(review, review.bundle.entries[0], "like");
  assert.equal(vote.preview_sha256, preview.sha256);
  assert.throws(() => validateObservation({ ...vote, entry_id: review.bundle.entries[1].id }, review), /different artwork/);
  assert.throws(() => validateObservation({ ...vote, preview_sha256: "b".repeat(64) }, review), /different artwork/);
  const feedback = await feedbackDocument(review, [vote]);
  assert.equal(feedback.content_sha256, await canonicalDigest(feedback));
});

test("tampered, unresolved, duplicate and test-set review data is refused", async () => {
  const original = await fixtureReview();
  await assert.rejects(loadReview(JSON.stringify({ ...original, name: "Changed" })), /digest differs/);
  for (const mutate of [
    (b: typeof original) => { b.entries[0].artwork_sha256 = "c".repeat(64); },
    (b: typeof original) => { b.entries.push(b.entries[0]); },
    (b: typeof original) => { (b.entries[0] as { source_split: string }).source_split = "test"; },
    (b: typeof original) => { b.entries[0].preparation.stats.time_estimate.modeled_s = -1; },
  ]) {
    const bundle = structuredClone(original); mutate(bundle); bundle.content_sha256 = await canonicalDigest(bundle);
    await assert.rejects(loadReview(JSON.stringify(bundle)));
  }
});

test("append-only feedback merges tabs, survives reload and refuses conflicting history", async () => {
  const review = await loadReview(JSON.stringify(await fixtureReview())), storage = memoryStorage();
  const a = observe(review, review.bundle.entries[0], "like"), b = observe(review, review.bundle.entries[1], "dislike");
  saveObservation(storage, review, a); saveObservation(storage, review, b);
  assert.equal(readObservations(storage, review).length, 2);
  assert.equal(mergeObservations([a], [a, b]).length, 2);
  assert.throws(() => saveObservation(storage, review, { ...a, preference: "dislike" }), /Conflicting/);
  assert.throws(() => mergeObservations([a], [{ ...a, preference: "dislike" }]), /Conflicting/);
  const other = await fixtureReview(); other.name = "Other review"; other.content_sha256 = await canonicalDigest(other);
  assert.equal(readObservations(storage, await loadReview(JSON.stringify(other))).length, 0);
});

test("review cannot restore legacy artwork even with valid record and bundle hashes", async () => {
  const bundle = await fixtureReview();
  const art = Object.values(bundle.artworks)[0];
  art.conversion = { ...art.conversion, adapter: "tatbot-svg-paint/1", recipe_sha256: null };
  art.content_sha256 = await canonicalDigest(art);
  bundle.artworks = { [art.content_sha256]: art };
  for (const entry of bundle.entries) entry.artwork_sha256 = art.content_sha256;
  bundle.content_sha256 = await canonicalDigest(bundle);
  await assert.rejects(loadReview(JSON.stringify(bundle)), /Generate with DrawingBot V3/);
});

test("quota failures remain explicit and feedback can still be exported", async () => {
  const review = await loadReview(JSON.stringify(await fixtureReview())), storage = memoryStorage();
  storage.setItem = () => { throw new Error("quota exhausted"); };
  const vote = observe(review, review.bundle.entries[0], "like");
  assert.throws(() => saveObservation(storage, review, vote), /quota/);
  assert.equal((await feedbackDocument(review, [vote])).observations[0].preference, "like");
});
