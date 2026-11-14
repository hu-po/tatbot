import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { computeSmoothNormals } from "../src/core/anchor.ts";
import { cardHtml, turnWords, type CardSection } from "../src/core/card.ts";
import { clockwiseFrom } from "../src/core/landmarks.ts";

const card = (over: Partial<CardSection> = {}): CardSection => ({
  name: "Swallow", sentence: "a swallow on the left bicep", sizeMm: [45, 46.2], mirror: false, flipArt: false,
  tile: { name: "Forearm", sizeMm: [45, 63], fits: true },
  guide: { landmarks: [{ name: "elbow", distanceMm: 140 }, { name: "shoulder", distanceMm: 165 }], top: { towards: "shoulder", clockwiseDeg: 72 } },
  inks: [{ name: "True Black", rgb: [60, 60, 63] }], estimate: "about 1 h, 40 dips", detail: [], artUrl: "data:image/png;base64,AA", mockupUrl: "data:image/jpeg;base64,BB", ...over,
});

test("a card prints the design and tile at actual size, with a bar to check the printer", () => {
  const html = cardHtml("Arm", "abc123def4567890", [card()]);
  assert.match(html, /width:45mm;height:63mm/);
  assert.match(html, /width:45mm;height:46\.2mm/);
  assert.match(html, /<div class="bar" style="width:50mm">/);
  assert.match(html, /about 140 mm from the elbow, about 165 mm from the shoulder/);
  assert.match(html, /Tile top turned 72° clockwise/);
  assert.match(html, /Inkmap brief abc123def456\./);
});

test("a design past the page is scaled, and the card says so rather than show a bar", () => {
  const html = cardHtml("Arm", "abc", [card({ sizeMm: [285, 292], tile: { name: "Forearm", sizeMm: [45, 63], fits: false } })]);
  assert.match(html, /Shown at 38%, not actual size/);
  assert.doesNotMatch(html, /class="bar"/);
  assert.match(html, /the design does not fit it/);
});

test("names from the art are text, not markup", () => {
  const html = cardHtml("<b>", "abc", [card({ name: `<script>alert("x")</script>`, sentence: "a & b" })]);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&#60;script&#62;/);
});

test("a mirrored design is flipped on the card and named so", () => {
  const html = cardHtml("Arm", "abc", [card({ mirror: true, flipArt: true })]);
  assert.match(html, /scaleX\(-1\)/);
  assert.match(html, /mm, mirrored/);
  assert.doesNotMatch(cardHtml("Arm", "abc", [card({ mirror: true })]), /scaleX\(-1\)/, "a render was separated mirrored");
});

test("a turn reads as a person would hold the tile, the short way round", () => {
  assert.equal(turnWords({ towards: "shoulder", clockwiseDeg: 3 }), "Tile top points at the shoulder");
  assert.match(turnWords({ towards: "shoulder", clockwiseDeg: 90 }), /^Tile top turned 90° clockwise/);
  assert.match(turnWords({ towards: "shoulder", clockwiseDeg: -40 }), /^Tile top turned 40° anticlockwise/);
  assert.match(turnWords({ towards: "hip", clockwiseDeg: 300 }), /^Tile top turned 60° anticlockwise/);
});

test("the turn is measured on the skin, clockwise as seen from outside it", () => {
  // A sheet facing -y, seen from -y with +z up and +x to the right: the frame's up is +z at rotation 0.
  const g = new THREE.PlaneGeometry(0.2, 0.2, 4, 4).toNonIndexed();
  g.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  g.deleteAttribute("normal");
  computeSmoothNormals(g);
  const anchor = { face: 12, barycentric: [1 / 3, 1 / 3, 1 / 3] as [number, number, number] };
  const up = new THREE.Vector3(0, 0, 1), right = new THREE.Vector3(1, 0, 0);
  assert.ok(Math.abs(clockwiseFrom(g, anchor, 0, up)) < 1e-6, "its top points up");
  assert.ok(Math.abs(Math.abs(clockwiseFrom(g, anchor, 0, right)) - 90) < 1e-6, "a quarter turn from the side");
  assert.ok(Math.abs(clockwiseFrom(g, anchor, 0.5, up) - clockwiseFrom(g, anchor, 0.2, up) - clockwiseFrom(g, anchor, 0.3, up)) < 1e-6, "turns add");
});
