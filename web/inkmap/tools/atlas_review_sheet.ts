#!/usr/bin/env node
// Emit one reviewable card per InkLang placement: the resolved anchor drawn on
// the body view that faces it, next to the phrase that produced it. Feeds the
// human review pass that checks whether the atlas actually agrees with the
// words — the checks in the suite prove determinism and containment, not that
// "on the left hip" looks like a hip.
//
//   npm run atlas:review-sheet -- /path/to/review.json
import { readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { anchorToPoint } from "../src/core/anchor.ts";
import { AtlasIndex, parseAtlas } from "../src/core/atlas.ts";
import { BODY_SPEC, buildSkin } from "../src/core/body.ts";
import {
  SITES,
  intentFromSite,
  realizePlacement,
  resolveIntent,
  type SitePhrase,
} from "../src/core/inklang/index.ts";

const destination = process.argv[2];
if (!destination) throw new Error("usage: atlas_review_sheet.ts OUTPUT.json");

const W = 300, H = 460, PAD = 22;
type ViewId = "front" | "back" | "left" | "right";
const views: { id: ViewId; normal: THREE.Vector3; project: (p: THREE.Vector3) => [number, number] }[] = [
  { id: "front", normal: new THREE.Vector3(0, -1, 0), project: (p) => [p.x, p.z] },
  { id: "back", normal: new THREE.Vector3(0, 1, 0), project: (p) => [-p.x, p.z] },
  { id: "left", normal: new THREE.Vector3(1, 0, 0), project: (p) => [p.y, p.z] },
  { id: "right", normal: new THREE.Vector3(-1, 0, 0), project: (p) => [-p.y, p.z] },
];

interface Card {
  id: string;
  body: string;
  view: ViewId;
  prompt: string;
  phrase: string;
  site_id: string;
  laterality: string | null;
  aspect: string | null;
  level: string | null;
  face: number;
  /** True when the anchor's surface faces into the body from every view. */
  inward: boolean;
  /** Subsampled projected outline of every face the region covers. */
  region: [number, number][];
  region_faces: number;
  kind: "leaf" | "aspect" | "level" | "relative";
  detail: string | null;
  x: number;
  y: number;
}

const silhouettes: Record<string, string> = {};
const cards: Card[] = [];

const spec = BODY_SPEC;
{
  const bytes = readFileSync(new URL(`../public/${spec.path}`, import.meta.url));
  const gltf = await new GLTFLoader().parseAsync(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
  const skin = buildSkin(gltf.scene);
  const raw = JSON.parse(readFileSync(
    new URL(`../public/bodies/${spec.id}.regions.json`, import.meta.url), "utf8"));
  const index = new AtlasIndex(parseAtlas(raw, skin.centroids.length / 3), skin.geometry, skin.centroids);
  const positions = skin.geometry.getAttribute("position") as THREE.BufferAttribute;

  // One shared silhouette per body view; every card of that view reuses it.
  const mappers: Record<string, (p: THREE.Vector3) => [number, number]> = {};
  for (const view of views) {
    const flat: [number, number][] = [];
    for (let v = 0; v < positions.count; v += 11) {
      flat.push(view.project(new THREE.Vector3().fromBufferAttribute(positions, v)));
    }
    const xs = flat.map((p) => p[0]), ys = flat.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min((W - PAD * 2) / (maxX - minX), (H - PAD * 2) / (maxY - minY));
    const map = (p: [number, number]): [number, number] => [
      W / 2 + (p[0] - (minX + maxX) / 2) * scale,
      H / 2 - (p[1] - (minY + maxY) / 2) * scale,
    ];
    mappers[view.id] = (p: THREE.Vector3) => map(view.project(p));
    silhouettes[`${spec.id}:${view.id}`] = flat
      .map((p) => { const [x, y] = map(p); return `M${x.toFixed(0)} ${y.toFixed(0)}h1`; })
      .join("");
  }

  // Faces belonging to each atlas region, so a card can show the region it
  // resolved inside and not just the single anchor. Whether the region is
  // right and whether the anchor sits well inside it are different bugs.
  const regionFaces = new Map<string, number[]>();
  for (let face = 0; face < raw.faces.length; face++) {
    const code = raw.faces[face];
    if (code < 0) continue;
    const site = raw.sites[code >> 2];
    const lat = ["", "left", "right"][code & 3] ?? "";
    const key = lat ? `${site}:${lat}` : site;
    let list = regionFaces.get(key);
    if (!list) regionFaces.set(key, (list = []));
    list.push(face);
  }

  const push = (phrase: SitePhrase, kind: Card["kind"], detail: string | null): void => {
    let prompt: string;
    try {
      // Aspects and levels are constrained per site; skip combinations the
      // lexicon does not accept rather than inventing a card for them.
      prompt = realizePlacement(phrase);
    } catch {
      return;
    }
    const result = resolveIntent(intentFromSite(prompt, phrase), index);
    if (result.status !== "resolved" || !result.anchor || !result.actual) return;
    const { p, n } = anchorToPoint(skin.geometry, result.anchor);
    // Pick the view that can actually SEE this anchor. Scoring on the face
    // normal alone puts a medial arm surface on the opposite side's view,
    // where the marker lands over the far edge of the body and reads as the
    // wrong limb entirely — "on the left forearm" drawn 32 cm behind the
    // right-side camera. Require the anchor to be on the camera's half first;
    // only then prefer the most face-on view.
    const visible = views.filter((candidate) => p.dot(candidate.normal) > 0
      && n.dot(candidate.normal) > 0);
    const nearSide = views.filter((candidate) => p.dot(candidate.normal) > 0);
    const pool = visible.length ? visible : nearSide.length ? nearSide : views;
    const view = pool.reduce((best, candidate) =>
      n.dot(candidate.normal) > n.dot(best.normal) ? candidate : best);
    // No view both reaches this anchor and faces it: the surface points into
    // the body in the rest pose. Say so on the card instead of pretending.
    const inward = visible.length === 0;
    const [x, y] = mappers[view.id](p);
    const key = result.actual.laterality
      ? `${result.actual.site_id}:${result.actual.laterality}`
      : result.actual.site_id;
    const all = regionFaces.get(key) ?? [];
    const step = Math.max(1, Math.ceil(all.length / 150));
    const outline: [number, number][] = [];
    for (let i = 0; i < all.length; i += step) {
      const c = new THREE.Vector3(
        skin.centroids[3 * all[i]], skin.centroids[3 * all[i] + 1], skin.centroids[3 * all[i] + 2]);
      const [rx, ry] = mappers[view.id](c);
      outline.push([Number(rx.toFixed(0)), Number(ry.toFixed(0))]);
    }
    cards.push({
      id: `${spec.id}:${kind}:${prompt}`,
      body: spec.id,
      view: view.id,
      prompt,
      phrase: result.actual.canonical_phrase,
      site_id: result.actual.site_id,
      laterality: result.actual.laterality,
      aspect: result.actual.aspect,
      level: result.actual.level,
      face: result.anchor.face,
      kind,
      detail,
      inward,
      region: outline,
      region_faces: all.length,
      x: Number(x.toFixed(1)),
      y: Number(y.toFixed(1)),
    });
  };

  for (const [id, site] of Object.entries(SITES)) {
    const lateralities = site.laterality === "sided" ? (["left", "right"] as const) : ([null] as const);
    for (const laterality of lateralities) {
      push({ id, laterality, aspect: null, level: null }, "leaf", null);
    }
  }
  for (const [id, laterality, aspect] of [
    ["forearm", "left", "inner"],
    ["forearm", "right", "outer"],
    ["shoulder_cap", "left", "front"],
    ["shoulder_cap", "right", "back"],
    ["neck", null, "side"],
    ["hand", "left", "top"],
  ] as [string, any, string][]) {
    push({ id, laterality, aspect, level: null }, "aspect", aspect);
  }
  for (const level of ["upper", "mid", "lower"] as const) {
    push({ id: "forearm", laterality: "left", aspect: null, level }, "level", level);
  }
  for (const [id, laterality, kind, offset] of [
    ["collarbone", "left", "below", 0.0508],
    ["collarbone", "left", "above", 0.03],
    ["ear", "left", "behind", 0.03],
    ["knee_ditch", "left", "above", 0.05],
    ["spine", null, "beside", 0.05],
  ] as [string, any, any, number][]) {
    push(
      { id, laterality, aspect: null, level: null, relation: { kind, offset_m: offset, render: `${Math.round(offset * 1000)} mm` } },
      "relative",
      `${Math.round(offset * 1000)} mm ${kind}`,
    );
  }
}

writeFileSync(destination, JSON.stringify({ width: W, height: H, silhouettes, cards }));
process.stderr.write(`wrote ${cards.length} review cards, ${Object.keys(silhouettes).length} silhouettes\n`);
