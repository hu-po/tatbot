// Render every atlas default anchor on the canonical SOMA rest surface. Each
// anchor appears exactly once, assigned to the orthographic view nearest its
// outward face normal. This is a review artifact, not a resolver input.
//
//   npm run atlas:contact-sheet -- /path/to/atlas-contact-sheet.svg
import { readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { anchorToPoint, type Anchor } from "../src/core/anchor.ts";
import { BODY_SPEC, buildSkin } from "../src/core/body.ts";

const destination = process.argv[2];
if (!destination) throw new Error("usage: atlas_contact_sheet.ts OUTPUT.svg");

const WIDTH = 1600;
const HEIGHT = 2200;
const GROUP_WIDTH = WIDTH;
const HEADER = 130;
const PANEL_WIDTH = GROUP_WIDTH / 3;
const PANEL_HEIGHT = (HEIGHT - HEADER) / 2;
const PAD_X = 55;
const PAD_Y = 55;

type ViewId = "front" | "back" | "left" | "right" | "top" | "bottom";
interface View {
  id: ViewId;
  title: string;
  normal: [number, number, number];
  project: (point: THREE.Vector3) => [number, number];
}

const views: View[] = [
  { id: "front", title: "front (-Y)", normal: [0, -1, 0], project: (p) => [p.x, p.z] },
  { id: "back", title: "back (+Y)", normal: [0, 1, 0], project: (p) => [-p.x, p.z] },
  { id: "left", title: "anatomical left (+X)", normal: [1, 0, 0], project: (p) => [p.y, p.z] },
  { id: "right", title: "anatomical right (-X)", normal: [-1, 0, 0], project: (p) => [-p.y, p.z] },
  { id: "top", title: "top (+Z)", normal: [0, 0, 1], project: (p) => [p.x, p.y] },
  { id: "bottom", title: "bottom (-Z)", normal: [0, 0, -1], project: (p) => [p.x, -p.y] },
];

const escapeXml = (value: string) => value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const lines: string[] = [
  `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">`,
  "<rect width=\"100%\" height=\"100%\" fill=\"#11151b\"/>",
  "<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.body{fill:#f3f7ff}.muted{fill:#8d9aaa}.label{fill:#edf4ff;font-size:12px}.leader{stroke:#7892be;stroke-width:1;opacity:.8}.anchor{fill:#ffbd5a;stroke:#1b2028;stroke-width:2}.cloud{fill:#607086;opacity:.14}.panel{fill:#171c24;stroke:#303a48;stroke-width:1}</style>",
];

interface RegionRecord {
  site_id: string;
  laterality: "left" | "right" | null;
  default_anchor: Anchor;
}

const spec = BODY_SPEC;
{
  const asset = readFileSync(new URL(`../public/${spec.path}`, import.meta.url));
  const buffer = asset.buffer.slice(asset.byteOffset, asset.byteOffset + asset.byteLength);
  const gltf = await new GLTFLoader().parseAsync(buffer, "");
  const skin = buildSkin(gltf.scene);
  const atlas = JSON.parse(readFileSync(new URL(`../public/bodies/${spec.id}.regions.json`, import.meta.url), "utf8")) as {
    body: { rest_surface_sha256: string };
    regions: Record<string, RegionRecord>;
  };
  const offsetX = 0;
  lines.push(`<text class="body" x="${offsetX + 34}" y="48" font-size="27" font-weight="700">MHR through SOMA canonical rest surface</text>`);
  lines.push(`<text class="muted" x="${offsetX + 34}" y="78" font-size="13">${spec.id} · surface ${atlas.body.rest_surface_sha256.slice(0, 16)}… · ${Object.keys(atlas.regions).length} labeled default anchors</text>`);

  const positions = skin.geometry.getAttribute("position") as THREE.BufferAttribute;
  const anchorViews = new Map<ViewId, { key: string; point: THREE.Vector3 }[]>();
  for (const view of views) anchorViews.set(view.id, []);
  for (const [key, region] of Object.entries(atlas.regions).sort(([a], [b]) => a.localeCompare(b))) {
    const { p, n } = anchorToPoint(skin.geometry, region.default_anchor);
    const view = views.reduce((best, candidate) => {
      const score = n.x * candidate.normal[0] + n.y * candidate.normal[1] + n.z * candidate.normal[2];
      const bestScore = n.x * best.normal[0] + n.y * best.normal[1] + n.z * best.normal[2];
      return score > bestScore ? candidate : best;
    });
    anchorViews.get(view.id)!.push({ key, point: p });
  }

  for (const [viewIndex, view] of views.entries()) {
    const column = viewIndex % 3;
    const row = Math.floor(viewIndex / 3);
    const x0 = offsetX + column * PANEL_WIDTH;
    const y0 = HEADER + row * PANEL_HEIGHT;
    lines.push(`<rect class="panel" x="${x0 + 7}" y="${y0 + 7}" width="${PANEL_WIDTH - 14}" height="${PANEL_HEIGHT - 14}" rx="10"/>`);
    lines.push(`<text class="body" x="${x0 + 24}" y="${y0 + 38}" font-size="16" font-weight="700">${escapeXml(view.title)}</text>`);

    const projected: [number, number][] = [];
    for (let vertex = 0; vertex < positions.count; vertex += 3) {
      projected.push(view.project(new THREE.Vector3().fromBufferAttribute(positions, vertex)));
    }
    const minX = Math.min(...projected.map((point) => point[0]));
    const maxX = Math.max(...projected.map((point) => point[0]));
    const minY = Math.min(...projected.map((point) => point[1]));
    const maxY = Math.max(...projected.map((point) => point[1]));
    const scale = Math.min(
      (PANEL_WIDTH - PAD_X * 2) / Math.max(maxX - minX, 1e-9),
      (PANEL_HEIGHT - PAD_Y * 2) / Math.max(maxY - minY, 1e-9),
    );
    const mapPoint = ([x, y]: [number, number]): [number, number] => [
      x0 + PANEL_WIDTH / 2 + (x - (minX + maxX) / 2) * scale,
      y0 + PANEL_HEIGHT / 2 - (y - (minY + maxY) / 2) * scale,
    ];

    // A sparse projected point cloud gives enough body context without making
    // the evidence file depend on a browser/WebGL renderer.
    const cloud = projected.filter((_, index) => index % 3 === 0).map((point) => {
      const [x, y] = mapPoint(point);
      return `<circle class="cloud" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.1"/>`;
    });
    lines.push(...cloud);

    const anchors = anchorViews.get(view.id)!.map(({ key, point }) => {
      const [x, y] = mapPoint(view.project(point));
      return { key, x, y };
    });
    for (const side of ["left", "right"] as const) {
      const items = anchors.filter((item) => side === "left" ? item.x < x0 + PANEL_WIDTH / 2 : item.x >= x0 + PANEL_WIDTH / 2).sort((a, b) => a.y - b.y);
      let labelY = y0 + 58;
      for (const item of items) {
        labelY = Math.max(labelY, item.y);
        const maximum = y0 + PANEL_HEIGHT - 22 - (items.length - items.indexOf(item) - 1) * 15;
        labelY = Math.min(labelY, maximum);
        const labelX = side === "left" ? x0 + 18 : x0 + PANEL_WIDTH - 18;
        const anchorX = side === "left" ? labelX + 4 : labelX - 4;
        lines.push(`<line class="leader" x1="${item.x.toFixed(1)}" y1="${item.y.toFixed(1)}" x2="${anchorX.toFixed(1)}" y2="${labelY.toFixed(1)}"/>`);
        lines.push(`<circle class="anchor" cx="${item.x.toFixed(1)}" cy="${item.y.toFixed(1)}" r="4"/>`);
        lines.push(`<text class="label" x="${labelX}" y="${(labelY + 4).toFixed(1)}" text-anchor="${side === "left" ? "start" : "end"}">${escapeXml(item.key)}</text>`);
        labelY += 15;
      }
    }
    lines.push(`<text class="muted" x="${x0 + PANEL_WIDTH - 22}" y="${y0 + 38}" font-size="12" text-anchor="end">${anchors.length} anchors</text>`);
  }
}

lines.push("</svg>");
writeFileSync(destination, `${lines.join("\n")}\n`);
console.log(`wrote ${destination}`);
