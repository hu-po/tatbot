/** One bounded SVG paint materializer for the browser and the offline compiler.
 * Paint becomes metric regions: cap/join geometry, holes, and layer order survive
 * without adding ambiguous style fields to TattooProgram/1. This is a visual
 * artwork adapter; tool-specific fill planning remains the InkProgram compiler.
 */
import { DOMParser as XmlParser, XMLSerializer } from "@xmldom/xmldom";
import { Color, ShapeUtils, Vector2, type Curve, type Path, type Matrix3 } from "three";
import { SVGLoader, type StrokeStyle } from "three/examples/jsm/loaders/SVGLoader.js";
import { canonicalDigest, ContractError, type JsonObject } from "./schema.ts";
import { validateTattooProgram, type TattooElement, type TattooProgram } from "./tattoo-program.ts";
import { sha256Hex } from "../sha256.ts";

export const SVG_ADAPTER = "tatbot-svg-paint/1";
const MAX_BYTES = 2_000_000;
const MAX_POINTS = 100_000;
const MAX_ELEMENTS = 50_000;
const TAGS = new Set(["svg", "g", "path", "line", "polyline", "polygon", "circle", "ellipse", "rect", "title", "desc"]);
const STYLE = new Set(["fill", "fill-rule", "fill-opacity", "stroke", "stroke-opacity", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "opacity"]);
const ATTRS = new Set(["xmlns", "version", "id", "viewBox", "width", "height", "transform", "d", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r", "rx", "ry", "points", "style", ...STYLE]);
const NUMBER = /[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g;

function refuse(detail: string, path = "$.svg"): never {
  throw new ContractError("tattoo_program_unsupported", path, detail);
}

function numericList(text: string, at: string): number[] {
  if (text.replace(NUMBER, "").replace(/[\s,]/g, "")) refuse("invalid numeric list", at);
  const values = (text.match(NUMBER) ?? []).map(Number);
  if (!values.every(Number.isFinite)) refuse("non-finite numeric value", at);
  return values;
}

function pathSyntax(text: string, at: string): void {
  const token = /[MmZzLlHhVvCcSsQqTtAa]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g;
  if (text.replace(token, "").replace(/[\s,]/g, "")) refuse("unsupported path command or malformed number", at);
  const tokens = text.match(token) ?? [];
  const counts: Record<string, number> = { M: 2, L: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, T: 2, A: 7, Z: 0 };
  if (tokens.length && tokens[0]!.toUpperCase() !== "M") refuse("path must start with moveto", at);
  let i = 0;
  while (i < tokens.length) {
    const command = tokens[i++].toUpperCase();
    if (!Object.hasOwn(counts, command)) refuse("missing path command", at);
    const values: number[] = [];
    while (i < tokens.length && !/^[a-z]$/i.test(tokens[i])) values.push(Number(tokens[i++]));
    const count = counts[command];
    if (!values.every(Number.isFinite) || (count === 0 ? values.length !== 0 : !values.length || values.length % count !== 0)) refuse(`invalid ${command} arguments`, at);
    if (command === "A") for (let offset = 0; offset < values.length; offset += count) {
      if (values[offset] < 0 || values[offset + 1] < 0 || ![0, 1].includes(values[offset + 3]) || ![0, 1].includes(values[offset + 4])) refuse("invalid arc radius or flag", at);
    }
  }
}

function transformSyntax(text: string, at: string): void {
  const call = /([a-zA-Z]+)\s*\(([^()]*)\)/g;
  if (text.replace(call, "").replace(/[\s,]/g, "")) refuse("malformed transform", at);
  const sizes: Record<string, number[]> = { matrix: [6], translate: [1, 2], scale: [1, 2], rotate: [1, 3], skewX: [1], skewY: [1] };
  for (const match of text.matchAll(call)) {
    const args = numericList(match[2], at);
    if (!sizes[match[1]]?.includes(args.length)) refuse(`unsupported transform or arity ${match[1]}`, at);
  }
}

/** Normalize inline styles into attributes so browser and XML DOMs agree. */
function normalize(svg: string): { text: string; box: number[] } {
  if (new TextEncoder().encode(svg).length > MAX_BYTES) refuse("SVG exceeds 2 MB");
  if (/<!DOCTYPE|<!ENTITY|<\?/i.test(svg)) refuse("XML declarations, entities, and processing instructions are unsupported");
  const doc = new XmlParser({ onError: (_level, message) => { refuse(`malformed XML: ${message}`); } }).parseFromString(svg, "image/svg+xml");
  const root = doc.documentElement;
  if (!root || root.tagName !== "svg" || root.namespaceURI !== "http://www.w3.org/2000/svg") refuse("expected SVG namespace and root");
  const nodes = Array.from(doc.getElementsByTagName("*"));
  if (nodes.length > 10_000) refuse("too many SVG nodes");
  for (const [index, node] of nodes.entries()) {
    const at = `$.svg.${node.tagName}[${index}]`;
    if (!TAGS.has(node.tagName) || (node !== root && node.tagName === "svg")) refuse(`unsupported element ${node.tagName}`, at);
    for (const attribute of Array.from(node.attributes)) {
      if (!ATTRS.has(attribute.name)) refuse(`unsupported attribute ${attribute.name}`, at);
    }
    if (node.hasAttribute("style")) {
      for (const declaration of node.getAttribute("style")!.split(";")) {
        if (!declaration.trim()) continue;
        const parts = declaration.split(":");
        const key = parts[0].trim();
        if (parts.length !== 2 || !STYLE.has(key)) refuse(`unsupported style ${key}`, at);
        node.setAttribute(key, parts[1].trim());
      }
      node.removeAttribute("style");
    }
    for (const attribute of Array.from(node.attributes)) {
      if (/url\(|!important|inherit|currentColor|var\(/i.test(attribute.value)) refuse(`unsupported paint expression ${attribute.name}`, at);
      if (["opacity", "fill-opacity", "stroke-opacity"].includes(attribute.name) && Number(attribute.value) !== 1) {
        refuse("translucent paint requires a versioned compositing contract; expected opacity 1", at);
      }
      if (["fill", "stroke"].includes(attribute.name) && attribute.value !== "none") color(attribute.value);
      if (attribute.name === "fill-rule" && !["evenodd", "nonzero"].includes(attribute.value)) refuse("unknown fill rule", at);
      if (attribute.name === "transform") transformSyntax(attribute.value, at);
      if (attribute.name === "d") pathSyntax(attribute.value, at);
      if (attribute.name === "points") {
        const points = numericList(attribute.value, at);
        if (points.length < 4 || points.length % 2) refuse("expected coordinate pairs", at);
      }
      if (["x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "stroke-width", "stroke-miterlimit", ...(node === root ? [] : ["width", "height"])].includes(attribute.name)) {
        const values = numericList(attribute.value.replace(/px$/, ""), at);
        if (values.length !== 1) refuse(`expected scalar ${attribute.name}`, at);
        if (["r", "rx", "ry", "width", "height", "stroke-width", "stroke-miterlimit"].includes(attribute.name) && values[0] < 0) refuse(`negative ${attribute.name}`, at);
      }
    }
  }
  const box = (root.getAttribute("viewBox") ?? "").trim().split(/[\s,]+/).map(Number);
  if (box.length !== 4 || !box.every(Number.isFinite) || box[2] <= 0 || box[3] <= 0) refuse("positive finite viewBox required");
  return { text: new XMLSerializer().serializeToString(doc), box };
}

function color(paint: string): [number, number, number] {
  let hex = paint.trim().toLowerCase();
  if (Object.hasOwn(Color.NAMES, hex)) hex = `#${Color.NAMES[hex as keyof typeof Color.NAMES].toString(16).padStart(6, "0")}`;
  if (/^#[0-9a-f]{3}$/.test(hex)) hex = "#" + [...hex.slice(1)].map(c => c + c).join("");
  if (/^#[0-9a-f]{6}$/.test(hex)) return [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16) / 255) as [number, number, number];
  const match = hex.match(/^rgb\(\s*([\d.]+)(%?)\s*,\s*([\d.]+)(%?)\s*,\s*([\d.]+)(%?)\s*\)$/);
  if (match) {
    const values = [1, 3, 5].map(index => Number(match[index]) / (match[index + 1] === "%" ? 100 : 255));
    if (values.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) return values as [number, number, number];
  }
  return refuse(`unsupported color ${paint}`);
}

function distanceToSegment(point: Vector2, a: Vector2, b: Vector2): number {
  const delta = b.clone().sub(a);
  const t = delta.lengthSq() ? Math.max(0, Math.min(1, point.clone().sub(a).dot(delta) / delta.lengthSq())) : 0;
  return point.distanceTo(a.clone().addScaledVector(delta, t));
}

function flattened(path: Path, tolerance: number): Vector2[] {
  const output: Vector2[] = [];
  const walk = (curve: Curve<Vector2>, lo: number, hi: number, a: Vector2, b: Vector2, depth: number) => {
    if (output.length > MAX_POINTS) refuse("curve exceeds point budget");
    const middle = (lo + hi) / 2;
    const samples = [0.25, 0.5, 0.75].map(fraction => curve.getPoint(lo + (hi - lo) * fraction));
    const error = Math.max(...samples.map(point => distanceToSegment(point, a, b)));
    if (!Number.isFinite(error)) refuse("non-finite path geometry");
    if (error <= tolerance) { output.push(b); return; }
    if (depth >= 20) refuse("curve exceeds subdivision budget");
    walk(curve, lo, middle, a, samples[1], depth + 1);
    walk(curve, middle, hi, samples[1], b, depth + 1);
  };
  for (const curve of path.curves) {
    const start = curve.getPoint(0);
    if (!output.length) output.push(start);
    walk(curve, 0, 1, start, curve.getPoint(1), 0);
  }
  if (path.autoClose && output.length && !output[0].equals(output[output.length - 1])) output.push(output[0].clone());
  return output.filter((point, index) => index === 0 || point.distanceToSquared(output[index - 1]) > 1e-24);
}

function isSimilarity(matrix: Matrix3, sx: number, sy: number): boolean {
  const e = matrix.elements;
  const a = new Vector2(e[0] * sx, e[1] * sy);
  const b = new Vector2(e[3] * sx, e[4] * sy);
  const scale = Math.max(a.lengthSq(), b.lengthSq());
  return scale > 0 && Math.abs(a.lengthSq() - b.lengthSq()) <= scale * 1e-8 && Math.abs(a.dot(b)) <= scale * 1e-8;
}

/** How a stroked SVG path reaches the compiler. `outline` (the default)
 * regionises it at its `stroke-width`, so the fill planner rings the line and
 * the pen draws both edges; `centerline` keeps the line the artist drew as one
 * open `path` element at the planning width — one pen pass, no fill planning.
 */
export const STROKE_MODES = ["outline", "centerline"] as const;
export type StrokeMode = (typeof STROKE_MODES)[number];

export interface SvgProgramOptions {
  canvas_m: [number, number];
  semantic_intent: string;
  provenance?: JsonObject;
  width_m?: number;
  deposition?: number;
  chord_error_m?: number;
  strokes?: StrokeMode;
}

export async function tattooProgramFromSvg(svg: string, options: SvgProgramOptions): Promise<TattooProgram> {
  const { text, box } = normalize(svg);
  const [width, height] = options.canvas_m;
  if (![width, height].every(value => Number.isFinite(value) && value > 0)) refuse("physical canvas must be positive finite metres");
  const sx = width / box[2], sy = height / box[3];
  const chordError = options.chord_error_m ?? 0.0001;
  if (!Number.isFinite(chordError) || chordError <= 0 || chordError > 0.0001) refuse("chord error must be in (0, 0.1 mm]");
  const tolerance = chordError / Math.max(sx, sy) / 4;
  const strokeMode: StrokeMode = options.strokes ?? "outline";
  if (!STROKE_MODES.includes(strokeMode)) refuse(`unsupported stroke mode ${String(options.strokes)}`, "$.options.strokes");
  const sourceHash = await sha256Hex(new TextEncoder().encode(svg).buffer);
  if (options.provenance?.source_sha256 && options.provenance.source_sha256 !== sourceHash) {
    throw new ContractError("wrong_hash", "$.provenance.source_sha256", "does not bind the SVG bytes");
  }
  const provenance = options.provenance ?? { producer: SVG_ADAPTER, version: "1", created_utc: "1970-01-01T00:00:00Z", source_sha256: sourceHash };
  const result: TattooProgram = { schema: "tatbot.tattoo-program/1", content_sha256: "0".repeat(64),
    canvas_m: { width, height }, inks: [], layers: [], negative_space_masks: [],
    semantic_intent: options.semantic_intent, preview_sha256: sourceHash, provenance: { ...provenance, source_sha256: sourceHash } };
  let total = 0;
  const metric = (point: Vector2): [number, number] => {
    const x = (point.x - box[0]) * sx, y = height - (point.y - box[1]) * sy;
    if (!Number.isFinite(x + y) || x < -1e-10 || x > width + 1e-10 || y < -1e-10 || y > height + 1e-10) {
      refuse("paint extends outside its physical canvas; crop or resize explicitly");
    }
    const q = (value: number, max: number) => Math.max(0, Math.min(max, Number(value.toPrecision(14))));
    return [q(x, width), q(y, height)];
  };
  const layer = (paint: string, polygons: Vector2[][]) => {
    const rgb = color(paint);
    const id = result.inks.find(ink => ink.color_srgb.every((value, i) => value === rgb[i]))?.id ?? `ink-${result.inks.length}`;
    if (!result.inks.some(ink => ink.id === id)) result.inks.push({ id, color_srgb: rgb });
    const elements: TattooElement[] = [];
    for (const polygon of polygons) {
      if (polygon.length < 3 || Math.abs(ShapeUtils.area(polygon)) * sx * sy < 1e-16) continue;
      if (++total > MAX_ELEMENTS) refuse("paint exceeds element budget");
      elements.push({ id: `paint-${total}`, kind: "region", closed: true, fill: true,
        width_m: options.width_m ?? 0.0008, deposition: options.deposition ?? 0.7, points_m: polygon.map(metric) });
    }
    if (elements.length) result.layers.push({ id: `layer-${result.layers.length}`, ink_id: id, elements });
  };
  const centerlines = (paint: string, polylines: Vector2[][]) => {
    const rgb = color(paint);
    const id = result.inks.find(ink => ink.color_srgb.every((value, i) => value === rgb[i]))?.id ?? `ink-${result.inks.length}`;
    if (!result.inks.some(ink => ink.id === id)) result.inks.push({ id, color_srgb: rgb });
    const elements: TattooElement[] = [];
    for (const polyline of polylines) {
      const closed = polyline.length > 2 && polyline[0].equals(polyline[polyline.length - 1]);
      const points = closed ? polyline.slice(0, -1) : polyline;
      if (points.length < 2) continue;
      if (++total > MAX_ELEMENTS) refuse("paint exceeds element budget");
      elements.push({ id: `line-${total}`, kind: "path", closed, fill: false,
        width_m: options.width_m ?? 0.0008, deposition: options.deposition ?? 0.7, points_m: points.map(metric) });
    }
    if (elements.length) result.layers.push({ id: `layer-${result.layers.length}`, ink_id: id, elements });
  };
  const parsed = new SVGLoader().parse(text);
  for (const shapePath of parsed.paths) {
    const { style, transform } = shapePath.userData as { style: StrokeStyle & { fill?: string; stroke?: string }; transform: Matrix3 };
    if (style.fill !== "none" && style.fill !== undefined) {
      const polygons: Vector2[][] = [];
      for (const shape of shapePath.toShapes()) {
        const contour = flattened(shape, tolerance);
        const holes = shape.holes.map(hole => flattened(hole, tolerance));
        if (!holes.length) polygons.push(contour);
        else {
          // Triangulation preserves local holes without globally erasing other layers.
          const triangles = ShapeUtils.triangulateShape(contour, holes);
          const points = [...contour, ...holes.flat()];
          polygons.push(...triangles.map(triangle => triangle.map(index => points[index])));
        }
      }
      layer(style.fill, polygons);
    }
    if (style.stroke !== undefined && style.stroke !== "none" && style.strokeWidth > 0) {
      if (strokeMode === "centerline") {
        // The source width, cap and join are deliberately not carried: the
        // line is drawn once at the planning width whatever the artist's
        // stroke-width was, and its points are already transformed.
        centerlines(style.stroke, shapePath.subPaths.map(subpath => flattened(subpath, tolerance)));
        continue;
      }
      if (!isSimilarity(transform, sx, sy)) refuse("nonuniformly scaled or skewed strokes require explicit outline conversion");
      if (!["butt", "round", "square"].includes(style.strokeLineCap) || !["miter", "round", "bevel"].includes(style.strokeLineJoin)) refuse("unsupported stroke cap/join");
      const radius = style.strokeWidth / 2;
      const divisions = Math.max(12, Math.ceil(Math.PI / Math.acos(Math.max(-1, 1 - tolerance / Math.max(radius, tolerance)))));
      if (divisions > 4096) refuse("stroke exceeds arc budget");
      const polygons: Vector2[][] = [];
      for (const subpath of shapePath.subPaths) {
        const points = flattened(subpath, tolerance);
        if (points.length < 2) continue;
        const geometry = SVGLoader.pointsToStroke(points, style, divisions, 0);
        if (!geometry) continue;
        const positions = geometry.getAttribute("position");
        for (let i = 0; i < positions.count; i += 3) {
          polygons.push([0, 1, 2].map(j => new Vector2(positions.getX(i + j), positions.getY(i + j))));
        }
        geometry.dispose();
      }
      layer(style.stroke, polygons);
    }
  }
  if (!result.layers.length) refuse("SVG contains no visible supported paint");
  result.content_sha256 = await canonicalDigest(result);
  return validateTattooProgram(result);
}
