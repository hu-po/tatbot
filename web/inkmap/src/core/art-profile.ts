/** What an image holds for a load of inks, read off its pixels: how much of the art rectangle is inked, how much of
 *  it each ink takes, and how fat each ink's fattest spot is. Pure, so the browser and the tests read the same thing.
 *
 * The pipeline widens the darkest ink's lines to one drawn line, and opens every other colour so none is thinner than
 * one (python/tatbot_art separate.py), then fills inside a half line's margin. A colour whose fattest spot is thin
 * therefore loses its line first. Over the art corpus's 124 colour-ink cases at 60 mm, a radius under 1 mm caught 7 of
 * the 8 inks that drew nothing and flagged 4 that did draw, so this is a warning, not a prediction.
 */
import { lab, isPaper } from "./art.ts";
import { INKS, MAX_INKS, type Ink } from "./inks.ts";

export interface InkArea {
  ink: string;
  /** The share of the art rectangle nearest this ink. */
  share: number;
  /** The fattest spot's inscribed radius, as a fraction of the art's width. */
  radiusOfWidth: number;
}

export interface ArtProfile {
  /** The share of the art rectangle that is inked at all. */
  coverage: number;
  inks: InkArea[];
}

export const THIN_INK_RADIUS_MM = 1;
/** Inks under this share of the rectangle are the image's antialiasing, not a colour of the design. */
const MIN_SHARE = 0.002;
const SUGGEST_MIN_SHARE = 0.01;

/** Each pixel's distance to the nearest pixel outside `mask`, in pixels: a two-pass chamfer. */
function insetRadius(mask: Uint8Array, width: number, height: number): number {
  const distance = new Float32Array(mask.length);
  const diagonal = Math.SQRT2;
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= width || y >= height ? 0 : distance[y * width + x]);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (!mask[y * width + x]) continue;
    distance[y * width + x] = Math.min(at(x - 1, y) + 1, at(x, y - 1) + 1, at(x - 1, y - 1) + diagonal, at(x + 1, y - 1) + diagonal);
  }
  let largest = 0;
  for (let y = height - 1; y >= 0; y--) for (let x = width - 1; x >= 0; x--) {
    if (!mask[y * width + x]) continue;
    const index = y * width + x;
    distance[index] = Math.min(distance[index], at(x + 1, y) + 1, at(x, y + 1) + 1, at(x + 1, y + 1) + diagonal, at(x - 1, y + 1) + diagonal);
    if (distance[index] > largest) largest = distance[index];
  }
  return largest;
}

/** The profile of an RGBA art rectangle for `inks`; paper is transparent, or light and neutral by the pipeline's rule. */
export function profileArt(rgba: Uint8ClampedArray, width: number, height: number, inks: readonly Ink[]): ArtProfile {
  const colours = inks.map((ink) => lab(...ink.rgb));
  const masks = inks.map(() => new Uint8Array(width * height));
  const counts = new Array<number>(inks.length).fill(0);
  let inked = 0;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const [r, g, b, a] = rgba.subarray(4 * pixel, 4 * pixel + 4);
    if (a < 128 || isPaper(r, g, b)) continue;
    inked++;
    const [L, A, B] = lab(r, g, b);
    let nearest = 0, best = Infinity;
    colours.forEach(([l, u, v], index) => {
      const distance = (L - l) ** 2 + (A - u) ** 2 + (B - v) ** 2;
      if (distance < best) { best = distance; nearest = index; }
    });
    masks[nearest][pixel] = 1;
    counts[nearest]++;
  }
  const total = width * height;
  return {
    coverage: inked / total,
    inks: inks.map((ink, index) => ({ ink: ink.id, share: counts[index] / total, radiusOfWidth: insetRadius(masks[index], width, height) / width })),
  };
}

/** Connected runs of ink, four-neighbour, each with its size and box. */
function inkShapes(mask: Uint8Array, width: number, height: number): { size: number; box: [number, number, number, number] }[] {
  const seen = new Uint8Array(mask.length), shapes = [], stack: number[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    stack.push(start); seen[start] = 1;
    const box: [number, number, number, number] = [width, height, -1, -1];
    let size = 0;
    while (stack.length) {
      const at = stack.pop()!, x = at % width, y = (at - x) / width;
      size++;
      box[0] = Math.min(box[0], x); box[1] = Math.min(box[1], y); box[2] = Math.max(box[2], x); box[3] = Math.max(box[3], y);
      for (const next of [x > 0 ? at - 1 : -1, x + 1 < width ? at + 1 : -1, y > 0 ? at - width : -1, y + 1 < height ? at + width : -1]) {
        if (next >= 0 && mask[next] && !seen[next]) { seen[next] = 1; stack.push(next); }
      }
    }
    shapes.push({ size, box });
  }
  return shapes;
}

/** Marks under this share of the ink are specks, not parts of the design. */
const SPECK_SHARE = 0.02;
/** Past this share of the art's width or height, a speck has moved the box a person would notice. */
const SPECK_SLACK = 0.1;
/** A picture this much inked, this much of it pale and weakly coloured, is a background that did not read as paper. */
const PALE_COVERAGE = 0.8, PALE_SHARE = 0.6;

/** What the art box hides, for an RGBA art rectangle with paper transparent. Nothing here refuses: the art box is the
 *  pipeline's rule, so a stray speck really does set the drawn width, and a pale background really would be inked. */
export function artNotes(rgba: Uint8ClampedArray, width: number, height: number, padPx: number): string[] {
  const mask = new Uint8Array(width * height);
  let inked = 0, pale = 0;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const [r, g, b, a] = rgba.subarray(4 * pixel, 4 * pixel + 4);
    if (a < 128 || isPaper(r, g, b)) continue;
    mask[pixel] = 1;
    inked++;
    const [lightness, u, v] = lab(r, g, b);
    if (lightness > 60 && Math.hypot(u, v) < 30) pale++;
  }
  const notes: string[] = [];
  if (inked / mask.length > PALE_COVERAGE && pale / inked > PALE_SHARE) {
    notes.push("Most of this image is a pale background, which would be drawn as ink. Use a white or transparent background.");
  }
  const kept = inkShapes(mask, width, height).filter((shape) => shape.size >= SPECK_SHARE * inked);
  if (kept.length) {
    const core = [Math.min(...kept.map((s) => s.box[0])), Math.min(...kept.map((s) => s.box[1])), Math.max(...kept.map((s) => s.box[2])), Math.max(...kept.map((s) => s.box[3]))];
    const [coreWidth, coreHeight] = [core[2] - core[0] + 1 + 2 * padPx, core[3] - core[1] + 1 + 2 * padPx];
    if (coreWidth < (1 - SPECK_SLACK) * width || coreHeight < (1 - SPECK_SLACK) * height) {
      notes.push(`A small mark away from the design stretches its box, so the design is about ${Math.round(100 * Math.min(1, coreWidth / width))}% of the width you set. Remove the mark to size the design itself.`);
    }
  }
  return notes;
}

/** The inks a design needs, most used first, at most a load's worth: a suggestion for art that names none. */
export function suggestInks(rgba: Uint8ClampedArray, width: number, height: number): string[] {
  const { inks } = profileArt(rgba, width, height, INKS.filter((ink) => ink.suggest !== false));
  const used = [...inks].sort((a, b) => b.share - a.share).filter((area) => area.share >= SUGGEST_MIN_SHARE);
  return (used.length ? used : inks.slice(0, 1)).slice(0, MAX_INKS).map((area) => area.ink);
}

/** The colours this width puts at risk, each with the width that would hold it. The darkest ink is traced, not opened. */
export function thinInks(profile: ArtProfile, inks: readonly Ink[], widthMm: number): { ink: Ink; minWidthMm: number }[] {
  const darkest = inks.reduce<Ink | null>((dark, ink) => (dark && lab(...dark.rgb)[0] <= lab(...ink.rgb)[0] ? dark : ink), null);
  return profile.inks.flatMap((area) => {
    const ink = inks.find((item) => item.id === area.ink);
    if (!ink || ink === darkest || area.share < MIN_SHARE) return [];
    const minWidthMm = THIN_INK_RADIUS_MM / area.radiusOfWidth;
    return widthMm < minWidthMm ? [{ ink, minWidthMm }] : [];
  });
}
