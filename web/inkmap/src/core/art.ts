/** Source art (`tatbot.inkmap-art/1`): the design as an image, which is what Inkmap places.
 *
 * Inkmap is about the design and its placement. A record holds the image bytes, the rectangle of them that
 * is art (`art_px`), provenance and licence, and the art pipeline's hints (the inks it was made for). Strokes
 * come later and elsewhere: the exported tattoo brief gives `tatbot art separate` the image and the width
 * chosen here, and DrawingBot V3 computes the paths at that size.
 *
 * `art_px` follows the art pipeline's own rule (python/tatbot_art image.py), so the width set here is the
 * width that gets drawn: light, nearly neutral pixels are paper (CIELAB L* > 85, chroma < 12), the art is
 * the bounding box of the rest (any channel under 0.98), padded by 3% of the image's longer side.
 */
import { canonicalDigest, canonicalJson, ContractError } from "./human-representation/schema.ts";
import { sha256Hex } from "./sha256.ts";

export const ART_SCHEMA = "tatbot.inkmap-art/1";
export const MAX_ART_BYTES = 8_000_000;
/** A design's width when it is first placed, mm; the height follows its art's aspect. */
export const DEFAULT_ART_WIDTH_MM = 60;
/** A tall design's height when it is first placed, mm, so a strip starts on the limb rather than past it. */
const DEFAULT_ART_HEIGHT_MM = 120;
export const MIN_WIDTH_MM = 10;
/** The longest a design's side may be, either way round. */
export const MAX_SIDE_MM = 300;

export type ArtMediaType = "image/png" | "image/jpeg";

export interface ArtImage {
  media_type: ArtMediaType;
  sha256: string;
  width_px: number;
  height_px: number;
  data_base64: string;
}

export interface ArtSource {
  kind: "stock" | "imported" | "generated";
  identifier: string | null;
  license: string | null;
  attribution: string | null;
  generation: { prompt: string; model: string | null; generator: string | null } | null;
}

/** What the art pipeline needs beyond the image: the inks the design was made for and its separation style. */
export interface ArtPalette { inks: string[]; style: string | null }

export interface ArtRecord {
  schema: typeof ART_SCHEMA;
  content_sha256: string;
  name: string;
  image: ArtImage;
  /** The art inside the image, [x, y, width, height] in pixels from the top left. */
  art_px: [number, number, number, number];
  source: ArtSource;
  palette: ArtPalette | null;
}

const fail = (path: string, detail: string, code = "art_record_invalid"): never => { throw new ContractError(code, path, detail); };

function keys(value: unknown, required: string[], path: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(path, "expected object");
  if (Object.keys(value as object).sort().join("|") !== [...required].sort().join("|")) fail(path, "missing or unknown fields");
}

const text = (value: unknown, path: string, nullable = false): void => {
  if (nullable && value === null) return;
  if (typeof value !== "string" || !value.trim() || value.length > 10_000) fail(path, "expected bounded nonempty string");
};

/** SHA-256 of bytes, however they sit in their buffer. */
export const sha256Bytes = (bytes: Uint8Array): Promise<string> => sha256Hex(bytes.slice().buffer);

export function base64ToBytes(data: string): Uint8Array {
  const raw = atob(data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let raw = "";
  for (let i = 0; i < bytes.length; i += 0x8000) raw += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(raw);
}

/** Pixel size from a PNG's IHDR or a JPEG's frame header, without decoding the image. */
export function imageDimensions(bytes: Uint8Array, mediaType: ArtMediaType): [number, number] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (mediaType === "image/png") {
    const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (bytes.length < 24 || signature.some((b, i) => bytes[i] !== b)) fail("$.image", "not a PNG");
    return [view.getUint32(16), view.getUint32(20)];
  }
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) fail("$.image", "not a JPEG");
  for (let i = 2; i + 9 < bytes.length;) {
    if (bytes[i] !== 0xff) { i++; continue; }
    const marker = bytes[i + 1];
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { i += 2; continue; }
    const length = view.getUint16(i + 2);
    if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
      return [view.getUint16(i + 7), view.getUint16(i + 5)];
    }
    i += 2 + length;
  }
  return fail("$.image", "JPEG has no frame header");
}

/** The EXIF orientation a JPEG asks for, 1 when it asks for none or its metadata cannot be read. */
export function jpegOrientation(bytes: Uint8Array): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  try {
    for (let i = 2; bytes[i] === 0xff && bytes[i + 1] !== 0xda;) {
      if (bytes[i + 1] === 0xe1 && String.fromCharCode(...bytes.subarray(i + 4, i + 10)) === "Exif\0\0") {
        const tiff = i + 10, little = view.getUint16(tiff) === 0x4949;
        const entries = tiff + view.getUint32(tiff + 4, little);
        for (let index = 0; index < view.getUint16(entries, little); index++) {
          const entry = entries + 2 + 12 * index;
          if (view.getUint16(entry, little) === 0x0112) return view.getUint16(entry + 8, little);
        }
        return 1;
      }
      i += 2 + view.getUint16(i + 2);
    }
  } catch { /* a truncated header asks for nothing */ }
  return 1;
}

function validateSource(value: unknown): asserts value is ArtSource {
  keys(value, ["kind", "identifier", "license", "attribution", "generation"], "$.source");
  if (!["stock", "imported", "generated"].includes(String(value.kind))) fail("$.source.kind", "unknown source kind");
  for (const field of ["identifier", "license", "attribution"]) text(value[field], `$.source.${field}`, true);
  if (value.generation === null) return;
  keys(value.generation, ["prompt", "model", "generator"], "$.source.generation");
  const generation = value.generation as Record<string, unknown>;
  text(generation.prompt, "$.source.generation.prompt");
  text(generation.model, "$.source.generation.model", true);
  text(generation.generator, "$.source.generation.generator", true);
}

/** Synchronous structural admission; `validateArtRecord` also checks digests and the image bytes. */
export function validateArtShape(value: unknown): asserts value is ArtRecord {
  keys(value, ["schema", "content_sha256", "name", "image", "art_px", "source", "palette"], "$");
  if (value.schema !== ART_SCHEMA) fail("$.schema", "unsupported art version: Inkmap places source art (PNG or JPEG), not stroke artwork");
  text(value.name, "$.name");
  if (typeof value.content_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.content_sha256)) fail("$.content_sha256", "expected SHA-256");
  keys(value.image, ["media_type", "sha256", "width_px", "height_px", "data_base64"], "$.image");
  const image = value.image as Record<string, unknown>;
  if (image.media_type !== "image/png" && image.media_type !== "image/jpeg") fail("$.image.media_type", "PNG or JPEG only");
  if (typeof image.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(image.sha256)) fail("$.image.sha256", "expected SHA-256");
  for (const field of ["width_px", "height_px"]) {
    const v = image[field];
    if (!Number.isSafeInteger(v) || (v as number) < 8 || (v as number) > 16_384) fail(`$.image.${field}`, "expected 8..16384 pixels");
  }
  if (typeof image.data_base64 !== "string" || !image.data_base64 || image.data_base64.length > MAX_ART_BYTES * 4 / 3 + 4) fail("$.image.data_base64", "expected the image bytes, at most 8 MB");
  const box = value.art_px;
  if (!Array.isArray(box) || box.length !== 4 || !box.every(v => Number.isSafeInteger(v) && v >= 0)) fail("$.art_px", "expected four whole pixel values");
  const [x, y, w, h] = box as number[];
  if (w < 1 || h < 1 || x + w > (image.width_px as number) || y + h > (image.height_px as number)) fail("$.art_px", "the art lies outside its image");
  validateSource(value.source);
  if (value.palette !== null) {
    keys(value.palette, ["inks", "style"], "$.palette");
    const palette = value.palette as Record<string, unknown>;
    if (!Array.isArray(palette.inks) || palette.inks.length < 1 || palette.inks.length > 5 || !palette.inks.every(ink => typeof ink === "string" && /^[a-z0-9_]+$/.test(ink))) {
      fail("$.palette.inks", "expected one to five ink ids");
    }
    text(palette.style, "$.palette.style", true);
  }
  canonicalJson(value);
}

export async function validateArtRecord(value: unknown): Promise<ArtRecord> {
  validateArtShape(value);
  if (value.content_sha256 !== await canonicalDigest(value)) fail("$.content_sha256", "record digest mismatch", "wrong_hash");
  const bytes = base64ToBytes(value.image.data_base64);
  if (bytes.length > MAX_ART_BYTES) fail("$.image", "image exceeds 8 MB");
  if (await sha256Bytes(bytes) !== value.image.sha256) fail("$.image.sha256", "image bytes differ from their digest", "wrong_hash");
  const [width, height] = imageDimensions(bytes, value.image.media_type);
  if (width !== value.image.width_px || height !== value.image.height_px) fail("$.image", "image size differs from the record");
  return structuredClone(value);
}

export async function makeArtRecord(input: Omit<ArtRecord, "schema" | "content_sha256">): Promise<ArtRecord> {
  const record: ArtRecord = { ...structuredClone(input), schema: ART_SCHEMA, content_sha256: "0".repeat(64) };
  record.content_sha256 = await canonicalDigest(record);
  return validateArtRecord(record);
}

/** The placed size of `record` at `widthMm`: its art's aspect, the width to 0.1 mm (the pipeline's 10 px/mm). */
export function artSizeMm(record: Pick<ArtRecord, "art_px">, widthMm?: number): [number, number] {
  const aspect = record.art_px[2] / record.art_px[3];
  const width = Math.round((widthMm ?? Math.max(MIN_WIDTH_MM, Math.min(DEFAULT_ART_WIDTH_MM, DEFAULT_ART_HEIGHT_MM * aspect))) * 10) / 10;
  return [width, Number((width / aspect).toFixed(3))];
}

/** The widest a design of this shape may be, with its height inside the limit too. */
export function maxWidthMm([width, height]: readonly [number, number]): number {
  return Math.max(MIN_WIDTH_MM, Math.floor(10 * Math.min(MAX_SIDE_MM, MAX_SIDE_MM * width / height)) / 10);
}

// ---- the art pipeline's paper and crop rules, on decoded RGBA pixels ----------------------------------------

const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const labF = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);

/** An 8-bit sRGB colour in CIELAB (D65). */
export function lab(r: number, g: number, b: number): [number, number, number] {
  const [lr, lg, lb] = [linear(r / 255), linear(g / 255), linear(b / 255)];
  const fx = labF((0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047);
  const fy = labF(0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb);
  const fz = labF((0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** True for paper: light and nearly neutral in CIELAB, the pipeline's `flatten_paper`. */
export function isPaper(r: number, g: number, b: number): boolean {
  const [lightness, a, bb] = lab(r, g, b);
  return lightness > 85 && Math.hypot(a, bb) < 12;
}

/** Python's round: halves to even, so a margin matches the pipeline's to the pixel. */
const roundHalfEven = (value: number) => {
  const floor = Math.floor(value), rest = value - floor;
  return rest > 0.5 ? floor + 1 : rest < 0.5 ? floor : floor % 2 === 0 ? floor : floor + 1;
};

/** The art's rectangle in an RGBA image: ink is any non-paper pixel with a channel under 0.98, padded by 3%. */
export function artBox(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number, margin = 0.03): [number, number, number, number] {
  let top = height, bottom = -1, left = width, right = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = 4 * (y * width + x), r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      if (isPaper(r, g, b) || Math.min(r, g, b) / 255 >= 0.98) continue;
      if (y < top) top = y;
      if (y > bottom) bottom = y;
      if (x < left) left = x;
      if (x > right) right = x;
    }
  }
  if (bottom < 0) fail("$.image", "the image has no artwork: every pixel is paper");
  const pad = roundHalfEven(margin * Math.max(width, height));
  const x0 = Math.max(left - pad, 0), y0 = Math.max(top - pad, 0);
  return [x0, y0, Math.min(right + pad + 1, width) - x0, Math.min(bottom + pad + 1, height) - y0];
}

/** The art as Inkmap shows it: its rectangle only, paper made transparent so the skin shows through. */
export function inkOnly(rgba: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba);
  for (let i = 0; i < out.length; i += 4) if (isPaper(out[i], out[i + 1], out[i + 2]) || Math.min(out[i], out[i + 1], out[i + 2]) / 255 >= 0.98) out[i + 3] = 0;
  return out;
}
