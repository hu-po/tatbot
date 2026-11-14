/** Source art in the browser: the image decoded, the ink-only canvas a decal and a thumbnail show, and a
 *  record made from a dropped file. Pure record rules live in art.ts. */
import * as THREE from "three";
import { artBox, base64ToBytes, bytesToBase64, jpegOrientation, inkOnly, makeArtRecord, MAX_ART_BYTES, sha256Bytes, type ArtMediaType, type ArtRecord, type ArtSource } from "./art.ts";
import { artNotes, profileArt, suggestInks, type ArtProfile } from "./art-profile.ts";
import { inkById, type Ink } from "./inks.ts";
import type { InkmapRender } from "./render-record.ts";

/** Textures stay at most this many pixels on their longer side; 1024 px over a 60 mm design is 17 px/mm. */
const MAX_TEXTURE_PX = 2048;
/** A profile reads the art at most this many pixels on its longer side: an ink's fattest spot is millimetres wide. */
const MAX_PROFILE_PX = 512;
const canvases = new Map<string, Promise<HTMLCanvasElement>>();
const profiles = new Map<string, Promise<ArtProfile>>();

async function decode(bytes: Uint8Array, mediaType: ArtMediaType): Promise<ImageBitmap> {
  return createImageBitmap(new Blob([bytes as BlobPart], { type: mediaType }));
}

function context(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  return [canvas, canvas.getContext("2d", { willReadFrequently: true })!];
}

/** A canvas of white paper: what a transparent image is laid on here and in the art pipeline (image.py `on_paper`),
 *  so the two read one picture. Drawn on nothing, its transparent pixels would read as black ink. */
function paper(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const [canvas, ctx] = context(width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  return [canvas, ctx];
}

/** The record's art rectangle with paper transparent: what is inked, on whatever the skin is. */
export function artCanvas(record: ArtRecord): Promise<HTMLCanvasElement> {
  const key = `${record.image.sha256}:${record.art_px.join()}`;   // a change of inks draws the same picture
  let held = canvases.get(key);
  if (!held) {
    held = (async () => {
      const bitmap = await decode(base64ToBytes(record.image.data_base64), record.image.media_type);
      const [x, y, w, h] = record.art_px;
      const scale = Math.min(1, MAX_TEXTURE_PX / Math.max(w, h));
      const [canvas, ctx] = paper(Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)));
      ctx.drawImage(bitmap, x, y, w, h, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
      ctx.putImageData(new ImageData(inkOnly(pixels.data) as ImageDataArray, canvas.width, canvas.height), 0, 0);
      return canvas;
    })();
    if (canvases.size >= 64) canvases.delete(canvases.keys().next().value!);
    canvases.set(key, held);
    held.catch(() => canvases.delete(key));
  }
  return held;
}

/** The art rectangle's pixels, scaled to what a profile reads. */
async function profilePixels(record: ArtRecord): Promise<[Uint8ClampedArray, number, number]> {
  const source = await artCanvas(record);
  const scale = Math.min(1, MAX_PROFILE_PX / Math.max(source.width, source.height));
  const [canvas, ctx] = context(Math.max(1, Math.round(source.width * scale)), Math.max(1, Math.round(source.height * scale)));
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return [ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height];
}

/** The record's inks that Inkmap knows, whose colours it can match. */
export const knownInks = (record: ArtRecord): Ink[] => (record.palette?.inks ?? []).flatMap((id) => inkById.get(id) ?? []);

/** What the art holds for its own inks; the digest covers them, so a change of inks reads afresh. */
export function artProfile(record: ArtRecord): Promise<ArtProfile> {
  let held = profiles.get(record.content_sha256);
  if (!held) {
    held = profilePixels(record).then(([rgba, width, height]) => profileArt(rgba, width, height, knownInks(record)));
    if (profiles.size >= 64) profiles.delete(profiles.keys().next().value!);
    profiles.set(record.content_sha256, held);
    held.catch(() => profiles.delete(record.content_sha256));
  }
  return held;
}

/** A texture for one decal; each decal owns its texture because mirroring mutates it. */
export async function artTexture(record: ArtRecord): Promise<THREE.CanvasTexture> {
  const texture = new THREE.CanvasTexture(await artCanvas(record));
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return texture;
}

/** A robot render as a texture. It is ink over white, shown by multiplying, so nothing here cuts the white away. */
export async function renderTexture(render: InkmapRender): Promise<THREE.CanvasTexture> {
  const bitmap = await decode(base64ToBytes(render.image.data_base64), "image/png");
  const [canvas, ctx] = context(bitmap.width, bitmap.height);
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return texture;
}

/** A small PNG of the art for the library and lists. */
export async function artThumbnail(record: ArtRecord, size = 256): Promise<string> {
  const source = await artCanvas(record);
  const scale = Math.min(1, size / Math.max(source.width, source.height));
  const [canvas, ctx] = context(Math.max(1, Math.round(source.width * scale)), Math.max(1, Math.round(source.height * scale)));
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

/** `tatbot.art-generation/1`, the art corpus's sidecar beside a generated image: its prompt and model. */
interface GenerationSidecar { schema: "tatbot.art-generation/1"; prompt: string; model?: string; generator?: string; subject?: string }

function mediaTypeOf(file: File, bytes: Uint8Array): ArtMediaType {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return "image/jpeg";
  throw new Error(`${file.name}: Inkmap places PNG or JPEG art; convert other images first`);
}

async function uprightJpeg(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
  if (!blob) throw new Error("the browser could not re-encode this photo");
  return new Uint8Array(await blob.arrayBuffer());
}

/** A record from an image file, with the generation sidecar's provenance when one is given. */
export async function artFromFile(file: File, sidecar?: unknown): Promise<ArtRecord> {
  if (file.size > MAX_ART_BYTES) throw new Error(`${file.name} exceeds 8 MB`);
  const original = new Uint8Array(await file.arrayBuffer());
  const mediaType = mediaTypeOf(file, original);
  const bitmap = await decode(original, mediaType);   // a browser turns a JPEG as its EXIF says
  const { width, height } = bitmap;
  const [canvas, ctx] = paper(width, height);
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const box = artBox(ctx.getImageData(0, 0, width, height).data, width, height);
  // A record's pixels are its header's, so a turned photo is kept upright rather than turned on every read.
  const bytes = mediaType === "image/jpeg" && jpegOrientation(original) !== 1 ? await uprightJpeg(canvas) : original;
  const side = (sidecar && typeof sidecar === "object" && (sidecar as { schema?: unknown }).schema === "tatbot.art-generation/1")
    ? sidecar as GenerationSidecar : null;
  const stem = file.name.replace(/\.[^.]+$/, "");
  const source: ArtSource = side
    ? { kind: "generated", identifier: stem, license: null, attribution: null,
      generation: { prompt: side.prompt, model: side.model ?? null, generator: side.generator ?? null } }
    : { kind: "imported", identifier: stem, license: null, attribution: null, generation: null };
  const record = await makeArtRecord({ name: side?.subject ?? stem, art_px: box, source, palette: null,
    image: { media_type: mediaType, sha256: await sha256Bytes(bytes), width_px: width, height_px: height, data_base64: bytesToBase64(bytes) } });
  return withInks(record, suggestInks(...await profilePixels(record)));
}

/** What a person should know about the art just imported, in plain words: see `artNotes`. */
export async function importNotes(record: ArtRecord): Promise<string[]> {
  const [rgba, width, height] = await profilePixels(record);
  return artNotes(rgba, width, height, 0.03 * Math.max(record.image.width_px, record.image.height_px) * width / record.art_px[2]);
}

/** The record with these inks, its style kept; a new digest, the same picture. */
export function withInks(record: ArtRecord, inks: string[]): Promise<ArtRecord> {
  const { schema: _schema, content_sha256: _digest, ...rest } = record;
  return makeArtRecord({ ...rest, palette: { inks, style: record.palette?.style ?? null } });
}
