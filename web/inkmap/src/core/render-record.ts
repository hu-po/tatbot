/** A robot render (`tatbot.inkmap-render/1`, config/inkmap/render.schema.json): what `tatbot art separate --brief`
 *  gives back, the predicted ink of one placement as a picture on white, with what it measured.
 *
 * It is a picture, never strokes: Inkmap shows it on the body so the client approves what will be drawn, not the source
 * image. Strokes are made at one size, so a render holds only for its art, width and mirror. Python's half is
 * `tatbot_contracts.art`. */
import { base64ToBytes, imageDimensions, sha256Bytes } from "./art.ts";
import { canonicalDigest } from "./human-representation/schema.ts";

export const RENDER_SCHEMA = "tatbot.inkmap-render/1";
/** Renders kept in this browser: each is a picture of up to a few hundred kilobytes. */
export const MAX_RENDERS = 24;

export interface InkmapRender {
  schema: typeof RENDER_SCHEMA;
  content_sha256: string;
  art_sha256: string;
  width_mm: number;
  height_mm: number;
  /** The placement was mirrored, and so already is this picture. */
  mirror: boolean;
  image: { media_type: "image/png"; sha256: string; width_px: number; height_px: number; data_base64: string };
  minutes: number;
  dips: number;
  rinses: number;
  lost_inks: string[];
  weakest_source: string;
}

const FIELDS = "art_sha256|content_sha256|dips|height_mm|image|lost_inks|minutes|mirror|rinses|schema|weakest_source|width_mm";
const fail = (detail: string): never => { throw new Error(`render_invalid: ${detail}`); };

export async function validateRender(value: unknown): Promise<InkmapRender> {
  if (!value || typeof value !== "object") fail("not an object");
  const render = value as InkmapRender;
  if (render.schema !== RENDER_SCHEMA) fail("this is not a robot render");
  if (Object.keys(render).sort().join("|") !== FIELDS) fail("missing or unknown fields");
  if (render.content_sha256 !== await canonicalDigest(render)) fail("content digest differs");
  if (!/^[0-9a-f]{64}$/.test(render.art_sha256)) fail("art_sha256 must be a SHA-256");
  if (!(render.width_mm > 0 && render.height_mm > 0)) fail("needs a positive size");
  const bytes = base64ToBytes(render.image.data_base64);
  if (await sha256Bytes(bytes) !== render.image.sha256) fail("image bytes differ from their digest");
  const [width, height] = imageDimensions(bytes, "image/png");
  if (width !== render.image.width_px || height !== render.image.height_px) fail("image size differs from the record");
  return structuredClone(render);
}

const sameWidth = (a: number, b: number) => Math.abs(a - b) < 0.05;

/** The render of exactly this art at this size and mirror. The art's digest covers its inks, so new inks need a new one. */
export function findRender(renders: readonly InkmapRender[], artSha: string | undefined, widthMm: number, mirror: boolean): InkmapRender | null {
  return renders.find((render) => render.art_sha256 === artSha && render.mirror === mirror && sameWidth(render.width_mm, widthMm)) ?? null;
}

/** The nearest render of this art at another width, to say why none is showing. */
export function renderAtOtherWidth(renders: readonly InkmapRender[], artSha: string | undefined, widthMm: number, mirror: boolean): InkmapRender | null {
  const others = renders.filter((render) => render.art_sha256 === artSha && render.mirror === mirror && !sameWidth(render.width_mm, widthMm));
  return others.sort((a, b) => Math.abs(a.width_mm - widthMm) - Math.abs(b.width_mm - widthMm))[0] ?? null;
}
