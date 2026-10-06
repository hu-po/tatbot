/** Opening a portable design: read it, then send it where it belongs.
 *
 * The user has one file and one "Open design" control. Which editor can show it
 * is a property of the file, not a question to ask them — and when neither can
 * show all of it, the honest answer is to say which feature is unsupported and
 * keep the original bytes for download, not to rebuild a lossy approximation
 * and call it the same file.
 */
import { requireAcquiredArtwork } from "./artwork-record.ts";
import { parseDesign, MAX_DESIGN_BYTES, type InkmapDesign } from "./design.ts";

export type DesignRoute =
  | { kind: "body"; design: InkmapDesign }
  | { kind: "chart"; design: InkmapDesign }
  | { kind: "unsupported"; design: InkmapDesign; reason: string };

/** Where a validated design can be edited, and why not, when it cannot be. */
export function routeDesign(design: InkmapDesign): DesignRoute {
  for (const record of Object.values(design.artworks)) requireAcquiredArtwork(record);
  const kinds = new Set(design.placements.map(item => item.placement.target.kind));
  if (kinds.size !== 1) {
    return { kind: "unsupported", design, reason:
      `this design mixes ${[...kinds].sort().join(" and ")} targets in one file; each editor shows one kind at a time` };
  }
  const [kind] = kinds;
  if (kind === "body") return { kind: "body", design };
  if (kind !== "plane" && kind !== "cylinder") {
    return { kind: "unsupported", design, reason: `unsupported placement target ${String(kind)}` };
  }
  const warped = design.placements.find(item => item.placement.warp !== null);
  if (warped) {
    return { kind: "unsupported", design, reason:
      `placement ${warped.id} carries a surface warp, which this editor cannot represent` };
  }
  return { kind: "chart", design };
}

/** One file in, one route out. Size and schema refusals come from the reader. */
export async function openDesignFile(file: File): Promise<DesignRoute> {
  if (file.size > MAX_DESIGN_BYTES) throw new Error("Design exceeds 20 MB");
  return routeDesign(await parseDesign(await file.text()));
}
