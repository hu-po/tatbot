import { artworkFromSvg } from "../tools/artwork-import.ts";
import type { ArtworkSource } from "../src/core/artwork-record.ts";

export function fixtureArtwork(svg: string, name = "fixture", size: [number, number] = [30, 30], source?: ArtworkSource) {
  return artworkFromSvg({ name, original_svg: svg,
    source: source ?? { kind: "fixture", identifier: name, license: "CC0-1.0", attribution: null, generation: null },
    conversion: { canvas_m: size.map(mm => mm / 1000) as [number, number], semantic_intent: name,
      width_m: .0003, deposition: 1, chord_error_m: .000005 } });
}
