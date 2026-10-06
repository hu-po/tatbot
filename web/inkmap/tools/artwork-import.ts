/** Synthetic contract/source tooling only; editor and ROS admit native DBV3 acquisitions. */
import { makeArtworkRecord, type ArtworkRecord, type ArtworkSource } from "../src/core/artwork-record.ts";
import { SVG_ADAPTER, tattooProgramFromSvg, type SvgProgramOptions } from "../src/core/human-representation/svg-program.ts";

export async function artworkFromSvg(input: {
  name: string; original_svg: string; source: ArtworkSource;
  conversion: SvgProgramOptions & { adapter?: string };
}): Promise<ArtworkRecord> {
  const program = await tattooProgramFromSvg(input.original_svg, input.conversion);
  return makeArtworkRecord({ name: input.name, source_sha256: program.provenance.source_sha256 as string,
    source: input.source, program, conversion: { adapter: SVG_ADAPTER, recipe_sha256: null,
      chord_error_m: input.conversion.chord_error_m ?? .0001 } });
}
