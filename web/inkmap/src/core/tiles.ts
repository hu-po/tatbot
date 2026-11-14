/** The fiducial stencil tiles a design is drawn on (config/stencil-tiles.json, which the tile generator reads too).
 *  The whole tile is the page, and the compile centres the design on it without turning it, so a design fits when
 *  its own width and height do. A size that does not fit is advice here: the compile is what refuses it. */
import tilesJson from "../../../../config/stencil-tiles.json" with { type: "json" };

export interface StencilTile {
  id: string;
  name: string;
  /** What the layout goes on: `skin`, or `silicone` for the practice pad. */
  for: string;
  size_mm: [number, number];
}

export const STENCIL_TILES: StencilTile[] = Object.entries(tilesJson.tiles).map(([id, tile]) =>
  ({ id, name: tile.name, for: tile.for, size_mm: tile.tile_mm as [number, number] }));

export const DEFAULT_TILE_ID = "forearm";

/** How far a design passes the tile's width and height, mm; zero where it fits. */
export function overflowMm(tile: StencilTile, size: readonly [number, number]): [number, number] {
  return [Math.max(0, size[0] - tile.size_mm[0]), Math.max(0, size[1] - tile.size_mm[1])];
}

/** The widest a design of this aspect fits the tile, to the 0.1 mm a width is kept to. */
export function fitWidthMm(tile: StencilTile, size: readonly [number, number]): number {
  return Math.floor(10 * Math.min(tile.size_mm[0], tile.size_mm[1] * size[0] / size[1]) + 1e-9) / 10;
}
