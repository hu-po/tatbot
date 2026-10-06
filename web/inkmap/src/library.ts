import type { DesignMeta } from "./core/schema.ts";

/** How many designs the short library shows before "Browse all". */
export const SHORT_LIBRARY = 6;

/** Artwork made in this session (generated, imported) first, newest first, then the curated
 *  stock in its manifest order — never a recency invented for a fresh project. */
export function shortLibrary(designs: DesignMeta[], limit = SHORT_LIBRARY): DesignMeta[] {
  const visible = visibleLibrary(designs);
  const session = visible.filter((d) => !d.sourcePath).reverse();
  const stock = visible.filter((d) => d.sourcePath);
  return [...session, ...stock].slice(0, limit);
}

/** What the picker offers: every design the session made, and the stock
 *  acquired examples in manifest order. */
export function visibleLibrary(designs: DesignMeta[]): DesignMeta[] {
  return designs.filter((d) => d.library !== false);
}
