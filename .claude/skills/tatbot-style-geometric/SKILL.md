---
name: tatbot-style-geometric
description: Geometric and sacred-geometry tattoo style for tatbot, including the seed-of-life brand motif. Exact construction, thin one-weight lines, symmetric cells filled in two inks, mapped to DBV3 recipes, with the exact seed-of-life construction, a tested Flow prompt and its failure modes. Use when a design is asked for as geometric, sacred geometry, mandala, seed or flower of life, or built from circles and polygons.
---

# Geometric and sacred geometry

Geometric work is built with compass and straightedge: circles, arcs, polygons and facets, all in one
precise line weight, with colour laid into chosen cells. Its brand motif is the **seed of life**, the
card that ends every tatbot short. Of all the styles, it is the one where exact geometry beats a
generated raster. Read `tatbot-color` and `tatbot-dbv3-recipes`.

## The look

- **Line.** Thin, even linework of one weight, with no hand wobble. The construction circles stay
  visible as part of the design.
- **Structure.** Symmetry: radial (mandala), mirror (animal heads) or lattice.
- **Colour.** Flat fills in some cells (vesicae, petals, facets), alternating or radiating. Leave most
  cells empty; colour in at most about 40% of them.
- **Subject.** The subject is built *from* the geometry: facets for a head, petals for a flower,
  links for a robot arm. Its silhouette must still read.

## Palette on the owned inks

Black lines plus two fill inks, usually a complementary or analogous pair:

| Pair | Effect |
| --- | --- |
| `bright_orange` + `light_purple` | Complementary warm-cool: flowers, mandalas |
| `blue_sky` + `golden_yellow` | Bright and clean: animals, sun |
| `bright_red` + `light_green` | Bold. Keep them in 60-30 proportion |
| `rose_pink` + `light_purple` | Analogous, soft |

Two-ink cells can carry a third colour as a hatch mix. Show it with `tatbot art swatch`.

## Constructing the seed of life exactly

Seven circles of radius r: one at the centre, and six with centres on the central circle at 0°, 60°,
120°, 180°, 240° and 300°. The flower of life adds a second ring of twelve:

- six at distance 2r on the same angles;
- six at distance r·√3 on the angles 30°, 90°, … 330°;

and is usually clipped by a bounding circle of radius 3r. Every intersection is a vesica (petal)
cell, which is a natural region for a fill.

**Write the grid as exact SVG circles; do not trace a raster of it.**
- Tracing an antialiased thin line with ECS edges finds both of its edges, so the line is drawn twice
  and wobbles.
- The intended route is DBV3's SVG Converter PFM with the SVG as source. It is untested on this
  decoder, so verify it once before relying on it (`tatbot-dbv3-recipes`).
- Until then, render the grid as a raster at 4 px or more per line width and trace it with ECS edges.
  Check the render for doubled lines, and keep the subject and fills as separate layers.

## Layers, tools and recipes

1. **Fill inks, one acquisition each.** Use an angled hatch at the ink's screen angle, eroded half a
   line width so fills stay inside the cell lines.
2. **Black construction and subject lines**, last, as one-weight 3RL lines.

Fine cells under about 2 mm across cannot hold a fill. Leave them empty.

## Flow prompt (tested 2026-10-30, Nano Banana Pro, 1:1)

> Geometric tattoo design of {subject} constructed on a seed-of-life circle grid. Precise thin black
> linework of one weight, compass and straight-edge construction circles kept as part of the design,
> symmetrical; flat fills in only two colours, {ink a} and {ink b}, inside some of the cells. No
> gradients, no shading, no texture. Plain white background, centered, isolated, no text.

Results:
- **Colours.** Kept exactly (orange and violet, sky blue and yellow, red and green).
- **Grid.** It dominates. The poppy turned into an abstract rosette. The sea lion head and the robot
  arm stayed legible.
- **Palette score.** Thin construction lines antialias to grey, so a palette-adherence measure
  undercounts them (24–51% on-palette). Threshold the lines before snapping colours.
- **The legibility fix (verified).** Add "The {subject}'s silhouette is bold and clearly readable,
  with {its defining parts}; the grid stays light behind it."
  - The poppy then came back as a poppy: four cupped petals and a stem, three times out of three.
  - Inked area halved (26% to 12%), which also makes the piece cheaper.
  - One of the three gave the silhouette a heavy outline in a second line weight; normalise it.
  - Alternatively, generate the subject alone and compose it with an exact grid.

## Separating a Flow image

`tatbot art separate --style geometric` keeps the darker lines and draws a one-line band as a single
pass. Flow's construction lines are thin and light grey, though.
- At 60 mm, faint grid lines fall to paper, and only the darkest circles survive, in fragments.
- The fills separate cleanly: about 33 min for the poppy.

Take the fills and the subject from Flow, and draw the grid from exact geometry.

## Failure modes

| Failure | Fix |
| --- | --- |
| Circles not quite concentric or tangent | Rebuild the grid as exact SVG; keep only the subject and fill choice from Flow |
| Too many cells filled, so the piece is heavy and slow | Fill alternate petals only. Check the dips in the render |
| Line weight varies | Normalise it: one tool, one width, a single pass |
| The subject is lost in the grid | Reduce the grid to the seed of life (7 circles) instead of the flower (19) |
