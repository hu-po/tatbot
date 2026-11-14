---
name: tatbot-dbv3-recipes
description: DrawingBotV3 path-finding recipes for tatbot. Which PFM and settings give outlines, solid fills, angled hatching, contour shading, magnum swaths and continuous fills, what each costs in strokes and line, and the traps that break acquisition. Use when writing tatbot.dbv3-job/3 jobs, choosing a fill style for a region, or debugging a DBV3 acquisition.
---

# DBV3 recipes

DrawingBotV3 (DBV3) makes every drawing path tatbot draws. These recipes are the ones exercised on
this project. The setting names are DBV3's own, and they are read back after every acquisition, so a
typo fails loudly. Background: `docs/drawingbot.md`. Job format: `tatbot.dbv3-job/3`.

## Running it

```bash
tatbot drawingbot generate job.json --out NEW_DIR --app /opt/drawingbotv3
```

- **Version.** Premium 1.6.22 only. Its activation is per machine; check with a known job (the
  multicolour fixture in `scripts/tests/fixtures/dbv3-multicolor/`) before a batch.
- **One acquisition per (ink, fill style).** Every acquisition of a piece shares one canvas size, and
  `tatbot ros compile` stacks them as layers.
- **Physical size.** Generate at the size you will draw. A size change is a new acquisition, never a
  rescale.
- **Pens.** Pens are fixed colours and separation is `Default`. `pens.py` refuses DBV3's CMYK and
  Colour Match modes.

## The recipe table

Widths are a 3RL-class liner at 0.6 mm generation width unless stated.

| Style | PFM | Settings | Distribution | Use and cost |
| --- | --- | --- | --- | --- |
| outline | ECS Drawing | `Edge Blur 5, Plotting Resolution 2, Draw Contours false, Draw Shading false` | Single pen | Edges only. Traces each edge out and back, so contact length about doubles. Outlines are often the largest time item. |
| solid | ECS Drawing | `Plotting Resolution 2, Draw Edges false, Draw Contours false` | Single pen | A packed fill of a region |
| hatch | Sketch Lines | defaults (Line Density 75) | Even weighted | Reads like hand-inked linework. At density 30–50 it loses a detailed drawing, though 50 held a dense linocut. |
| angled hatch, per ink | Streamlines Flow Field | `Threshold 250, Start Angle (−θ mod 180), Amplitude 0.001, X Frequency 0.001, Y Frequency 0.001, Min Spacing w, Max Spacing 10w, Min Length 5, Max Length 2000` | Luminance weighted | Optical mixing. Parallel lines at the ink's screen angle θ, spaced by tone, ending where the source is white. 0.001 is DBV3's floor for those three settings (0 does not read back). Narrow or ragged regions get only their cores, so use it for tints and mixes and `solid` for full regions. `tatbot art separate` writes it |
| magnum swath (9M) | Sketch Lines | `Start Angle Min 90, Start Angle Max 90, Drawing Delta Angle 0` | Even weighted | Pen 3.0 mm. Erode regions 1.6 mm first, because the needle row lays ink only moving along the print's y. Overruns fine silhouettes. |
| contour shading | Streamlines Edge Field | `Threshold 250, Min Spacing 1, Max Spacing 4` | Luminance weighted | Lines follow form, but many short paths (median about 2 mm), so stroke count is the cost |
| flow fill | Streamlines Flow Field | `Threshold 250` | Luminance weighted | Organic texture |
| sweep | Sketch Sweeping Curves | defaults | Even weighted | Long continuous paths (median about 130 mm): the fewest strokes per area, but they cross white between a layer's regions (see below) |
| waves | Sketch Waves | defaults | Even weighted | Diagonal wavy fill; crosses white between regions |
| maze | Voronoi TSP | `Ignore White true` | Single pen | One continuous path through every region of the layer, so it crosses the white between them |
| squares | Sketch Squares | defaults, source grey 40 | Even weighted | Right-angled squiggles. Test region: 100% of the line inside, 97% covered, 38 paths |
| curves | Sketch Curves | defaults, source grey 40 | Even weighted | Loose curved strokes. 99% inside, 98% covered, 15 paths |
| superformula | Sketch Superformula | defaults, source grey 40 | Even weighted | Star-like spokes; long spokes cross to other regions |

**Never use Sketch Lines at a fixed angle for regions.** With `Drawing Delta Angle 0`, its squiggles
run straight on out of their region for 10–20 mm. Measured: only 17% of the line stayed inside the
region. Free-angle Sketch Lines (the defaults) are fine for a whole-image sketch.

**DBV3's angle runs clockwise.** Start Angle a draws lines at −a in the artwork's y-up frame: 45 gives
lines at 135°.

**Tone response.** Grey to coverage, measured with the art studio's step wedge:
- the flattened flow field at 0.6 mm: grey 80 → 92%, 120 → 57%, 160 → 40%, 200 → 28%;
- below grey 80 DBV3 only adds line (429 mm/cm² at grey 0 against 172 at grey 80), so a solid uses
  grey 70.
`tatbot art tone` re-measures it.

**Distribution must match the PFM family.** Sketch PFMs need `Even weighted`. Streamlines need
`Luminance weighted`. ECS and TSP take `Single pen`. With the wrong one the PFM silently draws little
or nothing.

## Region containment

A fill for a region must keep its line inside the region, or it draws ink across bare skin. On real
designs, a layer is many separate regions of one ink, and a PFM that travels between them inks the gap.
Measured over every acquisition in the art corpus on 2026-10-30 (409 layers, 60 mm designs): the share
of each fill's line more than 0.4 mm outside its own layer's source:

| Fill | Layers | Line | Outside | Worst layer |
| --- | --- | --- | --- | --- |
| packed (ECS) | 163 | 343 m | 0.0% | 0% |
| hatch (flattened flow field) | 94 | 10 m | 0.0% | 0% |
| contour (Streamlines Edge Field) | 46 | 35 m | 0.0% | 0% |
| flow (Streamlines Flow Field) | 23 | 15 m | 0.1% | 1% |
| squares (Sketch Squares) | 13 | 23 m | 0.5% | 1% |
| edges (ECS outline) | 62 | 213 m | 0.7% | 5% |
| curves (Sketch Curves) | 13 | 40 m | 3.0% | 5% |
| superformula (Sketch Superformula) | 13 | 44 m | 4.5% | 11% |
| waves (Sketch Waves) | 47 | 112 m | 6.4% | 26% |
| sweep (Sketch Sweeping Curves) | 22 | 57 m | 7.8% | 42% |
| maze (Voronoi TSP) | 61 | 65 m | 9.5% | 57% |

Before that, a single 30 mm test source (a disc, a 2 mm bar and a ring, its regions close together)
screened candidates. It passes fills that leak on real designs, so treat it as a first filter only:

| PFM | Inside | Covered | Paths | Verdict |
| --- | --- | --- | --- | --- |
| ECS Drawing (packed) | 100% | 97% | 53 | use |
| Sketch Squares | 100% | 97% | 38 | use |
| Sketch Curves | 99% | 98% | 15 | leaks on real designs (above) |
| Sketch Superformula | 98% | 92% | 52 | leaks on real designs (above) |
| Adaptive Circular Scribbles (`Ignore White`) | 95% | 83% | 4 | avoid: its merged path crosses white between regions |
| Voronoi Tree (`Ignore White`) | 98% | 65% | 189 | avoid: median path 1.4 mm |
| Voronoi Triangulation (`Ignore White`) | 86% | 96% | 214 | avoid: edges span the gaps between regions |
| Voronoi Diagram (`Ignore White`) | 65% | 96% | 662 | avoid: cells fill the whole canvas |
| Hatch Circular Scribbles | 75% | 68% | 10 | avoid |
| Hatch Sawtooth | refused | | | its paths leave the canvas |
| Streamlines Spiral | refused | | | DBV3 raised `NoSuchElementException` |

## Cost

Measured on one region with the same source (10-24 fill study):

| Recipe | Paths | Line (mm) |
| --- | --- | --- |
| Sketch Lines | 33 | 2,081 |
| Sketch Sweeping Curves | 15 | 2,848 |
| ECS default | 56 | 2,934 |
| Streamlines Edge Field | 370 | 968 |
| Adaptive Dashes | 626 | 486 (median path 0.7 mm: avoid) |

Time is roughly:

- **Strokes:** about 3 s each, for descent and lift.
- **Contact:** line length at the draw speed.
- **Dips:** line ÷ `mm_per_dip` (60 mm for colour inks, 130 mm for black on the 3RL) × about 48 s.
- **Rinses:** one per ink change, about 40 s.

`tatbot ros compile` gives the modeled seconds, and `tatbot art render` adds the gaps.

Every recipe above lays three or four passes per spot of a solid region: ECS about 4 mm of line per mm² at any
pen width. For line art and flat letter fills, `tatbot art separate --fill offset` draws the regions without DBV3,
one pass per mark (about 1 mm of line per mm² for a 1 mm mark), with a cartridge per ink by `--tool` (docs/art.md,
"The offset fill").

## Traps

- **No pen colour channel at exactly 255** (and no exact primaries other than black). DBV3 exports
  such pens as colour names the decoder refuses. Use 254: `[254, 0, 0, 255]`.
- **Stippling PFMs** (Voronoi or Adaptive Stippling) export SVG circles, which the decoder refuses.
- **Adaptive TSP** exports nothing.
- **Export naming.** `svgLayerNaming` must be `%NAME%`, and `Random Seed` must be an integer.
- **Region edges.** Erode a region's mask before filling it, by half the tool's swath, or the fill
  overruns its outline.
- **aarch64 nodes** need a JDK with JavaFX inside the installation.

## Untested but promising

- **SVG Converter PFM** (Premium) imports an SVG and hatches flat regions with per-pen angle and
  spacing. It could take exact geometry, such as a seed-of-life grid written as SVG circles, without
  tracing a raster. Test that the decoder accepts its output before relying on it.
- **Colour Match and CMYK separation**, as a baseline against the art studio's own separation. They
  are blocked by `pens.py` until the owner decides.
