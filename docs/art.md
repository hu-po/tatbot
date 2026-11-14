# Art studio

`python/tatbot_art` is the offline art studio: what each ink looks like, which colours a palette load
can draw, and how a drawing program is predicted to look on paper, the silicone tile or skin. It runs
in the shared offline drawing interpreter (numpy, PyYAML, scipy) and contacts no robot, camera or
network. DrawingBotV3 makes the robot's paths ([DrawingBotV3](drawingbot.md)), and the studio decides colour
around them. Two kinds of path are the studio's own acquired paths: `tatbot art shade` makes whip and pendulum
shading, and the offset fill ([below](#the-offset-fill)) draws regions one pass per mark.

## Ink evidence

`config/inks.yaml` keeps each ink's `rgb` as its display colour. An optional `appearance` list records
observations, each with `srgb`, `source`, `substrate` and `date`. Sources rank from weakest to
strongest:

| Source | Meaning |
| --- | --- |
| `nominal` | the display `rgb`, used when nothing else is recorded |
| `vendor-photo` | the bottle's ink in a vendor's product photo: brightened, never on skin |
| `photo` | a photograph of drawn ink without a colour reference |
| `swatch` | a drawn swatch card measured against a colour reference |

The studio takes the strongest observation for the requested substrate, then one recorded for `any`.
Every output names the weakest source it used, so a render built on a nominal ink says so.

## The achievable set

`tatbot art swatch --inks a,b,c --out DIR` lists every colour a palette load can draw. An ink alone gives
tints by line pitch: coverage is the deposited width over the pitch. Two inks hatched at their own angles
give optical mixes. Each ink of a load keeps one hatch angle, at least 15 degrees from the next, and
black-like inks take 45 degrees.

The far colour of a hatched area follows a Yule-Nielsen modified Neugebauer model in linear sRGB
reflectance over the substrate, each ink, and their overlap, with `n = 2` until a swatch card fits it. A
translucent ink filters what lies under it. An opaque ink (`viscosity: opaque`, such as white or silver)
covers it. Two recipes within `--min-delta-e` CIEDE2000 of each other count as one colour, and the one
with less line per cm² is kept, because line length is what costs dips.

The command writes:

- `swatches.png`: each colour hatched as drawn, above its predicted far colour, ordered by hue.
- `swatches.json`: each colour's hex value, its layers (ink, angle, coverage, pitch) and its mm of line
  per cm² for each ink.

## Predicted render and minutes

`tatbot art render PROGRAM --substrate S --out DIR` draws a `tatbot-program` as ink:

- **Width.** Each stroke uses the tool's measured `line_width_m`, else `--width-mm`, else the stroke's
  generation width.
- **Dose.** A compiled program carries its needle pokes (`stats.dose`, from the planner's time law and the
  machine's rate). Each stroke's first and last millimetre, its middle and its turns then darken as that many
  steady passes, with a spot where the pen came down and lifted with the machine running, in place of the pass
  count alone. That a poke darkens as its share of a pass does is an assumption no ink has measured; where a
  stroke already covers fully the render shows no difference. A program without the record is drawn by its passes.
- **Passes.** Repeated passes of one ink darken, counted as `tatbot_ink.passes` counts them: another
  stroke, or one coming back over its own line after going more than its width along it (a retrace, a
  pendulum's next swing). Its joints and turns stay one pass, and so does a magnum's swept row. A render
  from before 2026-11-02, which took a stroke as one pass, differs by +0.02 ΔE on average in fidelity over
  46 corpus separations (at most +0.14) and ranks every fill study the same. Inks composite in program
  order, translucent ones filtering and opaque ones covering.
- **Fade.** A dipped ink fades over the last quarter of its `mm_per_dip` to 40% strength, so the dip
  schedule shows in the picture.
- **Deposition.** A stroke's `deposition`, one value or one per point, scales its ink linearly, so a
  whip's ramp from 1 to 0 fades along it. No ink has measured that curve; the report's `partial_mm` is
  each ink's line drawn at less than full deposition. A pendulum's overlapping swings darken as the same
  marks drawn as whips would. `tatbot ros compile` writes it on each fading stroke; `drawn.png` keeps each
  acquired path's own.
- **Spread.** On skin, `--spread-mm` blurs coverage before compositing; a larger value previews ageing.
- **Substrates.** `paper`, `silicone` and six skin tones, `skin-1` (light) to `skin-6` (deep). These
  are display colours, not measurements.

The report gives each ink's contact length, area and evidence, plus the time. Minutes are the compile's
modeled seconds plus fixed gaps for the operations it leaves out:

| Operation | Gap | Basis |
| --- | --- | --- |
| dip | 48 s | median gap measured on a full colour piece |
| rinse | 40 s | allowance |
| cartridge swap | 180 s | allowance |

Initial resource selections and changes keeping the same cartridge and ink incur no rinse or swap,
as in the executor. A workflow component already included by the compiler is retained once; the gaps
fill only missing components. These defaults are historical planning inputs, not current-rig measurements.

The render is a prediction. It is not `preview.svg` from `tatbot ros compile`, which colours strokes by
display rgb to show their order.

## Drawing-time profile

`tatbot art profile PROGRAM --out DIR` explains a saved compiled program's time without a robot or camera.
It prints the component minutes and shares and writes `profile.html` (standalone, no network assets) and
`profile.json` (`tatbot.art-time-profile/1`). The HTML report shows drawing, descent, settling, lifting,
travel between strokes, dipping, rinsing and cartridge swaps, sorted by cost. Resource rows show each
ink and cartridge's path length, stroke chunks, touchdowns, dips, rinses and swaps. Chunks continuing
pen-down add no touchdown; closed paths include their closing edge.

The profile reads the program's saved `tatbot-duration/2` components and motion digest, never today's
motion configuration, and records a SHA-256 of the input bytes. Use an original compiled program:
rehearsal transforms and changed workflow counts are refused because their saved statistics are stale.
Unitemized compiler time stays labelled as such. This is an estimated program duration, excluding page
setup, IK retiming, tracking holds, interruptions and recovery; it is not a live ETA.

`--dip-s S`, `--rinse-s S` and `--swap-s S` supply nonnegative finite gap allowances for comparisons.
They apply only where the compiler has no corresponding workflow component; the report distinguishes
compiler estimates, the historical dip-gap median, fixed allowances and supplied allowances. The render
and profile use the same accounting. Neither command rewrites a program or changes paths or dip cadence.

## Whip and pendulum shading

`tatbot art shade IMAGE --width-mm W --technique whip|pendulum --out DIR` turns a tone image into shading
whose deposition follows the tone along each stroke:

- **Input.** Darker is more ink; white and transparent draw nothing. Give it a tone map of the shading
  only. Outlines stay DBV3 lines: a whole design shaded at a 0.6 mm pitch turns its fine lines into dashes.
- **Cartridge.** `--tool` (default `lutin-3rl-bugpin`) sets the mark. The needle count and diameter are the
  datasheet's; round liner or magnum is its `config/sim2/tools/` file's; the tips lie as the simulator lays
  them (`tatbot_site.needle.groups`), a liner converged to 0.89 of a full ring and a magnum weaved (M1).
  A 3RL lays about 0.55 mm whichever way it moves, an 11RL about 1.15 mm.
  A magnum lays its row only across it: the 9M 3.15 mm, the 5M 1.75 mm, and either about 0.5 mm along it.
  The render sweeps a magnum's row, so its swaths stop square.
- **Strokes.** A round mark's strokes run from dark to light along the tone's mean descent inside the
  region (its hard edges ignored). A magnum's run across its row, whichever way across is nearer that
  descent, because the planner holds one tool heading per page. The row lies along the print's x by
  default, as the rig's 9M and 5M did (2026-10-08, within about 22 deg); `--row-deg` gives another
  placement. `--angle-deg` overrides either. Strokes lie `--pitch-mm` apart, by default the mark's width
  at their angle, so tone 1 is solid. Each point's deposition is the tone under it. Tone under `--floor`
  (0.05) draws nothing.
- **Off the descent.** `off_descent_deg` in the report is how far the strokes stray from the tone's
  descent. At 90 deg a magnum lays the gradient as constant bands, one swath each: turn the page (and
  `--row-deg` with it) or shade with a liner.
- **whip.** One stroke per run of tone, begun at its darker end. Each pays its own descent and lift.
- **pendulum.** Runs joined into zigzags where their ends lie within 3 mm or two pitches, whichever is
  more. A turn at the light end is lifted out (deposition 0), one at the dark end stays in, and a join
  through untoned paper between two dark ends is refused.
- **Rise.** Deposition never rises along a stroke faster than 1 per `--rise-mm` (4 mm: the skin ride's 0.5
  slope over an assumed 2 mm lift). A pendulum's swings toward a steep dark edge therefore reach it
  lighter; whips suit such edges. A stroke's start is its descent, so a steep climb there (an antialiased
  edge) is trimmed rather than ramped.

It writes `render.png` (no dip fade), `shade.json` (the settings, the angle used, the strokes and the
render's report), `artwork.json` and `preview-program.json`. The artwork is acquired paths
(`tatbot-art-shade/1`, its recipe digest binding the settings to the tone image's bytes) that
`tatbot ros compile artwork.json --ee-tool <tool>` places and draws, each point riding `(1 - deposition)`
of the tool's `fade_lift_mm` higher ([tools](tools.md)); a tool without one is refused, as every needle
cartridge but the 3RL is until a ride ladder measures its fade. Inkmap admits only DBV3 artwork, so it does not open
one. The preview program only previews: it has no format or measured page. Its minutes are the planner's
own estimate. On a 30 x 15 mm block fading left to right, 3RL whips model 5.7 min and one 3RL pendulum 3.8 min,
which still slows to a stop at each sharp turn. A 24 mm square fading downward takes 45 3RL whips (8.9 min)
or 8 9M whips (1.6 min).

## Separation

`tatbot art separate IMAGE --inks a,b,c --width-mm W --style S --out DIR --app INSTALLATION` turns a
flat-colour image, such as generated flash, into a drawable multi-ink program. DBV3 makes every path; the
separation writes only the greyscale sources DBV3 draws from.

1. **Prepare.** The image is read as a person sees it: turned as its EXIF says, transparency over white.
   Light, nearly neutral pixels become paper. The image is cropped to its artwork and
   resampled to the drawn width at 10 px/mm. Line-art styles first spread dark pixels so that lines
   thinner than one output pixel survive the resample.
2. **Assign.** Each pixel takes the nearest colour the load can draw over white, in CIEDE2000: a tint, a
   two-ink mix, or paper. A mix must beat the nearest single ink by 4 ΔE00 to be chosen; `--no-mixes`
   allows single inks only.
3. **Clean.** Colour fills are opened so none is thinner than one line. Specks under 0.5 mm² join their
   nearest colour. In line-art styles the darkest ink's regions are widened to at least one drawn line.
4. **Layers.** Each ink gets up to three DBV3 jobs on one canvas, lightest ink first:

   | Fill | PFM | For |
   | --- | --- | --- |
   | `hatch` | Streamlines Flow Field, field flattened (amplitude and frequencies 0.001), Luminance weighted, spacing 1–10 line widths | Tints and mix layers. Parallel lines at the ink's angle, so two inks mix optically. Tone sets the spacing through the measured curve. |
   | `solid` | ECS Drawing, edges and contours off | Regions the ink covers fully, packed along their contours |
   | `edges` | ECS Drawing, edges only | The darkest ink's solid regions, traced, in the `flash` style |

5. **Build.** Jobs are acquired with `tatbot drawingbot generate`. A layer DBV3 refuses is recorded and
   left out. A `tatbot-inks/3` draft binds each ink to an assumed cap: water in a large cap, the inks
   drawing the most line in the next largest. It is a draft; the palette decides which cap really holds
   which ink. `tatbot ros compile` makes the program, which is rendered on `--substrate`.

Before DBV3 runs, the canvas is checked against the page the compile will use: the print `--stencil` names, else
the nominal page's 90 x 134 mm clear centre. A design that cannot fit is refused there, and a plan only says so.
The page check and any failure are kept in `separation.json` (`page`, `failed`), and the summary line names any ink
that drew nothing, which a design's thinnest colour does at small sizes.

`--fit` instead of `--width-mm` draws at the widest width whose canvas that page holds at `--at`, measured on the
crop the separation makes, and records the room as `fit`. It sizes a design composed for that page, such as a tile
round's ([Designs for a stencil tile](#designs-for-a-stencil-tile)); it is not for shrinking a larger design.

On a fiducial-tile print `--fit` has a second bound. A design drawn over the tile covers the markers the print is
tracked by, so the width is stepped down, a tenth at a time, until the marker budget holds (`stack.yaml`
`page.markers`; `ros/README.md`, section 4.2): first on the separation's own regions, each taken
as inked all over, then, with `--app`, on the compiled program's count of the drawn paths. `fit.markers` records the
width the page holds (`page_mm`), the width the print can carry (`carried_mm`) and every size tried with what the
overhead and the wrist would still read at it. A design no size of which keeps the markers is refused.
`--allow-marker-loss` skips the second bound and prepares the program past the budget, for a sheet fixed on its mount.

`separation.json` also records `detail`, what the drawn size costs by the style skills' limits:

- per ink, its area, the part of it in colour regions narrower than 2 mm (`under_mm2`, `share`), which reads as
  outline rather than colour; for the darkest ink this is mostly its lines;
- per ink, `dropped_mm2`: what cleaning took from it, fills thinner than a line, specks and colour under a widened line;
- `gaps_mm2`: paper between inks narrower than 1 mm, which ink spread closes.

The summary line gives the share of colour under 2 mm, naming each colour ink mostly under it, and the narrow gaps.

The `separation.json` report gives the colours used and their shares, each layer, line per ink, the
compile's statistics, minutes, refused layers and the weakest ink evidence. It also scores **fidelity**:
CIEDE2000 between the target and the acquired paths drawn over white, after a blur standing in for viewing
from 30 cm (lightness 0.5 mm, a and b 1.5 mm; an approximation of S-CIELAB, not its fitted filters).

`--fill` picks the texture for regions an ink covers fully:
- `packed` (ECS, the default);
- `contour` (Streamlines Edge Field);
- `flow` (Streamlines Flow Field);
- `sweep` (Sketch Sweeping Curves);
- `maze` (Voronoi TSP);
- `waves` (Sketch Waves).
Only `packed` is measured. The others are for the voting loop to explore, and the drawn view shows
whether each stays inside its regions. `offset` is not DBV3's ([the offset fill](#the-offset-fill)).

Styles: `flash` traces and packs its black. `geometric` draws a black band one line wide as a single pass.
`linocut` keeps no line widening and traces no edges, because the mass edge is the outline.

Why these fills:
- **Sketch Lines at a fixed angle runs straight out of its regions,** 10–20 mm past them, so it is not
  used.
- **Streamlines seed only beside earlier streamlines,** so in narrow or ragged shapes they fill the cores
  and miss the rest. That is why solid regions are packed by ECS instead.

### The offset fill

DBV3 lays three or four passes over each spot of a region.
- **ECS** packs about 4 mm of line per mm² whatever the pen width.
- **Its edge trace** goes round both edges of a band, out and back.
- **The edge field** breaks its lines into 1–2 mm dashes.

On a 105 × 142 mm chicano lettering piece (2026-11-10), its ~1 mm black outlines and flourishes, 2.3 m of them,
came to 9.8 m of black line, and the red letters to 7.4 m. Two cartridges' worth modelled 405 min, and the thin
flourishes still broke up. `--fill offset` draws every ink's solid regions with `tatbot_art.offset` instead:

1. **Rings.** The region's edge, moved in half a pitch, then one pitch more each time. Each is one closed pass where
   what it encloses is at least a pitch across.
2. **Middles.** A band narrower than that is drawn once along its middle, from the region's own skeleton, so it
   stays one pass where the band narrows under a mark. Branches join where they continue straightest.
3. **Gaps.** Paper the marks leave, such as the core between two rings or a strip two pixels across between two
   passes, gets its skeleton too.

The same piece came to 2.8 m of black and 2.5 m of red, 99 min, with each region 96–98% covered at the marks'
widths.

Its regions are neither eroded nor traced: the outer pass is set in from each region's own edge. It needs no DBV3,
so an all-offset separation compiles and renders without `--app`. The artwork's adapter is `tatbot-art-offset/1`.

- `--tool INK=TOOL` draws an ink with another cartridge than the fitted one. Each cartridge's inks are drawn together,
  lightest first. In the inks draft, the first ink on a cartridge is a manual exchange, the arm landed while the
  operator swaps it. An ink after another on the same cartridge is a rinse.
- `--pitch-mm INK=MM` spaces an ink's passes. The default is its cartridge's simulated mark (`tatbot_art.cartridge`):
  the 3RL's 0.55 mm and the 11RL's 1.16 mm. Pass the mark at the depth you will draw at: the 3RL laid about 0.9 mm
  at a -2.5 mm trim on silicone (lab, 2026-11-09). A magnum lays its row only across a stroke, so it is refused.
- `--overlap-mm MM` grows a lighter ink's regions that far under the inks drawn after it, never into paper. A small
  misplacement between two cartridges' calibrations then hides under the darker line instead of showing paper.
  That is for the lightest ink drawn first. A negative value separates for the darkest ink first: the layers come
  darkest first and each lighter ink keeps that far clear of the darker ones, a gap in place of the underlap, because
  needles drawn through a darker ink carry it into their own cap. `tatbot ros compile` refuses a lighter ink over a
  darker one's mark in any other case (`ros/README.md` §6).

### From an Inkmap brief

`tatbot art separate --brief brief.json [--placement ID] --out DIR --app INSTALLATION` separates a design
placed in [Inkmap](inkmap.md). Inkmap places the source image and chooses its width; its tattoo brief
(`tatbot.inkmap-brief/1`, [design format](design-format.md#tattoo-brief)) carries the image and that width,
so DBV3 computes the strokes at the size the tattoo will be. The command checks the brief's digests, writes
the placement's image to `DIR/source/` with a `tatbot.art-generation/1` sidecar when the art was generated,
and separates it at the placed width. The inks and style are the art's own unless `--inks` or `--style`
names others; an image or a `--width-mm` beside `--brief` is refused, since the width is chosen in Inkmap.

- The brief's `art_px` is the rectangle Inkmap sized. The crop here must agree with it to within 2 px, or
  the design would be drawn at another size; a disagreement is refused.
- A mirrored placement is separated mirrored, because the compile never flips a design.
- Rotation is recorded in `separation.json`'s `brief`, not applied: it is measured against the body's
  surface, not the page.
- It writes `inkmap-render.json` and `ink-on-white.png`, the predicted ink over white at the design's canvas, for
  Inkmap to [show on the body](inkmap.md#the-robot-render). `tatbot art stock-renders` makes the ones Inkmap ships for
  its stock designs.
- A brief with several placements separates each in drawing order, into `OUT/01-<name>/`, `02-<name>/` and so on.
  `--placement` names one instead. One that fails does not stop the rest, and the command then exits nonzero.

`tatbot art tone --app INSTALLATION --out DIR` re-measures the hatch fill's grey-to-coverage curve with a
step wedge. Pass the result to `separate --tone` when the DBV3 version or pen width changes.

## The voting loop

The owner votes on designs on a private claude.ai page, `python/tatbot_art/gallery.html`. Each design
opens a review panel with three views:

- **Original** shows the generated art.
- **Robot rendition** shows the acquired paths or the ink separation while paths are pending. The caption lists
  recorded stroke types, inks and their cartridges, and the modelled minutes and dips when available. Use the
  rendition selector to compare available versions. Open a published fill study from the selector to vote on it.
- **On stencil** shows that rendition over the recorded print, with the canvas and print sizes in millimetres.
  It uses the compiled placement or the planned canvas placement for a sketch.

The panel offers Keep, Drop, and Shortlist. Use Left and Right to review the next or previous design, U to
keep, D to drop, and S to shortlist. The page keeps these collections in its database:

- `designs` and `rounds`, written only by the owner;
- one `votes` document per viewer, `{v: {design: ±1}, s: {design: 1}}`, written only by that viewer.
  `v` holds keeps and drops, `s` the stars;
- `map/current`, the map's layout, written only by the owner;
- `asks`, the owner's requests for the next round.

The page shows the same designs three ways:

- **Wall** opens on the latest round. Pick another round or use Filters to narrow by style or your review
  status and to change the grid preview. Choose Unvoted to resume a review. The review panel keeps its
  browsing order while you vote.
- **Map**: every design placed by what made it and what it looks like. Each neighbourhood is an island.
  Thin lines join look-alikes, and coloured lines run to the kept designs a design was built on.
- **Pyramid**: what each design has earned.

### Preparing renditions and scale previews

Run `tatbot art wall` with `--stencil-previews DIR` to export scale previews for designs with a recorded
stencil placement. The command overlays `drawn.png` when acquired, or `target.png` for a sketch, on the
verified stencil reference at the separation's recorded size and position. The dimensions describe the print
reference. Upload each PNG as a wall asset under its filename stem and rerun `tatbot art wall` with those
asset IDs. The filename identifies the print, placement, and rendition together.

To compare another stroke type, cartridge, or ink load, acquire the same source image with
`tatbot art separate` into `CORPUS/separations/<id>~<label>` and upload its `<id>~<label>-drawn` asset.
Use the design's recorded size and print for a scale comparison. The wall checks the source image digest
before adding the rendition. These extra renditions share the source design's vote. Published fill studies
keep their own votes.

### The pyramid

| Tier | Holds | Promoted by | Costs |
| --- | --- | --- | --- |
| sketch | every Flow image, and its plan-only separation into the load's inks | Keep | seconds |
| drawn | DBV3 robot paths and the render model's minutes | Keep | one separation |
| fill study | the design re-acquired in every other contained fill, each variant its own design | Shortlist | one separation per fill |
| shortlist | starred designs, bundled for review and the rig | | none: `tatbot art bundle` reuses the acquisitions |

- `tatbot art promote --designs FILE --votes FILE --corpus DIR --out WORK [--budget N] [--app DBV3]`
  plans a pass and, with `--app`, runs it.
  - Kept sketches are drawn first, then kept drawings are studied.
  - The budget counts DBV3 separations and is shared out one design at a time across styles, so one
    style's run of keeps cannot take every slot.
- A study variant lives in `separations/<id>~<fill>/` with a `study.json`.
  - It starts from its source's acquisitions, so only the solid layers go back to DBV3.
  - `tatbot art wall` writes its document with `study_of` and `parents` naming the source.
  - Because a variant differs from its source only in fill, a vote on it scores the fill gene, not the
    image.

### The map

`tatbot art map --designs FILE [--votes FILE] --corpus DIR --out MAP` measures how far apart two designs
are as a blend of two distances:

- **genes**: style, subject, ink-load overlap and fill. Styles that share a separation family count as
  half apart.
- **look**: a 12 × 12 OKLab thumbnail and a chroma-weighted hue histogram of the Flow image.

Ward-linkage clustering groups the designs into neighbourhoods of comparable size, about one per ten designs. Classical MDS places each neighbourhood
from the mean distances between neighbourhoods, then lays out its members by their own distances. Each
design links to its five nearest, and a fill-study variant stays beside its source. The page relaxes this
layout with a force simulation.

### Proposing the next round

- `tatbot art wall --round N --plan PROMPTS --corpus DIR --assets ASSETS --out FILE` turns a round of
  generated, separated designs, and any fill-study variants of them, into the page's documents.
  - Each document carries its **genes**: style, subject, palette (the ink load), variant and fill.
  - It also carries its **parents**.
- `tatbot art propose --designs FILE --votes FILE [--asks FILE] --round N --out PROMPTS` writes the next
  round's Flow prompts.
  1. **Open asks first, oldest first.** The owner uses Request a variation in the review panel or Request more on the map for:
     - more like a design (same style and inks, new subject);
     - its subject in other styles;
     - a blend of two designs (one's style with the other's inks);
     - more of a neighbourhood (genes sampled from its members, weighted by keeps).

     An ask may carry a steer of up to 200 characters, which is appended to its prompts. An ask that
     does not fit the round waits.
  2. **Scored genes for the rest.** Each gene value scores its designs' summed votes over (count + 2), so
     a single vote cannot decide a direction. Most prompts reuse the best genes, and `--explore` of them
     try untried ones: an untried style, subject and fill, and a random palette of one to four owned
     colour inks plus black.
  - No prompt repeats a style, subject and palette already on the wall.
  - No style takes more than its share of a round.
  - Each prompt names its **parents**: the kept designs that share at least two of its style, subject
    and palette, or the ask's anchors.
- The genes are style (22 templates, from traditional to Bauhaus), subject, palette and fill.
- The fill gene and the fill studies use only the fills whose line stays inside its own regions:
  packed, contour, flow and squares. The other textures in `acquire.SOLID_FILLS` (curves, superformula,
  waves, sweep, maze) cross into neighbouring regions until a layer is clipped to its own; the
  `tatbot-dbv3-recipes` skill has the measured table.

The agent running a round follows the `tatbot-art-loop` skill.

### From the shortlist to the rig

`tatbot art bundle [ID ...] --corpus DIR [--votes FILE [--kept]] --out DIR` turns drawn designs into
what review and the rig take. `--votes` adds every starred design, and `--kept` every design with
a positive vote total. A design without robot paths yet is listed as waiting for `promote`. Bundling runs no
DBV3: it reuses the separation's acquisitions.

Each `OUT/<id>/` holds:

- `layers/`: each acquired layer's `artwork.json`, byte for byte, named `<nn>-<ink>-<fill>.json` in
  drawing order.
- `inks.yaml`: the separation's `tatbot-inks/3` draft, which cap holds which ink.
  `--ink-substrate human_skin` is for a person.
  `--one-load` gives every bundled design the same caps, so designs drawn one after another need no cap
  swapped between them. The inks that draw the most line across all of them take the largest caps, and
  the strokes stay the same. More inks than the palette has caps beside water are refused.
  `bundles.json` records the shared load as `load`.
- `program/`: `tatbot ros compile layers/*.json --inks inks.yaml --at=X,Y`, every layer on one canvas.
  At the page centre its strokes and dips are the separation's own. `--at X,Y` centres the design on
  the stencil page and `--stencil` names the print. `--at ID=X,Y` places one design at its own
  position, so several designs can share one page, for example stacked on one silicone skin. Each design
  is still its own program, drawn in turn.
- `render.png` on `--substrate`, beside the separation's `target.png` and `drawn.png`.
- `bundle.json` (`tatbot.art-bundle/1`): layers and their artwork digests, inks and caps, line per ink,
  minutes, dips, rinses, fidelity, weakest evidence, refused layers and every file's digest. Its review
  is `pending`: read the render, `program/preview.svg` and the minutes before
  `tatbot ros draw program/program.json`.

`OUT/bundles.json` lists the bundles and the designs still waiting.

A bundle never turns, flips or scales a design: DBV3 drew it at its size. A size change means a new
separation at that width, so new artwork and a new `inks.yaml`. To choose the size and place on a body,
place the source image in Inkmap and separate its brief.

## Designs for a stencil tile

The lab draws one design on each small tile print. A forearm18 tile holds 40 x 62 mm over its six 18 mm markers,
and the design goes over them. A design for a tile is composed for it, never a larger design fitted down. At 40 mm
wide, a five-ink koi composed for the wall's 60 mm put 65% of its colour in regions under 2 mm, its black packed
over most of the rest, and modelled 94 min with 47 dips. The traditional style's budget at 35 mm is 15–25 min.

A tile's minutes follow how much of it is inked. The first tile round modelled about 9 min per 100 mm² inked
(correlation 0.90 over its 27 designs; about 17 min per metre of line, 0.97). Asked to fill the tile, its designs
inked 40–78% of it and ran 9–155 min, median 82.

1. **Propose.** `tatbot art propose ... --tile forearm18` composes the round for a tile of
   `config/stencil-tiles.json`:
   - every prompt ends with the tile's size and how little of it to ink. It asks for a small tattoo inside an
     upright 40 by 62 mm area, mostly outline and bare skin. Ink covers no more than a third of it, colour sits in
     a few small shapes, and there is no filled background, panel or frame. The shapes are few, large and thick,
     with nothing finer than 2 mm and no gap narrower than 1 mm;
   - a load holds black and at most two colours (`loop.TILE_INKS`, the style skills' 35 mm budget);
   - a style the tile cannot draw is left out, whatever its votes: dotwork, because programs have no dot
     (`loop.TILE_AVOID_STYLES`). Styles that only read poorly at this size, such as fine-line, stay for the votes
     to judge;
   - an ask the tile cannot draw, too many inks or such a style, waits for another round;
   - the plan records the tile as `tile`: the Flow aspect nearest it (9:16 for forearm18), the inks, the styles
     left out, and `minutes_budget`, the traditional style's 25 min at 35 mm scaled by area (51 min for
     forearm18, about a third of the tile inked).
2. **Generate** the round in Flow at the plan's aspect and file it as any round.
3. **Separate.** `tatbot art tiles --plan prompts-rN.json --corpus DIR --stencil PRINT --out DIR/tiles-rN.json
   [--app DBV3]` separates every design of the round, or the ids named, with `tatbot art separate --fit --stencil
   PRINT`:
   - each at the widest width the print's page holds and its markers carry, at its centre, into the corpus's
     `separations/<id>/`, where the wall, `promote` and `bundle` find it as any round's; the index gives both
     widths as `markers` where the markers are the tighter bound;
   - with `--app`, each is a program compiled onto the print, with its render and minutes: one program a tile,
     for `tatbot ros draw separations/<id>/program/program.json` once that tile's page is set up;
   - the index, `tatbot.art-tiles/1`, lists per design its canvas, the inks that drew nothing, its `detail` and,
     once acquired, its minutes, dips, fidelity, program and render, with `over_budget` when its minutes pass the
     tile's budget. The flag is advisory: the design stays on the wall and in the index, and the votes decide. A
     design that fails is listed under `failed` and the rest go on.
   - Without `--app` it plans: a design already acquired is planned again and its report loses its minutes, so
     re-run an acquired round with `--app`, which reuses the acquisitions.

   A plan composed for no tile, or a print whose page is not the tile's size, is refused.
4. **Promote** draws a kept design at its sketch's width and on its print, so a tile round's drawings stay
   tile-sized.

The generation sidecar travels as for any separation, in `separation.json` and every DBV3 job's provenance.

## Limits

- The mix model is analytic until a drawn swatch card measures the inks and their overlaps on each
  substrate. Tattoo pigment in skin mixes; ink on the tile layers.
- Ink colours listed as `vendor-photo` come from product photos, not from ink on a substrate.
- Shading's linear deposition and its 4 mm rise are assumptions; only the 3RL's fade on silicone is
  measured. There, at 0.7 mm, a whip thins about 8x along its stroke and a pendulum about 2.5x, so whip
  is the better of the two. Two faults are open: every stroke ends in a dark dot, stopping 0.66 mm up
  inside the range where the needle stops marking, and the rows band with the dip cycle, their spacing
  the strokes per dip times the pitch. That fade is from a -1.75 mm trim, just past where the needle
  starts marking and a dip lasts 40-60 mm. From -2.5 mm a dip lasts over 200 mm, but a whip square goes
  solid with no visible fade, at 60 or 200 mm per dip, with fine ink speckle around it.
- Read a shaded patch by measurement, from wrist frames taken before and after it, which cancel the print:
  at a glance a fade reads as full-length lines, needles mark the print even dry, and wet ink may hide a fade.
- Shading runs well over its modelled minutes, which leave out dips and each whip's own descent and lift.
  Four 20 mm patches took 70 min against 24 modelled. Whether a gradient made by lifting the needle survives healing is untested.
