---
summary: Tier-0 stencil design bench — seeded 2-D transfer scenes scored in millimetres
tags: [vision, stencil, benchmark]
updated: 2026-09-26
audience: [dev]
---

# Stencil bench (tier 0)

`tatbot vision stencil bench` scores a stencil design and a tracker together.
It renders a deterministic bank of camera views of the stencil's *transfer* on
skin, runs the tracker on each image, and measures where the tracker says each
page point is against where that point's ink actually landed, in millimetres
on the skin. The bench runs on the CPU only. It reads no camera, arm or network.

```bash
tatbot vision stencil bench --seed tatbot-42                              # legacy frame, SIFT baseline
tatbot vision stencil bench --artwork docs/assets/stencil-frames/tatbot-43
tatbot vision stencil bench --seed 1 --set frame_mm=10 --set stroke_mm=0.3
tatbot vision stencil bench --seed 1 --degradation clean                   # control: crisp dark transfer
tatbot vision stencil bench --seed 1 --bank holdout                        # final report only
tatbot vision stencil bench --candidate coded-flower-of-life --tracker coded
tatbot vision stencil bench --tracker lightglue                            # learned matcher; pulls torch + kornia
```

The default is 128 scenes. With the SIFT baseline and eight worker processes a
run takes about a minute. Outputs go to
`~/tatbot-logs/stencil-bench/<run-id>/`.

## Scenes

Each scene index draws its parameters from a seeded stream. The `train` and
`holdout` banks use separate streams with the same distribution. Tune a design
on `train`, and use `holdout` only to report a result.

- **Camera.** 60 % of scenes use the wrist D405 and 40 % an overhead PoE camera.
  The wrist camera streams at 640×480 (visiond) or 1280×720 (ROS inspect). Its
  intrinsics come from the camera registry, as the nominal FOV when none are
  declared. It sits 100–200 mm from the skin, tilted 0–35° from the normal,
  with any roll. It aims anywhere on the page, so partial framing is common.
  Overhead cameras use the calibration bundle's intrinsics, distortion and
  poses around the drawing pivot. Cameras that see the pivot at more than 70°
  are dropped. Without a bundle (a bare clone) the bench uses a nominal camera
  0.8 m away. The scorecard records `overhead_views` with each camera's
  measured px/mm at the page.
- **Surface.** 40 % of scenes are flat. The rest are a cylinder with a 30–60 mm
  radius, its axis mostly along the page's long side. The transfer wraps it
  isometrically. The skin lies on a cutting-mat table with clutter.
- **Transfer.** Page point `p` lands at `s = mirror(p) + wobble(p)`:
  - The wobble is a smooth field with 0.5–1.5 mm RMS and a 12–30 mm correlation
    length.
  - 20 % of transfers are mirrored.
  - The ink is violet or light blue on one of five skin tones.
  - Lines spread 1.5–2.4× and fade in patches.
  - Correlated wash-off removes 0–60 % of the frame, with fine speckle on top.
  - Ink pools and smudges.

  The violet colour, the spread and the wobble scale come from one photo of a
  seed-1 print beside its transfer on fake skin; `PHOTO_FIT` in
  `scripts/vision/stencil_bench_scene.py` holds the fitted values. The
  light-blue ink is assumed, not yet photographed.
- **Imaging.** Lambert shading, a lighting gradient, defocus and occasional
  motion blur, sensor noise, and JPEG compression (harsher for the H.264
  overhead streams).
- **Negatives.** Every eighth scene is blank skin. Every eighth shows another
  print from the same generator (seed `<seed>-distractor`).

`--degradation clean` renders the same views with a crisp dark transfer. It
separates transfer damage from viewing geometry. It also checks the bench
itself: the SIFT baseline localizes clean transfers to about 0.2 mm (p50).

## Truth

Truth is dense. For any page-UV point, the scene knows the skin point where its
transferred ink sits, and the pixel where that point appears, including wobble,
curvature and lens distortion. A correspondence `(uv, pixel)` is scored by
casting the pixel's ray onto the true skin. The error is that hit's distance
from the transferred point, in page millimetres (unrolled arc length on a
cylinder). A pixel whose ray misses the skin counts as a failed point.

## Candidates

- **`flower-of-life`** is the legacy generator (`scripts/lib/stencil_frame.py`):
  a seeded flower-of-life band with random gaps, filled lenses and dots.
  `--marked` adds its 6×34 print-ID grid.
- **`coded-flower-of-life`** (`scripts/lib/stencil_coded.py`) keeps the
  flower-of-life hex lattice but codes it. The decoder needs only two things
  from the artwork: a round blob on every junction, and one two-state mark per
  edge. Everything else is style.
  - **Beads.** Every junction carries a solid knot (`knot_mm`), and `halo_mm`
    stops the petal arcs short of it. Bare line junctions do not stand out at
    2-3 px/mm once the transfer spreads, and a knot crowded by petals merges
    with them. An isolated round bead is what a low-resolution view finds
    first.
  - **Bits.** The default `bit=teardrop` draws both petal arcs of every edge
    and fills a stretch of the petal at one end or the other (`teardrop_from`
    to `teardrop_to`, as fractions of the chord). `bit=seed` puts a small dot
    there instead (`seed_at`). `bit=side` draws only one of the two arcs, and
    its bulge side is the bit. Every reader compares the two states, so a
    washed-off edge reads as an erasure, not as the other bit. Seed and
    teardrop bits do not flip under a mirror, and the code and decoder
    account for it.
  - **Code.** The print ID seeds the bits, so the whole frame is the ID and
    there is no separate grid. A deterministic search then raises the worst
    window's distance: every window of edges within three lattice steps of a
    junction differs from every other window, under all twelve lattice
    rotations and reflections, in at least `window.distance_min` edges
    (`coded.json` records the achieved margins).
  - **Layout.** The lattice phase is chosen to fit the most whole edges into
    the band.
  - **Decoration.** `knot=disk|ring|dot-ring`, and `ornament=none|dots|rings`:
    uncoded decoration at triangle centres. Ring ornaments look like knots to
    the detector and cost most of the success rate.

  The default is the design chosen in the style search: 6.5 mm lattice,
  14 mm frame, 0.3 mm stroke, 2.2 mm beads with 1.5 mm halos, and teardrops
  from 0.30 to 0.45 of each petal. Other settings: `spacing_mm`, `frame_mm`,
  `stroke_mm`, `window_radius`, `optimize_rounds`. The generator writes
  `coded.json` (lattice, bits, style and window margins) beside
  `tracking.json`.

  ```bash
  tatbot vision stencil bench --candidate coded-flower-of-life --tracker coded      # the default design
  tatbot vision stencil bench --candidate coded-flower-of-life --tracker coded \
    --set bit=side --set spacing_mm=6 --set frame_mm=12 --set knot_mm=2.2 --set stroke_mm=0.4 --set halo_mm=0 --set ornament=dots
  ```

## Trackers

A tracker implements `prepare(artworks)` and `locate(image, intrinsics)`. It
returns a `Located`:

- a status: `accepted`, `rejected` or `ambiguous`;
- a pattern id;
- `(uv, pixel)` correspondences with optional per-point confidence;
- optionally a dense `model` (uv → pixel).

The tracker never sees the truth. Register new trackers in `TRACKERS`
(`scripts/vision/stencil_bench_trackers.py`) and new generators in `CANDIDATES`
(`scripts/vision/stencil_bench_candidates.py`).

- **`sift`** is stencild's acquisition of legacy artwork: `StencilScene` over a
  scene-scale `ReferenceBank`, one fresh detection per image. It reports its
  inlier landmarks as correspondences and its homography as the model.
- **`coded`** (`scripts/vision/stencil_coded_tracker.py`) decodes the coded
  candidate:
  1. It finds knot candidates on the skin with a scale-normalised Laplacian of
     Gaussian over the red and green ink channels.
  2. It grows local lattices from strong knots and reads each edge's bit.
  3. It votes symmetry and offset against every registered print's code. A
     component decodes only when its agreement is far beyond chance over the
     whole hypothesis space. On the train bank no wrong hypothesis came within
     two decades of the threshold.
  4. It then works in the page frame. It grows over the page's junctions,
     attaches components too weak to decode alone (the page model fixes their
     symmetry and bounds their offset), and bridges washed-off gaps. It keeps
     only junctions and reached groups that their own bits support.
  5. The model is the page plane's pose through the junctions, bent locally by
     their residuals.

  The decoder reports `mirrored` and `print_id` in `extra`. The bench decodes
  skin (`substrate="skin"`). stencild and `vision stencil replay` run the same
  decoder with `substrate="any"` inside search regions (a paper print and the
  grey overhead frames carry no warm skin), use each decode to seed a flow
  track, and re-read the print's bits at the tracked junctions every frame
  (`scripts/vision/stencil_coded_live.py`, see [stencil frames](stencil-frames.md)).
- **`lightglue`** (`scripts/vision/stencil_learned_tracker.py`) is the learned
  baseline: ALIKED features and LightGlue from kornia, on the CPU. It matches
  the red channel of the skin crop against the artwork rendered as a spread
  transfer at two scales, takes MAGSAC homography inliers as correspondences,
  and uses the coded tracker's model. The verb adds `torch` and `kornia` to the
  plan only for this tracker. The pretrained weights download into the torch
  hub cache on first use.

## Scorecard

`scorecard.json` holds:

- **overall** and **by_kind**: these rates.
  - `success_rate`: an accepted scene with the right pattern and a point error
    p50 ≤ 1 mm.
  - `accept_rate`.
  - `false_accept_rate`: an accept on a negative, a wrong pattern, or a point
    error p50 > 5 mm. The target is 0.
  - `ambiguous_rate`.

  Point error p50 and p95 are pooled over every correspondence of the correctly
  identified scenes.
- **localized_fraction**: the share of visible 3.5 mm frame cells holding a
  correspondence within 1 mm. **model_error_mm** and
  **model_localized_fraction** are the same measures for the tracker's dense
  model.
- **curves**: the metrics above binned by measured wash-off, cylinder radius,
  ink colour, camera and px/mm.
- **subtlety**: the printed artwork's black fraction (page and frame), its
  frame width and ink extent, and its largest solid blob in mm² (what survives
  a 1 mm opening).
- **processing_ms**: the tracker's time per image, measured while worker
  processes run in parallel.
- **reasons**: a histogram of kind, status and reason.

`scenes.jsonl` has one record per scene, including every point error.
`worst.jpg` shows false accepts first, then missed scenes that were well in
view. `sample.jpg` shows the first eight scenes. Each thumbnail carries the true
frame outline (green) and the correspondences coloured by error (green < 1 mm,
yellow < 3 mm, red worse).

## Fitting the degradation

`scripts/vision/stencil_bench_fit.py --photo P --reference tracking.json
--output DIR` measures one photo showing a print beside its transfer. It
locates both copies (SIFT on the print; frame-band corners and ECC on the
transfer, where SIFT fails). It rectifies both to page millimetres and aligns
them densely to the artwork. From that it reports:

- line width and spread;
- ink density and transmittance;
- wobble;
- washed and faded strokes;
- stray ink.

The printed copy passes through the same pipeline, so photo blur cancels out of
the ratios.

## Limits

Tier 0 is a 2-D renderer. It has no subsurface scattering, specular skin, hair,
occluding tools or arm shadows, and it does not model the overhead depth camera.
Its colours and damage are fitted from one photo. The later tiers are a
calibrated simulator render and real wrist captures.
