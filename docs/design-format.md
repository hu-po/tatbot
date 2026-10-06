---
summary: Public vector design and placement format
tags: [art, design, schema]
updated: 2026-09-04
audience: [artist, dev]
---

# Design format

Artists can contribute vector artwork and placement fixtures without access to
robot hardware.

## Artwork

Supply licensed raster or SVG source artwork to a version 3 DBV3 acquisition.
Finished drawing artwork must be acquired `InkmapArtwork/2` with
`dbv3-batik-paths/1` and a frozen recipe identity. SVG exports and previews remain
valid; the former public SVG importer is retired.

## Placement

`tatbot.surface-placement/2` expresses target-independent placement intent.
Its shared fields bind the artwork digest, physical size, rotation, mirror,
review and provenance. The `target` is one of:

- `body`: the existing body identity, surface digest, barycentric anchor,
  semantic site and supported faces;
- `plane`: a metric canvas, anchor `(u,v)` and boundary margin;
- `cylinder`: the same metric chart plus its radius. The `u` axis runs along
  the cylinder; `v` is circumferential arc length. The chart origin lies on
  the crest, not at the cylinder axis.

The readers refuse a rotated artwork canvas outside the target margin and
a cylinder chart covering a full circumference. Analytic placement currently
requires `warp: null`. These are nominal authoring surfaces: no measured
robot pose, contact height or execution permission is implied.

The JSON Schema is `config/human-representation/surface-placement-v2.schema.json`.
Browser and Python constructors produce the same digests. Body v1 imports
have an explicit migration preserving their binding and constraints; existing
v1 artifacts retain their original interpretation. Material-stroke planning
accepts both versions through the same decomposition logic. `InkProgram/2`
supports both body coordinates and target-bound metric chart coordinates.
Plane/cylinder curves retain the placement digest and their arc length is
verified from chart coordinates; readers reject mixed coordinate types and
rehashed target or length mismatches. Body v2 compilation preserves the same
material events as v1. Material planning is shared across targets, including
initial charge, replenishment and color changes. A charged program can resolve
without a palette; an exhausted charge still requires available supply.

This contract alone does not enable drawing: the robot compiles plane designs
through `tatbot ros compile` (see [portable design handoff](#portable-design-handoff)).

Use `config/inkmap/placement.schema.json` for the placement record. Keep the
design identifier, surface/frame, anchor, scale, rotation, mirror state, schema
version, and provenance together. Validate records before rendering or mapping.

[InkLang](inklang.md) owns the path from placement description to the exact
anchor. A PlacementFile v6 embeds the fixed model specification and identity,
the original description, normalized intent, surface-bound resolution, and any
explicitly accepted candidate.
Consumers must reject provenance whose body digest, face, barycentric weights,
actual site, or region UV disagrees with the placement and current atlas.

Placement v6 distinguishes two surface hashes:

- `asset_sha256` identifies the complete GLB bytes for provenance.
- `surface_sha256` identifies canonical non-indexed, Z-up XYZ quantized to
  little-endian signed integer 10-micrometre units behind face/barycentric
  anchors. A rig or material export may change
  the asset hash without changing what an anchor means.

Pose and robot-relative positioning do not belong in a placement. A compiled
legacy boundary-trace realization uses `config/inkmap/tattoo-scenario.schema.json`, which
records the resolved design, rig pose, support, transforms, declared tool, and
derived surface trace needed for deterministic replay.
Typed bundle compilation uses `config/inkmap/tattoo-scenario-v3.schema.json`
and retains the bound source bundle and InkProgram; see below for its gates.

The preview is an interchangeable front end. A placement file is not a motion
program and must not be treated as evidence of physical reachability.

## Compiling a scenario

### Portable editor projects

`tatbot.inkmap-project/3` encloses an embedded-artwork PlacementFile v6, editor
pose/appearance/camera settings, bounded placement history, a recoverable
pending-edit base, and the paper/cylinder editor's working draft. Its content
digest covers the entire document. Editor-only settings do not alter the
enclosed placement body/anchor contract. Project readers in TypeScript and
Python validate the same strict JSON Schema and canonical digest, reject
unknown versions/fields and missing artwork, and check all history frames.
Loading a project is not simulation compilation or permission to execute it.
`config/inkmap/project.schema.json` defines the format; the offline reader is
`tatbot_sim.inkmap.project.load_project`.

Projects store the shared artwork records directly in their body and chart
registries. `chart` and `chart_parked` are required and nullable. Earlier project
and placement schemas, and records from the retired tracer, require explicit
DBV3 regeneration and import; readers do not rebuild artwork
from their original SVG. Rotation stays in radians so reopening does not move
an unchanged design.

### Placement-to-scenario compilation

The offline compiler takes a self-contained placement file with frozen artwork,
binds the requested pose and tool in a typed bundle, and emits a v3 scenario
through the same compiler as the editor's simulation export:

```bash
tatbot sim compile config/inkmap/examples/forearm-placement-v6.json -- \
  --pose reclined-left-arm-supported --output /tmp/forearm-scenario.json
```

The compiled scenario retains the bundle, resolved body and pose checksums,
placement provenance, tool identity, support and deterministic typed trace.
Historical v2 boundary-trace scenarios remain readable for replay.

### Typed artwork conversion

`tatbot.inkmap-artwork/2` contains a frozen `TattooProgram/1`, source hash and
provenance, acquisition adapter, optional recipe digest and metric decoding
error bound. Its program is authoritative for geometry, physical dimensions,
pen widths, path order and ink assignment. Previews derive from that geometry.
Readers validate nested hashes and source binding without re-running the
original conversion. They never fetch source identifiers.

The installed native DBV3 worker emits ordered centerlines directly. Inkmap's
**Import artwork** accepts acquired JSON records only. Generic paint conversion
remains source tooling and contract fixtures; it cannot enter the drawing editor.
A record contains no embedded original SVG or independently stored preview.
See `config/inkmap/artwork.schema.json`, `tatbot_contracts.artwork` for the
stdlib path reader, and `tatbot_sim.inkmap.artwork` for public paint programs.

Project save/recovery is available independently. **Export simulation bundle**
downloads `tatbot.inkmap-sim-bundle/1`: the v6 placement file, immutable artwork
records, ordered typed surface placements, pose/support request, camera/skin
settings, seed, nominal tool ID, and pinned local body-asset manifest. The
manifest contains local keys and license-source hashes, never fetch URLs.
Unknown artwork licensing stays explicit; export is not redistribution approval.
This artwork record alone does not constitute a compiled scenario.

`tatbot sim compile FILE -- --output SCENARIO` recognizes a bundle and produces
a distinct v3 scenario with its source bundle and deterministic InkProgram.
For multiple placements, add `--placement-id ID`; no implicit first-placement
selection occurs. Request values are immutable: pose, tool, seed, support, and
world-placement CLI overrides refuse even when equal to a default. Author a
new bundle to change them. `--created-at` and `--git-sha` still describe the
compilation provenance. V2 remains the older boundary-trace format and is not
silently reinterpreted. V3 validation recompiles the typed program and checks
the derived trace, placement, pose/world realization, and local asset bytes.

V3 scene materialization derives a deterministic metric reference raster from
the bound `TattooProgram/1` and `SurfacePlacement/1`; it never falls back to the
v2 SVG boundary trace. At 4 px/mm with 4x supersampling it writes effective
soft coverage and unassociated sRGB color separately from nearest-sampled
integer layer/placement IDs. Row zero is chart negative-y and column zero is
chart negative-x. Placement rotation belongs to the intrinsic surface frame
and is not baked into the texture a second time; mirror is a horizontal texture
operation on both browser and simulator. `target-reference.png` is review evidence only: the runtime
patch starts as the requested bare skin color and deposited pigment remains a
separate `InkField`. The editor's **Open compiled preview** reader validates
the embedded bundle, typed-program hashes, derived trace, and immutable request
bindings before showing the scenario. It explicitly does not replace the CLI's
local tool-profile and compiler reconstruction checks. Cross-renderer Gate D
qualification remains required before the bundle milestone is complete.

Run `python -m tatbot_sim.inkmap.parity_evidence --output DIR` from the
`python/tatbot_sim` environment to reproduce the cross-renderer qualification
packet. The Gate D mask comparison uses 12 px/mm with 4x supersampling so the
smallest stipple fixture is not dominated by one-pixel quantization; the
runtime reference target retains its documented 4 px/mm minimum. The report
aligns browser row zero (TattooProgram positive-y) to simulator array row zero
(chart negative-y) explicitly before comparing masks. It also exercises the
browser and Python flattenability checks, including complete-wrap refusal.

The `TattooProgram/1` SVG adapter uses the shared TypeScript paint materializer
in `web/inkmap/src/core/human-representation/svg-program.ts`, invoked by the
Python `human_rep.tattoo_program` API through a bounded, local Node subprocess.
This adapter preserves painted metric regions: fills, compound-path holes,
stroke widths/caps/joins, nested transforms, colors, and layer order. It does
not infer centerlines from filled shapes. Existing typed InkProgram fill
planning remains responsible for turning regions into tool-intent strokes.
That planner unions equal-width/deposition paint within each ink layer before
adding connected inset contours. It never inks internal tessellation edges.
Paths are inset by the finite footprint radius, including around holes;
ideal footprint coverage must retain at least 98% of every connected paint
component, or the artwork/width pair refuses. This is geometric qualification,
not calibrated deposition or actual tool-width validation. Capacity splitting
and ink changes still use the shared ink-supply planner.

Planar set operations are pinned to Shapely 2.1.2; its runtime and GEOS versions
are bound into the compiler digest. A fixed 0.1 nm snap grid and 10 nm seam
closure reconcile floating-point tessellation T-junctions before planning. The
union is snapped again after buffering to remove sub-grid numeric slivers. A
0.1 micrometre conservative inset guard compensates for buffer roundoff. Short
links between contours must lie wholly within eroded paint; holes cannot be
bridged.
These are explicit geometry tolerances, not permission to bridge visible
negative space. Source artwork and its canonical program are not rewritten.

#### Fill styles

The compiled InkProgram may carry an optional top-level `fill_style` naming the
paint planner it was compiled with, following the legacy `operating_budget_s`
precedent: absent means `concentric` (the inset rings above, the default), and
`hatch` is contour-then-hatch — two boundary rings (the depth-0 outline and one
inner ring at 0.45 × width), then the centre domain eroded by two ring pitches
is filled with parallel rows at the same pitch along the long axis of each
connected region's minimum-area bounding rectangle, so a feather becomes a few
long rows rather than a stack of nested slivers. Row pieces join into a
serpentine only when the joining segment lies wholly inside the centre domain
and is at most 2 × width long — the ring linker's own admissibility test — so
visible negative space is never bridged; a hole splits a row into pieces that
continue separate chains. Hatch rows are reported at inset depth 3 and so
schedule as tier 2 behind the rings. Both styles pass the same spill, narrow
and per-component coverage gates. The style is a compile option
(`design check --fill-style` and `sim qualify-artwork --fill-style`); it never enters the artwork record's
`conversion`; the frozen program retains the acquisition geometry.

#### The stroke schedule

The planner's strokes are then scheduled before capacity splitting, by one
pass the simulator-side consumers share (`design check` and the simulator's
rehearsal corpus call `scheduled_material_strokes`; `tatbot_ink` ports the
same three stages). Within each consecutive (source layer, ink)
group — so colour changes, dips and barriers keep their sequence — three
stages run:

Acquired DBV3 paths bypass this ordering and deduplication pass: native sequence,
direction, closedness and repeated traversals are retained. The following pass
applies to generic source/fixture paint programs.

1. **Tier.** Every stroke carries a tier: explicit elements (`path`,
   `cubic_bezier`, `dots`, `stipple`) and the fill planner's boundary rings
   (inset depth 0) are tier 0; insets 1–2 are tier 1; deeper insets are
   tier 2. Tier 0 draws first, so a run stopped early has drawn the
   silhouette and a full run fills detail last.
2. **Dedup at the tool's footprint.** Walking in tier order with a union of
   the footprints kept so far, a fill ring whose own footprint adds less
   than 25 % of its area is dropped, unless the drop would take its paint
   component below the coverage floor or cost it more than 1 % of its area
   in total (both re-checked before every drop against the coverage the
   component has with every ring kept: a drop is a deliberate loss and may
   not spend the qualification budget). The footprint is
   the fitted tool's recorded line width when the datasheet has one, else
   the planning width, so an over-fine planning width stops doubling ink.
   Explicit elements are never dropped. Every drop is reported with its
   index, source primitive, length and new-area fraction.
3. **Order within a tier.** Greedy nearest neighbour from the previous
   stroke's end, ties to the lower original index. A closed ring is rotated
   to start at the vertex nearest the previous end; an open stroke is
   reversed when its far end is nearer. The point arrays are the truth;
   `start_choice` and `direction` are unchanged.

The compiled InkProgram carries the schedule only through its stroke order and
each stroke's `ordering_rationale`, which keeps its `source layer N,` prefix
and appends `; tier T depth D` (and `; rotated to vertex K` / `; reversed`
where the order pass edited the points). Nothing new enters the document
schema; `content_sha256` of a compiled program reflects the order. The pass is
covered by the compiler digest. The audit — per-tier counts and lengths, the
dropped list and the pen-up travel before and after — is the `schedule` field
of `design check` output.

The source/fixture SVG paint parser accepts opaque, self-contained geometry with presentation
attributes or inline styles. Text, external references, filters, gradients,
translucent paint, unknown style operations, and malformed syntax refuse with
an element path. Nonuniformly scaled strokes require explicit outline
conversion. Curves are adaptively flattened with a maximum requested chord
error of 0.1 mm. Source SVG bytes and their digest remain review provenance.

The shared program preview renders fill regions without an extra outline and
negative-space masks as transparency rather than white pigment. New placement files use the typed v3 bundle compiler.
Do not infer filled-ink episode support from a faithful artwork preview alone.

The shared `web/inkmap/public/designs/manifest.json` collection supplies acquired
CC0 examples to Inkmap, normal body/flat simulation and the perception pilot.
It binds artwork JSON hashes, source metadata, acquired physical dimensions and
pen widths, and disjoint artwork-family train/validation/test splits.
The editor's picker and every catalogue ID contain only three native DBV3
acquisitions: orbit, sprout and ridges. The original finished SVG designs,
hidden showcase artwork and compiler fixtures have been removed. Each example
is acquired at 30 × 30 mm; a different physical size requires regeneration.

Normal sampling emits typed v3 scenarios. The versioned `spiral-v1` is an
explicit calibration control. Historical geometric generators remain available
only through explicit legacy tasks/test APIs; normal sampling cannot fall back
to them. Random procedural SVGs are never normal artwork.

`tatbot sim materialize` and Inkgen batches produce source imagery and source
traces. Their directories cannot supply finished simulator artwork. Acquire
those sources through native DBV3 first; `--artwork-dir` accepts complete native
acquisition directories with matching `result.json`, recipe and artwork hashes.

`tatbot sim qualify-artwork -- --output-dir DIR` checks every starter artwork at
its size-domain endpoints against source/preview paint, canonical simulator
target, finite-footprint coverage, complete trajectory duration, and runtime
InkField deposition. It writes source/program/target/deposition views and retains
all rejected cases. See [simulation](simulation.md#shared-artwork-and-calibration).

## Portable design handoff

`tatbot.inkmap-design/1` contains a name, frozen `InkmapArtwork/2` records, and
an ordered list of `SurfacePlacement/2` records with unique placement IDs and
explicit artwork references. The registry contains exactly the used artwork.
The content digest binds the order, original artwork, conversion, and placement
intent. Readers validate the frozen artwork, nested hashes and cross-references;
rehashing an envelope cannot detach a placement from its source program.

The same design format admits body, plane and cylinder targets. It contains no
robot pose, tool, palette, simulation request, or execution authority. The
robot compile binds the selected tool and the registered page separately.
Nominal chart dimensions are not measured pad dimensions.

The body editor's **Export design** action uses the same artwork materializer and
body-placement derivation as simulation export, explicitly upgrading the latter's
v1 body bindings. Generated source provenance is retained. Existing placement
and simulation bundle imports remain versioned interfaces; their contents are
not silently reinterpreted. Plane/cylinder design construction and Python reading
are available through the shared design API and the **Paper** and **Cylinder**
workspaces. Each edits its own analytic canvas — a project holds a pad draft
and a cylinder draft that never mix — with ordered artwork placements,
rotation and mirroring, choosing artwork from the same library and
generator as the body and placing it the same way — a ghost under the pointer,
a click, then drag and handles. Both are previewed in 3D on the bench's
own fixtures — the 7.5 × 11 in paper pad, 1 cm thick, and the ⌀85 mm × 7.5 in
paper cylinder, each white with a faint blue ¼ in grid — with the artwork bent
onto the surface through the same chart frame the placement is checked with;
`u` follows the cylinder's axis and `v` is arc length from its crest. The
fixture sizes are the editor's defaults, not measurements of any bench
setup. **Portable design** and **Open design** (under File) use the same
portable handoff as body export; **Artwork SVG** exports the selected artwork
as the editor draws it. Bounds and margins are validated before saving. The
in-memory draft survives switching workspaces and a reload; save a design file
before clearing the browser.

`tatbot ros compile design.json` turns a plane design into the ROS 2 stack's
`program.json` and a `preview.svg` (contact by schedule tier, dashed pen-up
travel). It accepts plane charts only; a 100 × 150 mm chart is the stencil
page. The compile, its refusals and the program format are in
`ros/README.md` section 4.2. A preview is not a deposition prediction.
