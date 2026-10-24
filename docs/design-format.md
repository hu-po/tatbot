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

Pose and robot-relative positioning do not belong in a placement.

The preview is an interchangeable front end. A placement file is not a motion
program and must not be treated as evidence of physical reachability.

## Projects and artwork

### Portable editor projects

`tatbot.inkmap-project/3` encloses an embedded-artwork PlacementFile v6, editor
pose/appearance/camera settings, bounded placement history, a recoverable
pending-edit base, and the paper/cylinder editor's working draft. Its content
digest covers the entire document. Editor-only settings do not alter the
enclosed placement body/anchor contract. The project reader validates the
strict JSON Schema and canonical digest, rejects unknown versions/fields and
missing artwork, and checks all history frames. Loading a project is not
permission to execute it. `config/inkmap/project.schema.json` defines the
format.

Projects store the shared artwork records directly in their body and chart
registries. `chart` and `chart_parked` are required and nullable. Earlier project
and placement schemas, and records from the retired tracer, require explicit
DBV3 regeneration and import; readers do not rebuild artwork
from their original SVG. Rotation stays in radians so reopening does not move
an unchanged design.

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
stdlib path reader, and `tatbot_surface.artwork` for public paint programs.

The `TattooProgram/1` SVG adapter is the shared TypeScript paint materializer
in `web/inkmap/src/core/human-representation/svg-program.ts`. It preserves
painted metric regions: fills, compound-path holes, stroke widths/caps/joins,
nested transforms, colors, and layer order, and it does not infer centerlines
from filled shapes. Acquired DBV3 paths need none of it: their native
sequence, direction, closedness and repeated traversals are what the robot
compile consumes.

The source/fixture SVG paint parser accepts opaque, self-contained geometry with presentation
attributes or inline styles. Text, external references, filters, gradients,
translucent paint, unknown style operations, and malformed syntax refuse with
an element path. Nonuniformly scaled strokes require explicit outline
conversion. Curves are adaptively flattened with a maximum requested chord
error of 0.1 mm. Source SVG bytes and their digest remain review provenance.

The shared program preview renders fill regions without an extra outline and
negative-space masks as transparency rather than white pigment.

The shared `web/inkmap/public/designs/manifest.json` collection supplies acquired
CC0 examples to Inkmap and the offline consumers that read the collection.
It binds artwork JSON hashes, source metadata, acquired physical dimensions and
pen widths, and disjoint artwork-family train/validation/test splits.
The editor's picker and every catalogue ID contain only three native DBV3
acquisitions: orbit, sprout and ridges. The original finished SVG designs,
hidden showcase artwork and compiler fixtures have been removed. Each example
is acquired at 30 × 30 mm; a different physical size requires regeneration.

## Portable design handoff

`tatbot.inkmap-design/1` contains a name, frozen `InkmapArtwork/2` records, and
an ordered list of `SurfacePlacement/2` records with unique placement IDs and
explicit artwork references. The registry contains exactly the used artwork.
The content digest binds the order, original artwork, conversion, and placement
intent. Readers validate the frozen artwork, nested hashes and cross-references;
rehashing an envelope cannot detach a placement from its source program.

The same design format admits body, plane and cylinder targets. It contains no
robot pose, tool, palette or execution authority. The
robot compile binds the selected tool and the registered page separately.
Nominal chart dimensions are not measured pad dimensions.

The body editor's **Export design** action derives each body placement from
the frozen artwork and its v1 surface binding (`web/inkmap/src/core/body-design.ts`),
explicitly upgrading that binding. Generated source provenance is retained.
Existing placement imports remain a versioned interface; their contents are
not silently reinterpreted. Plane/cylinder design construction is available
through the shared design API and the **Paper** and **Cylinder**
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
travel). It accepts plane charts only; a 128 × 172 mm chart is the default
stencil page. The compile, its refusals and the program format are in
`ros/README.md` section 4.2. A preview is not a deposition prediction.
