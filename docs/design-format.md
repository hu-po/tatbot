---
summary: Public tattoo art, placement and brief formats, and the robot's acquired artwork
tags: [art, design, schema]
updated: 2026-11-02
audience: [artist, dev]
---

# Design format

Artists can contribute art and placement fixtures without access to robot
hardware. There are two halves with one hand-off between them:

- **Design and placement** work with the source image. Inkmap places it on a
  body and sizes it freely; nothing there is a stroke.
- **Strokes** are made from that image at the chosen size by the art pipeline
  and DrawingBot V3 (`tatbot art separate`), as one acquired artwork per ink.
  The robot draws those.

The **tattoo brief** is the hand-off: each placed design's image and the width
to draw it at.

## Source art

`tatbot.inkmap-art/1` (`config/inkmap/art.schema.json`) is one design as an
image:

- `image`: `image/png` or `image/jpeg`, its SHA-256, pixel size and base64
  bytes, at most 8 MB.
- `art_px`: `[x, y, width, height]`, the rectangle the art occupies. It is the
  art pipeline's crop: the box around every pixel that is not paper (light and
  neutral, CIELAB L\* above 85 and chroma under 12), padded by 3% of the image's
  longer side, halves rounded to even. A placement's width is this rectangle's
  width.
- `source`: `stock`, `imported` or `generated`, with an identifier, licence,
  attribution and, for generated art, the prompt, model and generator.
- `palette`: the catalogue inks (one to five) and style it was made for, or
  `null`. Inkmap suggests inks for imported art from its colours, and the artist can
  change them.

The content digest covers the whole record, image bytes included. Readers check
the digest, the image digest, the pixel size read from the PNG or JPEG header,
and that `art_px` lies inside the image. `web/inkmap/src/core/art.ts` and
`tatbot_contracts.art` are the two readers.

The stock collection, `web/inkmap/public/designs/manifest.json`
(`tatbot.inkmap-art-collection/1`), names each design's image, digest, pixel
size, `art_px`, source and palette. Inkmap builds each record from those bytes
and refuses an image that differs. Its designs are Flow-generated and not
dedicated to the public domain.

## Placement

Use `config/inkmap/placement.schema.json` for the placement record. Keep the
design identifier, surface/frame, anchor, size, rotation, mirror state, schema
version, and provenance together. Validate records before rendering or mapping.

PlacementFile v7 carries its designs as source-art records. Earlier versions
carried stroke artwork and have no reader.

[InkLang](inklang.md) owns the path from placement description to the exact
anchor. A PlacementFile embeds the fixed model specification and identity,
the original description, normalized intent, surface-bound resolution, and any
explicitly accepted candidate.
Consumers must reject provenance whose body digest, face, barycentric weights,
actual site, or region UV disagrees with the placement and current atlas.

A placement file distinguishes two surface hashes:

- `asset_sha256` identifies the complete GLB bytes for provenance.
- `surface_sha256` identifies canonical non-indexed, Z-up XYZ quantized to
  little-endian signed integer 10-micrometre units behind face/barycentric
  anchors. A rig or material export may change
  the asset hash without changing what an anchor means.

A placement's `rotation_rad` turns the design about the surface normal from the
body's +z projected onto the rest surface. On an arm that is rarely upright:
Inkmap adds the turn that is ([Inkmap](inkmap.md#projects-editing-and-recovery)).

Pose and robot-relative positioning do not belong in a placement.

The preview is an interchangeable front end. A placement file is not a motion
program and must not be treated as evidence of physical reachability.

`tatbot.surface-placement/2`
(`config/human-representation/surface-placement-v2.schema.json`) binds a
stroke program to a body: the program digest, physical size, rotation, mirror,
review, provenance, and a `body` target with the body identity, surface digest,
barycentric anchor, semantic site and supported faces. It is the input to
material-stroke planning (`InkProgram/2`); the plane and cylinder targets were
removed with the paper mounts (2026-11-02). Browser and Python constructors
produce the same digests. Body v1 imports have an explicit migration preserving
their binding and constraints.

## Tattoo brief

`tatbot.inkmap-brief/1` (`config/inkmap/brief.schema.json`) is what Inkmap's
**Export brief** writes: a name, the body identity, the art records used, and
the accepted placements in drawing order. Each placement names its `art_id`,
`width_mm` (to 0.1 mm, the pipeline's 10 px/mm), height, rotation, mirror,
anchor, site and the sentence that placed it. The art registry holds exactly
the placed art, and the content digest covers everything.

`tatbot art separate --brief brief.json --out DIR` reads it
([art pipeline](art.md)), checks it with `tatbot_contracts.art`, and separates
the placement's image at its width. A brief carries no strokes, tool, palette
cap or execution authority.

## Robot render

`tatbot.inkmap-render/1` (`config/inkmap/render.schema.json`) is what comes back: `inkmap-render.json`, which
`tatbot art separate --brief` writes beside each placement's separation. It holds a PNG of the program's predicted ink
over white at the design's canvas, the digest of the art it was separated from, its `width_mm`, `height_mm` and
`mirror`, and the separation's `minutes`, `dips`, `rinses`, `lost_inks` and `weakest_source`. It carries no strokes.
`tatbot_contracts.art` and `web/inkmap/src/core/render-record.ts` are the two readers. The stock collection lists its
own under each entry's `renders`, in `public/designs/renders/`.

## Projects

`tatbot.inkmap-project/5` encloses a PlacementFile v7, editor
pose/appearance/camera settings, bounded placement history and a recoverable
pending-edit base. Its content digest covers the entire document. Editor-only
settings do not alter the enclosed placement body/anchor contract. The project
reader validates the strict JSON Schema and canonical digest, rejects unknown
versions/fields and missing art, and checks all history frames. Loading a
project is not permission to execute it. `config/inkmap/project.schema.json`
defines the format.

Earlier versions have no reader or migration: **Open project** refuses them, and
one saved in the browser gives way to a new project.
Rotation stays in radians so reopening does not move an unchanged design.

## Acquired artwork

`tatbot.inkmap-artwork/2` (`config/inkmap/artwork.schema.json`) is what DBV3
acquisition produces and the robot compiles: a frozen `TattooProgram/1`, source
hash and provenance, the `dbv3-batik-paths/1` adapter, the frozen recipe
identity and a metric decoding error bound. Its program is authoritative for
geometry, physical dimensions, pen widths, path order and ink assignment.
Previews derive from that geometry. Readers validate nested hashes and source
binding without re-running the original conversion. They never fetch source
identifiers. `tatbot_contracts.artwork` is the stdlib reader and
`tatbot_surface.artwork` builds public paint programs.

A DBV3 drawing is optimized for its acquisition size and pen width, so it is
never resized: a different size is a new acquisition from the source image.
That is why Inkmap places the image and the brief names the width.

`tatbot ros compile artwork.json` turns one acquisition, or several drawn as
layers (one per ink, on one canvas), into the ROS 2 stack's `program.json` and
a `preview.svg`. `--at X,Y` centres it on the 128 × 172 mm stencil page and
`--stencil DIR` names a generated print. The compile, its refusals and the
program format are in `ros/README.md` section 4.2. A preview is not a
deposition prediction. The simulator's single-ink CC0 acquisitions (orbit,
sprout and ridges, 30 × 30 mm) are in `config/sim2/acquired/`.
