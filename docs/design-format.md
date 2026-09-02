---
summary: Public vector design and placement format
tags: [art, design, schema]
updated: 2026-09-01
audience: [artist, dev]
---

# Design format

Artists can contribute vector artwork and placement fixtures without access to
robot hardware.

## Artwork

Prefer SVG with a declared viewBox, consistent winding, and a license or
provenance note. Keep paths in design coordinates; do not bake a machine-specific
transform into the artwork.

## Placement

Use `config/inkmap/placement.schema.json` for the placement record. Keep the
design identifier, surface/frame, anchor, scale, rotation, mirror state, schema
version, and provenance together. Validate records before rendering or mapping.

Placement v4 distinguishes two hashes:

- `asset_sha256` identifies the complete GLB bytes for provenance.
- `surface_sha256` identifies canonical non-indexed, Z-up XYZ quantized to
  little-endian signed integer 10-micrometre units behind face/barycentric
  anchors. A rig or material export may change
  the asset hash without changing what an anchor means.

Pose and robot-relative positioning do not belong in a placement. A compiled
offline realization uses `config/inkmap/tattoo-scenario.schema.json`, which
records the resolved design, rig pose, support, transforms, fitted tool, and
derived surface trace needed for deterministic replay.

The preview is an interchangeable front end. A placement file is not a motion
program and must not be treated as evidence of physical reachability.

## Compiling a scenario

The offline compiler resolves a built-in or embedded SVG, flattens curves to a
maximum 0.1 mm chord error, applies the placement size and mirror, unfolds
adjacent body triangles around the anchor, and emits face/barycentric samples
at no more than 0.5 mm spacing:

```bash
tatbot sim compile config/inkmap/examples/forearm-placement-v4.json -- \
  --pose reclined-left-arm-supported --output /tmp/forearm-scenario.json
```

The supported SVG geometry is `path` (`M/L/H/V/C/S/Q/T/A/Z`, absolute or
relative), `line`, `polyline`, `polygon`, `circle`, `ellipse`, and rounded or
square `rect`. Element transforms currently fail closed with a named error;
they are not silently ignored. Filled outlines are compiled as their boundary
paths. The compiled scenario includes immutable SVG text, verified body and
rig checksums, the full named pose, placement provenance, robot/tool identity,
support identity, and a deterministic trace digest.

The checked-in manifest contains ten compiler fixtures: anchor, crescent,
dagger, heart, lightning, rose, snake, star, sun, and wave. A generated SVG
does not enter the simulator by URL. First save it as a local `.svg`, then use
the procedural materializer's `--generated-design-dir`; the resolved SVG text
and checksum are embedded into each accepted placement and scenario.
