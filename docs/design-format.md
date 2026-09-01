---
summary: Public vector design and placement format
tags: [art, design, schema]
updated: 2026-08-31
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

The preview is an interchangeable front end. A placement file is not a motion
program and must not be treated as evidence of physical reachability.
