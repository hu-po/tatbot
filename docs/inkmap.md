---
summary: Inkmap tattoo design placement and preview
tags: [web, design, preview]
updated: 2026-08-31
audience: [artist, dev, contributor]
---

# Inkmap

`web/inkmap/` is a browser application for placing a vector design on a 3D
body preview. It produces a versioned placement record; it does not command a
robot or upload a design by itself.

## Develop

```bash
cd web/inkmap
npm ci
npm run check
npm run dev
```

The app should work with local fixtures and no private service. Keep generated
images and personal designs outside Git unless their license permits sharing.

## Placement record

The canonical public schema is `config/inkmap/placement.schema.json`. A record
should identify the design, body surface/frame, anchor, scale, rotation, mirror
state, schema version, and provenance. See [design format](design-format.md).

Preview geometry is a design aid, not a claim that a physical tool can safely
reach the same surface.
