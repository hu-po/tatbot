---
summary: Inkmap tattoo design placement and preview
tags: [web, design, preview]
updated: 2026-09-01
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
The built-in offline library has ten SVGs in
`web/inkmap/public/designs/manifest.json`; every one is also a surface-compiler
fixture for the procedural simulator.

### Procedural simulation showcase

The guided milestone view uses the production Three.js renderer and checked-in
scenario fixtures; it is not a painted mockup. It switches between both rigged
bodies and all five tattoo-session poses, shows each pose's bed, chair, or
armrest proxy, projects each scenario's real SVG placement, and draws the
compiled face/barycentric surface trace in cyan:

```bash
tatbot inkmap dev
# open http://127.0.0.1:4180/?showcase=1
```

The evidence panel records the deterministic CPU sampling and reach-audit run
behind the five examples. Its last pipeline stage remains visibly pending:
the showcase does not claim a GPU-rendered ManiSkill episode, deformable skin,
MediaPipe tracking, powered-arm behavior, or safe human contact.

## Placement record

The canonical public schema is `config/inkmap/placement.schema.json`. A record
should identify the design, body surface/frame, anchor, scale, rotation, mirror
state, schema version, and provenance. See [design format](design-format.md).

Preview geometry is a design aid, not a claim that a physical tool can safely
reach the same surface.

## Named body poses

Both checked-in Human Base Mesh bodies have the shared
`hbm-humanoid-v1` skeleton. Standing neutral is the editor default and the
rigging reference. The tattoo-session pose set is supine on a bed, prone on a bed,
reclined with the legs on the chair rest, and reclined with either arm lowered
onto an armrest. The
pose control bakes the pose into the geometry used for display, region lookup,
raycasts, and decals, and displays the named support proxy. Placement anchors
still name the unchanged canonical rest-surface face and barycentric
coordinates.

The authored skeleton and IK targets live in `config/inkmap/body-rig.json`.
Generated `*.rigged.glb`, deterministic `*.rig.npz`, and
`config/inkmap/body-poses.json` files are checked in so normal development does
not require Blender. To regenerate them, use Blender 4.0 or newer with NumPy
available to its Python runtime:

```bash
tatbot inkmap rig
# one body while iterating:
tatbot inkmap rig -- --body hbm-male-stylized
```

Generation fails if a tattooable vertex has no weight, the rig changes
canonical face order, NumPy linear-blend skinning differs from Blender by more
than 0.1 mm, or a pose exceeds the configured joint-rotation, edge-length, or
triangle-area deformation gates. Reclined poses also have semantic anatomy
gates: each knee must bend upward in the sagittal plane instead of sideways or
behind the hip-to-ankle line, supported elbows must remain visibly flexed, and
supported wrists must remain aligned with their forearms.
`npm --prefix web/inkmap run check`
independently checks all named poses in Three.js against Blender-authored
samples. These numerical gates complement, rather than replace, browser render
review of every body/pose combination.
