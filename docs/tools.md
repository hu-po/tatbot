---
summary: Public tool registry and geometry conventions
tags: [tools, geometry, configuration]
updated: 2026-09-02
audience: [dev, contributor]
---

# Tool registry

Every end-effector tool used by code should have one versioned datasheet under
`config/tools/`. Simulation geometry, URDF links, calibration, and dataset
metadata must derive from that record instead of repeating dimensions.

## Datasheet contract

- Identify the schema version and coordinate origin.
- State units, mounting frame, tip geometry, and contact assumptions.
- Validate the fitted tool before generating derived geometry.
- Record the datasheet revision with every run or placement artifact.

## Resolved geometry

The datasheet body profile, a planted touch-off, and the working TCP are
different facts. `scripts/lib/tool_spec.py` resolves them once for the real
URDF, simulation URDF, FK, and dataset metadata:

- `mount_from_body` places the unchanged physical body profile.
- `body_tip_offset_m` is where rendered/contact material ends.
- `tip_offset_m` is what the calibration physically planted.
- `tcp_offset_m` is the working point used by FK and planning.

For contact tools those three endpoints must agree within 0.5 mm. A
well-conditioned fixed-point/pivot touch-off qualifies the contact vector. For
an axisymmetric profile whose mount origin lies on its centreline, the vector
also determines the contact-relevant axis; roll is unobservable but does not
change the geometry. Metadata therefore separates
`contact_geometry_status: pivot-calibrated` from
`body_pose_status: axis-inferred` instead of calling the
whole result provisional. Optional independent body evidence can promote the
body envelope to `independent-qualified` for asymmetric clearance work.
Coordinates or a hand-edited status are not evidence, and a new touch-off
clears prior independent body evidence because the seat may have changed.
Non-contact tools may separate material and TCP only by their explicit
datasheet standoff.

Dataset metadata always records the concrete geometry that ran, including a
nominal fallback and its source; null offsets are not a geometry contract.

Public examples should use a clearly marked fixture tool. Physical dimensions,
calibration values, and deployment choices that are not needed to build the
software remain private.

Contact qualification is evidence-bound by the touch-off's tool/frame identity,
pose count, condition number, rotation spread, residual, datasheet tip range,
and seat-angle range. Held-out and leave-one-out disagreement travel as
uncertainty metadata; they never widen the collision/marking band. A named
simulation recipe may use that bound to draw one mount-frame tip offset per
shard. The draw remains fixed for the shard, like one physical seating, and
the visible body, collision endpoint, IK TCP, and metadata all use the same
resolved geometry. Different seeds span plausible calibration outcomes without
allowing marks away from the simulated surface. Optional
explicit body geometry separately revalidates its report digest and
current-seat identity, falling back to the inferred axis if that evidence is
missing or stale. Physical capture procedures and calibration values stay in
the private operator documentation rather than this public contract.

## Optional independent body qualification

`tatbot tool qualify-body` validates an independently measured tool-body axis
against the current touch-off. This is useful for asymmetric bodies and
close-clearance studies; it is not a substitute for contact qualification and
does not authorize powered operation. The command only reads files. Without
`--write` it is a dry run, and even a writing invocation never connects to or
commands an arm.

The input is a JSON report containing at least five chronological remove and
reseat samples. Each sample identifies its touch-off session and records a
body-profile origin, a unit body axis, and an independently planted tip in the
tool-mount frame. The selected sample must be the final reseat and must match
the current workspace touch-off. Measurements must come from an instrument
independent of the planted-tip fit; deriving them from that same fit is not
independent evidence.

Dry-run before writing:

```bash
scripts/tatbot --ee-tool fixture-pen tool qualify-body -- \
  --report /path/to/body-reseat-report.json
scripts/tatbot --ee-tool fixture-pen tool qualify-body -- \
  --report /path/to/body-reseat-report.json --write
```

A validation failure writes nothing. A passing write stores canonical report
bytes, binds their digest and computed metrics to the workspace, and
revalidates the result. Existing evidence is immutable unless the bytes are
identical. A later touch-off clears the independent body evidence because the
physical seat may have changed. Site-specific capture procedures, measured
coordinates, and qualification records remain in private operator documents.
