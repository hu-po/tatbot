---
summary: Public tool registry and geometry conventions
tags: [tools, geometry, configuration]
updated: 2026-09-30
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

A datasheet's `calibration:` block declares how the station probe may touch
the tool: `face_kind: halo`, a tube's end, with the ring's outside
(`wall_radius_m`), its hole (`face_radius_m`), where a rim touch puts the ball
(`rim_touch_radius_m`) and how far up the wall a side touch meets it
(`side_touch_height_m`). The laser's nose and the ballpoint cartridge's tube
declare one; the probe calibration refuses a tool whose datasheet does not
(`ros/README.md` §7). The block's other fields (`contact_reference`,
`accuracy_profile`, `feature`) belong to the retired one-arm mechanical recipe,
and nothing reads them; they stay because a datasheet's bytes are its tool
identity, which the URDF and every recorded dataset carry. The touch probe on
the installed palette replaces that recipe's pits (`docs/palette.md`).

A fitted mechanical point is stored apart from `tcp_z_m`: a standoff tool
keeps its nominal working point, and a contact tool's seated point carries an
unresolved writing-contact correction.

`config/workspace.yaml` retains `mechanical_contact_*` as calibration records:
coordinates, contact reference, accuracy profile, fit status and source session.
`tatbot ros calib apply` updates them with the adopted fit; a tool change clears
them. A pivot touch-off without a separate contact reference leaves those keys
null. Motion consumers use `pen_tip_offset_*` and the resolved working TCP;
the historical contact record alone grants no motion or fit qualification.

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

## The drawn line

`contact_radius_m` is collision and TCP geometry and `tip_detail` is render
geometry; neither says how wide a line the tool lays. That is the optional
`line:` mapping, recorded like `measured:` and validated at load:

```yaml
line:
  width_mm: 0.5           # in (0.05, 2.0]
  status: assumed         # measured | assumed
  utc: 2026-09-15
  method: "datasheet ball diameter; swatch not yet measured"
  substrate: paper_pad
```

It is a deposition measurement on one substrate at one working speed, exposed
as `ToolSpec.line_width_m`; a partial block refuses. The DBV3 recipe records its
acquisition pen width. ROS preparation binds the fitted tool's physical footprint
without editing the acquired geometry. The fill planner's source/fixture tooling
can use the footprint for deduplication; native DBV3 paths retain repeated passes.
A `status: assumed` value is a placeholder for a measurement: draw a short swatch
at the working speed on the substrate, measure the line against a known pitch
(or a loupe), then record `measured` with the UTC and method.

## The stroke: press or ride

A rotary machine's tip reciprocates through its stroke. A page touch trips
only once the tip is pushed up to the top of that stroke, where it stops, so
the touched page is where the tip meets the paper at the top of its stroke,
whether the machine runs or not. Two optional fields place the stroke, and
`ToolSpec.stroke_m` and `ToolSpec.tip_out_at_top_m` validate them at load:

```yaml
stroke_mm: 3.5            # the machine's stroke with this cartridge, in (0, 6]
tip_out_at_top_mm: 2.0    # the tip out of the cartridge's tube at the top of the stroke; negative: inside it
```

The collar that sets the tip's protrusion moves the TCP, so a change needs a
new touch-off. How the drawing path uses these is motion tuning, `motion.yaml`
`pen.mode`, read at every draw against the fitted tool's datasheet:

- **press:** the machine stays off, the tip stays at the top of its stroke,
  and the path presses `pen.press.lift_m` into the page; the arm's and mount's
  give keeps the tip on the paper.
- **ride:** the drawing stack runs the machine through its switch and the
  path rides `pen.ride.fraction` of `stroke_mm` over the touched page, so the
  running tip meets the paper that far down each stroke and would reach the
  rest past it.

A tool that records neither field, or whose tip sits inside its tube at the
top of the stroke (a touch would meet the tube), cannot ride, unless the
tube's end is its contact reference (`calibration.contact_reference:
tube_end`, a needle cartridge). Its touches and wrist gauge then measure the
tube's end, which is the TCP, so the path is that end's height and riding
needs only `stroke_mm`. `needle_reach_mm` records how far the needles pass the
tube's end at the bottom of the stroke; the run's `pen down` event then says
how deep they go.

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

`scripts/tool_body_qualify.py` validates an independently measured tool-body
axis against the current touch-off. This optional file-based study is useful for
asymmetric bodies and close-clearance work; without `--write` it is a dry run.

The input is a JSON report containing at least five chronological remove and
reseat samples. Each sample identifies its touch-off session and records a
body-profile origin, a unit body axis, and an independently planted tip in the
tool-mount frame. The selected sample must be the final reseat and must match
the current workspace touch-off. Measurements must come from an instrument
independent of the planted-tip fit; deriving them from that same fit is not
independent evidence.

Dry-run before writing:

```bash
python3 scripts/tool_body_qualify.py --ee-tool fixture-pen \
  --report /path/to/body-reseat-report.json
python3 scripts/tool_body_qualify.py --ee-tool fixture-pen \
  --report /path/to/body-reseat-report.json --write
```

A validation failure writes nothing. A passing write stores canonical report
bytes, binds their digest and computed metrics to the workspace, and
revalidates the result. Existing evidence is immutable unless the bytes are
identical. A later touch-off clears the independent body evidence because the
physical seat may have changed. Site-specific capture procedures, measured
coordinates, and qualification records remain in private operator documents.

## Ballpoint cartridge

The installed ballpoint uses `ink.mode: cartridge`: it writes from its internal
refill and never dips. There is no cartridge clock or supply declaration; ink
coverage is judged from the drawn marks, not elapsed wall time.

The Inlumino Heart Ink saturated-color ballpoint dot cartridges share the
`lutin-ballpoint-dot` tool geometry. Each color has a separate pigment identity
in `config/inks.yaml`: `inlumino_saturated_sky_blue`,
`inlumino_saturated_purple`, `inlumino_saturated_green`,
`inlumino_saturated_pink` and `inlumino_saturated_yellow`. Catalog RGB values
are nominal preview colors, not measured deposited color responses.

The ten-cartridge pack contains two of each color; `config/inventory.yaml`
records the package quantities. Bind each acquired DBV3 pen to a color resource
with `tatbot-inks/3`. At a color change the arm lands and the operator swaps the
cartridge by hand; the cartridges share the tip geometry, so the resumed run
keeps its tip calibration and measured page, with no new touches. A different
physical geometry needs calibration and new page touches. The set never dips.
