---
summary: Public tool registry and geometry conventions
tags: [tools, geometry, configuration]
updated: 2026-11-10
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
(`side_touch_height_m`). The needle cartridges and the laser's nose declare
one. The probe calibration refuses a tool whose datasheet declares none
(`ros/README.md` §7). `contact_reference: tube_end` makes the planner ride the
tube's end on the surface. `accuracy_profile: needle-tube` tells the probe
calibration that an edge pass will not trip the probe, because a cartridge's
floating needles graze the ball. `feature` belongs to the retired one-arm
mechanical recipe, and nothing reads it; it stays because a datasheet's bytes
are its tool identity, which the URDF and every recorded dataset carry. The
touch probe on the installed palette replaces that recipe's pits
(`docs/palette.md`).

A fitted mechanical point is stored apart from `tcp_z_m`: a standoff tool
keeps its nominal working point, and a contact tool's seated point carries an
unresolved writing-contact correction.

`config/workspace.yaml` retains `mechanical_contact_*` as calibration records:
coordinates, contact reference, accuracy profile, fit status and source session.
Motion consumers use `pen_tip_offset_*` and the resolved working TCP; the
historical contact record alone grants no motion or fit qualification.

On an arm's ROS owner the fitted cartridge and its tip are runtime state, not
git: `tatbot ros cartridge fit <tool> --arm <arm>` records the cartridge in the
mount and `tatbot ros calib apply` its adopted tip, each in
`~/tatbot-ros/calib/cartridge-<arm>.json` (`scripts/lib/cartridge_fit.py`).
`tool_spec.read_workspace` lays that record over the section's `tool_id` and tip
keys there, so a swap or an adoption needs no commit or deploy. The record keeps
a tip per cartridge type: fitting a type again starts from its last measured
tip, marked carried until a run measures this cartridge's, and a type never
measured on the arm starts at its datasheet's nominal tip. The section in git is
the seed the arm's first fit or adoption copies.

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
whole result provisional. Non-contact tools may separate material and TCP only by their explicit
datasheet standoff.

Dataset metadata always records the concrete geometry that ran, including a
nominal fallback and its source; null offsets are not a geometry contract.

## The drawn line

`contact_radius_m` is collision and TCP geometry and `tip_detail` is render
geometry; neither says how wide a line the tool lays. That is the optional
`line:` mapping, recorded like `measured:` and validated at load:

```yaml
line:
  width_mm: 0.3           # in (0.05, 2.0]
  status: assumed         # measured | assumed
  utc: 2026-09-15
  method: "needle grouping's width; swatch not yet measured"
  substrate: silicon_skin
```

It is a deposition measurement on one substrate at one working speed, exposed
as `ToolSpec.line_width_m`; a partial block refuses. The DBV3 recipe records its
acquisition pen width. ROS preparation binds the fitted tool's physical footprint
without editing the acquired geometry. The fill planner's source/fixture tooling
can use the footprint for deduplication; native DBV3 paths retain repeated passes.
A `status: assumed` value is a placeholder for a measurement: draw a short swatch
at the working speed on the substrate, measure the line against a known pitch
(or a loupe), then record `measured` with the UTC and method.

The pen's passes over each spot (`tatbot_ink.passes`: a program's
`stats.passes`, a run's `passes.json`) are counted at `line.width_mm`, else at
each path's generation width; `tatbot ros compile --mark-width-mm` counts at
another width, and `stats.passes.marks` says which. No needle cartridge's mark
on skin is measured, so the 3RL counts at its 0.3 mm paths.

## The fade: `fade_lift_mm`

A stroke may fade: its `deposition` per point (a whip from `tatbot art shade`) rides
`(1 - deposition) * fade_lift_mm` higher than the pen's path, so deposition 0 lifts
the mark out (ros/README.md 4.2). `fade_lift_mm`, in (0, 4] mm like the pen trim's
reach, is the lift over which the tool's mark goes from full to nothing: read it from
a ride ladder on the tool's substrate. The 3RL records 0.7 mm: a fade ladder on the
silicone skin, five lines rising evenly from a -1.75 mm trim, inked 4-6 mm of each
(2026-11-03). It is measured from that depth, so a deeper ride fades later. The other
cartridges record none yet. A tool without it draws no fading stroke: the
compiler refuses one. On paper a fade is a line breaking
up; whether a lifted needle's gradient survives healing on skin is untested.

## The stroke: press or ride

A rotary machine's tip reciprocates through its stroke. A page touch trips
only once the tip is pushed up to the top of that stroke, where it stops, so
the touched page is where the tip meets the surface at the top of its stroke,
whether the machine runs or not. Two optional fields place the stroke, and
`ToolSpec.stroke_m` and `ToolSpec.tip_out_at_top_m` validate them at load:

```yaml
stroke_mm: 4.0            # the machine's stroke with this cartridge, in (0, 6]
tip_out_at_top_mm: -1.0   # the tip out of the cartridge's tube at the top of the stroke; negative: inside it
```

The collar that sets the tip's protrusion moves the TCP, so a change needs a
new touch-off. How the drawing path uses these is motion tuning, `motion.yaml`
`pen.mode`, read at every draw against the fitted tool's datasheet:

- **press:** the machine stays off, the tip stays at the top of its stroke,
  and the path presses `pen.press.lift_m` into the page; the arm's and mount's
  give keeps the tip on the surface.
- **ride:** the drawing stack runs the machine through its switch and the
  path rides `pen.ride.fraction` of `stroke_mm` over the touched page, so the
  running tip meets the surface that far down each stroke and would reach the
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
uncertainty metadata; they never widen the collision/marking band. Optional
explicit body geometry separately revalidates its report digest and
current-seat identity, falling back to the inferred axis if that evidence is
missing or stale. Physical capture procedures and calibration values stay in
the private operator documentation rather than this public contract.

## Cartridge exchanges

Every fitted needle cartridge dips (`ink.mode: real`, `ros/README.md` §6).
`ink.mode: cartridge` stays in the vocabulary for a self-inking refill, which
writes from its own supply and never dips; no shipped datasheet declares one.

Bind each acquired DBV3 pen to a resource with `tatbot-inks/3`. A resource may
set its own `dip:` values (`mm_per_dip`, `dwell_s`, `above_ink_m`, `drain_s`,
`drain_above_rim_m`) over the datasheet's. `drain_s` is a hold over the cap
after the dwell with the machine running, `drain_above_rim_m` over its rim; no
datasheet states one yet, so no dip drains until a lab drain ladder measures
it, and a drain without a stated height holds 3 mm over the rim.

A cartridge fitted on another's dip values says so: `dip.card: {status:
borrowed, from: [...]}`. `tatbot ros compile` refuses it for a design until
its dip card is read ([lab](lab.md)) and the datasheet says `status: read`
with the study; `--allow-borrowed-dip` draws on the borrowed values anyway,
and the program's resource then carries `dip_card: borrowed`. The 1007RL
(`lutin-7rl`) is the first: its depth and dwell are the 11RL's and its
`mm_per_dip` the 3RL's first figure. At a resource change the arm lands and the operator swaps the
cartridge by hand. Cartridges of
one machine share its body, so the resumed run keeps its measured page; a
different physical geometry needs calibration and new page touches. A fresh
cartridge of the same type can still seat its tube's end millimetres from the
last one's: a new 3RL on the xArm (2026-11-09) put its ink 2-3 mm off the plan.
Calibrate the tip again after one (`tatbot ros calib sweep`, `apply`), then the
station touch and its height, then each wrist camera's `calib wristcam` and
`calib gauge`, which hold the tip in the camera's frame.
