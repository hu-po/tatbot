---
summary: Public fiducial configuration concepts
tags: [vision, fiducials, calibration]
updated: 2026-09-27
audience: [dev, contributor]
---

# Fiducials

Fiducials provide repeatable reference points for camera and end-effector pose
estimation. The machine-readable inventory lives in `config/fiducials.json`;
the Rust and Python consumers must read that inventory rather than duplicate
IDs or sizes.

## Configuration rules

- Give every marker family, id, and physical size an explicit record.
- Keep units and frame names beside the value.
- Separate board-only markers from markers that may appear on an end effector.
- Version calibration outputs and record the exact input profile.

Schema 2 requires an explicit `targets.<name>.family` for every target. Both
`apriltag_16h5` and `apriltag_36h11` are supported in the same inventory; schema
1 retains its single-family interpretation. Detections retain both family and numeric ID. Calibration captures store
family-scoped corner maps, so the same numeric ID in different families
cannot enter the wrong target solve. Legacy numeric-only records omit IDs
that are ambiguous in their frozen inventory.
Same-family duplicates require an explicit ambiguity group and phase separation.
A pending layout can contain no transforms; generated tag placement and pose
tracking remain unavailable until a measured layout passes the existing gates.

The palette target is a world anchor, not an end-effector layout: one 36h11 tag
on the palette's enclosure roof, turned a quarter turn against the palette
body. The `palette_tag` joint in `urdf/palette.urdf` carries that measured
turn, and the palette pose solve crosses it rather than assuming the tag is
aligned with the body. See [palette](palette.md).

Run calibration against a static, non-human fixture. Private geometry,
identifiers, and acceptance measurements do not belong in a public page.

## Positioning a parked wrist for camera inspection

`tatbot --ee-tool <fitted-tool> calib inspect --arm blue --operator-go`
retains current measured joints before selecting a new pose. Connection takes
a guarded position hold at the measured pose, so this is an active hold rather
than a passive network probe. It uses the same native calibration owner,
build identity, driver lease, physical E-stop and verified idle release as
`calib pose`. It selects no wrist, base, staging or landing target. The
`calib-inspect` run retains connection, status, telemetry and release records;
these measure state, not a camera registration or task clearance.

Native re-seeding retains `native-reseed ` JSON lines in `owner.log.stderr`.
Each uses the receipt family and `native-reseed` kind. They record the measured
pose, legal hold target, mode-change requirement and
wall/elapsed timestamps around mode selection and the timed-target wrapper.
Target timestamps include the wrapper's feedback and stop checks; they are not
SDK send timestamps. Correlate these phases with the complete native flight,
including rows before the connected acknowledgement. Commanded limits and a
quiet post-connection hold alone do not establish a measured startup speed
bound. These diagnostics add no feedback query or controller setting change.

`tatbot --ee-tool <fitted-tool> calib pose --arm pink --wrist-deg 180`
positions one parked wrist for camera inspection.
Angles are absolute wrist-roll joint degrees. Optional `--base-deg` selects
absolute base yaw within +/-30 degrees; omission retains the measured yaw.
Joints 1–4 must already be within 0.1 rad of their configured parked targets
and return to those targets. The wrist travel is at most a quarter turn plus
0.05 rad of readback tolerance. The command uses the existing guarded recovery
worker: legal measured takeover followed by small, timed vendor moves, with
arrival checked after each increment. It never streams below-zero hard-stop
feedback as a new motion target. No controller limit is changed.

Each increment lasts four seconds. Planned base/wrist increments use a
0.08 rad/s twice-mean speed bound, leaving room for the 0.02 rad arrival
readback tolerance within 0.1 rad/s. The recovery path retains its carriage
qualification, live limits, exclusive driver lease, E-stop and contact cap.
The workspace and complete tool sweep must be clear, both tools non-emitting,
and the other arm parked and stable. After the final arrival it holds for
`--hold-seconds` (default 5, maximum 120), then idles and releases the arm.
It does not return to the previous angle. The `calib-pose` run retains every
increment, feedback tick and measured release. Camera visibility does not
establish calibration accuracy or return repeatability.

## Calibration

The fixed-camera calibration that used these targets (board captures, the
per-arm recipe, the rig candidate and its adoption, and motion-envelope
teaching) was removed in 2026-09 with the overhead PoE cameras. The inventory
stays the source for tracking, stencil and station tags; see
[Vision](vision.md) for what the retained registrations still feed.


For a retained current-pose wrist observation, add `--hold-seconds 5
--capture-wrist` to `calib inspect`. The bounded hold requests status from the
existing native owner, then subscribes to the already running wrist capture
owner. It retains the shared original RGB-D capture with unspecified joint
labels, releases to verified idle, and binds each normalized exposure to
bracketing native telemetry in `wrist-pose-binding.json`. That report includes
nearest measured joints, carriage, timestamp skew, bracket gaps and observed
motion across the camera window. It supplies nominal host-clock correspondence;
physical accuracy and world registration remain unqualified. It adopts no
camera, arm or board calibration. Capture failures follow the same terminal
release path; no new pose target is selected.

Add `--capture-overhead` to `calib inspect`, or to a reviewed `calib pose
--operator-go`, to retain the fixed camera during the same native hold. It
queries the existing overhead-depth owner through the shared capture API;
no ROS node, second arm owner or camera SDK is started. The run's `overhead/`
contains `owner-packet.bin` (the exact frame-set reply), its SHA256-bound
`capture.json`, a decoded BGR array/PNG and aligned depth in metres when present.
After verified idle release, `overhead-pose-binding.json` binds every original
frame's normalized exposure to bracketing native telemetry. An unbracketed
exposure fails the binding while preserving capture and release evidence.
The report retains measured joints/carriage, host skew/gaps and observed
motion; cross-device clock accuracy and physical accuracy remain unqualified.

`tatbot calib register-fit --python /path/to/validated/python --arm blue
--capture /path/to/hold-a /path/to/hold-b --bundle /path/to/calibration.json
--setup-id table-setup --out /path/to/new-fit` fits retained overhead views
offline. Select captures from one physical setup after moving an arm base or
the fixed camera; old registrations and start positions cannot establish the
new relationship. Tool- and wrist-local geometry and unchanged camera optics
can be reused.

The fit verifies original packet hashes, metadata and reconstructed exposure
bindings, clean capture source revisions and unchanged model/configuration
inputs. Seven measured axes feed the existing URDF, including carriage and
joint corrections. The target detector and robust registration fitter are
shared with the calibration stack. Repeated captures at the same pose do not
increase distinct-view coverage. Existing quality gates require twelve
distinct poses, repeated tag coverage, corner residuals and leave-one-view-out
stability. The output retains observations, source hashes, the generated model,
bundle and an explicitly unadopted candidate; exit 3 records a quality refusal.
This command starts no ROS node, arm driver or camera owner and installs no
calibration. Host-time correspondence and fitted pixel residuals alone do not
establish physical tracing accuracy.
Add `--seat-report` when the existing interpreter also has SciPy to compare
the shared rigid and tag-seat models. Each holdout removes a complete shared
distinct-view group, including its repeated captures; camera and seat are
refitted without that group. All original sightings remain in this diagnostic.
`seat-diagnostics.json` retains the comparison and original holds per view;
`wrist-layout.json` is a measured-layout candidate whose per-tag pose counts
use those distinct groups. It changes no rigid registration gate and installs
nothing. Validate a proposed layout through `tatbot vision tags export`, then
collect a fresh registration against the adopted model. A better fit does not
prove mount movement or provide a physical accuracy bound.
This option changes no motion limit, posture target or adoption gate. Captured
pose motion requires the explicit `--operator-go` declaration; standing task
authorization can cover it. Multiple qualified views are inputs to the shared
registration fit, not automatic calibration adoption.
