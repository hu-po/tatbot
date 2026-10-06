# Surface, samples and capture formats

The ROS 2 stack (`ros/`), the simulator, the offline samples planner and the
wrist-camera tools read or write the files below. Every file has a `schema` value; readers refuse
an unknown schema, and a preflight failure refuses a path rather than silently
clamping it.

## Frames

A surface is stored in the configured robot-root frame. Samples are stored in
the configured arm-base frame. The profile, URDF and tool datasheet supply the
conversion between them, the tool-tip transform, the tool axis, the camera
extrinsics, the workspace and the joint limits. A consumer never substitutes
values copied from another rig.

`rotation` is always the commanded rotation of the final arm link. The tool-tip
point and the tool axis come from the selected tool configuration.

## Shared constants

Every number the C++ samples planner (`cpp/teleop/square_probe.cpp`, run
offline as `path_plan_check`) and the Python planner (`scripts/lib/pen_path.py`)
must agree on lives in one file, `config/motion_constants.json` (schema
`tatbot.draw-constants/1`). It holds the control period, the carriage-IK
envelope, the planner's joint-velocity and model-error caps, the path's lean
budget and dip speeds, and the legacy audio-rule constants.

- `scripts/gen_motion_constants.py` renders `cpp/teleop/motion_constants.hpp` from it.
- `scripts/lib/motion_constants.py` reads it.
- `scripts/check motion-constants` fails when the header is stale.

The file's short SHA is written into every samples file as `constants_sha`.
The C++ parser refuses a file whose `constants_sha` differs from the value it
was compiled with. The file is authoritative, not this page.

## Samples files: `orbit.csv` and `path.csv`

A samples file starts with `key,value` header lines, then a `columns,...`
declaration, then one row per control tick:

```text
schema,tatbot.draw-samples/1
kind,orbit | path
frame,arm/base_link
period_s,0.0025
constants_sha,<config/motion_constants.json short SHA>
sample_count,N
capture_count,K
start_tolerance_m,<profile-derived tolerance>
columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture,dip
...
```

- `p` is the tool-tip position and `v` the feedforward velocity.
- `r` is the row-major target rotation of the final link.
- `pen` is zero for retracted travel and one for a drawing segment.
- `capture` is zero except at an orbit sample that requests a numbered capture.
- `dip` is zero except on the row where dip `k` bottoms out in a cap.
- The first sample must agree with the arm's current pose within the declared
  start tolerance.

Dips are path rows like any other. Each is pen-up travel from the standoff to
the cap's transit point, a descent to the hover above the rim, the plunge, a
dwell, the retract, and the travel back. Pen-down rows are checked against
`planner.max_joint_velocity_rad_s`. Pen-up rows are checked against
`planner.max_joint_velocity_pen_up_rad_s`, with the looser model-error cap
`max_model_error_pen_up_m`.

The parser ignores unknown header keys. It refuses:

- a malformed schema, or a missing or mismatched `constants_sha`;
- non-finite values or discontinuities;
- profile-limit violations or excessive model error;
- an invalid starting pose.

`cpp/teleop/build/path_plan_check <samples.csv> <period> <joint values...>`
runs that parser and planner offline. The ROS CLIK parity test
(`ros/tatbot_motion/test/test_clik_parity.py`) compares the ROS planner
against it.

## Wrist captures: `capture-<k>.npz`

`tatbot vision capture` (`scripts/vision_capture.py`) writes one
`capture-<k>.npz` per capture and then `capture-<k>.done`. A reader waits for
the `.done` marker before opening the archive.

A capture holds, for each configured camera role:

- the depth image, its validity counts and its depth units;
- the depth intrinsics;
- one color frame, when the camera provides one.

It also holds the observed joints, the carriage position, the capture index
and the wall time. Readers use the stored depth units and intrinsics; they
never infer them from a camera model name.

## `surface.npz`

`surface.npz` uses schema `tatbot.surface/1`. It stores:

- a plane or cylinder chart and the chart frame;
- the canvas dimensions;
- a displacement grid with per-cell sample counts and residuals;
- the contact anchor.

Grid rows index `v` and columns index `u`. The chart rotation columns are
`(e_u, e_v, n)`.

A mapped chart puts `(0, 0)` at the touched reference's projection, and the
height anchoring places the surface through that reference. Reparameterizing
a cylinder keeps its fitted radius and axis. A plane's `u` direction follows
the projected robot-root X hint, or Y when that is degenerate. A cylinder's
`u` follows its fitted axis. These coordinate conventions do not establish
physical registration.

An optional `surface.json` sidecar records provenance and fit statistics
without changing the numeric contract. The NumPy reader
(`scripts/lib/surface_model.py`) and the simulator's loader
(`python/tatbot_sim/src/tatbot_sim/surface_io.py`) read the same
representation, and parity tests pin their frame points and normals.
