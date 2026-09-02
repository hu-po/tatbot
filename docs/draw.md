# `tatbot draw` — mapped-surface drawing contract

`tatbot draw` maps a non-human target with wrist cameras, compiles a design
against that surface, runs the same planner used by the executor as an offline
preflight, and only then makes a path available to a motion session. This page
documents the public interfaces between those components. It does not contain
deployment topology, measured calibration, powered-run evidence, or permission
for human use.

The implementation is split across:

- the C++ executor and samples parser under `cpp/teleop/`;
- the mapper, kinematics, path compiler, capture server, and shadow viewer under
  `scripts/`;
- the simulator's compatible displaced-surface implementation under
  `python/tatbot_sim/`; and
- the `tatbot draw` verbs registered under `scripts/lib/tatbot_cli/verbs/`.

Every exchanged file has a `schema` value. Readers refuse unknown schemas and
preflight failures refuse the path rather than silently clamping it.

## Safety boundary

Motion verbs require a complete hardware profile, the configured tool, the
hardware E-stop path, the repository's arm gate, and an operator-provided
single-use nonce. `tatbot draw plan`, `tatbot draw shadow`, synthetic capture,
and the samples-file checker are hardware-free. Passing those offline checks
does not qualify a physical system or authorize operation near a person.

Start with:

```text
tatbot status --json
tatbot draw run --explain
tatbot draw run --dry-run
```

Do not work around an exit-code 3 refusal. See [Safety and operating
boundaries](safety.md), [E-stop contract](estop.md), and [Tool
profiles](tools.md).

## Session flow

A motion session owns one draw directory, referred to below as `D`:

```text
wrapper
  -> write D/draw.json
  -> start wrist-camera capture and the optional shadow viewer
executor
  -> establish a guarded contact pose
  -> write D/trigger.json
mapper
  -> write D/orbit.csv
executor + capture server
  -> execute the orbit and exchange capture requests/results
  -> write D/hold.json
mapper/compiler
  -> write D/surface.npz, D/surface.json, D/path.csv,
     D/preflight.json, and the shadow recording
executor
  -> accept only a preflighted path, execute it, retract, and land
```

`tatbot draw scan` stops after mapping and shadow generation. It never streams
a drawing path. `tatbot draw plan` recompiles an existing surface offline, and
`tatbot draw shadow` opens existing artifacts without connecting to an arm.

## Frames

The surface is stored in the configured robot-root frame. Executor samples are
stored in the configured arm-base frame. The conversion, tool-tip transform,
tool axis, camera extrinsics, workspace, and joint limits come from the selected
profile, URDF, and tool datasheet; consumers must not substitute values copied
from another rig.

`rotation` always means the executor-controlled final arm-link rotation. The
tool-tip point and tool axis are derived from the selected tool configuration.

## `D/draw.json`

The wrapper writes a versioned configuration before starting any stage. The
shape below is illustrative; actual bounds and defaults are reported by the CLI
and selected profile.

```json
{
  "schema": "tatbot.draw-config/1",
  "tool": "lutin-ballpoint-dot",
  "design": {
    "kind": "spiral",
    "radius_mm": 10,
    "turns": 3,
    "rotation_deg": 0
  },
  "draw_speed_mm_s": 2,
  "scan_only": true,
  "orbit": {
    "mode": "camera",
    "poses": 5
  },
  "map": {
    "chart": "auto",
    "cell_mm": 1,
    "hole_fill_mm": 0
  },
  "lean_budget_deg": 20,
  "lean_deadband_deg": 0
}
```

The compiler derives segment duration from path length and configured speed.
Approach, descent, lift, orientation, and carriage behavior remain planner
contracts; they are not caller-supplied waypoints.

By default the planner follows the mapped surface normal. The optional lean
deadband lets the tool lean by the configured angle before the wrist follows
the normal. This does not bypass `lean_budget_deg`: preflight still reports and
refuses excessive lean.

## `D/trigger.json` and `D/hold.json`

The executor writes a guarded pose for the stages to consume:

```json
{
  "schema": "tatbot.draw-pose/1",
  "frame": "arm/base_link",
  "period_s": 0.0025,
  "joints": [0, 0, 0, 0, 0, 0],
  "carriage_m": 0,
  "tip": [0, 0, 0],
  "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "tool": "lutin-ballpoint-dot",
  "t_wall": 0
}
```

The numbers above demonstrate shape only. A real pose must come from the live
executor; it must never be copied from documentation.

## Samples files: `orbit.csv` and `path.csv`

Samples files use `key,value` header lines, a `columns,...` declaration, and one
row per control tick:

```text
schema,tatbot.draw-samples/1
kind,orbit | path
frame,arm/base_link
period_s,0.0025
sample_count,N
capture_count,K
start_tolerance_m,<profile-derived tolerance>
columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture
...
```

- `p` is tool-tip position and `v` is feedforward velocity.
- `r` is the row-major final-link target rotation.
- `pen` is zero for retracted travel and one for a drawing segment.
- `capture` is zero except at an orbit sample that requests a numbered capture.
- The first sample must agree with the executor's current pose within the
  declared start tolerance.

The executor parser ignores unknown report keys but refuses malformed schemas,
non-finite values, discontinuities, profile-limit violations, excessive model
error, or an invalid starting pose. `path_plan_check` uses that same parser and
planner offline; the compiler runs it before exposing a path to the executor.

## Capture handshake

The capture server writes `D/capture/server.ready` only after every configured
wrist camera is streaming. At a capture sample:

1. the executor holds and writes `request-<k>.json`;
2. the server writes `capture-<k>.npz`; and
3. the server writes `capture-<k>.done` last.

Each capture contains depth, validity counts, depth units, intrinsics, optional
color, the observed joints, carriage position, capture index, and wall time for
each configured camera role. Readers use the stored depth units and intrinsics;
they never infer them from a camera model name.

The executor treats a missing or late completion marker as a refusal and does
not continue scripted motion.

## `D/surface.npz` and `D/surface.json`

`surface.npz` uses schema `tatbot.surface/1`. It stores a plane or cylinder
chart, the chart frame, canvas dimensions, a displacement grid, per-cell sample
counts and residuals, and the contact anchor. Grid rows index `v`, columns index
`u`, and the chart rotation columns are `(e_u, e_v, n)`.

`surface.json` records provenance and fit statistics without changing the
numeric contract. The NumPy mapper and Torch simulator load the same surface
representation; parity tests pin their frame points and normals.

## Preflight: refuse, never clamp

`D/preflight.json` records either an accepted plan and its measured margins or
a named refusal. Refusal classes include:

- design extent outside the chart's injective region;
- missing observed cells under the design;
- excessive tool-axis lean;
- unreachable poses, joint limits, or joint-velocity limits;
- discontinuities or non-finite samples;
- excessive executor-model error; and
- profile, tool, schema, or start-pose mismatch.

A refusal exits with code 3. The caller may change the design, remap the target,
or correct the selected profile, but must not reinterpret a refused path as
safe.

## Hardware-free validation

The public tree supports these checks without an arm:

```text
scripts/check docs
scripts/check cli tests
draw_capture.py serve <dir> --fake
tatbot draw plan <draw-dir>
tatbot draw shadow <draw-dir> --save
cpp/teleop/build/path_plan_check <samples.csv> <period> <joint values...>
```

Synthetic capture and simulation validate schemas, transforms, path
continuity, rejection behavior, and renderer compatibility. Physical
qualification remains a separate, operator-observed process on an instrumented
non-human fixture.
