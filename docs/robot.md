---
summary: Public robot model and control concepts
tags: [robot, arms, kinematics]
updated: 2026-08-31
audience: [dev, contributor]
---

# Robot model

Tatbot uses a pair of Trossen WidowX AI arms in a leader/follower research
configuration. The public repository includes the model and software interfaces;
deployment-specific addresses and calibration records are private.

## Model

The URDF at `urdf/tatbot.urdf` is consumed by vision replay and visualization
tools. Treat it as a shared interface: when geometry changes, rebuild consumers
and record the model revision with the run.

The fitted visual model has a Lutin ballpoint on the pink-taped `right` arm
and a mirrored laser attachment on `left`, with one D405 on each wrist.
Left/right are persistent physical identities; their positions on the screen
depend on the viewing direction. Leader/follower are control assignments.
The old left handle and finger links retain frames only, without geometry.
The left laser uses the V5 CAD assembly, whose meshes are authored in
`left/carriage_right` but which is bolted to `left/carriage_left` rolled half a
turn about the carriage axis — the placement the measured wrist fiducials and
the palette contact holds agree on. The pen is rendered from its datasheet on
the tool datum carried through that placement, with its lens face at the
2026-09-16 touch-off. Nothing there establishes a laser focus or collision
clearance.

Left and right are the robot's own sides, looking forward through the fixed
cameras: +X forward, +Y left, +Z up. `root` and `rig_center` coincide at the
arm-pair midpoint. A registration must associate the observed wrist with that
same arm's measured joints; similar leader/follower poses do not prove identity.
A changed whole-URDF digest requires fresh source bindings wherever an
acquisition binds that digest.

The control path is joint-space. It keeps the mapping from a leader arm to a
follower explicit instead of hiding a separate inverse-kinematics service in the
public API.

## Control layers

- `cpp/teleop/` provides low-latency leader/follower teleoperation.
- `python/lerobot_robot_tatbot/` adapts the robot to LeRobot episode and policy
  interfaces.
- `rust/visiond/` records camera data and can replay a run with the URDF.
- `python/tatbot_sim/` provides hardware-independent development and tests.

See [teleoperation](teleop_tuning.md), [vision](vision.md), and
[simulation](simulation.md) for the public entry points.

## Hardware profiles: `tatbot profile`

Which arms, addresses and e-stop a checkout may drive is a *profile*
(`config/profiles/`), resolved before any motion verb connects. `tatbot
profile show [name]` prints the resolved one (backend, arms, e-stop,
provenance) and says whether it may drive hardware; `tatbot profile list`
lists the profiles this checkout carries and marks each `hardware` or
`synthetic/incomplete`; `tatbot profile check [name]` runs exactly the
validation the motion gate runs and exits 3 when it would refuse, without
starting anything. A public clone carries only gate-incapable profiles.

## Recovery after a controller fault

`tatbot arm recover` lands the follower first and starts leader recovery only
after the follower succeeds. Failure, interruption or timeout stops the sequence.
An explicit `leader` or `follower` argument selects one arm. Each single-arm
attempt retains an `arm-recover` run log.

Recovery automatically terminates existing processes holding the shared arm
driver lease, including teleop, LeRobot and stuck recovery children. It sends
SIGTERM, allows up to 10 seconds for cleanup, then sends SIGKILL to remaining
holders and waits up to two seconds for the lock. This also stops a workflow
using both arms when recovering an explicit single role.
Only actual lease holders are signaled; unrelated processes are left running.
Recovery retains the lock file and must acquire its exclusive lock before
connecting to an arm. If ownership cannot be obtained, it refuses with exit 6.

The landing runs on the C++ arm stack: `cpp/teleop/arm_recover`, a small
program on the vendor SDK that the teleop executor also uses, built by
`scripts/check cpp` and staged on the arm node by `tatbot deploy`. No
Python environment is involved. It opens a fresh driver session with the
controller's error state cleared, pushes the arm's golden when the controller
boots faulted or with measured feedback outside its admitted tolerance, takes
position control softly at the measured pose, then moves staged → sleep → idle
and verifies the sleep pose was actually reached. An arm joint measured past its
limits by more than that tolerance is refused instead, before any command (exit
7): a power cycle can leave a joint counted a full turn off, and clamping it into
its limits would snap the arm. Power the controller off, turn that joint by hand
to near 0, and power it on before recovering again. The carriage stays at its
measured position through the staged sweep, then returns to its configured
rest position (0 m in the Tatbot profile) during the sleep phase; recovery
fails if that return is not measured. The e-stop device is mandatory: a latched or silent button
refuses the landing before any command (exit 3), and a press mid-move freezes
the arm where it is until the button is released. Each attempt runs in its own
child process, because the vendor driver cannot be reused after a failed
connection; a controller that never answers exits 5 with nothing commanded. The
LeRobot plugins keep their own copy of this ritual for a failed in-session
disconnect.

The standalone launcher enforces a 45-second budget around that process,
including termination of the previous owners, with a two-second termination
grace. The SDK's connection timeout alone does not
bound later TCP reads in older drivers. Expiry terminates the driver and reports
**unknown arm state**, never a successful landing or release. Landing
verification and Ctrl+C shielding remain in the recovery process while it runs.
This outer timeout does not repair controller communication or establish that a
blocked driver can handle an e-stop.

Before any controller power cycle, support both arms and stop all existing
recovery/teleop processes. A pending recovery could otherwise reconnect and
start a landing when the controller returns. Do not force a stiff arm or start
another driver alongside a hung one. Trossen documents controller and driver
hangs in older versions; see the
[vendor troubleshooting guide](https://docs.trossenrobotics.com/trossen_arm/v1.9/troubleshooting.html#connection-issues).

## Hardware boundary

The public model is not a calibration or an operating procedure. Do not infer
joint limits, tool offsets, contact behavior, or permission for human use from
the example files. Those values belong to the private acceptance contract.
