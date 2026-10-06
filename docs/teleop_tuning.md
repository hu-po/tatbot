---
summary: Public teleoperation and tuning concepts
tags: [teleoperation, control]
updated: 2026-09-14
audience: [dev, contributor]
---

# Teleoperation

Teleoperation maps a human-guided leader arm to a follower arm through an
explicit joint-space interface. The implementation is in `cpp/teleop/`.

## Physical arms and session roles

`left` and `right` identify physical arms. Leader and follower are session
roles. `config/arms.json` binds each physical arm to its controller profile
field, controller YAML, SDK end effector, workspace section and URDF prefix.
The legacy words in profile keys and YAML filenames identify the installed
hardware; assigning a session role does not exchange those references.

Inspect either assignment without opening a driver, camera or serial device:

```bash
tatbot teleop plan --leader left --follower right
tatbot teleop plan --leader right --follower left --json
```

Omitting both roles selects left leader and right follower. Specifying one
selects the other physical arm for the remaining role. Selecting the same arm
twice is a usage error. The plan reports the receiving arm's workspace/tool
reference and whether the current executor implements the assignment. A plan
does not acquire ownership, validate calibration or authorize motion.

`tatbot teleop start` accepts the same role selectors. Existing default
execution remains unchanged; explicit `--leader left --follower right` selects
that same path. General reverse execution refuses before routing or hardware
access. `--wrist-calibration` selects the supported right-led, mirrored-arm
mode for free-space capture, described below. The direct shell launcher checks
that mode and its roles as well. Editing the registry cannot enable another
mapping.

Teleop retains exclusive ownership of both arms. Per-arm leases and independent
concurrent execution are not introduced by this registry. Camera-to-arm and USB
owner assignments live in the vision registry, independently of teleop roles.
A moved camera requires a new wrist extrinsic. Tool and camera configuration
stay with the physical arm when session roles change.

## Development path

Use the simulator or recorded replay first. Build the C++ component with CMake,
then exercise the loop with no hardware attached. Keep tuning parameters in the
component configuration and record units with every change.

## One-arm measurement before tuning

`tatbot calib joint-measure` is an attended, free-air diagnostic for the
installed right-arm ballpoint. It opens the current Rust single-arm owner,
checks its native source identity and fitted tool, and records the right
carriage after opposite 0.2 mm arrivals. It then steps one rotary joint at a
time, physical joint 6 back to 1, by at most 0.02 rad in each direction at
speeds below, near and above that joint's configured friction transition.
The other five rotary joints hold their measured positions. The physical
E-stop, exclusive driver lease, contact cap and trip retract remain active.
If parked feedback rests outside a nominal rotary command limit, the owner
first uses its guarded recovery path to bring at most two such joints up to
0.03 rad each into the legal range; a larger correction is refused.
The owner may use up to two smaller follow-up commands to reach the same
20 mrad-or-less target when the first commanded trajectory ends short; it
refuses if the joint departs the bounded interval, another axis drifts more
than 5 mrad, progress stalls, or the final arrival error exceeds 5 mrad.
To continue a partially completed attended sequence, pass
`--resume-run <prior-run-id>`. The conductor checks the prior run's hashes,
six carriage direction rests, sequential joint prefix and verified idle
release, then repeats the unfinished joint from its first speed band.
If an attended retry refuses that joint for no progress, add
`--skip-blocked-run <retry-run-id>` alongside `--resume-run`. The conductor
checks the linked refusal and idle release, records that joint as incomplete,
and continues with the next joint without commanding the blocked one again.

The `calib-joint-measure` run retains the owner telemetry, command and rest
windows, native build identity, fitted tool and configuration hashes. It
applies no controller or touch-search setting. The current owner records
positions, velocities and external efforts in `telemetry.bin` (`TBGUIDE1`),
plus SDK accelerations, joint efforts and compensation efforts in a paired
`dynamics.bin` (`TBDYNA01`). These small local movements can characterize
friction and hold behavior. They cannot identify a complete end-effector
inertia tensor or replace a broader clearance-checked gravity orientation
sweep and separate dynamic excitation.

## Invariants

- The leader/follower connection is exclusive; two drivers must not command the
  same controller.
- E-stop state is checked before and during motion-capable workflows.
- A stopped loop must not replay stale targets when it resumes.
- A tool or calibration is selected explicitly, or resolved from the fitted-tool
  pointer a touch-off wrote; it is never inferred from a screenshot or a host
  name, and a stated tool that is wrong is refused rather than replaced.
- Real-time scheduling is a prerequisite, not an option: the control loop
  refuses before either arm driver is constructed when this login session cannot
  reach its priority. `tatbot teleop check` reports that (and the other
  prerequisites) without connecting to anything.

## Diagnosing lost feedback

If the arms unexpectedly become stiff, stop the experiment through the original
teleop terminal; do not force the leader. Fresh flight-log timestamps do not
prove fresh controller readings: the SDK can repeat its cached measurements.
UDP-loss warnings and a slow loop are evidence to investigate before tuning
friction or force feedback. A loop-timing warning alone cannot identify CPU
contention as the cause.

For a short diagnostic trial, start a passive trace in a separate terminal:

```bash
tatbot teleop trace-network --seconds 90
```

It routes to the arm owner. Wait for `tcpdump` to print `listening on any`, then
start normal manual teleop in its own desktop terminal with the fitted tool.
Use small free-space movements within the cleared workspace, keeping the pen
clear of the fixture, and end the trial if stiffness returns. Leave precision
teaching until communication is stable.

The trace requires installed `tcpdump` and noninteractive `sudo` permission on
the arm owner. It captures only the configured controllers' ARP and TCP/UDP
control-port traffic, with 128-byte packet snapshots, a four-file ring of about
16 MB and a 15–120 second duration. Network counters and routes are saved before
and after; capture diagnostics include kernel packet drops. Evidence lives in
`teleop-network/` under the run-log root. An empty/failed trace returns exit 3.
This command never connects an SDK, opens the e-stop serial device, changes
network configuration, or starts/stops/releases teleop. A retained trace is
diagnostic evidence, not communication or motion qualification.

## Mirrored wrist calibration

`teleop start --wrist-calibration` (the mode the retired wrist calibration
phase selected) makes physical right the input and physical left receive
commands. Joints 0, 4 and 5 reverse
movement from the measured starting poses, with no startup or resume alignment.
Shoulder, elbow and wrist pitch (joints 1–3) retain their signs.
The position, velocity and reflected-effort mappings use the same sign.
Controller addresses, per-arm profiles and SDK end-effector models remain
attached to their physical arms. The mode is for free-space wrist capture;
it rejects contact/reference capture. Its position mapping
is `q_follower_start + sign * (q_leader - q_leader_start)`, preserving each
arm's initial wrist orientation. This is an offset-preserving teleoperation
mapping, not a geometrical reflection around encoder zero.

These sessions write `WXTLOG2` with the same 64-byte header and record width as
`WXTLOG1`. Header offset 48 is a flags word: bit 0 absolute carriage, bit 1
right leader, bit 2 mirrored base, bit 3 mirrored joint 4 and bit 4 mirrored
joint 5; bit 5 means start-pose anchoring. New captures write 63. Older absolute
mirrors wrote 31 (joints 0/4/5) or 7 (joint 0 only). All remain readable; readers
and fused-pose metadata expose both the mirrored joints and anchoring policy.
Position, velocity, effort and target columns describe **control roles**, not fixed physical sides.
The right arm is the leader channel; the left arm is the follower channel.
The physical-arm-aware fuser explicitly accepts this version. Other Python
consumers refuse it until they support its role assignment. Live telemetry
uses version 2 with `right_leader: true`; old viewer receivers reject it rather
than animate the wrong arm. Version 1 recordings remain readable unchanged.
