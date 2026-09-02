---
summary: Public teleoperation and tuning concepts
tags: [teleoperation, control]
updated: 2026-09-01
audience: [dev, contributor]
---

# Teleoperation

Teleoperation maps a human-guided leader arm to a follower arm through an
explicit joint-space interface. The implementation is in `cpp/teleop/` and the
LeRobot adapter provides a separate episode interface.

## Development path

Use the simulator or recorded replay first. Build the C++ component with CMake,
then exercise the loop with no hardware attached. Keep tuning parameters in the
component configuration and record units with every change.

## Invariants

- The leader/follower connection is exclusive; two drivers must not command the
  same controller.
- E-stop state is checked before and during motion-capable workflows.
- A stopped loop must not replay stale targets when it resumes.
- A tool or calibration is selected explicitly; it is never inferred from a
  screenshot or a host name.

## Native-Cartesian square capability probe

`tatbot teleop square` separates follower accuracy from leader-arm stiction. An
operator hand-guides the fitted tip to light paper contact using the normal C++
teleop. After both arms remain below 0.10 rad/s for 0.20 seconds, the probe
prints `READY`. Readiness stays latched while the operator reaches for the
keyboard unless the follower itself exceeds 0.30 rad/s. One SPACE press—with
no Enter—preflights the complete square through a damped-least-squares model,
then streams its joint positions at 400 Hz. Each edge uses a quintic time
profile that starts and ends at zero velocity and acceleration. The default is
a 6 mm edge over 12 seconds (0.5 mm/s average, 0.9375 mm/s nominal Cartesian
peak). Z and orientation stay at the trigger pose.

This is autonomous motion after the trigger. Run it locally on the arm node;
the CLI refuses `--on`. A fresh literal nonce, explicit tool, live E-stop
operator, contact cap, measured-velocity guard and rolling-effort guard are all
mandatory:

```console
tatbot --ee-tool lutin-ballpoint-dot teleop square \
  --size-mm 6 --edge-s 12 --nonce <fresh-literal>
```

The probe chooses the first base-X and base-Y signs toward the arm base, then
uses the opposite signs to close the square. This avoids extending farther out
from a hand-guided start that may already be near the IK workspace boundary.
The chosen sequence is printed before motion. This tests a square in the
controller frame; it does not register the square to printed grid lines. On
completion the carriage retracts the pen and both arms hold until the operator
supports them and presses Enter. An E-stop, contact cap, measured velocity or
rolling arm-effort trip also retracts and holds, but terminates the probe
permanently. Scripted motion never auto-resumes.

Before motion, the local model must agree with the SDK's live FK within 0.25
mm, and every planned sample must pass guarded joint limits, model tracking and
a 0.25 rad/s planned joint-velocity cap. During motion, a planned position may
never lead a measured joint by more than 0.05 rad. This position-controlled
path lets a small error accumulate across drivetrain stiction; the earlier
Cartesian-velocity path left loaded joints 0 and 1 completely stationary at
sub-millimetre-per-second TCP speeds. Measured joint velocity and rolling
effort remain independently guarded.

The arm must reach each corner within 0.25 mm. It may settle for up to 3 seconds
after the nominal edge time; failure retracts the pen and terminates the probe.
After any non-emergency terminal outcome, Enter releases the hold and the
wrapper lands follower then leader through the shared staged-to-sleep recovery
routine. Ctrl+C at that prompt is an emergency release and deliberately skips
automatic landing.

Each run writes `square_probe.csv` beside its normal flight log. Its endpoint
errors come from the same encoders and forward kinematics that drive the
controller, so they diagnose command tracking but do **not** establish physical
tip accuracy. Measure the inked side lengths, closure gap, corner overshoot and
line waviness on the paper. The physical mark is the acceptance evidence.

## Expanding-spiral distortion probe

`tatbot teleop spiral` uses the same one-key handoff, full joint-plan preflight,
runtime guards, terminal hold and automatic landing as the square probe, but
draws one continuous Archimedean spiral. The trigger point is the spiral center,
not an outer endpoint. Leave at least the selected radius clear on the paper in
every base-X/Y direction.

```console
tatbot --ee-tool lutin-ballpoint-dot teleop spiral \
  --radius-mm 6 --turns 3 --duration-s 180 --ease-s 2 \
  --nonce <fresh-literal>
```

The default three-turn path is about 57.2 mm long. It advances at approximately
constant arc-length speed, with a two-second quintic speed ease at each endpoint
so velocity and acceleration join the stationary hold continuously. Over 180
seconds its cruise speed is about 0.321 mm/s; a 120-second run cruises at about
0.485 mm/s. Unlike full-duration quintic progress, it leaves the center promptly
instead of spending 41 seconds inside the first 0.5 mm. The continuous curve
makes base-X/Y scale mismatch visible as an ellipse and changing backlash or
tracking error visible as direction-dependent distortion over several
revolutions.

Each run writes `spiral_probe.csv` beside the flight log. At 10 Hz it records
the planned and encoder/FK X, Y and Z, per-axis error, and planned/measured
radius. This can diagnose joint command tracking and model-coordinate Z drift.
It cannot measure compliance in the pen mount, paper height or actual tip depth.
Use line darkness, width, skips and paper deformation as the physical depth
evidence; measure the inked major/minor diameters at several radii for X/Y
distortion.

For a physical A/B against the same path, the experimental ballpoint-only mode
adds the tool carriage as a seventh coordinated degree of freedom:

```console
tatbot --ee-tool lutin-ballpoint-dot teleop spiral \
  --radius-mm 6 --turns 3 --duration-s 120 --ease-s 2 \
  --carriage-ik --nonce <fresh-literal>
```

Keep the pen clear until the printed off-paper carriage preflight passes. It
slowly checks a 2.0 -> 1.5 -> 2.5 -> 2.0 mm reversal, then holds a 2 mm bias so
the operator establishes contact in the same configuration used by the plan.
At SPACE, all 48,000 samples must pass arm limits, modeled tip tracking and a
guarded 0.5--3.5 mm carriage envelope before any scripted command is sent.
Runtime arm lead, carriage lead, contact and E-stop trips retract the pen and
terminate the candidate. The CSV adds target/measured carriage position so its
tracking can be compared with the ink; the ordinary six-joint mode is unchanged.

Public docs describe interfaces only. Powered tuning and acceptance records are
private and must remain separate from public examples.
