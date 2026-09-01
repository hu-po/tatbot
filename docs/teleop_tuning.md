---
summary: Public teleoperation and tuning concepts
tags: [teleoperation, control]
updated: 2026-08-31
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

Public docs describe interfaces only. Powered tuning and acceptance records are
private and must remain separate from public examples.
