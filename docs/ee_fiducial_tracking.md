---
summary: End-effector fiducial tracking interface
tags: [vision, fiducials, end-effector]
updated: 2026-10-23
audience: [dev, contributor]
---

# End-effector tracking

The end-effector tracker estimates a tool pose from wrist or tool-mounted
fiducials and camera observations. It is a software interface for replay and
instrumented testing.

## Pipeline

1. Load the versioned fiducial inventory and camera calibration.
2. Reject frames with missing identity, stale timestamps, or invalid geometry.
3. Solve pose in a named coordinate frame.
4. Attach confidence, source frames, and calibration revisions to the result.
5. Compare against a fixture or replay before any physical integration.

The live tracker, `trackd`, publishes `tatbot.tracking-pose/1`: the tracked
frame's pose in the matched camera calibration world (`world_from_ee`, with
`tracking_frame` naming the frame), the tags and cameras it used, and the
inventory, layout and calibration identities. The historical robot-world key
`world_from_base` describes the full URDF root; it must not be confused with an
arm-base pose.

The implementation and detailed measured baselines remain in the private
engineering tree. Public contributions should add fixtures and failure cases.
