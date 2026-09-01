---
summary: End-effector fiducial tracking interface
tags: [vision, fiducials, end-effector]
updated: 2026-08-31
audience: [dev, contributor]
---

# End-effector tracking

The end-effector tracker estimates a tool pose from wrist or tool-mounted
fiducials and camera observations. It is a software interface for replay and
instrumented testing, not a permission to mount a tool for human use.

## Pipeline

1. Load the versioned fiducial inventory and camera calibration.
2. Reject frames with missing identity, stale timestamps, or invalid geometry.
3. Solve pose in a named coordinate frame.
4. Attach confidence, source frames, and calibration revisions to the result.
5. Compare against a fixture or replay before any physical integration.

The implementation and detailed measured baselines remain in the private
engineering tree. Public contributions should add fixtures and failure cases.
