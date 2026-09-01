---
summary: Public fiducial configuration concepts
tags: [vision, fiducials, calibration]
updated: 2026-08-31
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

Run calibration against a static, non-human fixture. Private geometry,
identifiers, and acceptance measurements do not belong in a public page.
