---
summary: Public tool registry and geometry conventions
tags: [tools, geometry, configuration]
updated: 2026-08-31
audience: [dev, contributor]
---

# Tool registry

Every end-effector tool used by code should have one versioned datasheet under
`config/tools/`. Simulation geometry, URDF links, calibration, and dataset
metadata must derive from that record instead of repeating dimensions.

## Datasheet contract

- Identify the schema version and coordinate origin.
- State units, mounting frame, tip geometry, and contact assumptions.
- Validate the fitted tool before generating derived geometry.
- Record the datasheet revision with every run or placement artifact.

Public examples should use a clearly marked fixture tool. Physical dimensions,
calibration values, and deployment choices that are not needed to build the
software remain private.
