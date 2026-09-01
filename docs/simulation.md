---
summary: Hardware-independent Tatbot simulation workflow
tags: [simulation, testing]
updated: 2026-08-31
audience: [dev, contributor]
---

# Simulation

Use `python/tatbot_sim/` for offline development, dataset-shape checks, and
control experiments that do not connect to an arm or camera.

## Quick start

```bash
cd python/tatbot_sim
uv sync
uv run python -m tatbot_sim --help
```

Keep generated episodes and renders outside the repository. Include the source
revision and simulator configuration in any artifact manifest.

## What simulation proves

Simulation can validate pure transforms, schema handling, deterministic replay,
and software integration. It does not prove camera calibration, contact force,
e-stop behavior on physical hardware, or safe human use.

## Contribution checklist

1. Add a deterministic fixture for the new behavior.
2. Assert units and coordinate frames at the boundary.
3. Run the simulator tests and `scripts/check --light`.
4. Label simulated results as simulated in the run manifest.
