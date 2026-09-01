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

## A dataset to read without generating one

[`tatbot/sim-paper-draw-demo`](https://huggingface.co/datasets/tatbot/sim-paper-draw-demo)
is a published shard of the `paper-draw` distribution — 8 episodes, 2836
frames, wrist RGB and depth, generated with seed 0 and no flags beyond the
recipe. Use it to inspect the dataset shape, feature names and metadata that
this repository produces before running the factory yourself:

```bash
uv run --project python/lerobot_robot_tatbot python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('tatbot/sim-paper-draw-demo')
print(ds.meta.info['total_episodes'], 'episodes;', ds.meta.info['total_frames'], 'frames')
print(sorted(ds.meta.features))"
```

Regenerate an equivalent shard locally with:

```bash
tatbot sim generate paper-draw -- --out-dir <dir> --num-episodes 8 --num-envs 4 --seed 0
```

## What simulation proves

Simulation can validate pure transforms, schema handling, deterministic replay,
and software integration. It does not prove camera calibration, contact force,
e-stop behavior on physical hardware, or safe human use.

## Contribution checklist

1. Add a deterministic fixture for the new behavior.
2. Assert units and coordinate frames at the boundary.
3. Run the simulator tests and `scripts/check --light`.
4. Label simulated results as simulated in the run manifest.
