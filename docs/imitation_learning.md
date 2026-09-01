---
summary: Public LeRobot and dataset interfaces
tags: [imitation-learning, lerobot, datasets]
updated: 2026-08-31
audience: [dev, contributor]
---

# Imitation learning

The LeRobot integration turns teleoperated episodes into datasets and exposes a
matching policy interface. Code lives in `python/lerobot_robot_tatbot/`.

## Dataset contract

An episode should include task text, action and observation schemas, sampling
rate, tool/schema version, source revision, and a clear simulated-versus-real
label. Keep personal data, raw recordings, credentials, and private model
artifacts out of the public repository.

## Reproducibility

Pin the code revision and dependency lockfile. Validate shapes and units offline
before training. Compare policy results on held-out fixtures and report failed
or rejected runs rather than selecting only successful examples.

Training and physical evaluation policies are private acceptance material; this
page documents only the public adapter boundary.
