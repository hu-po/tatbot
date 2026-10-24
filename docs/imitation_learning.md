---
summary: Public LeRobot and dataset interfaces
tags: [imitation-learning, lerobot, datasets]
updated: 2026-10-23
audience: [dev, contributor]
---

# Imitation learning

The LeRobot integration is an environment, not a robot: `python/lerobot_robot_tatbot/`
pins the LeRobot install that training, the policy server, the dataset tools
and `vision capture` run in, and holds the `depth-v1` encoder training applies
to recorded D405 depth. Training uses the archived datasets (`tatbot data
hub`): teleoperated recording was removed in 2026-09 and the ManiSkill data
factory in 2026-10, so no source produces new ones. The follower robot class
that ran policies on the blue arm was deleted with that arm's use in 2026-10;
no robot client consumes the policy server today.

A checkpoint's camera keys name wrist views of one physical arm, as the vision
registry assigns them. A checkpoint trained on two wrist views must be
replaced or retrained for a one-view setup; the missing view is never filled
with the other arm's image.

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

## Evaluation ladder

Offline dataset loss checks tensor and preprocessing compatibility; `tatbot
rollout bench plausibility` checks a policy's action chunks against a
demonstration-derived contract, offline or against a running policy server,
with no arm. There is no simulated policy screen: the ManiSkill evaluation was
deleted with that simulator, and none replaces it yet.
