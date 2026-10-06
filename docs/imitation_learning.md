---
summary: Public LeRobot and dataset interfaces
tags: [imitation-learning, lerobot, datasets]
updated: 2026-09-14
audience: [dev, contributor]
---

# Imitation learning

The LeRobot integration is the follower plugin and its policy interface;
training datasets come from the simulator (teleoperated recording was removed
in 2026-09). Code lives in `python/lerobot_robot_tatbot/`.

Wrist observations come from cameras assigned to the receiving physical arm in
the vision registry. A rollout reads one or more local views; it does not
open the other arm's camera. `tatbot rollout run` uses the physical left arm;
the follower plugin's `physical_arm` selects the controller preset, fitted tool,
URDF chain and floor receipt independently of its LeRobot action interface.
The launcher resolves the address and controller file from `config/arms.json`
and the hardware profile.
Async rollout checks the checkpoint's complete RGB/depth key set against
those local views before acquiring motion authority. A checkpoint trained on
two wrist views must be replaced or retrained for a one-view setup; the missing
view is never filled with the other arm's image.

The required floor receipt belongs to that same physical arm. A missing pad
touch, stale receipt or mismatched generated tool block refuses startup.
Motion aborts terminate the policy session; the retired recording-resume API
is absent. Repeated staging in one connection preserves the flight CSV and
the existing warning throttle.

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

Offline dataset loss checks tensor and preprocessing compatibility. `tatbot sim
eval policy` is the next screen: a feature-only Tatbot follower client
sends exact simulated RGB/depth/state through the deployed async LeRobot policy
server, executes the returned chunks in ManiSkill with the rollout filter/slew
semantics, then compares pigment against each episode's exact generated design.
It retains checkpoint and protocol identity, chunk accepts/rejects,
intended/drawn/overlay evidence, and confidence intervals. It refuses to compare
contaminated seed splits, dirty producers, or mixed tool/task contracts. A
physical rollout remains a separate, operator-observed acceptance stage; sim
evaluation never advances or authorizes it.
