---
summary: Public robot model and control concepts
tags: [robot, arms, kinematics]
updated: 2026-08-31
audience: [dev, contributor]
---

# Robot model

Tatbot uses a pair of Trossen WidowX AI arms in a leader/follower research
configuration. The public repository includes the model and software interfaces;
deployment-specific addresses and calibration records are private.

## Model

The URDF at `urdf/tatbot.urdf` is consumed by vision replay and visualization
tools. Treat it as a shared interface: when geometry changes, rebuild consumers
and record the model revision with the run.

The control path is joint-space. It keeps the mapping from a leader arm to a
follower explicit instead of hiding a separate inverse-kinematics service in the
public API.

## Control layers

- `cpp/teleop/` provides low-latency leader/follower teleoperation.
- `python/lerobot_robot_tatbot/` adapts the robot to LeRobot episode and policy
  interfaces.
- `rust/visiond/` records camera data and can replay a run with the URDF.
- `python/tatbot_sim/` provides hardware-independent development and tests.

See [teleoperation](teleop_tuning.md), [vision](vision.md), and
[simulation](simulation.md) for the public entry points.

## Hardware boundary

The public model is not a calibration or an operating procedure. Do not infer
joint limits, tool offsets, contact behavior, or permission for human use from
the example files. Those values belong to the private acceptance contract.
