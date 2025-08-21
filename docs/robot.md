---
summary: Robot arms, URDF models, and kinematics
tags: [robot, arms]
updated: 2025-08-21
audience: [dev, operator]
---

# Robot

## Arms

- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm)
- [Trossen WidowXAI arms](https://docs.trossenrobotics.com/trossen_arm/main/specifications.html)
- [Driver API Docs](https://docs.trossenrobotics.com/trossen_arm/main/api/library_root.html#)

see:

- `config/trossen/arm_{l|r}.yaml`: Low-level arm and motor parameters.
- `src/conf/arms/default.yaml`: High-level configuration, including IP addresses and end-effector offsets.
- `src/tatbot/data/arms.py`: The Pydantic data model that loads these configurations.

### Update Firmware

First update the firmware to the latest version:

https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html#software-upgrade

Download latest firmware:

https://docs.trossenrobotics.com/trossen_arm/main/downloads.html

```bash
cd ~/Downloads && wget <get link from above>
unzip firmware-wxai_v0-v1.8.4.zip
teensy_loader_cli --mcu=TEENSY41 -s firmware-wxai_v0-v1.8.4.hex
```

TODO: Set the velocity_tolerance to 0.2 times the velocity max
https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#joint-limits
TODO: Edit end effector parameters for tattoo ee
https://docs.trossenrobotics.com/trossen_arm/main/api/structtrossen__arm_1_1EndEffector.html#struct-documentation
TODO: Create trossen_arm.StandardEndEffector.wxai_v0_tatbot_l and trossen_arm.StandardEndEffector.wxai_v0_tatbot_r

### Home Arms

https://docs.trossenrobotics.com/trossen_arm/main/service.html#arm-homing

## URDF

- tatbot is defined using URDF at `tatbot/urdf/tatbot.urdf`
- based off the [official URDF](https://github.com/TrossenRobotics/trossen_arm_description)

## IK

- [`jax`](https://github.com/jax-ml/jax)
- [`pyroki`](https://github.com/chungmin99/pyroki)
- every stroke tatbot executes is a sequence of joint angles
- joint angles are computed using inverse kinematics (ik)
- ik is computed on the GPU in parallel

see:

- `src/tatbot/data/stroke.py`
- `src/tatbot/gen/batch.py`
