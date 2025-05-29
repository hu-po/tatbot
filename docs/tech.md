# Tech

technical description of tatbot

## Devices

tatbot consists of several computers, cameras, and robotos connected via ethernet:

- `ojo`: NVIDIA Jetson AGX Orin (12-core ARM Cortex-A78AE @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai`: System76 Meerkat PC (13th Gen Intel i5-1340P, 16-core @ 4.6GHz) (15GB RAM)
- `rpi1`: Raspberry Pi 5 (4-core ARM Cortex-A76 @ 2.4 GHz) (8GB RAM)
- `rpi2`: Raspberry Pi 5 (4-core ARM Cortex-A76 @ 2.4 GHz) (8GB RAM)
- `camera1`: Amcrest PoE cameras (5MP)
- `camera2`: Amcrest PoE cameras (5MP)
- `camera3`: Amcrest PoE cameras (5MP)
- `camera4`: Amcrest PoE cameras (5MP)
- `camera5`: Amcrest PoE cameras (5MP)
- `camera-a`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `camera-b`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-main`: 5-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l`: Trossen Arm Controller box connected to WidowXAI arm
- `arm-r`: Trossen Arm Controller box connected to WidowXAI arm
- `oop`: (only used for development)

## Dependencies

- [`pyroki`](https://github.com/chungmin99/pyroki) - inverse kinematics
- [`viser`](https://github.com/nerfstudio-project/viser) - browser visualizer and GUI
- [`librealsense`](https://github.com/IntelRealSense/librealsense) - RGBD depth cameras (pointclouds) for skin tracking
- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm) - robot arms
- [`jax`](https://github.com/jax-ml/jax) - gpu acceleration

## Networking

- `switch-main`: 
    (1) short black ethernet cable to `switch-poe`
    (2) short black ethernet cable to `trossen-ai`
    (3) short black ethernet cable to `ojo`
    (4) blue ethernet cable to `arm-r` controller box
    (5) blue ethernet cable to `arm-l` controller box
- `switch-poe`: 
    (1) long black ethernet cable to `camera1`
    (2) long black ethernet cable to `camera2`
    (3) long black ethernet cable to `camera3`
    (4) long black ethernet cable to `camera4`
    (5) long black ethernet cable to `camera5`
    (6) -
    (7) short black ethernet cable to `rpi1`
    (8) short black ethernet cable to `rpi2`
    (uplink-1) -
    (uplink-2) short black ethernet cable to `switch-main`

hardcoded ip addresses:

- `192.168.1.33` - `oop`
- `192.168.1.91` - `camera1`
- `192.168.1.92` - `camera2`
- `192.168.1.93` - `camera3`
- `192.168.1.94` - `camera4`
- `192.168.1.95` - `camera5`
- `192.168.1.96` - `ojo`
- `192.168.1.97` - `trossen-ai`
- `192.168.1.98` - `rpi1`
- `192.168.1.99` - `rpi2`

## Realsenses

Two D405 realsense cameras are used to get a pointcloud of the skin. Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).

`camera-a` is connected to `trossen-ai` via usb3 port and attached to the end effector of `arm-r`
`camera-b` is connected to `trossen-ai` via usb3 port and attached to alumnium frame, giving it an overhead view

## Setup

1. flip power strip in back to on
2. press power button on `trossen-ai`, it will glow blue
3. flip rocker switches to "on" on `arm-r` and `arm-l` control boxes

keys, tokens, passwords are stored in the `.env` file.

```bash
source .env
```

python dependencies are managed with environments using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
deactivate && \
rm -rf .venv && \
rm uv.lock && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r pyproject.toml && \
uv run python tatbot.py
```

jax is used for gpu acceleration, to check if jax has access to gpu:

```bash
uv run python -c "import jax; has_gpu = bool(jax.devices('gpu')); print(has_gpu)"
```

## URDF

tatbot is defined using a [custom URDF file](https://github.com/hu-po/tatbot-urdf), which might not be kept up to date with the official [trossen widowxai description](https://github.com/TrossenRobotics/trossen_arm_description)