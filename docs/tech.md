# Tech

description of tatbot (tattoo robot) technical stack

## Index

- [Software](#software)
  - [Run](#run)
  - [Setup](#setup)
  - [Dependencies](#dependencies)
  - [Devices](#devices)
  - [Networking](#networking)
  - [Models - VLAs](#models---vlas)
- [Hardware](#hardware)
  - [Trossen Robot Arms](#trossen-robot-arms)
  - [URDF](#urdf)
  - [Realsense Cameras](#realsense-cameras)
  - [PoE IP Cameras](#poe-ip-cameras)
  - [AprilTags](#apriltags)
- [Artwork](#artwork)

## Run

setup environment, optionally clean output directory

```bash
source scripts/env.sh
./scripts/clean.sh
```

tatbot is designed as a multi-node system, with the following roles:

`oop` 🦊 and `ook` 🦧 are the main nodes, they:
- generate images and create plans (using gpu)
- run the mcp server to interact with tatbot
- run the visualization server to view plans

```bash
# optionally install dev dependencies
uv pip install .[dev,viz] && \
uv pip install .[gen] && \
uv run -m tatbot.gen.from_svg --name "yawning_cat" --debug
```

`trossen-ai` 🦾 sends commands to robot arms, receives realsense camera images, and records lerobot datasets:

```bash
uv pip install .[bot] && \
# configure trossen arms
uv run -m tatbot.bot.trossen --debug
# run lerobot dataset recording from plan
uv run -m tatbot.bot.plan --debug
```

`ojo` 🦎 runs the policy servers for the VLA model and for the 3d reconstruction model

```bash
uv pip install .[vla] && \
# TODO
```

`rpi1` 🍓 runs apriltag tracking and camera calibration:

```bash
uv pip install .[tag] && \
uv run tatbot.tag.scan --bot_scan_dir ~/tatbot/outputs/ --debug
```

`rpi2` 🍇 hosts the network drive, which contains the output directory for all nodes

```bash
uv pip install .[viz] && \
uv run -m tatbot.viz.plan --plan_dir ~/tatbot/outputs/plans/yawning_cat
```

## Dependencies

tatbot makes use of python dependencies managed using [`uv`](https://docs.astral.sh/uv/getting-started/installation/): see `pyproject.toml`.

- [`mcp-python`](https://github.com/modelcontextprotocol/python-sdk) - model context protocol
- [`paramiko`](https://github.com/paramiko/paramiko) - multi-node management (ftl, ssh)
- [`jax`](https://github.com/jax-ml/jax) - gpu acceleration
- [`pyroki`](https://github.com/chungmin99/pyroki) - inverse kinematics
- [`gr00t`](https://github.com/hu-po/Isaac-GR00T) - VLA foundation model
- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm) - robot arms
- [`pyrealsense2`](https://github.com/IntelRealSense/librealsense) - depth cameras
- [`lerobot`](https://github.com/hu-po/lerobot) - dataset, finetuning
- [`viser`](https://github.com/nerfstudio-project/viser) - GUI
- [`pupil-apriltags`](https://github.com/pupil-labs/apriltags) - object tracking
- [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python) - cameras

## Setup

```bash
# Basic install
git clone --depth=1 https://github.com/hu-po/tatbot.git && \
cd ~/tatbot && \
git pull && \
# Optional: Clean old uv environment
deactivate && \
rm -rf .venv && \
rm uv.lock && \
# Setup new uv environment
uv venv --prompt="tatbot" && \
source .venv/bin/activate && \
uv pip install .
# source env variables (i.e. keys, tokens, camera passwords)
source .env
```

Turn on the robot

1. flip power strip in the back to on.
2. press power button on `trossen-ai`, it will glow blue.
3. flip rocker switches to "on" on `arm-r` and `arm-l` control boxes underneath workspace.
4. flip rocker switch on the back of the light to turn it on.

## Devices

tatbot consists of several computers, cameras, and robotos connected via ethernet:

- `ojo` 🦎: NVIDIA Jetson AGX Orin (ARM Cortex-A78AE, 12-core @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai` 🦾: System76 Meerkat PC (Intel i5-1340P, 16-core @ 4.6 GHz) (15GB RAM)
- `ook` 🦧: Acer Nitro V 15 w/ NVIDIA RTX 4050 (Intel i7-13620H, 16-core @ 3.6 GHz) (16GB RAM) (6GB VRAM) (194 TOPS)
- `rpi1` 🍓: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `rpi2` 🍇: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `camera1` 📷: Amcrest PoE cameras (5MP)
- `camera2` 📷: Amcrest PoE cameras (5MP)
- `camera3` 📷: Amcrest PoE cameras (5MP)
- `camera4` 📷: Amcrest PoE cameras (5MP)
- `camera5` 📷: Amcrest PoE cameras (5MP)
- `realsense1` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `realsense2` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-lan`: 8-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l` 🦾: Trossen Arm Controller box (back) connected to WidowXAI arm
- `arm-r` 🦾: Trossen Arm Controller box (front) connected to WidowXAI arm

during development *dev mode*, the following pc is also available:

- `oop` 🦊: Ubuntu PC w/ NVIDIA RTX 3090 (AMD Ryzen 9 5900X, 24-core @ 4.95 GHz) (66GB RAM) (24GB VRAM) (TOPS)

## Networking

tatbot uses shared ssh keys for nodes to talk, send files, and run remote commands: see `_net.py` and `config/nodes.yaml`.

- `switch-lan`:
    - (1) short black ethernet cable to `switch-poe`
    - (2) short black ethernet cable to `trossen-ai`
    - (3) short black ethernet cable to `ojo`
    - (4) long black ethernet cable to `ook`
    - (5) *dev mode* long ethernet cable to `oop`
    - (6) blue ethernet cable to `arm-r` controller box
    - (7) blue ethernet cable to `arm-l` controller box
    - (8)
- `switch-poe`:
    - (1) long black ethernet cable to `camera1`
    - (2) long black ethernet cable to `camera2`
    - (3) long black ethernet cable to `camera3`
    - (4) long black ethernet cable to `camera4`
    - (5) long black ethernet cable to `camera5`
    - (6) -
    - (7) short black ethernet cable to `rpi1`
    - (8) short black ethernet cable to `rpi2`
    - (uplink-1) -
    - (uplink-2) short black ethernet cable to `switch-lan`

to setup the network:

```bash
uv run tatbot.net.net
```

NFS setup: [wiki](wiki/nfs.md)

## Models - VLAs

[wiki](wiki/vlas.md)

Vision Language Action models are used to perform robot behaviors.
Finetuning pretrained VLAs is done by using lerobot dataset format, see [datasets here](https://huggingface.co/tatbot).
Use the [lerobot dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).
Experiment logs on [wandb](https://wandb.ai/hug/tatbot-calib).

## Trossen Robot Arms

tatbot uses two [Trossen WidowXAI arms](https://docs.trossenrobotics.com/trossen_arm/main/specifications.html).

- [Driver API Documentation](https://docs.trossenrobotics.com/trossen_arm/main/api/library_root.html#)
- [official URDF](https://github.com/TrossenRobotics/trossen_arm_description)

each arm has a config file in `config/trossen/arm_{l|r}.yaml`, update configs using `tatbot.bot.trossen`

## URDF

tatbot is defined using URDF at `tatbot/urdf/tatbot.urdf`.

## Realsense Cameras

tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- both realsense cameras are connected to `trossen-ai` via usb3 port
- both realsense cameras are mounted on adjustable goosenecks, so their extrinsic position changes often
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.
- Should be calibrated out of the box, but can be recalibrated: [example1](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_auto_calibration_example.py), [example2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_ucal_example.py).

TODO: these will somewhat randomly fail, need to create robust exception handling

## PoE IP Cameras

tatbot uses 5 poe ip cameras to create a 3d skin reconstruction: see `src/data/cam.py` and `config/cam/default.yaml`.
the cameras are currently set at:
- resolution: 1920x1080
- fps: 5
- bitrate CBR max 2048
- frameinterval: 10
- no substream, all watermarks off

## AprilTags

Objects (i.e. ink palette) in the scene are tracked using [AprilTags](https://chaitanyantr.github.io/apriltag.html).
- 

## Artwork

tatbot uses image generators to generate artwork, and then vectorizes the artwork into strokes executed by the arms.

- [Replicate Playground](https://replicate.com/playground)
- [DrawingBotV3](https://docs.drawingbotv3.com/en/latest/index.html)
- https://docs.drawingbotv3.com/en/latest/pensettings.html
