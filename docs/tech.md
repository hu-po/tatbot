# Tech

description of tatbot (tattoo robot) technical stack

## Index

- [Software](#software)
  - [Run](#run)
  - [Setup](#setup)
  - [Dependencies](#dependencies)
  - [Devices](#devices)
  - [Networking](#networking)
- [Hardware](#hardware)
  - [Trossen Robot Arms](#trossen-robot-arms)
  - [URDF](#urdf)
  - [Realsense Cameras](#realsense-cameras)
  - [PoE IP Cameras](#poe-ip-cameras)
  - [AprilTags](#apriltags)
- [Models - VLAs](#models---vlas)
  - [SmolVLA](#smolvla)
    - [Train](#train-smolvla)
    - [Eval](#eval-smolvla)
  - [Gr00t](#gr00t)
    - [Train](#train-gr00t) 
    - [Eval](#eval-gr00t)
- [Models - VGGT](#models---vggt)

## Run

tatbot is designed as a multi-node system, with the following roles:R

`ook` 🦧 creates plans heuristically or from generated images, using the local gpu for batch ik of paths

```bash
uv pip install .[gen] && \
# generate a benchmark plan (~12 paths of different types)
uv run gen_bench.py --debug
# use a generated image to create a plan
uv run gen_image.py --prompt "cat"
```

`trossen-ai` 🦾 sends commands to robot arms, receives realsense camera images, and records lerobot datasets:

```bash
uv pip install .[bot] && \
uv run run_bot.py
```

`ojo` 🦎 runs the policy servers for the VLA model and for the 3d reconstruction model

```bash
uv pip install .[vla] && \
# TODO
```

`rpi1` 🥧 runs apriltag tracking:

```bash
uv pip install .[tag] && \
uv run run_tag.py
```

`rpi2` 🫐 runs visualization:

```bash
uv pip install .[viz] && \
uv run run_viz.py
```

## Dependencies

tatbot makes use of the following dependencies:

- [`mcp-python`](https://github.com/modelcontextprotocol/python-sdk) - model context protocol
- [`paramiko`](https://github.com/paramiko/paramiko) - multi-node management (ftl, ssh)

`gen`
- [`jax`](https://github.com/jax-ml/jax) - gpu acceleration
- [`pyroki`](https://github.com/chungmin99/pyroki) - inverse kinematics

`vla`
- [`gr00t`](https://github.com/hu-po/Isaac-GR00T) - VLA foundation model

`bot`
- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm) - robot arms
- [`pyrealsense2`](https://github.com/IntelRealSense/librealsense) - depth cameras
- [`lerobot`](https://github.com/hu-po/lerobot) - dataset, finetuning

`viz`
- [`viser`](https://github.com/nerfstudio-project/viser) - GUI

`tag`
- [`pupil-apriltags`](https://github.com/pupil-labs/apriltags) - object tracking
- [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python) - cameras

these dependencies use custom forks:


## Setup

Various software versions of tatbot with different dependencies and designs are available under `tatbot/src`.
Python dependencies are managed using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and a `pyproject.toml` file.

```bash
# Basic install
git clone --depth=1 https://github.com/hu-po/tatbot.git && \
cd ~/tatbot && \
git pull
# Choose a release
cd src/0.3
# Optional: Clean old uv environment
deactivate && \
rm -rf .venv && \
rm uv.lock && \
# Setup new uv environment
uv venv && \
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

- `ojo`: NVIDIA Jetson AGX Orin (ARM Cortex-A78AE, 12-core @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai`: System76 Meerkat PC (Intel i5-1340P, 16-core @ 4.6 GHz) (15GB RAM)
- `ook`: Acer Nitro V 15 w/ NVIDIA RTX 4050 (Intel i7-13620H, 16-core @ 3.6 GHz) (16GB RAM) (6GB VRAM) (194 TOPS)
- `rpi1`: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `rpi2`: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `camera1`: Amcrest PoE cameras (5MP)
- `camera2`: Amcrest PoE cameras (5MP)
- `camera3`: Amcrest PoE cameras (5MP)
- `camera4`: Amcrest PoE cameras (5MP)
- `camera5`: Amcrest PoE cameras (5MP)
- `head`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `wrist`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-lan`: 8-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l`: Trossen Arm Controller box (back) connected to WidowXAI arm
- `arm-r`: Trossen Arm Controller box (front) connected to WidowXAI arm

during development *dev mode*, the following pc is also available:

- `oop`: Ubuntu PC w/ NVIDIA RTX 3090 (AMD Ryzen 9 5900X, 24-core @ 4.95 GHz) (66GB RAM) (24GB VRAM) (TOPS)

## Networking

tatbot uses a shared ssh key for nodes to talk, send files, and run remote commands: see `_net.py` and `config/nodes.yaml`.

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

hardcoded ip addresses:

- `192.168.1.51` - `oop`
- `192.168.1.90` - `ook`
- `192.168.1.91` - `camera1`
- `192.168.1.92` - `camera2`
- `192.168.1.93` - `camera3`
- `192.168.1.94` - `camera4`
- `192.168.1.95` - `camera5`
- `192.168.1.96` - `ojo`
- `192.168.1.97` - `trossen-ai`
- `192.168.1.98` - `rpi1`
- `192.168.1.99` - `rpi2`

## Trossen Robot Arms

tatbot uses two [Trossen WidowXAI arms](https://docs.trossenrobotics.com/trossen_arm/main/specifications.html).

- [Driver API Documentation](https://docs.trossenrobotics.com/trossen_arm/main/api/library_root.html#)
- [official URDF](https://github.com/TrossenRobotics/trossen_arm_description)

each arm has a config file in `config/trossen_arm_{l|r}.yaml`, update configs using `_bot.py`

## URDF

tatbot is defined using URDF at `tatbot/assets/urdf/tatbot.urdf`.

## Realsense Cameras

tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- `wrist` is connected to `trossen-ai` via usb3 port and attached to the end effector of `arm-r`
- `head` is connected to `trossen-ai` via usb3 port and attached to alumnium frame, giving it an overhead view
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.
- Should be calibrated out of the box, but can be recalibrated: [example1](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_auto_calibration_example.py), [example2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_ucal_example.py).

TODO: these will somewhat randomly fail, need to create robust exception handling

## PoE IP Cameras

tatbot uses 5 poe ip cameras to create a 3d skin reconstruction: see `_cam.py` and `config/cameras.yaml`.

## AprilTags

Objects (i.e. ink palette) in the scene are tracked using [AprilTags](https://chaitanyantr.github.io/apriltag.html).

## Models - VLAs

Vision Language Action models are used to perform robot behaviors.
Finetuning pretrained VLAs is done by using lerobot dataset format, see [datasets here](https://huggingface.co/tatbot).
Use the [lerobot dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).
Experiment logs on [wandb](https://wandb.ai/hug/tatbot-calib).

### SmolVLA

[blog](https://huggingface.co/blog/smolvla)
[model](https://huggingface.co/lerobot/smolvla_base)

#### Train

instructions for `oop`

```bash
# basic install
git clone --depth=1 https://github.com/hu-po/lerobot.git && \
cd lerobot/
# setup uv venv
uv venv && \
source .venv/bin/activate && \
uv pip install -e ".[smolvla]"
# run training
wandb login
uv run python ~/lerobot/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=tatbot/tatbot-calib-test \
  --output_dir=~/tatbot/output/train/calib-test/smolvla \
  --batch_size=64 \
  --wandb.enable=true \
  --wandb.project=tatbot-calib \
  --steps=1000
```

#### Eval

instructions for `trossen-ai` performing model inference and running robot

```bash
# TODO
```

### Gr00t

[blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
[model](https://huggingface.co/nvidia/GR00T-N1.5-3B)
[repo](https://github.com/NVIDIA/Isaac-GR00T)

#### Train

instructions for `oop`

```bash
# basic install
git clone --depth 1 https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# setup uv venv
uv venv --python=3.11 && \
source .venv/bin/activate && \
uv pip install .[base]
# download dataset locally
export DATASET_DIR="/home/oop/tatbot/output/train/tatbot-calib-test/dataset" && \
huggingface-cli download \
  --repo-type dataset tatbot/tatbot-calib-test \
  --local-dir $DATASET_DIR
# copy modality config file
cp /home/oop/tatbot/config/gr00t_modality.json $DATASET_DIR/meta/modality.json
# load dataset
python scripts/load_dataset.py \
  --dataset-path $DATASET_DIR \
  --embodiment-tag new_embodiment \
  --plot-state-action \
  --steps 64 \
  --video-backend torchvision_av
# train with docker
docker build -f Dockerfile -t gr00t-train .
docker run -it --gpus all --shm-size=8g --rm \
  -e WANDB_RUN_ID="gr00t-test" \
  -e WANDB_PROJECT="tatbot-calib" \
  -v $DATASET_DIR:/dataset \
  -v $HF_HOME:/root/.cache/huggingface \
  -v /home/oop/tatbot/output/train/tatbot-calib-test/gr00t:/output \
  -v /home/oop/Isaac-GR00T:/workspace \
  gr00t-train \
  bash -c "pip install -e . --no-deps && \
  python scripts/gr00t_finetune.py \
    --dataset-path /dataset \
    --embodiment-tag new_embodiment \
    --num-gpus 1 \
    --output-dir /output \
    --max-steps 10000 \
    --data-config tatbot \
    --batch_size 1 \
    --video-backend torchvision_av"
```

#### Eval

instructions for `ojo`, acting as the policy server

```bash
# basic install
git clone https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# copy policy checkpoint into ojo
scp oop@192.168.1.53:/home/oop/tatbot/output/train/tatbot-calib-test/gr00t /tmp/gr00t
# policy with dockerfile
docker build -f orin.Dockerfile -t gr00t-eval .
docker run -it --gpus all --rm \
  -v /tmp/gr00t:/checkpoint \
  -v /home/ojo/Isaac-GR00T:/workspace \
  gr00t-eval \
  bash -c "pip3 install .[orin] && \
  python scripts/inference_service.py --server \
    --model_path /checkpoint \
    --embodiment-tag new_embodiment \
    --data-config tatbot \
    --denoising-steps 4"
```

instructions for `trossen-ai` acting as the robot client

```bash
git clone https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# setup uv venv
uv venv --python=3.11 && \
source .venv/bin/activate && \
uv pip install .[base]
# run robot client
python getting_started/examples/eval_lerobot.py \
    --robot.type=tatbot \
    --policy_host=192.168.1.96 \
    --lang_instruction="move slightly upwards in z"
```

## Models - VGGT

