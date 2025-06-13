# Tech

description of tatbot technical stack

## Index

- [Setup](#setup)
- [Run](#run)
- [Devices](#devices)
- [Train](#train)
  - [SmolVLA](#smolvla)
  - [Gr00t](#gr00t)
- [Eval](#eval)
  - [SmolVLA](#smolvla-1)
  - [Gr00t](#gr00t-1)
- [URDF](#urdf)
- [AprilTags](#apriltags)
- [Profiling](#profiling)
  - [snakeviz](#snakeviz)
  - [scalene](#scalene)
  - [nsys](#nsys)
  - [JAX profiler](#jax-profiler)

## Setup

Various software versions of tatbot with different dependencies and designs are available under `tatbot/src`.
Python dependencies are managed using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and a `pyproject.toml` file.

```bash
# Basic install
git clone --depth=1 https://github.com/hu-po/tatbot.git && \
cd ~/tatbot && \
git pull
# Choose a release
cd src/0.2
# Optional: Clean old uv environment
deactivate && \
rm -rf .venv && \
rm uv.lock
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

Generate a tattoo pattern from a prompt and execute it on the robot:

```bash
uv run pattern_image.py --prompt="growling cat"
DISPLAY=:0 uv run record_pattern.py --pattern_dir ~/tatbot/output/patterns/growling_cat
```

## Devices

tatbot consists of several computers, cameras, and robotos connected via ethernet:

- `ojo`: NVIDIA Jetson AGX Orin (ARM Cortex-A78AE, 12-core @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai`: System76 Meerkat PC (13th Gen Intel i5-1340P, 16-core @ 4.6 GHz) (15GB RAM)
- `rpi1`: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `rpi2`: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `camera1`: Amcrest PoE cameras (5MP)
- `camera2`: Amcrest PoE cameras (5MP)
- `camera3`: Amcrest PoE cameras (5MP)
- `camera4`: Amcrest PoE cameras (5MP)
- `camera5`: Amcrest PoE cameras (5MP)
- `head`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `wrist`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-main`: 5-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l`: Trossen Arm Controller box (back) connected to WidowXAI arm
- `arm-r`: Trossen Arm Controller box (front) connected to WidowXAI arm

during development, the following pc is also available:

- `oop`: Ubuntu PC w/ NVIDIA GeForce RTX 3090 (AMD Ryzen 9 5900X, 24-core @ 4.95 GHz) (66 GB RAM) (24 GB VRAM) (TOPS)

## Dependencies

tatbot makes use of the following dependencies:

- [`jax`](https://github.com/jax-ml/jax) - gpu acceleration
- [`pyroki`](https://github.com/chungmin99/pyroki) - inverse kinematics
- [`viser`](https://github.com/nerfstudio-project/viser) - GUI
- [`librealsense`](https://github.com/IntelRealSense/librealsense) - depth cameras
- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm) - robot arms
- [`pupil-apriltags`](https://github.com/pupil-labs/apriltags) - object tracking

these dependencies use custom forks:

- [`lerobot`](https://github.com/hu-po/lerobot) - dataset, finetuning
- [`gr00t`](https://github.com/hu-po/Isaac-GR00T) - VLA foundation model

## Networking

- `switch-main`:
    - (1) short black ethernet cable to `switch-poe`
    - (2) short black ethernet cable to `trossen-ai`
    - (3) short black ethernet cable to `ojo`
    - (4) blue ethernet cable to `arm-r` controller box
    - (5) blue ethernet cable to `arm-l` controller box
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
    - (uplink-2) short black ethernet cable to `switch-main`

hardcoded ip addresses:

- `192.168.1.53` - `oop`
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

- [trossen_arm](ttps://github.com/TrossenRobotics/trossen_arm)
- [Driver API Documentation](https://docs.trossenrobotics.com/trossen_arm/main/api/library_root.html#)
- [official URDF](https://github.com/TrossenRobotics/trossen_arm_description)

each arm has a config file in `config/trossen_arm_{l|r}.yaml`, push/pull config with:

```bash
uv run ~/tatbot/config/trossen.py --arm r
```

## Realsense Cameras

tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- `wrist` is connected to `trossen-ai` via usb3 port and attached to the end effector of `arm-r`
- `head` is connected to `trossen-ai` via usb3 port and attached to alumnium frame, giving it an overhead view
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.

TODO: these will somewhat randomly fail, need to create robust exception handling

## VLAs

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

```


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
uv pip install PyQt5
MPLBACKEND=QtAgg python scripts/load_dataset.py \
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

## URDF

tatbot is defined using URDF at `tatbot/assets/urdf/tatbot.urdf`.

## AprilTags

Objects (i.e. ink palette) in the scene are tracked using [AprilTags](https://chaitanyantr.github.io/apriltag.html).

## Profiling

TODO: none of these feel great...

### [snakeviz](https://github.com/jiffyclub/snakeviz)

```bash
uv pip install snakeviz
snakeviz tatbot.prof
```

### [scalene](https://github.com/plasma-umass/scalene)

```bash
uv pip install scalene
scalene --cpu --gpu --memory tatbot.py
```

### [nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

```bash
nsys profile -t cuda,nvtx,osrt --stats=true -o tatbot uv run python tatbot.py --debug
nsys-ui tatbot.nsys-rep
```

### JAX profiler

```bash
uv run python tatbot.py 
tensorboard --logdir ./jax_trace
```