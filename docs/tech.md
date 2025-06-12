# Tech

technical description of tatbot

## URDF

tatbot is defined using URDF at `tatbot/assets/urdf/tatbot.urdf`.

various software versions of tatbot with different dependencies and designs are available under `tatbot/src`.

## Setup

python dependencies are managed with environments using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Basic install
git clone https://github.com/hu-po/tatbot.git && \
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
```

keys, tokens, passwords are stored in the `.env` file.

```bash
source .env
```

## Run

1. flip power strip in the back to on.
2. press power button on `trossen-ai`, it will glow blue.
3. flip rocker switches to "on" on `arm-r` and `arm-l` control boxes underneath workspace.
4. flip rocker switch on the back of the light to turn it on.

```bash
uv run python src/0.2/tatbot.py
```

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
- `head`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `wrist`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-main`: 5-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l`: Trossen Arm Controller box connected to WidowXAI arm
- `arm-r`: Trossen Arm Controller box connected to WidowXAI arm
- `oop`: TODO

## Dependencies

tatbot makes use of the following dependencies:

- [`pyroki`](https://github.com/chungmin99/pyroki) - inverse kinematics
- [`viser`](https://github.com/nerfstudio-project/viser) - GUI
- [`librealsense`](https://github.com/IntelRealSense/librealsense) - depth cameras
- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm) - robot arms
- [`jax`](https://github.com/jax-ml/jax) - gpu acceleration
- [`pupil-apriltags`](https://github.com/pupil-labs/apriltags) - object tracking

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

```bash
uv run python config/trossen.py
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
uv run python ~/lerobot/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=tatbot/tatbot-calib-test \
  --output_dir=~/tatbot/output/train/calib-test/smolvla \
  --batch_size=64 \
  --wandb.enable=true \
  --wandb.project=tatbot-calib \
  --steps=20000
```

#### Eval

instructions for `oop`
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
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T/
# setup uv venv
uv venv --python=3.10 && \
source .venv/bin/activate && \
uv pip install .[base]
# download dataset locally
export DATASET_DIR="~/tatbot/output/train/tatbot-calib-test/dataset"
huggingface-cli download \
    --repo-type dataset tatbot/tatbot-calib-test \
    --local-dir $DATASET_DIR
# copy modality config file
cp ~/tatbot/config/gr00t_modality.json $DATASET_DIR/meta/modality.json
# load dataset
uv run scripts/load_dataset.py \
    --dataset-path $DATASET_DIR \
    --plot-state-action \
    --video-backend torchvision_av
# train
export WANDB_RUN_ID="gr00t-test"
export WANDB_PROJECT="tatbot-calib"
uv run python ~/lerobot/lerobot/scripts/gr00t_finetune.py \
   --dataset-path $DATASET_DIR \
   --num-gpus 1 \
   --output-dir ~/tatbot/output/train/tatbot-calib-test/gr00t  \
   --max-steps 10000 \
   --data-config tatbot \
   --
   --video-backend torchvision_av
```

#### Eval

instructions for `ojo`

```bash
# basic install
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T/
# set uv venv
uv venv --python=3.10 && \
source .venv/bin/activate && \
uv pip install .[orin]
# eval
python getting_started/examples/eval_lerobot.py \
    --robot.type=tatbot \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, head: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab pens and place into pen holder."
```

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