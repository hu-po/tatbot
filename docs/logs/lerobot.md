## LeRobot Setup

https://docs.trossenrobotics.com/trossen_arm/v1.3/tutorials/lerobot/setup.html

```bash
conda activate lerobot
cd ~/lerobot && git pull
pip install -e ".[trossen_ai]"
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
```

change the ip address for the leader and follower in 

```bash
vim ~/lerobot/lerobot/common/robot_devices/robots/configs.py
```

```python
@RobotConfig.register_subclass("trossen_ai_solo")
@dataclass
class TrossenAISoloRobotConfig(ManipulatorRobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": TrossenArmDriverConfig(
                # wxai
                ip="192.168.1.2",
                model="V0_LEADER",
            ),
        }
    )
    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": TrossenArmDriverConfig(
                ip="192.168.1.3",
                model="V0_FOLLOWER",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=218622278376,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_wrist": IntelRealSenseCameraConfig(
                serial_number=230422273017,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --robot.max_relative_target=5 \
  --control.type=teleoperate
```

# Recording a dataset

https://docs.trossenrobotics.com/trossen_arm/v1.3/tutorials/lerobot/record_episode.html


```bash
cd ~/lerobot/
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(huggingface-cli whoami | head -n 1)
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Test recording episode using Trossen AI Solo." \
--control.repo_id=${HF_USER}/trossen_ai_solo_test \
--control.tags='["tutorial"]' \
--control.warmup_time_s=5 \
--control.episode_time_s=30 \
--control.reset_time_s=30 \
--control.num_episodes=2 \
--control.push_to_hub=true
```

https://huggingface.co/datasets/hu-po/trossen_ai_solo_test

online dataset visualizer:

https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=hu-po%2Ftrossen_ai_solo_test&episode=0

replay dataset:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --robot.max_relative_target=null \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/trossen_ai_solo_test \
  --control.episode=0 \
```

# Pick up Tattoo Wand

https://docs.trossenrobotics.com/trossen_arm/v1.3/tutorials/lerobot/record_episode.html#recording-configuration

meerkat has 16 cores, so 4 threads per camera seems reasonable.
using sounds requires DISPLAY, so ignore given we are using ssh
robot cannot reach below its base plate, so using a taller table top

```bash
source ~/dev/tatbot-dev/config/.env
cd ~/lerobot/ && git pull && conda activate lerobot
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Pick up tattoo wand." \
--control.repo_id=${HF_USER}/pickup_wand \
--control.tags='["pickup"]' \
--control.warmup_time_s=5 \
--control.episode_time_s=12 \
--control.reset_time_s=12 \
--control.num_episodes=5 \
--control.push_to_hub=true
```

installing lerobot on oop so we can train

```bash
cd ~/dev/
git clone -b trossen-ai https://github.com/Interbotix/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10 && conda activate lerobot
pip install -e ".[trossen_ai]"
conda install -y -c conda-forge ffmpeg
```

```bash
source ~/dev/tatbot-dev/config/.env
wandb login
huggingface-cli login --token ${HUGGINGFACE_TOKEN}
python lerobot/scripts/train.py \
  --dataset.repo_id=hu-po/pickup_wand \
  --policy.type=act \
  --output_dir=outputs/train/act_pickup_wand \
  --job_name=act_pickup_wand \
  --device=cuda \
  --wandb.enable=true
```

running into errors trying to train on bare metal, so going to try docker

```bash
cd ~/dev/lerobot
docker build -t lerobot-gpu-img -f docker/lerobot-gpu/Dockerfile .
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/lerobot/outputs \
  -e WANDB_API_KEY \
  -e HUGGINGFACE_TOKEN \
  lerobot-gpu-img \
  python lerobot/scripts/train.py \
    --dataset.repo_id=hu-po/pickup_wand \
    --policy.type=act \
    --output_dir=outputs/train/act_pickup_wand \
    --job_name=act_pickup_wand \
    --wandb.enable=true
```

the interbotix fork of lerobot seems outdated, so trying the official repo

```bash
cd ~/dev/
git clone https://github.com/huggingface/lerobot
cd lerobot
docker build -t lerobot-gpu-img -f docker/lerobot-gpu/Dockerfile .
docker run --gpus all --rm -it \
  --shm-size=8g \
  -v $(pwd)/outputs:/lerobot/outputs \
  -e WANDB_API_KEY \
  -e HUGGINGFACE_TOKEN \
  lerobot-gpu-img \
  python lerobot/scripts/train.py \
    --dataset.repo_id=hu-po/pickup_wand \
    --policy.type=act \
    --output_dir=outputs/train/act_pickup_wand \
    --job_name=act_pickup_wand \
    --wandb.enable=true
```

evaluating the model, first need to copy the checkpoint to meerkat

```bash
scp -r /home/oop/dev/lerobot/outputs/train/act_pickup_wand/checkpoints/020000/pretrained_model \
trossen-ai@192.168.1.97:/home/trossen-ai/lerobot/outputs/pickup_wand_2000
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up tattoo wand." \
  --control.repo_id=hu-po/eval_pickup_wand \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=12 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/pickup_wand_2000/pretrained_model \
  --control.num_image_writer_processes=1
```

policy is trash, doesn't even really center itself over the wand
the examples were only 5 episodes, the examples were very low quality themselves, and the model likely overfit

maybe try using the eval script? `lerobot/scripts/eval.py`
nope, that is definitely intended for sim

maybe try a different policy? lets start from pizero, on oop:

```bash
docker run --gpus all --rm -it \
  --shm-size=8g \
  -v $(pwd)/outputs:/lerobot/outputs \
  -e WANDB_API_KEY \
  -e HUGGINGFACE_TOKEN \
  lerobot-gpu-img \
  python lerobot/scripts/train.py \
    --dataset.repo_id=hu-po/pickup_wand \
    --policy.type=pi0 \
    --output_dir=outputs/train/pi0_pickup_wand \
    --job_name=pi0_pickup_wand \
    --wandb.enable=true
```

  need to modify docker container to install pi0 dependencies `pip install --no-binary=av -e ".[pi0]"`

```bash
vim docker/lerobot-gpu/Dockerfile
docker build -t lerobot-gpu-img -f docker/lerobot-gpu/Dockerfile .
```

had to grant access to https://huggingface.co/google/paligemma-3b-pt-224
it is taking a while to authenticate, so going to try later

# Pick up Cube

trying a simpler task, going to record a higher quality dataset for cube pickup

first need to create my own fork of lerobot https://github.com/hu-po/lerobot/tree/main

everything in the trossen lerobot repo is on a branch called `trossen-ai`
https://github.com/huggingface/lerobot/compare/main...Interbotix:lerobot:trossen-ai

on meerkat, move the current lerobot repo to backup directory

```bash
cd ~
mv lerobot lerobot-trossen
conda deactivate
conda create --name lerobot-trossen --clone lerobot
conda env remove --name lerobot
git clone https://github.com/hu-po/lerobot
cd lerobot
git pull
conda create -y -n lerobot python=3.10 && conda activate lerobot
pip install -e ".[trossen_ai]"
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```

retry on meerkat to collect dataset of pick cube

```bash
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-sleep.py && \
rm -rf /home/trossen-ai/.cache/huggingface/lerobot/hu-po/pickup_cube && \
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Pick up cube." \
--control.repo_id=hu-po/pickup_cube \
--control.tags='["pickup"]' \
--control.warmup_time_s=10 \
--control.episode_time_s=12 \
--control.reset_time_s=6 \
--control.num_episodes=8 \
--control.push_to_hub=true
```

need to reduce gripper force, and the "stickyness" of joint 2

gripper force at 0.1 feels good, but the stickyness of joint 2 and 3 is still an issue

now lets try and finetune act

```bash
docker build -t lerobot-gpu-img -f docker/lerobot-gpu/Dockerfile .
docker run --gpus all --rm -it \
  --shm-size=8g \
  -v $(pwd)/outputs:/lerobot/outputs \
  -e WANDB_API_KEY \
  -e HUGGINGFACE_TOKEN \
  lerobot-gpu-img \
  python lerobot/scripts/train.py \
    --dataset.repo_id=hu-po/pickup_cube \
    --policy.type=act \
    --output_dir=outputs/train/act_pickup_cube \
    --job_name=act_pickup_cube \
    --wandb.enable=true
```

evaluate

```bash
scp -r /home/oop/dev/lerobot/outputs/train/act_pickup_cube/checkpoints/020000/pretrained_model \
trossen-ai@192.168.1.97:/home/trossen-ai/lerobot/outputs/pickup_cube_2000
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up cube." \
  --control.repo_id=hu-po/eval_pickup_cube \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=12 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/pickup_cube_2000/pretrained_model \
  --control.num_image_writer_processes=1
```

policy is trash

# Pick Cube Day 2

trying new home position

```bash
ssh trossen-ai@192.168.1.97
cd ~/lerobot
git pull
source ~/dev/tatbot-dev/config/.env
conda activate lerobot
rm -rf /home/trossen-ai/.cache/huggingface/lerobot/hu-po/pickup_cube_2
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-sleep.py
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Pick up cube." \
--control.repo_id=hu-po/pickup_cube_2 \
--control.tags='["pickup"]' \
--control.warmup_time_s=3 \
--control.episode_time_s=6 \
--control.reset_time_s=6 \
--control.num_episodes=8 \
--control.push_to_hub=true
```

robot is easier to control, but still very confusing whether it is warmup or reset or episode, need to try the UI

the UI uses yaml configs that seem to be stored at
`./miniconda3/envs/trossen_ai_data_collection_ui_env/lib/python3.10/site-packages/trossen_ai_data_collection_ui/configs/tasks.yaml`

copy the config files to the UI directory, clear out old recordings

```bash
bash ~/dev/tatbot-dev/scripts/trossen-ai/copy-ui-config.sh
bash ~/dev/tatbot-dev/scripts/trossen-ai/clear-hf-cache.sh
```

run the UI with consolidated scripts

```bash
bash ~/dev/tatbot-dev/scripts/trossen-ai/start-trossen-ui.sh
```

train act policy on new pick cube and place circle dataset
this is on the A100 in the cloud

```bash
cd ~/lerobot && git pull
sudo docker build -t lerobot-gpu-img -f docker/lerobot-gpu/Dockerfile .
sudo docker run --gpus all --rm -it --shm-size=16g -v $(pwd)/outputs:/lerobot/outputs lerobot-gpu-img bash
# once inside the container run:
wandb login
# copy paste WANDB_API_KEY=... from .env file
huggingface-cli login
# copy paste HF_TOKEN=... from .env file
python lerobot/scripts/train.py \
  --dataset.repo_id=hu-po/pick_cube_place_circle \
  --policy.type=act \
  --save_freq=10_000 \
  --output_dir=outputs/train/act_pick_cube_place_circle \
  --job_name=act_pick_cube_place_circle \
  --wandb.enable=true
```

copy over the checkpoint by zipping it, then downloading it to oop, then scp to meerkat

```bash
cd ~
tar -czvf act_pick_cube_place_circle_10000.tar.gz -C /home/ubuntu/lerobot/outputs/train/act_pick_cube_place_circle/checkpoints/010000/ pretrained_model
tar -czvf act_pick_cube_place_circle_20000.tar.gz -C /home/ubuntu/lerobot/outputs/train/act_pick_cube_place_circle/checkpoints/020000/ pretrained_model
```

on oop

```bash
scp ~/Downloads/act_pick_cube_place_circle_10000.tar.gz trossen-ai@192.168.1.97:/home/trossen-ai/Downloads/
scp ~/Downloads/act_pick_cube_place_circle_20000.tar.gz trossen-ai@192.168.1.97:/home/trossen-ai/Downloads/
```
on meerkat

```bash
mkdir ~/lerobot/outputs/act_pick_cube_place_circle_10000 && tar -xvf ~/Downloads/act_pick_cube_place_circle_10000.tar.gz -C ~/lerobot/outputs/act_pick_cube_place_circle_10000
mkdir ~/lerobot/outputs/act_pick_cube_place_circle_20000 && tar -xvf ~/Downloads/act_pick_cube_place_circle_20000.tar.gz -C ~/lerobot/outputs/act_pick_cube_place_circle_20000
```

evaluate

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up cube and place circle." \
  --control.repo_id=hu-po/eval_act_pick_cube_place_circle_10000 \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=5 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/act_pick_cube_place_circle_10000/pretrained_model \
  --control.num_image_writer_processes=1
```

policy kinda looks like it wants to work, but in practice trash

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up cube and place circle." \
  --control.repo_id=hu-po/eval_act_pick_cube_place_circle_20000 \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=5 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/act_pick_cube_place_circle_20000/pretrained_model \
  --control.num_image_writer_processes=1
```

policy is trash