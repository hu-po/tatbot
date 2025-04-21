# PiZero

## Pick Cube

use lerobot fork to finetune pizero on pick cube dataset, compare to ACT

```bash
docker build -t lerobot-gpu-pi0-img -f docker/lerobot-gpu/Dockerfile.pi0 .
docker run --gpus all --rm -it \
  --shm-size=8g \
  -v $(pwd)/outputs:/lerobot/outputs \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  lerobot-gpu-pi0-img \
  python lerobot/scripts/train.py \
    --dataset.repo_id=hu-po/pickup_cube \
    --policy.type=pi0 \
    --output_dir=outputs/train/pi0_pickup_cube \
    --job_name=pi0_pickup_cube \
    --wandb.enable=true
```

oop shut off when running this command, going to rent a cloud gpu to do it.
renting a A100 for $1.50/hour, no filesystem, using cloud ide in dark mode

```bash
git clone https://github.com/hu-po/lerobot.git
cd lerobot
sudo docker build -t lerobot-gpu-pi0-img -f docker/lerobot-gpu/Dockerfile.pi0 .
sudo docker run --gpus all --rm -it --shm-size=16g -v $(pwd)/outputs:/lerobot/outputs lerobot-gpu-pi0-img bash
# once inside the container run:
wandb login
# copy paste WANDB_API_KEY=... from .env file
huggingface-cli login
# copy paste HF_TOKEN=... from .env file
python lerobot/scripts/train.py \
  --dataset.repo_id=hu-po/pickup_cube \
  --policy.type=pi0 \
  --save_freq=2_000 \
  --output_dir=outputs/train/pi0_pickup_cube \
  --job_name=pi0_pickup_cube \
  --wandb.enable=true
```

this one trains slower, overfits faster, so changing save freq to 2000 from 20000
copy over the checkpoint by zipping it, then downloading it to oop, then scp to meerkat

```bash
cd ~
tar -czvf pi0_pickup_cube_002000.tar.gz -C /home/ubuntu/lerobot/outputs/train/pi0_pickup_cube/checkpoints/002000/ pretrained_model
tar -czvf pi0_pickup_cube_004000.tar.gz -C /home/ubuntu/lerobot/outputs/train/pi0_pickup_cube/checkpoints/004000/ pretrained_model
```

on oop

```bash
scp ~/Downloads/pi0_pickup_cube_002000.tar.gz trossen-ai@192.168.1.97:/home/trossen-ai/Downloads/
scp ~/Downloads/pi0_pickup_cube_004000.tar.gz trossen-ai@192.168.1.97:/home/trossen-ai/Downloads/
```
on meerkat

```bash
tar -xvf ~/Downloads/pi0_pickup_cube_002000.tar.gz -C ~/lerobot/outputs/
tar -xvf ~/Downloads/pi0_pickup_cube_004000.tar.gz -C ~/lerobot/outputs/
```

evaluate policies

had to install some extra dependencies

```bash
pip install transformers>=4.48.0
pip install pytest>=8.1.0 pytest-cov>=5.0.0 pyserial>=3.5
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up cube." \
  --control.repo_id=hu-po/eval_pi0_pickup_cube_002000 \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=12 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/pi0_pickup_cube_002000 \
  --control.num_image_writer_processes=1
```

get a `WARNING:root:No accelerated backend detected. Using default cpu, this will be slow.` error

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Recording evaluation episode of pick up cube." \
  --control.repo_id=hu-po/eval_pi0_pickup_cube_004000 \
  --control.tags='["pickup"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=12 \
  --control.reset_time_s=12 \
  --control.num_episodes=3 \
  --control.push_to_hub=true \
  --control.policy.path=/home/trossen-ai/lerobot/outputs/pi0_pickup_cube_004000 \
  --control.num_image_writer_processes=1
```

## Pick Cube and Place Circle

train pi0 policy on new pick cube and place circle dataset

```bash
cd ~/lerobot && git pull
sudo docker build -t lerobot-gpu-pi0-img -f docker/lerobot-gpu/Dockerfile.pi0 .
sudo docker run --gpus all --rm -it --shm-size=16g -v $(pwd)/outputs:/lerobot/outputs lerobot-gpu-pi0-img bash
# once inside the container run:
wandb login
# copy paste WANDB_API_KEY=... from .env file
huggingface-cli login
# copy paste HF_TOKEN=... from .env file
python lerobot/scripts/train.py \
  --dataset.repo_id=hu-po/pick_cube_place_circle \
  --policy.type=pi0 \
  --save_freq=2_000 \
  --output_dir=outputs/train/pi0_pick_cube_place_circle \
  --job_name=pi0_pick_cube_place_circle \
  --wandb.enable=true
```

copy over the checkpoint by zipping it, then downloading it to oop, then scp to meerkat

```bash
cd ~
tar -czvf pi0_pick_cube_place_circle_002000.tar.gz -C /home/ubuntu/lerobot/outputs/train/pi0_pick_cube_place_circle/checkpoints/002000/ pretrained_model
```

on oop

```bash
scp ~/Downloads/pi0_pick_cube_place_circle_002000.tar.gz trossen-ai@192.168.1.97:/home/trossen-ai/Downloads/
```
on meerkat

```bash
mkdir ~/lerobot/outputs/pi0_pick_cube_place_circle_002000 && tar -xvf ~/Downloads/pi0_pick_cube_place_circle_002000.tar.gz -C ~/lerobot/outputs/pi0_pick_cube_place_circle_002000
```
