Adding the WidowX AI robot to ManiSkill

https://github.com/hu-po/ManiSkill/tree/main

https://github.com/haosulab/ManiSkill/pull/904

https://github.com/hu-po/ManiSkill-WidowXAI

## Pick Cube using PPO (state based)

testing out the pick cube task with ppo, using a NVIDIA A10 node on Lambda Labs (0.75$/hr)

```bash
cd ~/maniskill
git clone https://github.com/hu-po/ManiSkill.git
cd ManiSkill
sudo chown -R $USER:$USER ~/maniskill/ManiSkill
sudo apt-get install libvulkan1 vulkan-tools libglvnd0 libegl1 libgl1 libgles2
vulkaninfo | grep GPU
python3 -m venv maniskill_env
source maniskill_env/bin/activate
pip install --upgrade pip setuptools
pip install wandb networkx torchrl tensordict tensorboard
pip install -e .
wandb login
```

run ppo_fast with 50 steps per episode

```bash
python3 examples/baselines/ppo/ppo_fast.py \
--env_id="PickCube-v1" --robot-uids="widowxai" \
--num_envs=4096 --num-steps=50 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=10_000_000 \
--num_eval_envs=16 \
--compile --track \
--seed=42 --exp-name="ppo-cube-state-seed42-steps50"
```

robot is reaching towards cube, but not enough steps for the robot to grasp the cube

```bash
python3 examples/baselines/ppo/ppo_fast.py \
--env_id="PickCube-v1" --robot-uids="widowxai" \
--num_envs=4096 --num-steps=100 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=10_000_000 \
--num_eval_envs=16 \
--compile --track \
--seed=42 --exp-name="ppo-cube-state-seed42-steps100"
```

typo in num_steps, you also need to change num_eval_steps

```bash
python3 examples/baselines/ppo/ppo_fast.py \
--env_id="PickCube-v1" --robot-uids="widowxai" \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--num_steps=100 --num_eval_steps=100 \
--total_timesteps=10_000_000 \
--num_eval_envs=16 \
--compile --track \
--seed=3 --exp-name="ppo-cube-state-seed3-steps100v3"
```

the num_steps was not overwritting the max_episode_steps, so had to change the ppo code, now try

```bash
python3 examples/baselines/ppo/ppo_fast.py \
--env_id="PickCube-v1" --robot-uids="widowxai" \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--num_steps=100 --num_eval_steps=100 \
--total_timesteps=100_000_000 \
--num_eval_envs=16 \
--compile --track \
--seed=6 --exp-name="ppo-cube-state-seed6-steps100"
```

## Draw Environment

on main ubuntu dev pc

create a new `wxai-draw.urdf` with a pen tip attached to the gripper

push it to the maniskill cache

```bash
cd ~/dev/ManiSkill-WidowXAI && git pull
cp -r . /home/oop/.maniskill/data/robots/widowxai
ls ~/.maniskill/data/robots/widowxai
```

test out changes using robot visualizer (this is in a `draw` branch)

```bash
cd ~/dev/ManiSkill
source .venv/bin/activate
git checkout draw
git pull
uv pip install -e .
python -m mani_skill.examples.demo_robot --robot-uid widowxai_draw
```

going to need to create motion planning trajectories for the draw environment

## Improving PPO for Pick Cube

trying out 50 steps but the robot starts closer and has the fingers open

```bash
python examples/baselines/ppo/ppo_fast.py \
--env_id="PickCube-v1" --robot_uids="widowxai" --seed=0 \
--num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=50_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-state-seed0-walltime_efficient" \
--track
```

## PPO Pick Cube on Cloud Machine

using a NVIDIA A10 node on Lambda Labs (0.75$/hr) (no filesystem)

```bash
git clone https://github.com/hu-po/ManiSkill.git
cd ManiSkill
sudo chown -R $USER:$USER ~/ManiSkill
sudo apt-get install libvulkan1 vulkan-tools libglvnd0 libegl1 libgl1 libgles2
vulkaninfo | grep GPU
python3 -m venv maniskill_env
source maniskill_env/bin/activate
pip install --upgrade pip setuptools
pip install wandb networkx torchrl tensordict tensorboard
pip install -e .
wandb login
sudo mkdir -p /usr/share/vulkan/icd.d
sudo bash -c 'echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"/usr/lib/x86_64-linux-gnu/libvulkan_nvidia.so\", \"api_version\": \"1.3.221\"}}" > /usr/share/vulkan/icd.d/nvidia_icd.json'
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
echo "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" >> ~/.bashrc
source ~/.bashrc
sudo mkdir -p /usr/share/glvnd/egl_vendor.d
sudo bash -c 'echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so\", \"api\": \"egl\"}}" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json'
sudo apt update
sudo apt install --reinstall nvidia-driver-550 nvidia-utils-550
unset DISPLAY
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
```

run the test script

```bash
cd ~/ManiSkill/examples/baselines/ppo
chmod +x test-widowxai.sh
./test-widowxai.sh
```

update the code and then run the test script again

```bash
cd ~/ManiSkill && git stash && git pull && cd ~/ManiSkill/examples/baselines/ppo
./test-widowxai.sh
```

extract specific sub-video from results video

```bash
ffmpeg -i working.mp4 -filter:v "crop=iw/4:ih/4:iw*2/4:ih*3/4" -c:a copy output.mp4
```

## PPO with push cube

```bash
cd ~/ManiSkill && git stash && git pull && cd ~/ManiSkill/examples/baselines/ppo
chmod +x test-widowxai-pushcube.sh
./test-widowxai-pushcube.sh
```

## TODO

waiting for big sweep to finish, and then we will

- filter down the control modes to just the ones that work
- remove testing scripts
- final polish
- post best 4x4 video grid
- ask for merge

draw on wacom as ultimate 2d sim 2 real test. pressure control sweeping


## Filtering Down Control Modes

- looking at wandb filter `pushcube.*p60.*`
    - "pd_joint_delta_pos"
        - high train/reward, zero eval/success and train/success
        - videos look bad
    - "pd_joint_pos"
        - medium low train and eval reward, low eval and train success
        - video look bad
    - "pd_ee_delta_pos"
        - top 2 on train/success_once and eval/reward|return|success_once 
        - videos look good
    - "pd_ee_delta_pose"
        - top 2 on train/success_once and eval/reward|return|success_once
        - similar to pd_ee_delta_pos
        - videos look good
    - "pd_ee_pose"
        - videos look quite bad
        - very low to no train or eval reward and success
    - "pd_joint_target_delta_pos"
        - low train success, low eval success, high train reward, med eval reward
        - some of the videos are good, some are bad
    - "pd_ee_target_delta_pos"
        - high train and eval success, mid to high train and eval reward
        - very bad videos, robot waves like crazy
    - "pd_ee_target_delta_pose"
        - high train and eval success, mid to high train and eval reward
        - very similar to pd_ee_target_delta_pos
        - very bad videos, robot waves like crazy
    - "pd_joint_vel"
        - low to zero train and eval success, high train and eval reward
        - videos look okay but mostly bad, slow robot
    - "pd_joint_pos_vel"
        - skipped
    - "pd_joint_delta_pos_vel"
        - low to zero train and eval success, high train and eval reward
        - videos look bad, robot is slow


commands to verify PR:

```bash
python ppo_fast.py --env_id="PushCube-v1" --robot_uids="widowxai" --seed=3 \
  --num_envs=4096 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000 \
  --num_eval_envs=16 \
  --compile --exp-name="ppo-PushCube-v1-state-widowxai-3-walltime_efficient" \
  --track
```

```bash
python ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=3 \
  --num_envs=4096 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000 \
  --num_eval_envs=16 \
  --compile --exp-name="ppo-PushCube-v1-state-widowxai-3-walltime_efficient" \
  --track
```

```bash
python ppo_rgb.py --env_id="PushCube-v1" --robot_uids="widowxai_cam" --seed=3 \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000 \
  --num_eval_envs=16 \
  --exp-name="ppo-PushCube-v1-rgb-widowxai-${seed}-walltime_efficient" \
  --track
```

## With new updates

using a NVIDIA A100 node on Lambda Labs (1.45$/hr) (no filesystem)

need to create these files:
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#ubuntusudo 

install nvidia-550 drivers and then reboot

```bash
git clone https://github.com/hu-po/ManiSkill.git
cd ManiSkill
sudo chown -R $USER:$USER ~/ManiSkill
sudo apt-get install libvulkan1 vulkan-tools libglvnd0 libegl1 libgl1 libgles2
vulkaninfo | grep GPU
python3 -m venv maniskill_env
source maniskill_env/bin/activate
pip install --upgrade pip setuptools
pip install wandb networkx torchrl tensordict tensorboard
pip install -e .
wandb login
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=41 \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=50_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-41-walltime_efficient" \
--track
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=41 \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=50_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-41-walltime_efficient" \
--track
```

inline sweep:

```bash
seed=32 && \
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=$seed \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=50_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-${seed}-num_minibatches_32" \
--track && \
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=$seed \
--num_envs=4096 --update_epochs=8 --num_minibatches=64 \
--total_timesteps=50_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-${seed}-num_minibatches_64" \
--track
```

seed sweep:

```bash
seed=0 && \
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=$seed \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=60_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-${seed}-mediumcube" \
--track && \
seed=1 && \
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=$seed \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=60_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-${seed}-mediumcube" \
--track && \
seed=2 && \
python ~/ManiSkill/examples/baselines/ppo/ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=$seed \
--num_envs=4096 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=60_000_000 \
--num_eval_envs=16 \
--compile --exp-name="ppo-PickCube-v1-widowxai-state-${seed}-mediumcube" \
--track
```

