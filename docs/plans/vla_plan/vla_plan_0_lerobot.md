---
summary: VLA plan 0 — training with LeRobot for tatbot
tags: [plans, vla]
updated: 2025-08-21
audience: [dev]
---

# 🎓 VLA Policy Training Guide for Tatbot Robot

This guide provides comprehensive documentation for finetuning Vision-Language-Action (VLA) policies, evaluating training with WandB, and performing inference on the Tatbot robot with RealSense cameras using the LeRobot framework.

## Table of Contents
1. [Overview](#overview)
2. [Available VLA Policies](#available-vla-policies)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Training VLA Policies](#training-vla-policies)
6. [Evaluation with WandB](#evaluation-with-wandb)
7. [Robot Inference on Tatbot](#robot-inference-on-tatbot)
8. [Code Examples](#code-examples)
9. [Troubleshooting](#troubleshooting)

## Overview

The LeRobot framework supports multiple Vision-Language-Action policies that can be trained on robotic manipulation tasks and deployed on real hardware. This guide focuses on two main VLA policies:
- **SmolVLA**: A lightweight vision-language-action model optimized for efficient robotics
- **π0 (Pi0)**: A vision-language-action flow model for general robot control

## Available VLA Policies

### SmolVLA
- **Paper**: https://arxiv.org/abs/2506.01844
- **Location**: `src/lerobot/policies/smolvla/`
- **Main Files**:
  - `modeling_smolvla.py`: Model implementation
  - `configuration_smolvla.py`: Configuration class
  - `smolvlm_with_expert.py`: VLM with expert module

### π0 (Pi0)
- **Paper**: https://www.physicalintelligence.company/download/pi0.pdf
- **Location**: `src/lerobot/policies/pi0/`
- **Main Files**:
  - `modeling_pi0.py`: Model implementation
  - `configuration_pi0.py`: Configuration class
  - `paligemma_with_expert.py`: PaliGemma with expert module

## Environment Setup

### Install Dependencies

```bash
# Basic installation
pip install -e .

# SmolVLA specific dependencies (includes transformers, accelerate, safetensors)
pip install -e ".[smolvla]"

# Pi0 specific dependencies (includes transformers)
pip install -e ".[pi0]"

# For Tatbot robot support
pip install -e ".[tatbot]"

# For RealSense camera support
pip install -e ".[intelrealsense]"

# For WandB logging (included in base requirements)
# WandB is already included in the base installation

# Complete installation for Tatbot with VLA policies
pip install -e ".[tatbot,intelrealsense,smolvla,pi0]"
```

## Dataset Preparation

### Using Existing Datasets

LeRobot provides access to various datasets through HuggingFace Hub:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset
dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")
```

### Creating Custom Datasets for Tatbot

For the Tatbot robot, you'll need to record data with proper camera configuration:

```python
# Dataset recording configuration for Tatbot
delta_timestamps = {
    "observation.image": [-0.1, 0.0],  # Previous and current frame
    "observation.state": [-0.1, 0.0],  # Previous and current state
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
```

## Training VLA Policies

### Training SmolVLA

#### From Scratch
```bash
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=your_dataset_repo \
    --batch_size=64 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project=tatbot_smolvla \
    --output_dir=outputs/train/smolvla_tatbot
```

#### From Pretrained Model
```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=your_dataset_repo \
    --batch_size=64 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_smolvla_finetune
```

### Training π0 (Pi0)

#### From Scratch
```bash
lerobot-train \
    --policy.type=pi0 \
    --dataset.repo_id=your_dataset_repo \
    --batch_size=32 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project=tatbot_pi0 \
    --output_dir=outputs/train/pi0_tatbot
```

#### From Pretrained Model
```bash
lerobot-train \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=your_dataset_repo \
    --batch_size=32 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_pi0_finetune
```

### Key Training Configuration Parameters

#### SmolVLA Configuration (`configuration_smolvla.py`)
```python
# Key parameters
n_obs_steps: int = 1
chunk_size: int = 50
n_action_steps: int = 50
max_state_dim: int = 32
max_action_dim: int = 32
resize_imgs_with_padding: tuple = (512, 512)

# Training settings
optimizer_lr: float = 1e-4
optimizer_grad_clip_norm: float = 10
scheduler_warmup_steps: int = 1_000
scheduler_decay_steps: int = 30_000

# Model settings
vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
freeze_vision_encoder: bool = True
train_expert_only: bool = True
```

#### Pi0 Configuration (`configuration_pi0.py`)
```python
# Key parameters (similar structure to SmolVLA)
n_obs_steps: int = 1
chunk_size: int = 50
n_action_steps: int = 50
```

## Evaluation with WandB

### WandB Setup

The training script automatically integrates with WandB through `src/lerobot/utils/wandb_utils.py`:

```python
# Key WandB configuration in training
--wandb.enable=true \
--wandb.project=your_project_name \
--wandb.entity=your_wandb_entity \
--wandb.notes="Training notes" \
--wandb.mode=online  # or offline for local logging
```

### Tracked Metrics

The following metrics are automatically logged to WandB:
- **Training Metrics**: loss, gradient norm, learning rate, update speed
- **Evaluation Metrics**: success rate, reward sum, evaluation speed
- **System Metrics**: GPU utilization, memory usage

### Evaluation Script Usage

```bash
lerobot-eval \
    --policy.path=outputs/train/smolvla_tatbot/checkpoints/last/pretrained_model \
    --env.type=tatbot \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --device=cuda
```

## Robot Inference on Tatbot

### Tatbot Configuration

The Tatbot robot configuration is defined in `src/lerobot/robots/tatbot/`:

#### Key Components (`tatbot.py`)
- Dual arm setup with left and right arms
- RealSense camera integration
- IP camera support
- Thread pool executor for parallel operations

#### Configuration Structure (`config_tatbot.py`)
```python
@dataclass
class TatbotConfig(RobotConfig):
    rs_cameras: dict[str, CameraConfig]  # RealSense cameras
    ip_cameras: dict[str, CameraConfig]  # IP cameras
    ip_address_l: str  # Left arm IP
    ip_address_r: str  # Right arm IP
    arm_l_config_filepath: str  # Left arm YAML config
    arm_r_config_filepath: str  # Right arm YAML config
    home_pos_l: list[float]  # Left arm home position
    home_pos_r: list[float]  # Right arm home position
    goal_time: float  # Default travel time
    connection_timeout: float  # Connection timeout
```

### RealSense Camera Setup

RealSense cameras are configured in `src/lerobot/cameras/realsense/`:

```python
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras import ColorMode, Cv2Rotation

# Configure RealSense camera
config = RealSenseCameraConfig(
    serial_number_or_name="your_camera_serial",
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.BGR,
    rotation=Cv2Rotation.NO_ROTATION,
    use_depth=True  # Enable depth capture
)

camera = RealSenseCamera(config)
camera.connect()
```

### Inference Script

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# OR for Pi0:
# from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.robots.tatbot.tatbot import Tatbot
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

# Load trained policy (SmolVLA example)
policy = SmolVLAPolicy.from_pretrained("outputs/train/smolvla_tatbot/checkpoints/last/pretrained_model")
# OR for Pi0:
# policy = PI0Policy.from_pretrained("outputs/train/pi0_tatbot/checkpoints/last/pretrained_model")
policy.eval()
policy.to("cuda")

# Initialize Tatbot
config = TatbotConfig(
    rs_cameras={
        "cam_left": RealSenseCameraConfig(serial_number_or_name="left_serial"),
        "cam_right": RealSenseCameraConfig(serial_number_or_name="right_serial")
    },
    ip_address_l="192.168.1.10",
    ip_address_r="192.168.1.11",
    # ... other config parameters
)

robot = Tatbot(config)
robot.connect()

# Main inference loop
try:
    while True:
        # Get observations from robot
        observation = robot.get_observation()
        
        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(observation)
        
        # Execute action on robot
        robot.send_action(action)
        
except KeyboardInterrupt:
    robot.disconnect()
```

## Code Examples

### Complete Training Pipeline

```python
# train_vla_tatbot.py
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.train import train

# Configure training
config = TrainPipelineConfig(
    policy_type="smolvla",
    dataset_repo_id="your_tatbot_dataset",
    output_dir="outputs/train/tatbot_vla",
    steps=100000,
    batch_size=32,
    eval_freq=5000,
    save_freq=10000,
    log_freq=100,
    wandb_enable=True,
    wandb_project="tatbot_vla_training"
)

# Run training
train(config)
```

### Custom Dataset Recording

```python
# record_tatbot_dataset.py
from lerobot.robots.tatbot.tatbot import Tatbot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

robot = Tatbot(config)
robot.connect()

# Record episodes
dataset = LeRobotDataset.create(
    repo_id="your_username/tatbot_task",
    fps=30,
    robot=robot
)

# Record data...
dataset.push_to_hub()
```

## Troubleshooting

### Common Issues and Solutions

1. **RealSense Camera Not Found**
   ```bash
   # Find available cameras
   lerobot-find-cameras realsense
   ```

2. **CUDA Out of Memory**
   - Reduce batch_size
   - Enable gradient accumulation
   - Use mixed precision training with `--policy.use_amp=true`

3. **Slow Data Loading**
   - Increase number of dataloader workers
   - Use local dataset cache
   - Optimize image preprocessing

4. **WandB Connection Issues**
   - Use offline mode: `--wandb.mode=offline`
   - Sync later with: `wandb sync outputs/train/your_run`

5. **Robot Connection Timeout**
   - Check network connectivity
   - Verify IP addresses in config
   - Increase connection_timeout parameter

## Additional Resources

### Key Files for Reference
- **Training Script**: `src/lerobot/scripts/train.py`
- **Evaluation Script**: `src/lerobot/scripts/eval.py`
- **WandB Utils**: `src/lerobot/utils/wandb_utils.py`
- **Tatbot Robot**: `src/lerobot/robots/tatbot/tatbot.py`
- **RealSense Camera**: `src/lerobot/cameras/realsense/camera_realsense.py`
- **SmolVLA Policy**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **Pi0 Policy**: `src/lerobot/policies/pi0/modeling_pi0.py`

### Documentation References
- Training with Script: `examples/4_train_policy_with_script.md`
- Policy README files in respective directories
- Camera configuration guides in `docs/source/cameras.mdx`

## Citation

If you use SmolVLA:
```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and others},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

For π0:
Refer to Physical Intelligence paper at https://www.physicalintelligence.company/download/pi0.pdf
