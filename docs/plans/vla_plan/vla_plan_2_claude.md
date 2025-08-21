---
summary: VLA plan 2 — training and inference on tatbot stroke datasets
tags: [plans, vla]
updated: 2025-08-21
audience: [dev]
---

# VLA Training and Inference Plan for Tatbot Stroke Datasets

## Executive Summary

This document outlines the complete pipeline for training Vision-Language-Action (VLA) models on datasets recorded via the `stroke.py` tool and implementing MCP-based inference for the Tatbot system. The plan leverages the existing LeRobot framework infrastructure while adding Tatbot-specific capabilities.

## Table of Contents
1. [Dataset Recording via stroke.py](#dataset-recording-via-strokepy)
2. [Dataset Format and Structure](#dataset-format-and-structure)
3. [Training Pipeline](#training-pipeline)
4. [MCP Tool for Inference](#mcp-tool-for-inference)
5. [Implementation Phases](#implementation-phases)
6. [Technical Requirements](#technical-requirements)

## Dataset Recording via stroke.py

### Current Capabilities
The `stroke.py` tool already implements comprehensive recording functionality:

- **LeRobot Dataset Creation**: Creates LeRobotDataset instances at `/nfs/tatbot/recordings/stroke-{scene}-{timestamp}/`
- **Multi-modal Recording**: 
  - Robot joint states and actions (10 Hz default, configurable)
  - Optional RealSense depth cameras
  - Optional IP cameras
  - Joystick inputs for human corrections
- **Episode Management**: Each stroke pair (left/right) forms an episode with:
  - Observation frames (robot state + camera images)
  - Action frames (joint commands)
  - Episode conditioning metadata (stroke descriptions, G-code parameters)
  - Episode logs for debugging

### Data Collection Strategy

1. **Human Demonstrations**:
   ```bash
   # Record expert demonstrations with joystick corrections
   mcp__eek__stroke '{"scene": "tatbotlogo", "enable_joystick": true, "enable_realsense": true, "fps": 30}'
   ```

2. **Automated Collection**:
   ```bash
   # Record autonomous executions for data augmentation
   mcp__eek__stroke '{"scene": "default", "enable_realsense": true}'
   ```

3. **Resume Capability**: Continue interrupted recordings:
   ```bash
   mcp__eek__stroke '{"scene": "tatbotlogo", "resume": true}'
   ```

## Dataset Format and Structure

### Directory Structure
```
/nfs/tatbot/recordings/
└── stroke-{scene}-{timestamp}/
    ├── meta_data/
    │   ├── data.parquet          # Episode metadata
    │   ├── stats.json             # Dataset statistics
    │   └── info.json              # Robot/camera configuration
    ├── videos/
    │   ├── observation.image_{camera_name}_{episode:06d}.mp4
    │   └── observation.depth_{camera_name}_{episode:06d}.mp4
    ├── logs/
    │   └── episode_{episode:06d}.txt
    ├── episode_{episode:06d}/
    │   ├── stroke_l.png           # Left stroke visualization
    │   └── stroke_r.png           # Right stroke visualization
    ├── scene.yaml                 # Scene configuration
    ├── strokes.yaml              # StrokeList with G-code data
    └── strokebatch.safetensors   # Pre-computed IK solutions
```

### Data Schema
```python
# Observation features
observation = {
    "image": {camera_name: np.ndarray},  # RGB images
    "depth": {camera_name: np.ndarray},  # Depth maps (optional)
    "state": np.ndarray[14],            # Joint positions (7 per arm)
}

# Action features  
action = {
    "joints": np.ndarray[14],           # Target joint positions
}

# Episode conditioning
episode_cond = {
    "stroke_l": {...},                  # Left stroke metadata
    "stroke_r": {...},                  # Right stroke metadata
    "task": str,                        # Task description
}
```

## Training Pipeline

### Phase 1: Data Preparation

1. **Dataset Aggregation**:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   from pathlib import Path
   
   # Aggregate multiple recording sessions
   recordings_dir = Path("/nfs/tatbot/recordings")
   datasets = []
   
   for dataset_dir in recordings_dir.glob("stroke-*"):
       repo_id = f"tatbot/{dataset_dir.name}"
       dataset = LeRobotDataset(repo_id, root=str(dataset_dir))
       datasets.append(dataset)
   
   # Merge datasets or train on multiple
   ```

2. **Data Validation**:
   ```python
   # Validate dataset compatibility
   from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
   
   for dataset in datasets:
       sanity_check_dataset_robot_compatibility(
           dataset, robot, fps=30, 
           dataset_features=expected_features
       )
   ```

3. **Push to HuggingFace Hub** (optional):
   ```python
   dataset.push_to_hub(
       repo_id="your_org/tatbot_strokes",
       private=True
   )
   ```

### Phase 2: VLA Model Training

#### Direct Training from Local Datasets

For quickest iteration, train directly from local recording directories without Hub uploads:

```bash
# SmolVLA finetune from base checkpoint using local dataset
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.root="$HOME/tatbot/nfs/recordings/stroke-tatbotlogo-2025y-08m-09d-17h-02m-10s" \
    --output_dir=outputs/train/tatbotlogo/smolvla \
    --batch_size=64 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_smolvla

# Pi0 training from scratch
lerobot-train \
    --policy.type=pi0 \
    --dataset.root="$HOME/tatbot/nfs/recordings/stroke-default-latest" \
    --output_dir=outputs/train/default/pi0 \
    --batch_size=32 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_pi0
```

#### SmolVLA Training Configuration
```yaml
# config/train_smolvla_tatbot.yaml
model:
  type: smolvla
  vlm_model_name: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
  n_obs_steps: 1  # Single frame (adjust to 2 for temporal context)
  chunk_size: 50  # Match stroke pose steps
  n_action_steps: 50  # Match scene.stroke_length
  resize_imgs_with_padding: [512, 512]  # Standard SmolVLA resolution
  freeze_vision_encoder: true
  train_expert_only: true  # Focus on action prediction

training:
  batch_size: 32
  steps: 100000
  optimizer_lr: 1e-4
  optimizer_grad_clip_norm: 10
  scheduler_warmup_steps: 1000
  scheduler_decay_steps: 30000
  eval_freq: 5000
  save_freq: 10000
  
dataset:
  # Option 1: Local dataset root
  root: "/nfs/tatbot/recordings/stroke-tatbotlogo-latest"
  # Option 2: Hub repo ID (if pushed)
  # repo_id: "tatbot/stroke-aggregated"
  
  delta_timestamps:
    observation.image: [-0.1, 0.0]
    observation.state: [-0.1, 0.0]
    action: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

wandb:
  enable: true
  project: "tatbot_vla_strokes"
  entity: "your_entity"
  mode: "online"  # Use "offline" if network issues
```

#### Checkpoint Layout
Typical training output structure:
```
outputs/train/<scene>/<policy>/
└── checkpoints/
    ├── last/
    │   └── pretrained_model/  # Latest checkpoint
    ├── step_50000/
    │   └── pretrained_model/  # Intermediate checkpoint
    └── best/
        └── pretrained_model/  # Best validation checkpoint
```

#### Launch Training
```bash
# Train from scratch with short validation run
uv run lerobot-train \
    --policy.type=smolvla \
    --dataset.root="$HOME/tatbot/nfs/recordings/stroke-tatbotlogo-latest" \
    --batch_size=8 \
    --steps=100 \
    --output_dir=outputs/train/test_run

# Full training run
uv run lerobot-train \
    --policy.type=smolvla \
    --dataset.root="$HOME/tatbot/nfs/recordings/stroke-tatbotlogo-latest" \
    --batch_size=32 \
    --steps=100000 \
    --output_dir=outputs/train/tatbotlogo/smolvla
```

### Phase 3: Evaluation and Monitoring

1. **Training Validation**:
   ```bash
   # Quick sanity check - train for a few steps
   uv run lerobot-train \
       --policy.type=smolvla \
       --dataset.root="$HOME/tatbot/nfs/recordings/stroke-tatbotlogo-latest" \
       --batch_size=4 \
       --steps=10 \
       --output_dir=outputs/train/sanity_check
   ```

2. **WandB Metrics**:
   - Loss curves (MSE for actions, cross-entropy for language)
   - Action prediction accuracy
   - Validation episode success rate
   - GPU utilization and training speed

3. **Checkpoint Validation**:
   ```python
   # Verify checkpoint loads correctly
   from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
   
   checkpoint_path = "outputs/train/tatbotlogo/smolvla/checkpoints/last/pretrained_model"
   policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
   print(f"Loaded policy with config: {policy.config}")
   ```

## MCP Tool for Inference

### Tool Design: `vla_infer`

```python
# src/tatbot/tools/robot/vla_infer_models.py
from typing import Literal, Optional
from tatbot.tools.base import ToolInput, ToolOutput

class VLAInferInput(ToolInput):
    policy: Literal["smolvla", "pi0"]
    checkpoint_path: str
    scene: str = "default"
    device: Literal["cuda", "cpu"] = "cuda"
    max_steps: int = 500
    enable_realsense: bool = False
    fps: int = 10
    debug: bool = False
    record_eval: bool = False
    dry_run: bool = False

class VLAInferOutput(ToolOutput):
    success: bool = True
    message: str = ""
    num_steps: int = 0
    eval_dir: Optional[str] = None
```

```python
# src/tatbot/tools/robot/vla_infer.py
import time
import torch
from pathlib import Path
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import busy_wait
from tatbot.main import compose_and_validate_scene
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.robot.vla_infer_models import VLAInferInput, VLAInferOutput
from tatbot.utils.log import get_logger

log = get_logger("tools.vla_infer", "🧠")

@tool(
    name="vla_infer",
    nodes=["hog"],
    description="Run VLA policy inference on Tatbot from a checkpoint",
    input_model=VLAInferInput,
    output_model=VLAInferOutput,
)
async def vla_infer(input_data: VLAInferInput, ctx: ToolContext):
    """
    Execute tattoo strokes using a trained VLA policy.
    
    Parameters:
    - policy (str): Policy type ("smolvla" or "pi0")
    - checkpoint_path (str): Path to model checkpoint
    - scene (str): Scene configuration name
    - device (str): Device for inference ("cuda" or "cpu")
    - max_steps (int): Maximum steps to execute
    - enable_realsense (bool): Use RealSense cameras
    - fps (int): Inference frequency
    - record_eval (bool): Record evaluation dataset
    - dry_run (bool): Load without execution
    
    Returns:
    - success (bool): Execution status
    - num_steps (int): Number of steps executed
    - message (str): Status message
    - eval_dir (str): Path to evaluation dataset (if recorded)
    """
    
    try:
        yield {"progress": 0.01, "message": "Loading scene configuration..."}
        scene = compose_and_validate_scene(input_data.scene)
        
        # Configure cameras if enabled
        rs_cameras = {}
        if input_data.enable_realsense:
            from lerobot.cameras.realsense import RealSenseCameraConfig
            rs_cameras = {
                cam.name: RealSenseCameraConfig(
                    fps=cam.fps, width=cam.width, height=cam.height,
                    serial_number_or_name=cam.serial_number,
                ) for cam in scene.cams.realsenses
            }
        
        robot: Robot = make_robot_from_config(TatbotConfig(
            ip_address_l=scene.arms.ip_address_l,
            ip_address_r=scene.arms.ip_address_r,
            arm_l_config_filepath=scene.arms.arm_l_config_filepath,
            arm_r_config_filepath=scene.arms.arm_r_config_filepath,
            goal_time=scene.arms.goal_time_slow,
            connection_timeout=scene.arms.connection_timeout,
            home_pos_l=scene.sleep_pos_l.joints,
            home_pos_r=scene.sleep_pos_r.joints,
            rs_cameras=rs_cameras,
            ip_cameras={},
        ))
        
        # Dry run validation
        if input_data.dry_run:
            yield VLAInferOutput(
                success=True, 
                message="Loaded scene and robot config successfully (dry run)", 
                num_steps=0
            )
            return
        
        yield {"progress": 0.05, "message": "Connecting to robot..."}
        robot.connect()
        
        # Load policy
        yield {"progress": 0.1, "message": f"Loading {input_data.policy} checkpoint..."}
        if input_data.policy == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            policy = SmolVLAPolicy.from_pretrained(input_data.checkpoint_path)
        else:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            policy = PI0Policy.from_pretrained(input_data.checkpoint_path)
        
        policy.eval()
        policy.to(input_data.device)
        
        # Optional evaluation recording
        dataset = None
        eval_dir = None
        if input_data.record_eval:
            output_dir = Path("/nfs/tatbot/recordings")
            eval_dir = output_dir / f"vla-eval-{scene.name}-{int(time.time())}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            action_features = hw_to_dataset_features(robot.action_features, "action", True)
            obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
            dataset = LeRobotDataset.create(
                repo_id=f"tatbot/{eval_dir.name}",
                fps=input_data.fps,
                root=str(eval_dir),
                robot_type=robot.name,
                features={**action_features, **obs_features},
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(rs_cameras) if rs_cameras else 0,
            )
        
        # Move to ready position
        yield {"progress": 0.15, "message": "Moving to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        # Inference loop
        yield {"progress": 0.2, "message": "Starting inference loop..."}
        num_steps = 0
        dt_target = 1.0 / max(1, input_data.fps)
        
        try:
            while num_steps < input_data.max_steps:
                t0 = time.perf_counter()
                
                # Get observation and predict action
                observation = robot.get_observation()
                with torch.no_grad():
                    action = policy.select_action(observation)
                
                # Convert action format if needed (VLA policies may output joint angles)
                if hasattr(action, 'shape') and len(action.shape) == 1 and len(action) == 14:
                    # Action is likely joint angles, convert to robot action format
                    robot_action = robot._urdf_joints_to_action(action)
                else:
                    robot_action = action
                
                # Send action (use fast goal time for continuous control)
                sent_action = robot.send_action(robot_action, scene.arms.goal_time_fast)
                
                # Record if evaluation dataset is enabled
                if dataset is not None:
                    obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    act_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
                    dataset.add_frame({**obs_frame, **act_frame})
                
                num_steps += 1
                
                # Update progress periodically
                if num_steps % 50 == 0:
                    yield {
                        "progress": 0.2 + (0.7 * num_steps / input_data.max_steps),
                        "message": f"Executed {num_steps}/{input_data.max_steps} steps"
                    }
                
                # Maintain target FPS
                dt = time.perf_counter() - t0
                if dt < dt_target:
                    busy_wait(dt_target - dt)
            
            # Save evaluation episode if recording
            if dataset is not None:
                dataset.save_episode()
                
        finally:
            # Return to safe position
            yield {"progress": 0.95, "message": "Returning to ready position..."}
            robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
            robot.disconnect()
        
        yield VLAInferOutput(
            success=True,
            message=f"✅ Inference completed: {num_steps} steps executed",
            num_steps=num_steps,
            eval_dir=str(eval_dir) if eval_dir else None
        )
        
    except Exception as e:
        error_msg = f"❌ VLA inference failed: {e}"
        log.error(error_msg)
        yield VLAInferOutput(
            success=False,
            message=error_msg,
            num_steps=0
        )
```

### MCP Server Integration

1. **Register Tool in Config**:
```yaml
# src/conf/mcp/hog.yaml
tools:
  - align
  - reset
  - sense
  - stroke
  - vla_infer  # New VLA inference tool (renamed for clarity)

vla:
  default_checkpoint: "outputs/train/tatbotlogo/smolvla/checkpoints/last/pretrained_model"
  device: "cuda"
  batch_size: 1
```

2. **Tool Registration**:
The tool will be automatically registered when the module is imported during `register_all_tools()`:
```python
# src/tatbot/tools/robot/__init__.py (ensure vla_infer.py is imported)
from . import align, reset, sense, stroke, vla_infer  # Add vla_infer import
```

The existing `get_tools_for_node()` function will automatically discover it.

3. **Restart MCP Server**:
```bash
# Kill existing processes and restart
./scripts/kill.sh
./scripts/mcp_run.sh hog

# Or restart on remote node
ssh hog "bash ~/tatbot/scripts/mcp_run.sh hog"
```

### Inference Modes

1. **Dry Run Validation**:
```json
{
  "policy": "smolvla",
  "checkpoint_path": "outputs/train/tatbotlogo/smolvla/checkpoints/last/pretrained_model",
  "scene": "tatbotlogo",
  "dry_run": true
}
```

2. **Full Inference with Recording**:
```json
{
  "policy": "smolvla",
  "checkpoint_path": "outputs/train/tatbotlogo/smolvla/checkpoints/last/pretrained_model",
  "scene": "tatbotlogo",
  "device": "cuda",
  "max_steps": 500,
  "enable_realsense": true,
  "fps": 10,
  "record_eval": true
}
```

3. **CPU Testing (Lower Performance)**:
```json
{
  "policy": "pi0",
  "checkpoint_path": "outputs/train/default/pi0/checkpoints/best/pretrained_model",
  "device": "cpu",
  "max_steps": 50,
  "fps": 5
}
```

## Implementation Phases

### Phase 1: Data Collection (Week 1-2)
- [ ] Record 100+ episodes using stroke.py with various scenes
- [ ] Validate dataset format and compatibility
- [ ] Create train/validation splits
- [ ] Document recording best practices

### Phase 2: Model Training (Week 3-4)
- [ ] Set up training configuration for SmolVLA
- [ ] Implement custom data transforms if needed
- [ ] Train baseline model (50k steps)
- [ ] Monitor training with WandB
- [ ] Evaluate on validation set

### Phase 3: MCP Tool Development (Week 5)
- [ ] Implement vla_stroke_tool
- [ ] Add model loading and caching
- [ ] Integrate with existing MCP server
- [ ] Test inference pipeline

### Phase 4: Deployment and Optimization (Week 6)
- [ ] Deploy model to hog node
- [ ] Optimize inference speed (quantization, caching)
- [ ] Implement safety checks and fallbacks
- [ ] Create monitoring dashboard

### Phase 5: Iteration and Improvement (Ongoing)
- [ ] Collect failure cases
- [ ] Fine-tune on new data
- [ ] Experiment with Pi0 model
- [ ] Add multi-task capabilities

## Technical Requirements

### Hardware
- **Training**: GPU with 24GB+ VRAM (RTX 3090/4090 or better)
- **Inference**: GPU with 8GB+ VRAM (RTX 4050 on ook node sufficient)
- **Storage**: 500GB+ for datasets and checkpoints on NFS

### Software Dependencies
LeRobot extras (must be installed in LeRobot repo directory, not tatbot):
```bash
# In your LeRobot checkout directory (e.g., ~/lerobot)
cd ~/lerobot
uv pip install -e .[smolvla]  # For SmolVLA policy
uv pip install -e .[pi0]      # For Pi0 policy  
uv pip install -e .[tatbot]   # For Tatbot robot support
uv pip install -e .[intelrealsense]  # For RealSense cameras
```

### Environment Setup
```bash
# In tatbot repo
source scripts/setup_env.sh
uv pip install -e .
uv pip install -e .[gen,gpu,img,viz,dev]  # Tatbot extras

# Install WandB explicitly for experiment tracking
uv pip install wandb

# Load environment variables
set -a; source .env; set +a
```

### Network Requirements
- High-speed NFS for dataset access
- GPU node accessibility for remote training
- Low-latency connection for real-time inference

## Risk Mitigation

1. **Data Quality Issues**:
   - Solution: Implement data validation pipeline
   - Use joystick corrections during recording
   - Filter low-quality episodes

2. **Model Overfitting**:
   - Solution: Data augmentation (camera angles, lighting)
   - Regularization techniques
   - Diverse scene configurations

3. **Inference Latency**:
   - Solution: Model quantization
   - Batch processing where possible
   - Caching repeated computations

4. **Safety Concerns**:
   - Solution: Confidence thresholds
   - Human override capability
   - Gradual rollout with monitoring

## Success Metrics

1. **Training Metrics**:
   - Final validation loss < 0.01
   - Action prediction accuracy > 95%
   - Training time < 48 hours

2. **Inference Metrics**:
   - Inference speed > 10 Hz
   - Episode success rate > 80%
   - Human intervention rate < 10%

3. **System Metrics**:
   - Model size < 2GB
   - GPU memory usage < 8GB
   - Network latency < 50ms

## Conclusion

This plan provides a comprehensive roadmap for training VLA models on Tatbot stroke datasets and deploying them via MCP tools. The approach leverages existing infrastructure while adding minimal complexity, ensuring maintainability and scalability. The phased implementation allows for iterative improvements and risk mitigation throughout the development process.
