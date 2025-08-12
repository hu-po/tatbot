# Tatbot VLA Plan 3 — Training on stroke_tool datasets and MCP inference tool

This guide specifies what’s needed to train Vision-Language-Action (VLA) policies on datasets recorded via `src/tatbot/tools/robot/stroke.py`, and how to add an MCP tool to run inference from a specific model checkpoint.

## Table of Contents
1. [Scope](#scope)
2. [Dataset recorded by stroke_tool](#dataset-recorded-by-stroke_tool)
3. [Training requirements](#training-requirements)
4. [Preparing datasets for training](#preparing-datasets-for-training)
5. [Policy training configs](#policy-training-configs)
6. [Validating the pipeline](#validating-the-pipeline)
7. [MCP inference tool: design and code skeleton](#mcp-inference-tool-design-and-code-skeleton)
8. [Registering and running the MCP tool](#registering-and-running-the-mcp-tool)
9. [Operational tips](#operational-tips)

## Scope

- Train VLA policies (e.g., SmolVLA, π0) using data recorded by `stroke_tool`.
- Keep the LeRobot-native dataset format produced by `stroke_tool` to avoid conversion steps.
- Provide an MCP tool to load a chosen checkpoint and run inference on the robot.

## Dataset recorded by stroke_tool

`stroke_tool` writes LeRobot-compatible episodic datasets into `/nfs/tatbot/recordings/` with names like `stroke-<scene>-<timestamp>`. Within each dataset directory:

- `scene.yaml`: Scene definition saved at recording start.
- `strokes.yaml`: Stroke list with metadata; large arrays are in `arrays/*.npy` (see `tatbot.data.stroke`).
- `strokebatch.safetensors`: Packed joint trajectories and EE poses (`tatbot.data.stroke.StrokeBatch`).
- `logs/episode_*.txt`: Per-episode logs.
- `episode_000000/`, `episode_000001/`, …: Episode folders created by `LeRobotDataset`, including recorded frames and `episode_cond` with references to `stroke_l`/`stroke_r` metadata and any preview images.

Example directory structure (may vary slightly by LeRobot version/settings):

```
/nfs/tatbot/recordings/
└── stroke-{scene_name}-{timestamp}/
    ├── meta_data/
    │   ├── data.parquet
    │   ├── stats.json
    │   └── info.json
    ├── videos/
    │   ├── observation.image_{camera_name}_{episode:06d}.mp4
    │   └── observation.depth_{camera_name}_{episode:06d}.mp4
    ├── logs/
    │   └── episode_{episode:06d}.txt
    ├── episode_{episode:06d}/
    │   ├── stroke_l.png
    │   └── stroke_r.png
    ├── scene.yaml
    ├── strokes.yaml
    └── strokebatch.safetensors
```

Key details from `src/tatbot/tools/robot/stroke.py`:
- The dataset is created (or resumed) via `LeRobotDataset.create(...)`/`LeRobotDataset(...)` with `features` derived from `robot.action_features` and `robot.observation_features`.
- When RealSense/IP cameras are enabled, images are written through LeRobot’s image writer threads.
- `fps` defaults to 10 (configurable via tool input).
- Each pose step adds a frame composed of observation and the action actually sent to the robot.

Implication: these datasets are immediately usable by LeRobot training scripts (no custom converter required).

## Training requirements

- Python environment with Tatbot installed via `uv`. Install LeRobot separately (in its own checkout) if you need policy code/extras:

```bash
source scripts/setup_env.sh
uv pip install -e .
uv pip install -e .[bot,cam,gen,gpu,img,viz,dev,docs]
# If training with LeRobot policies, do this in your LeRobot repo (not here):
#   cd ~/lerobot && uv pip install -e .[smolvla,pi0]
set -a; source .env; set +a  # optional secrets for WandB, etc.
```

- GPU recommended for training; CPU-only is possible for debugging.
- WandB optional; install/enable explicitly in your training environment.

## Preparing datasets for training

You can train directly from a locally recorded dataset directory. Two common options:

- Local path training (recommended initially):
  - Use the full path to a recording directory, e.g. `/nfs/tatbot/recordings/stroke-tatbotlogo-2025y-08m-09d-17h-02m-10s`.
  - Many LeRobot CLIs accept `--dataset.root` or a `repo_id` that points locally; prefer `--dataset.root` where available.

- Pushing to Hub (optional):
  - If desired, push the dataset to the Hugging Face Hub using LeRobot’s dataset utilities.

Minimum checks before training:
- Confirm `strokes.yaml` and `strokebatch.safetensors` exist.
- Confirm episodes exist and contain frames and actions.
- Skim `logs/` for anomalies; ensure joint/action ranges and fps look correct.

Aggregating multiple recordings (optional):

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

recordings_dir = Path("/nfs/tatbot/recordings")
datasets = []
for dataset_dir in recordings_dir.glob("stroke-*"):
    repo_id = f"tatbot/{dataset_dir.name}"
    datasets.append(LeRobotDataset(repo_id, root=str(dataset_dir)))
# Train across multiple datasets or merge per your pipeline
```

## Policy training configs

Guidance uses the same policy families as in `docs/models/claude_vla_guide.md`.

- Choose policy: `smolvla` for faster iteration or `pi0` as needed.
- Observation length vs. action chunking:
  - Typical: `n_obs_steps = 1`, `chunk_size = 50`, `n_action_steps = 50` to match stroke pose steps.
  - If training on sequences spanning multiple poses, adjust accordingly.
- Image size and preprocessing: ensure to match camera output (e.g., 512×512 with padding for SmolVLA).

Example commands (adjust flags to your CLI wrapper; standardize outputs under `outputs/train/`):

SmolVLA finetune from base on a local dataset root:
```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.root="/nfs/tatbot/recordings/stroke-tatbotlogo-..." \
  --batch_size=32 \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.project=tatbot_smolvla_finetune \
  --output_dir=outputs/train/smolvla_tatbot
```

Pi0 finetune from base:
```bash
lerobot-train \
  --policy.type=pi0 \
  --dataset.root="/nfs/tatbot/recordings/stroke-tatbotlogo-..." \
  --batch_size=32 \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.project=tatbot_pi0_finetune \
  --output_dir=outputs/train/pi0_tatbot
```

Notes:
- Prefer `--dataset.root` for local datasets; use `--dataset.repo_id` only if pushing to Hub.
- Do not assume fixed `chunk_size`/`n_action_steps`; align with actual `scene.stroke_length` and model config.
- Keep evaluation split: either reserve episodes for validation or use `--dataset.split.*` flags where available.

## Validating the pipeline

- Dry-run a few training steps and check WandB/logs for:
  - Loss decreasing, stable gradient norms, GPU utilization reasonable.
  - Sampled batches show correct image shapes and action ranges.
- Save and test a checkpoint by running local evaluation on a held-out episode set.

## MCP inference tool: design and code skeleton

We will add a new MCP tool `vla_infer` that:
- Loads a specified checkpoint directory into the chosen policy class.
- Builds a `Tatbot` instance from a scene, connects, streams observations, and sends policy actions in a timed loop.
- Optionally records an evaluation dataset for later analysis.

Suggested file layout (choose one and keep consistent):
- Place under `src/tatbot/tools/robot/` (recommended, like other robot tools):
  - `src/tatbot/tools/robot/models_vla.py`
  - `src/tatbot/tools/robot/infer_vla.py`
- Or create a dedicated package `src/tatbot/tools/vla/` and import it in the registry. This guide shows the robot/ layout.

Input model fields (proposed):
- `policy`: one of `"smolvla" | "pi0"`
- `checkpoint_path`: local path to policy checkpoint (e.g., `outputs/train/smolvla_tatbot/checkpoints/last/pretrained_model`)
- `scene_name`: optional scene to connect the robot (default `"default"`)
- `device`: `"cuda" | "cpu"` (default `"cuda"`)
- `max_steps`: integer safety cap on loop iterations (default `500`)
- `enable_realsense`, `fps`, `debug`: same semantics as `stroke_tool`
- `record_eval`: bool to write an eval dataset (default `false`)
- `dry_run`: bool to validate checkpoint/scene without moving robot (default `false`)

Output model fields (proposed):
- `success`: bool
- `message`: str
- `num_steps`: int
- Optional: `eval_dir`: path to saved eval dataset

Code skeleton for the tool:

```python
# src/tatbot/tools/robot/models_vla.py
from typing import Literal, Optional
from tatbot.tools.base import ToolInput, ToolOutput

class VLAInferInput(ToolInput):
    policy: Literal["smolvla", "pi0"]
    checkpoint_path: str
    scene_name: str = "default"
    device: Literal["cuda", "cpu"] = "cuda"
    max_steps: int = 500
    enable_realsense: bool = False
    fps: int = 10
    debug: bool = False
    record_eval: bool = False
    dry_run: bool = False
    task_prompt: Optional[str] = None

class VLAInferOutput(ToolOutput):
    success: bool = True
    message: str = ""
    num_steps: int = 0
    eval_dir: Optional[str] = None
```

```python
# src/tatbot/tools/robot/infer_vla.py
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
from tatbot.tools.robot.models_vla import VLAInferInput, VLAInferOutput
from tatbot.utils.log import get_logger

log = get_logger("tools.vla_infer", "🧠")

@tool(
    name="vla_infer",
    nodes=["hog"],  # add GPU nodes if you want remote-only
    description="Run VLA policy inference on Tatbot from a checkpoint",
    input_model=VLAInferInput,
    output_model=VLAInferOutput,
)
async def vla_infer(input_data: VLAInferInput, ctx: ToolContext):
    try:
        yield {"progress": 0.01, "message": "Loading scene..."}
        scene = compose_and_validate_scene(input_data.scene_name)

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

        yield {"progress": 0.05, "message": "Connecting to robot..."}
        robot.connect()

        # Load policy
        yield {"progress": 0.1, "message": "Loading policy checkpoint..."}
        if input_data.policy == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            policy = SmolVLAPolicy.from_pretrained(input_data.checkpoint_path)
        else:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            policy = PI0Policy.from_pretrained(input_data.checkpoint_path)
        policy.eval()
        policy.to(input_data.device)

        # Optional dry-run: validate loading without moving hardware
        if input_data.dry_run:
            yield VLAInferOutput(success=True, message="Loaded scene and policy (dry run)", num_steps=0)
            return

        # Optional eval recording
        dataset = None
        if input_data.record_eval:
            output_dir = Path("/nfs/tatbot/recordings")
            eval_dir = output_dir / f"vla-eval-{scene.name}-{int(time.time())}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            action_features = hw_to_dataset_features(robot.action_features, "action", True)
            obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
            # Align writer threads with stroke.py convention
            num_camera_threads = 0
            if hasattr(robot, "rs_cameras") and len(robot.rs_cameras) > 0:
                num_camera_threads += 4 * len(robot.rs_cameras)
            if hasattr(robot, "ip_cameras") and len(robot.ip_cameras) > 0:
                num_camera_threads += 4 * len(robot.ip_cameras)
            dataset = LeRobotDataset.create(
                repo_id=f"tatbot/{eval_dir.name}", fps=input_data.fps,
                root=str(eval_dir), robot_type=robot.name,
                features={**action_features, **obs_features},
                use_videos=True, image_writer_processes=0, image_writer_threads=num_camera_threads,
            )

        # Move to ready
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)

        yield {"progress": 0.2, "message": "Starting inference loop..."}
        num_steps = 0
        dt_target = 1.0 / max(1, input_data.fps)
        
        def preprocess_observation_for_policy(observation):
            """Map Tatbot observation to policy-expected dict. Include task text if needed."""
            # TODO: implement mapping for chosen policy; pass input_data.task_prompt
            return observation

        def prepare_robot_action(policy_output):
            """Convert policy output to robot action format if necessary."""
            try:
                # If policy outputs 14D URDF joints (left7+right7):
                return robot._urdf_joints_to_action(policy_output)
            except Exception:
                return policy_output
        try:
            while num_steps < input_data.max_steps:
                t0 = time.perf_counter()
                observation = robot.get_observation()
                policy_obs = preprocess_observation_for_policy(observation)
                with torch.no_grad():
                    action = policy.select_action(policy_obs) if hasattr(policy, "select_action") else policy(policy_obs)
                # Send action (fast time for continuous control)
                sent_action = robot.send_action(prepare_robot_action(action), scene.arms.goal_time_fast)

                if dataset is not None:
                    obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    act_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
                    dataset.add_frame({**obs_frame, **act_frame})

                num_steps += 1
                # FPS pacing (busy-wait for precision)
                dt = time.perf_counter() - t0
                busy_wait(dt_target - dt)

            if dataset is not None:
                dataset.save_episode()

        finally:
            robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
            robot.disconnect()

        yield VLAInferOutput(success=True, message="Inference completed", num_steps=num_steps,
                             eval_dir=str(dataset.root) if dataset is not None else None)

    except Exception as e:
        log.error(f"vla_infer failed: {e}")
        yield VLAInferOutput(success=False, message=f"❌ {e}", num_steps=0)
```

## Registering and running the MCP tool

1) Import the new tool so it auto-registers:
- Update `src/tatbot/tools/registry.py` `register_all_tools()` to include:

```python
# Import VLA tools
try:
    from tatbot.tools.vla import infer  # noqa: F401
    log.debug("Imported VLA tools")
except ImportError as e:
    log.debug(f"VLA tools not available: {e}")
```

2) Ensure the node config includes this tool (or allow wildcard):
- Edit `conf/mcp/hog.yaml` and/or GPU nodes to include `vla_infer` in `mcp.tools`.
- If the tool requires GPU, add `requires=["gpu"]` to the decorator and ensure the node has `extras: [gpu]`.

3) Restart the MCP server on the node:

```bash
./scripts/run_mcp.sh hog
```

4) Invoke the tool from your MCP client with input JSON like:

```json
{
  "policy": "smolvla",
  "checkpoint_path": "outputs/train/smolvla_tatbot/checkpoints/last/pretrained_model",
  "scene_name": "tatbotlogo",
  "device": "cuda",
  "max_steps": 500,
  "enable_realsense": true,
  "fps": 10,
  "record_eval": true
}
```

## Operational tips

- Always verify the robot is clear to move before inference runs.
- Use conservative `goal_time_*` and `max_steps` initially.
- Start with `device=cpu` for dry-runs if GPU memory is tight, then switch to `cuda`.
- For reproducibility, snapshot your `scene.yaml` and checkpoint path in run metadata (WandB or eval dataset name).

References:
- Existing end-to-end guide: `docs/models/claude_vla_guide.md`
- Stroke recording tool: `src/tatbot/tools/robot/stroke.py`
- Stroke datatypes: `src/tatbot/data/stroke.py`
