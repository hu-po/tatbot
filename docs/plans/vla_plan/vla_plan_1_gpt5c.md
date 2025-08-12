# VLA Plan 1: Train on stroke.py datasets and add an MCP inference tool

This guide outlines how to train a VLA policy on datasets recorded by `tatbot/tools/robot/stroke.py` and how to expose an MCP tool to run inference from a specific checkpoint.

## 1) Dataset produced by stroke.py

Datasets are written to `/nfs/tatbot/recordings/stroke-<scene>-<timestamp>` and registered with repo id `tatbot/<dataset_dir_name>` using LeRobot's `LeRobotDataset`.

- Frame schema is built from `robot.observation_features` and `robot.action_features`
- FPS is configurable (default 10)
- Per-episode logs and conditioning artifacts are saved alongside frames

Artifacts you should see per recording:
- `scene.yaml` — scene snapshot used for the run
- `strokes.yaml` — serialized `StrokeList` (large arrays saved separately)
- `strokebatch.safetensors` — packed joint trajectories from IK
- `logs/episode_*.txt` — per-episode logs
- `episode_*/` — episode directories with preview images and frames

Implications for training:
- The dataset is already in LeRobot format; use directly in LeRobot training pipelines
- Verify your policy expects the same input keys; otherwise add a small preprocessing/mapping step

## 2) Training plan

Choose a policy (SmolVLA or Pi0) based on compute and task complexity.

- Environment
  - Install tatbot and extras on the training node: `uv pip install -e .[bot,cam,gen,gpu,img]`
  - Install LeRobot with the appropriate extras for the chosen policy
  - Ensure the dataset path is accessible (NFS mount or local copy)

- Dataset selection
  - Use repo id like `tatbot/stroke-<scene>-<timestamp>` (or pass an explicit dataset path)
  - Where supported by your training CLI, you can also use a local `--dataset.root="/nfs/tatbot/recordings/stroke-..."`

- Example: finetune SmolVLA from base
  ```bash
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=tatbot/stroke-<scene>-<timestamp> \
    --output_dir=~/tatbot/output/train/<scene>/smolvla \
    --batch_size=64 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_vla
  ```

- Example: train Pi0
  ```bash
  lerobot-train \
    --policy.type=pi0 \
    --dataset.repo_id=tatbot/stroke-<scene>-<timestamp> \
    --output_dir=~/tatbot/output/train/<scene>/pi0 \
    --batch_size=32 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project=tatbot_vla
  ```

Tips
- Start with a short run to validate I/O and schema
- Confirm image modalities and state keys match your policy’s expected inputs
- Save frequent checkpoints; keep both best and last

## 3) Checkpoints layout

Typical outputs:
- `~/tatbot/output/train/<scene>/smolvla/checkpoints/last/pretrained_model/`
- or `.../checkpoints/step_<N>/pretrained_model/`

These folders should be loadable with the policy class `from_pretrained` method.

## 4) MCP inference tool: design

Goal: Run on robot from a chosen checkpoint via an MCP tool call; stream progress and allow safe shutdown.

 Proposed tool
 - Name: `infer_vla`
 - Node: `hog`

Input model (Pydantic)
- `checkpoint_path: str` — local path or Hugging Face repo id
- `scene_name: str = "default"`
- `policy_type: {"smolvla","pi0"} = "smolvla"`
- `device: {"cpu","cuda"} = "cuda"`
- `enable_realsense: bool = False`
- `fps: int = 10`
- Optional: `max_steps: int | None`, `dry_run: bool = False`, `debug: bool = False`

Output model
- `success: bool`, `message: str`

Implementation sketch
```python
# file: tatbot/tools/robot/infer_vla.py
import asyncio, torch
from pydantic import BaseModel
from tatbot.tools.registry import tool
from tatbot.tools.base import ToolContext
from tatbot.main import compose_and_validate_scene
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.robots import make_robot_from_config

class InferVLAInput(BaseModel):
    checkpoint_path: str
    scene_name: str = "default"
    policy_type: str = "smolvla"  # or "pi0"
    device: str = "cuda"
    enable_realsense: bool = False
    fps: int = 10
    max_steps: int | None = None
    dry_run: bool = False
    debug: bool = False

class InferVLAOutput(BaseModel):
    success: bool
    message: str

@tool(name="infer_vla", nodes=["hog"], description="Run VLA policy inference on robot")
async def infer_vla(input_data: InferVLAInput, ctx: ToolContext) -> InferVLAOutput:
    scene = compose_and_validate_scene(input_data.scene_name)

    rs_cameras = {}
    if input_data.enable_realsense:
        from lerobot.cameras.realsense import RealSenseCameraConfig
        rs_cameras = {
            c.name: RealSenseCameraConfig(
                fps=c.fps, width=c.width, height=c.height, serial_number_or_name=c.serial_number
            ) for c in scene.cams.realsenses
        }

    robot = make_robot_from_config(TatbotConfig(
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

    if input_data.policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
    elif input_data.policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as Policy
    else:
        return InferVLAOutput(success=False, message=f"Unknown policy_type: {input_data.policy_type}")

    policy = Policy.from_pretrained(input_data.checkpoint_path)
    policy.to(input_data.device)
    policy.eval()

    if input_data.dry_run:
        return InferVLAOutput(success=True, message="Loaded policy and scene successfully (dry run)")

    robot.connect()
    try:
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        step = 0
        # Optional: record eval frames similar to stroke.py using LeRobotDataset
        while input_data.max_steps is None or step < input_data.max_steps:
            obs = robot.get_observation()
            with torch.no_grad():
                action = policy.select_action(obs)
            robot.send_action(action)
            await asyncio.sleep(1 / input_data.fps)
            step += 1
    finally:
        robot.disconnect()

    return InferVLAOutput(success=True, message="Inference completed")
```

Wiring
- Save as `tatbot/tools/robot/infer_vla.py`
- Ensure it is imported on server start (decorator-based registry picks it up upon import). If needed, add an import in `tatbot/tools/robot/__init__.py`
- Restart the MCP server on the target node:
  ```bash
  ssh hog "bash ~/tatbot/scripts/run_mcp.sh hog"
  ```

Example call payload
```json
{
  "tool": "infer_vla",
  "params": {
    "checkpoint_path": "/home/oop/tatbot/output/train/<scene>/smolvla/checkpoints/last/pretrained_model",
    "scene_name": "tatbotlogo",
    "policy_type": "smolvla",
    "device": "cuda",
    "enable_realsense": true,
    "fps": 10,
    "max_steps": 1000
  }
}
```

Operational tips
- Start with `device="cpu"` and `max_steps=50` for a dry integration test; switch to `cuda` after.
- Use conservative goal times initially; only switch to fast goal time inside the loop when stable.
- Keep the workcell clear and have an emergency stop ready during first live tests.

## 5) Validation checklist
- [ ] Short sanity train on one dataset; confirm loss decreases
- [ ] Checkpoint reloads with `from_pretrained`
- [ ] MCP tool connects to robot and reaches `ready_pos_full`
- [ ] Inference loop maintains requested FPS
- [ ] Clean shutdown and error handling verified

## 6) Next steps
- Add telemetry (loop time, policy latency) and optional video logging
- Implement automatic policy-type detection from checkpoint metadata
- Provide a safety interlock and emergency stop binding

## Appendix: Corrected MCP tool skeleton (matches Tatbot MCP contract)

```python
# file: src/tatbot/tools/robot/infer_vla.py
import asyncio
import torch
from pydantic import BaseModel
from tatbot.tools.registry import tool
from tatbot.tools.base import ToolContext
from tatbot.main import compose_and_validate_scene
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.robots import make_robot_from_config

class InferVLAInput(BaseModel):
    checkpoint_path: str
    scene_name: str = "default"
    policy_type: str = "smolvla"  # or "pi0"
    device: str = "cuda"
    enable_realsense: bool = False
    fps: int = 10
    max_steps: int | None = None
    dry_run: bool = False
    debug: bool = False

class InferVLAOutput(BaseModel):
    success: bool
    message: str

@tool(
    name="infer_vla",
    nodes=["hog"],
    description="Run VLA policy inference on robot",
    input_model=InferVLAInput,
    output_model=InferVLAOutput,
)
async def infer_vla(input_data: InferVLAInput, ctx: ToolContext):
    # Progress: loading scene
    yield {"progress": 0.01, "message": "Loading scene configuration..."}
    scene = compose_and_validate_scene(input_data.scene_name)

    # Optional RealSense wiring mirrors stroke.py
    rs_cameras = {}
    if input_data.enable_realsense:
        from lerobot.cameras.realsense import RealSenseCameraConfig
        rs_cameras = {
            c.name: RealSenseCameraConfig(
                fps=c.fps, width=c.width, height=c.height, serial_number_or_name=c.serial_number
            ) for c in scene.cams.realsenses
        }

    # Complete robot configuration
    robot = make_robot_from_config(TatbotConfig(
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

    # Load policy
    if input_data.policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
    elif input_data.policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as Policy
    else:
        yield InferVLAOutput(success=False, message=f"Unknown policy_type: {input_data.policy_type}")
        return

    yield {"progress": 0.05, "message": "Loading policy checkpoint..."}
    policy = Policy.from_pretrained(input_data.checkpoint_path)
    policy.to(input_data.device)
    policy.eval()

    if input_data.dry_run:
        yield InferVLAOutput(success=True, message="Loaded policy and scene successfully (dry run)")
        return

    yield {"progress": 0.1, "message": "Connecting to robot..."}
    robot.connect()
    try:
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        step = 0
        yield {"progress": 0.2, "message": "Starting inference loop..."}
        while input_data.max_steps is None or step < input_data.max_steps:
            obs = robot.get_observation()
            with torch.no_grad():
                action = policy.select_action(obs)
            robot.send_action(action)
            await asyncio.sleep(1 / input_data.fps)
            step += 1
            if step % 50 == 0:
                yield {"progress": 0.2 + 0.7 * (step / (input_data.max_steps or step)),
                       "message": f"Executed {step} steps"}
    finally:
        robot.disconnect()

    yield InferVLAOutput(success=True, message="Inference completed")
```
