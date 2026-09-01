"""Tests for sim_cinematic.py CLI and supply resolution."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy/missing dependencies before importing sim_cinematic.
# Note: DO NOT set sys.modules["torch"] as a fake module if torch is not installed,
# because scipy checks sys.modules["torch"] for `Tensor` or expects torch not to be in sys.modules,
# and other tests check `assert "torch" not in sys.modules`.

_TORCH_WAS_IN_MODULES = "torch" in sys.modules

def _setup_mocks():
    mock_modules = {}

    # gymnasium
    if "gymnasium" not in sys.modules:
        mock_gym = types.ModuleType("gymnasium")
        mock_gym.make = MagicMock()
        mock_modules["gymnasium"] = mock_gym
    elif not hasattr(sys.modules["gymnasium"], "make"):
        sys.modules["gymnasium"].make = MagicMock()

    # sapien
    if "sapien" not in sys.modules:
        mock_sapien = types.ModuleType("sapien")
        mock_sapien.Pose = MagicMock()
        mock_modules["sapien"] = mock_sapien

    # torch - only create a mock torch module temporarily during sim_cinematic import if needed
    if "torch" not in sys.modules:
        mock_torch = types.ModuleType("torch")
        mock_torch.as_tensor = MagicMock()
        mock_torch.Tensor = MagicMock()
        mock_modules["torch"] = mock_torch

    # tyro
    if "tyro" not in sys.modules:
        mock_tyro = types.ModuleType("tyro")
        mock_tyro.cli = MagicMock()
        mock_modules["tyro"] = mock_tyro

    # cv2
    if "cv2" not in sys.modules:
        mock_cv2 = types.ModuleType("cv2")
        mock_cv2.imwrite = MagicMock()
        mock_modules["cv2"] = mock_cv2

    # PIL
    if "PIL" not in sys.modules:
        mock_pil = types.ModuleType("PIL")
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image
        mock_modules["PIL"] = mock_pil
        mock_modules["PIL.Image"] = mock_pil_image

    # mani_skill
    if "mani_skill" not in sys.modules:
        mock_ms = types.ModuleType("mani_skill")
        mock_ms_sensors = types.ModuleType("mani_skill.sensors")
        mock_ms_camera = types.ModuleType("mani_skill.sensors.camera")
        mock_ms_camera.CameraConfig = MagicMock()
        mock_ms_utils = types.ModuleType("mani_skill.utils")
        mock_ms_sapien = types.ModuleType("mani_skill.utils.sapien_utils")
        mock_ms_sapien.look_at = MagicMock()

        mock_modules["mani_skill"] = mock_ms
        mock_modules["mani_skill.sensors"] = mock_ms_sensors
        mock_modules["mani_skill.sensors.camera"] = mock_ms_camera
        mock_modules["mani_skill.utils"] = mock_ms_utils
        mock_modules["mani_skill.utils.sapien_utils"] = mock_ms_sapien

    # tatbot_sim
    if "tatbot_sim" in sys.modules:
        mock_ts = sys.modules["tatbot_sim"]
    else:
        mock_ts = types.ModuleType("tatbot_sim")
        mock_ts.__path__ = []
        mock_modules["tatbot_sim"] = mock_ts

    if "tatbot_sim.agent" not in sys.modules:
        mock_ts_agent = types.ModuleType("tatbot_sim.agent")
        class FakeTatbotWXAI:
            CAM_FOV = 0.96
            CAM_WIDTH = 640
            CAM_HEIGHT = 480
        mock_ts_agent.TatbotWXAI = FakeTatbotWXAI
        mock_ts.agent = mock_ts_agent
        mock_modules["tatbot_sim.agent"] = mock_ts_agent

    if "tatbot_sim.distributions" not in sys.modules:
        mock_ts_dist = types.ModuleType("tatbot_sim.distributions")
        @dataclass
        class FakeDist:
            name: str
            tool_id: str
            def build_args(self):
                recipe = MagicMock()
                recipe.task = "language"
                recipe.horizon = 100
                recipe.dr = MagicMock()
                recipe.draw_clearance = 0.001
                recipe.task_name = "task"
                recipe.maze_task_name = "maze"
                recipe.erase_passes = 1
                recipe.erase_seconds = 1
                return recipe

        mock_ts_dist.DISTRIBUTIONS = {
            "skin-tattoo": FakeDist("skin-tattoo", "lutin-3rl-bugpin"),
            "paper-draw": FakeDist("paper-draw", "lutin-ballpoint-dot"),
        }
        mock_ts.distributions = mock_ts_dist
        mock_modules["tatbot_sim.distributions"] = mock_ts_dist

    if "tatbot_sim.env" not in sys.modules:
        mock_ts_env = types.ModuleType("tatbot_sim.env")
        class FakeTatbotDrawEnv:
            _load_lighting = None
            _default_sensor_configs = None
        mock_ts_env.TatbotDrawEnv = FakeTatbotDrawEnv
        mock_ts.env = mock_ts_env
        mock_modules["tatbot_sim.env"] = mock_ts_env

    if "tatbot_sim.textures" not in sys.modules:
        mock_ts_textures = types.ModuleType("tatbot_sim.textures")
        mock_ts_textures.TEX_DIR = Path("/tmp")
        mock_ts.textures = mock_ts_textures
        mock_modules["tatbot_sim.textures"] = mock_ts_textures

    if "tatbot_sim.tasks" not in sys.modules:
        mock_ts_tasks = types.ModuleType("tatbot_sim.tasks")
        mock_ts_tasks.validate_task = MagicMock()
        mock_ts_tasks.validate_supply = MagicMock()
        mock_ts.tasks = mock_ts_tasks
        mock_modules["tatbot_sim.tasks"] = mock_ts_tasks

    if "tatbot_sim.tools" not in sys.modules:
        mock_ts_tools = types.ModuleType("tatbot_sim.tools")
        mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
        mock_ts_tools.active_tool = MagicMock(return_value=mock_tool)
        mock_ts_substrate = MagicMock()
        mock_ts_substrate.name = "skin"
        mock_ts_tools.active_substrate = MagicMock(return_value=mock_ts_substrate)
        mock_ts_tools.set_supply = MagicMock()
        mock_ts_tools.supply = MagicMock(return_value=("wet", "nighthawk_black"))
        mock_ts.tools = mock_ts_tools
        mock_modules["tatbot_sim.tools"] = mock_ts_tools
    else:
        tools_mod = sys.modules["tatbot_sim.tools"]
        if not hasattr(tools_mod, "set_supply"):
            tools_mod.set_supply = MagicMock()
        if not hasattr(tools_mod, "supply"):
            tools_mod.supply = MagicMock(return_value=("wet", "nighthawk_black"))
        if not hasattr(tools_mod, "active_substrate"):
            mock_ts_substrate = MagicMock()
            mock_ts_substrate.name = "skin"
            tools_mod.active_substrate = MagicMock(return_value=mock_ts_substrate)
        if not hasattr(tools_mod, "active_tool"):
            mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
            tools_mod.active_tool = MagicMock(return_value=mock_tool)

    if "tatbot_sim.expert" not in sys.modules:
        mock_ts_expert = types.ModuleType("tatbot_sim.expert")
        mock_ts_expert.StrokeExpert = MagicMock()
        mock_ts_expert.reachable_canvas_masks = MagicMock(return_value=[MagicMock(fraction=0.8)])
        mock_ts_expert.reachable_height_ceiling = MagicMock(return_value=0.05)
        mock_ts.expert = mock_ts_expert
        mock_modules["tatbot_sim.expert"] = mock_ts_expert

    if "tatbot_sim.planning" not in sys.modules:
        mock_ts_planning = types.ModuleType("tatbot_sim.planning")
        mock_ts_planning.plan_batch = MagicMock()
        mock_ts.planning = mock_ts_planning
        mock_modules["tatbot_sim.planning"] = mock_ts_planning

    if "tatbot_sim.language" not in sys.modules:
        mock_ts_lang = types.ModuleType("tatbot_sim.language")
        mock_ts_lang.CLEAN_STYLE = "clean"
        mock_ts_lang.DEFAULT_STYLE = "full"
        mock_ts_lang.MOTIFS = {"flower_of_life": None}
        mock_ts_lang.SceneStyle = MagicMock()
        mock_ts.language = mock_ts_lang
        mock_modules["tatbot_sim.language"] = mock_ts_lang

    if "tatbot_sim.config" not in sys.modules:
        mock_ts_config = types.ModuleType("tatbot_sim.config")
        class FakeDRConfig:
            noise = MagicMock()
            pen_lean = MagicMock(max_off_base_rad=0.1)
            depth_noise = MagicMock()
            corrupt_depth = True
            rgb = MagicMock()

        mock_ts_config.DRConfig = FakeDRConfig
        mock_ts.config = mock_ts_config
        mock_modules["tatbot_sim.config"] = mock_ts_config

    if "tatbot_sim.depth_noise" not in sys.modules:
        mock_ts_depth = types.ModuleType("tatbot_sim.depth_noise")
        class FakeDepthCorruptor:
            def __init__(self, num_envs, device, cfg, seed):
                pass

            def __call__(self, depth):
                return depth

        class FakeRGBJitter:
            def __init__(self, num_envs, device, cfg, seed):
                pass

            def __call__(self, rgb):
                return rgb

        mock_ts_depth.DepthCorruptor = FakeDepthCorruptor
        mock_ts_depth.RGBJitter = FakeRGBJitter
        mock_ts.depth_noise = mock_ts_depth
        mock_modules["tatbot_sim.depth_noise"] = mock_ts_depth

    for mod_name, mod_obj in mock_modules.items():
        sys.modules[mod_name] = mod_obj

_setup_mocks()

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sim_cinematic", REPO / "scripts" / "sim_cinematic.py")
sim_cinematic = importlib.util.module_from_spec(spec)
sys.modules["sim_cinematic"] = sim_cinematic
spec.loader.exec_module(sim_cinematic)

# Clean up torch mock if torch was not originally in sys.modules
if not _TORCH_WAS_IN_MODULES and "torch" in sys.modules:
    del sys.modules["torch"]

# Load sim_preview module using the same mocked environment (ensuring mock torch is available)
_need_torch_mock = "torch" not in sys.modules
if _need_torch_mock:
    mock_torch = types.ModuleType("torch")
    mock_torch.as_tensor = MagicMock()
    mock_torch.Tensor = MagicMock()
    sys.modules["torch"] = mock_torch

spec_preview = importlib.util.spec_from_file_location("sim_preview", REPO / "scripts" / "sim_preview.py")
sim_preview = importlib.util.module_from_spec(spec_preview)
sys.modules["sim_preview"] = sim_preview
spec_preview.loader.exec_module(sim_preview)

if _need_torch_mock and not _TORCH_WAS_IN_MODULES:
    del sys.modules["torch"]


def test_args_default_supply_values() -> None:
    args = sim_cinematic.Args(out="/tmp/out")
    assert args.supply == "bench"
    assert args.wet == ""
    assert args.supply_ink == "nighthawk_black"


def test_main_supply_resolution_bench(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tasks = sys.modules["tatbot_sim.tasks"]
    mock_tools.set_supply.reset_mock()
    mock_tasks.validate_supply.reset_mock()

    args = sim_cinematic.Args(out="/tmp/out", supply="bench", wet="", look="bench")
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    monkeypatch.setattr(sim_cinematic.gym, "make", MagicMock(side_effect=RuntimeError("stop_after_setup")))

    with pytest.raises(RuntimeError, match="stop_after_setup"):
        sim_cinematic.main(args, dist)

    mock_tools.set_supply.assert_called_once_with("bench", "nighthawk_black")
    mock_tasks.validate_supply.assert_called_once()


def test_main_supply_resolution_wet_override(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tasks = sys.modules["tatbot_sim.tasks"]
    mock_tools.set_supply.reset_mock()
    mock_tasks.validate_supply.reset_mock()

    args = sim_cinematic.Args(out="/tmp/out", supply="bench", wet="triple_black", look="bench")
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    monkeypatch.setattr(sim_cinematic.gym, "make", MagicMock(side_effect=RuntimeError("stop_after_setup")))

    with pytest.raises(RuntimeError, match="stop_after_setup"):
        sim_cinematic.main(args, dist)

    # When wet is set to "triple_black", set_supply should receive ("wet", "triple_black")
    mock_tools.set_supply.assert_called_once_with("wet", "triple_black")
    mock_tasks.validate_supply.assert_called_once()


def test_main_invalid_supply_raises_system_exit(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tools.set_supply.side_effect = ValueError("unknown supply kind 'invalid'")

    args = sim_cinematic.Args(out="/tmp/out", supply="invalid")
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    with pytest.raises(SystemExit) as exc_info:
        sim_cinematic.main(args, dist)

    assert "unknown supply kind 'invalid'" in str(exc_info.value)
    mock_tools.set_supply.side_effect = None


def test_cli_tool_id_mismatch_raises_system_exit(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sim_cinematic.py", "skin-tattoo", "--out", "/tmp/out"])
    monkeypatch.delenv(sim_cinematic.GUARD, raising=False)
    monkeypatch.setenv("TATBOT_TOOL_ID", "lutin-ballpoint-dot")

    with pytest.raises(SystemExit) as exc_info:
        sim_cinematic.cli()

    assert "TATBOT_TOOL_ID='lutin-ballpoint-dot' is set but 'skin-tattoo' runs 'lutin-3rl-bugpin'" in str(exc_info.value)


def test_cli_tool_id_match_reexecs(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sim_cinematic.py", "skin-tattoo", "--out", "/tmp/out"])
    monkeypatch.delenv(sim_cinematic.GUARD, raising=False)
    monkeypatch.setenv("TATBOT_TOOL_ID", "lutin-3rl-bugpin")

    execv_called = []
    def fake_execv(executable, args):
        execv_called.append((executable, args))
        raise RuntimeError("execv_called")

    monkeypatch.setattr(os, "execv", fake_execv)

    with pytest.raises(RuntimeError, match="execv_called"):
        sim_cinematic.cli()

    assert len(execv_called) == 1
    assert os.environ.get(sim_cinematic.GUARD) == "skin-tattoo"
    assert os.environ.get("TATBOT_TOOL_ID") == "lutin-3rl-bugpin"


def test_shots_palette_and_dip_defined() -> None:
    assert "palette" in sim_cinematic.SHOTS
    palette_shot = sim_cinematic.SHOTS["palette"]
    assert palette_shot.at == (0.126, 0.2675, 0.085)
    assert palette_shot.eye == (0.46, 0.05, 0.30)

    assert "dip" in sim_cinematic.SHOTS
    dip_shot = sim_cinematic.SHOTS["dip"]
    assert dip_shot.track == "tool"
    assert dip_shot.offset == (0.070, -0.060, 0.045)


def test_build_cameras_uses_shot_at_target(monkeypatch) -> None:
    mock_look_at = sys.modules["mani_skill.utils.sapien_utils"].look_at
    mock_look_at.reset_mock()

    cameras = sim_cinematic.build_cameras(["palette"], 1920, 1080)
    assert len(cameras) == 1

    palette_shot = sim_cinematic.SHOTS["palette"]
    mock_look_at.assert_called_once_with(
        eye=list(palette_shot.eye),
        target=list(palette_shot.at),
        up=list(palette_shot.up),
    )


def test_main_dip_schedule_and_ink_metadata(monkeypatch, tmp_path: Path, capsys) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tools.supply.return_value = ("wet", "nighthawk_black")

    mock_env = MagicMock()
    mock_base = MagicMock()
    mock_env.unwrapped = mock_base
    mock_base.pad_height = None
    mock_base.pad_sheets = MagicMock()
    mock_base.surface = MagicMock()
    mock_base.cap_rims_np.return_value = None
    mock_base.ink_field.coverage.return_value = [0.15]
    mock_base.ink_policy.mode = "dip"
    mock_base.ink_episode_stats.return_value = {"dips": [2.0], "capacity": [1.0]}

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]
    import numpy as np
    mock_robot.get_qpos.return_value = MagicMock(
        clone=MagicMock(return_value=MagicMock()),
        __getitem__=MagicMock(return_value=np.array([0.0])),
    )
    mock_base.agent.robot = mock_robot

    mock_planning_mod = sys.modules["tatbot_sim.planning"]
    mock_planning_mod.plan_batch.reset_mock()

    fake_plan = MagicMock()
    fake_plan.preink = None
    fake_plan.dips = [[{"before_stroke": 0, "reason": "capacity", "slot": 1, "step": 10, "steps": 5}]]
    fake_plan.n_app = 20
    fake_plan.targets = np.zeros((1, 5, 3))
    fake_plan.pen_normals = np.zeros((1, 5, 3))
    fake_plan.q_raised = None
    fake_plan.tasks = ["test prompt"]
    fake_plan.episode_steps = 1
    fake_plan.surface_points = np.zeros((1, 3))
    fake_plan.surface_normals = np.zeros((1, 3))
    sys.modules["tatbot_sim.planning"].plan_batch.return_value = fake_plan

    fake_expert = MagicMock()
    fake_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    fake_expert.solve_pose.return_value = np.array([0.0])
    fake_expert.act.return_value = {}
    sys.modules["tatbot_sim.expert"].StrokeExpert.return_value = fake_expert

    monkeypatch.setattr(sim_cinematic.gym, "make", MagicMock(return_value=mock_env))

    # Mock sensor obs
    fake_sensor_data = {"cine_hero": {"rgb": MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8)))))}}
    mock_env.step.return_value = ({"sensor_data": fake_sensor_data}, 0, False, False, {})

    def fake_encode(frames, path, fps, crf):
        path.touch()

    monkeypatch.setattr(sim_cinematic, "encode", fake_encode)
    mock_pil = sys.modules["PIL"]
    mock_pil.Image.fromarray = MagicMock(return_value=MagicMock())

    args = sim_cinematic.Args(out=str(tmp_path), shots=("hero",), wet="nighthawk_black", look="bench", max_frames=1)
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    sim_cinematic.main(args, dist)

    mock_base.set_dip_schedule.assert_called_once_with(fake_plan)
    captured = capsys.readouterr().out
    assert "[cinematic] dip before stroke 0 (capacity) into 1 at step 30, 5 steps" in captured

    json_path = tmp_path / "skin-tattoo-language-s0.json"
    assert json_path.exists()
    import json
    meta = json.loads(json_path.read_text())
    assert meta["ink"]["wet"] == "nighthawk_black"
    assert meta["ink"]["mode"] == "dip"
    assert meta["ink"]["n_dips"] == 2.0
    assert meta["ink"]["dips"] == fake_plan.dips[0]


def test_sim_preview_args_prompt_phrases(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tool = MagicMock(tool_id="lutin-ballpoint-dot", prompt_phrase="using ballpoint pen")
    monkeypatch.setattr(mock_tools, "active_tool", MagicMock(return_value=mock_tool))

    args = sim_preview.Args()
    assert "using ballpoint pen" in args.task_name
    assert args.task_name == "draw a {size_mm}mm {shape} using ballpoint pen on the paper pad"
    assert "using ballpoint pen" in args.maze_task_name
    assert args.maze_task_name == "draw a continuous squiggle using ballpoint pen on the grid lines of the paper pad."


def test_sim_preview_main_execution(monkeypatch, tmp_path: Path) -> None:
    mock_env = MagicMock()
    mock_base = MagicMock()
    mock_env.unwrapped = mock_base
    mock_base.device = "cpu"
    mock_base.surface = MagicMock()
    mock_base.pad_sheets = MagicMock()
    mock_substrate = MagicMock()
    mock_substrate.name = "paper"
    mock_base.substrate = mock_substrate

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]
    import numpy as np

    mock_qpos = MagicMock()
    mock_qpos.clone.return_value = mock_qpos
    mock_qpos.__getitem__.return_value = np.zeros((1, 1))

    mock_robot.get_qpos.return_value = mock_qpos
    mock_base.agent.robot = mock_robot

    fake_mask = MagicMock()
    fake_mask.fraction = 0.8
    mock_expert_mod = sys.modules["tatbot_sim.expert"]
    mock_expert_mod.reachable_canvas_masks.return_value = [fake_mask]
    mock_expert_mod.reachable_height_ceiling.return_value = 0.05

    fake_expert = MagicMock()
    fake_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    fake_expert.solve_pose.return_value = np.array([[0.0]])
    fake_expert.act.return_value = {}
    mock_expert_mod.StrokeExpert.return_value = fake_expert

    fake_plan = MagicMock()
    fake_plan.preink = np.ones((1, 10, 10))
    fake_plan.n_app = 10
    fake_plan.targets = np.zeros((1, 5, 3))
    fake_plan.pen_normals = np.zeros((1, 5, 3))
    fake_plan.q_raised = np.array([[0.1]])
    fake_plan.tasks = ["draw circle using pen"]
    fake_plan.episode_steps = 2
    fake_plan.surface_points = np.zeros((1, 3))
    fake_plan.surface_normals = np.zeros((1, 3))

    mock_planning_mod = sys.modules["tatbot_sim.planning"]
    mock_planning_mod.plan_batch.return_value = fake_plan
    mock_planning_mod.plan_batch.reset_mock()

    monkeypatch.setattr(sim_preview.gym, "make", MagicMock(return_value=mock_env))

    # Mock observation for env.step
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_depth = np.ones((480, 640, 1), dtype=np.float32) * 100.0
    def make_mock_tensor(arr):
        t = MagicMock()
        t.cpu.return_value = MagicMock(numpy=MagicMock(return_value=np.stack([arr])))
        return t

    obs_sensor_data = {
        "thirdperson": {"rgb": make_mock_tensor(frame_rgb)},
        "topdown": {"rgb": make_mock_tensor(frame_rgb)},
        "wrist_upper": {
            "rgb": make_mock_tensor(frame_rgb),
            "depth": make_mock_tensor(frame_depth),
        },
        "wrist_lower": {"rgb": make_mock_tensor(frame_rgb)},
    }
    mock_env.step.return_value = ({"sensor_data": obs_sensor_data}, 0, False, False, {})

    mock_pil = sys.modules["PIL"]
    mock_im = MagicMock()
    mock_pil.Image.fromarray.return_value = mock_im

    args = sim_preview.Args(out=str(tmp_path), num_envs=1, clip_stride=1, depth=False)
    sim_preview.main(args)

    mock_base.preink.assert_called_once_with(fake_plan.preink)
    mock_planning_mod.plan_batch.assert_called_once()
    assert mock_planning_mod.plan_batch.call_args.kwargs["task_name"] == args.task_name
    assert mock_planning_mod.plan_batch.call_args.kwargs["maze_task_name"] == args.maze_task_name
    mock_env.close.assert_called_once()
