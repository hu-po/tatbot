"""Tests for sim_preview.py CLI and batch preview rendering."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def _setup_mocks():
    mock_modules = {}

    # gymnasium
    if "gymnasium" not in sys.modules:
        mock_gym = types.ModuleType("gymnasium")
        mock_gym.make = MagicMock()
        mock_modules["gymnasium"] = mock_gym
    elif not hasattr(sys.modules["gymnasium"], "make"):
        sys.modules["gymnasium"].make = MagicMock()

    # torch
    if "torch" not in sys.modules:
        mock_torch = types.ModuleType("torch")
        mock_torch.as_tensor = MagicMock(side_effect=lambda x, **kwargs: x)

        class Tensor:
            pass

        mock_torch.Tensor = Tensor
        mock_modules["torch"] = mock_torch

    # tyro
    if "tyro" not in sys.modules:
        mock_tyro = types.ModuleType("tyro")
        mock_tyro.cli = MagicMock()
        mock_modules["tyro"] = mock_tyro

    # PIL
    if "PIL" not in sys.modules:
        mock_pil = types.ModuleType("PIL")
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image
        mock_modules["PIL"] = mock_pil
        mock_modules["PIL.Image"] = mock_pil_image

    # mani_skill
    class CameraConfig:
        def __init__(self, uid, pose, width, height, fov, near, far):
            self.uid = uid
            self.pose = pose
            self.width = width
            self.height = height
            self.fov = fov
            self.near = near
            self.far = far

    if "mani_skill" not in sys.modules:
        mock_ms = types.ModuleType("mani_skill")
        mock_ms_sensors = types.ModuleType("mani_skill.sensors")
        mock_ms_camera = types.ModuleType("mani_skill.sensors.camera")
        mock_ms_camera.CameraConfig = CameraConfig
        mock_ms_utils = types.ModuleType("mani_skill.utils")
        mock_ms_sapien = types.ModuleType("mani_skill.utils.sapien_utils")
        mock_ms_sapien.look_at = MagicMock()

        mock_modules["mani_skill"] = mock_ms
        mock_modules["mani_skill.sensors"] = mock_ms_sensors
        mock_modules["mani_skill.sensors.camera"] = mock_ms_camera
        mock_modules["mani_skill.utils"] = mock_ms_utils
        mock_modules["mani_skill.utils.sapien_utils"] = mock_ms_sapien
    else:
        sys.modules["mani_skill.sensors.camera"].CameraConfig = CameraConfig

    for mod_name, mod_obj in mock_modules.items():
        sys.modules[mod_name] = mod_obj

    # tatbot_sim and submodules
    if "tatbot_sim" in sys.modules:
        ts = sys.modules["tatbot_sim"]
    else:
        ts = types.ModuleType("tatbot_sim")
        ts.__path__ = []
        sys.modules["tatbot_sim"] = ts

    if "tatbot_sim.config" not in sys.modules:
        mock_ts_config = types.ModuleType("tatbot_sim.config")

        class FakeDRConfig:
            noise = MagicMock()
            pen_lean = MagicMock(max_off_base_rad=0.1)
            depth_noise = MagicMock()
            corrupt_depth = True
            rgb = MagicMock()

        mock_ts_config.DRConfig = FakeDRConfig
        sys.modules["tatbot_sim.config"] = mock_ts_config
        ts.config = mock_ts_config

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
        sys.modules["tatbot_sim.depth_noise"] = mock_ts_depth
        ts.depth_noise = mock_ts_depth

    if "tatbot_sim.env" not in sys.modules:
        mock_ts_env = types.ModuleType("tatbot_sim.env")

        class FakeTatbotDrawEnv:
            _default_sensor_configs = property(lambda self: [])

        mock_ts_env.TatbotDrawEnv = FakeTatbotDrawEnv
        sys.modules["tatbot_sim.env"] = mock_ts_env
        ts.env = mock_ts_env
    else:
        env_mod = sys.modules["tatbot_sim.env"]
        if not hasattr(env_mod, "TatbotDrawEnv") or not isinstance(
            getattr(env_mod.TatbotDrawEnv, "_default_sensor_configs", None), property
        ):
            class FakeTatbotDrawEnv:
                _default_sensor_configs = property(lambda self: [])

            env_mod.TatbotDrawEnv = FakeTatbotDrawEnv

    if "tatbot_sim.expert" not in sys.modules:
        mock_ts_expert = types.ModuleType("tatbot_sim.expert")
        mock_ts_expert.StrokeExpert = MagicMock()
        mock_ts_expert.reachable_canvas_masks = MagicMock(return_value=[MagicMock(fraction=0.8)])
        mock_ts_expert.reachable_height_ceiling = MagicMock(return_value=0.05)
        sys.modules["tatbot_sim.expert"] = mock_ts_expert
        ts.expert = mock_ts_expert
    else:
        sys.modules["tatbot_sim.expert"].reachable_canvas_masks = MagicMock(return_value=[MagicMock(fraction=0.8)])
        sys.modules["tatbot_sim.expert"].reachable_height_ceiling = MagicMock(return_value=0.05)

    if "tatbot_sim.planning" not in sys.modules:
        mock_ts_planning = types.ModuleType("tatbot_sim.planning")
        mock_ts_planning.plan_batch = MagicMock()
        sys.modules["tatbot_sim.planning"] = mock_ts_planning
        ts.planning = mock_ts_planning

    if "tatbot_sim.interaction" not in sys.modules:
        mock_interaction = types.ModuleType("tatbot_sim.interaction")
        mock_interaction.WORKING_OFFSET_M = 0.0
        sys.modules["tatbot_sim.interaction"] = mock_interaction
        ts.interaction = mock_interaction

    if "tatbot_sim.tools" not in sys.modules:
        mock_ts_tools = types.ModuleType("tatbot_sim.tools")
        mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
        mock_ts_tools.active_tool = MagicMock(return_value=mock_tool)
        mock_ts_substrate = MagicMock()
        mock_ts_substrate.name = "skin"
        mock_ts_tools.active_substrate = MagicMock(return_value=mock_ts_substrate)
        mock_ts_tools.set_supply = MagicMock()
        mock_ts_tools.supply = MagicMock(return_value=("wet", "nighthawk_black"))
        sys.modules["tatbot_sim.tools"] = mock_ts_tools
        ts.tools = mock_ts_tools
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


_setup_mocks()

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sim_preview", REPO / "scripts" / "sim_preview.py")
sim_preview = importlib.util.module_from_spec(spec)
sys.modules["sim_preview"] = sim_preview
spec.loader.exec_module(sim_preview)


def test_args_defaults_include_active_tool_phrase(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
    monkeypatch.setattr(mock_tools, "active_tool", MagicMock(return_value=mock_tool))

    args = sim_preview.Args()
    assert "using 3RL bugpin cartridge" in args.task_name
    assert "using 3RL bugpin cartridge" in args.maze_task_name


def test_pacing_and_clip_stride_calculation(tmp_path: Path, monkeypatch) -> None:
    mock_env = MagicMock()
    mock_base = MagicMock()
    mock_env.unwrapped = mock_base
    mock_base.device = "cpu"
    mock_base.substrate.name = "paper"
    mock_base.pad_sheets = MagicMock()
    mock_base.surface = MagicMock()

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]
    mock_qpos = MagicMock()
    mock_qpos.clone.return_value = mock_qpos
    mock_qpos.__getitem__.return_value = np.zeros((1, 1))
    mock_robot.get_qpos.return_value = mock_qpos
    mock_base.agent.robot = mock_robot

    mock_expert = MagicMock()
    mock_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    mock_expert.solve_pose.return_value = np.zeros((1, 1))
    mock_expert.act.return_value = {}
    sys.modules["tatbot_sim.expert"].StrokeExpert.return_value = mock_expert

    fake_plan = MagicMock()
    fake_plan.preink = None
    fake_plan.targets = np.zeros((1, 10, 3))
    fake_plan.pen_normals = np.zeros((1, 10, 3))
    fake_plan.q_raised = None
    fake_plan.n_app = 10
    fake_plan.surface_points = np.zeros((1, 3))
    fake_plan.surface_normals = np.zeros((1, 3))
    fake_plan.episode_steps = 10
    fake_plan.tasks = ["pacing task"]

    plan_batch_mock = sys.modules["tatbot_sim.planning"].plan_batch
    plan_batch_mock.reset_mock()
    plan_batch_mock.return_value = fake_plan

    monkeypatch.setattr(sim_preview.gym, "make", MagicMock(return_value=mock_env))

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    fake_obs = {
        "sensor_data": {
            "wrist_upper": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8)), "depth": FakeTensor(np.ones((1, 480, 640, 1), dtype=np.float32))},
            "wrist_lower": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
            "thirdperson": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
            "topdown": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
        }
    }
    mock_env.reset.return_value = (fake_obs, {})
    mock_env.step.return_value = (fake_obs, 0, False, False, {})

    durations = []

    class FakePILImage:
        def __init__(self, arr):
            self.arr = arr

        def save(self, path, **kwargs):
            if "duration" in kwargs:
                durations.append(kwargs["duration"])

    mock_pil = sys.modules["PIL"]
    mock_pil.Image.fromarray = lambda arr: FakePILImage(arr)

    # clip_stride = 4
    args = sim_preview.Args(
        out=str(tmp_path),
        num_envs=1,
        horizon=100,
        clip_stride=4,
    )

    sim_preview.main(args)

    # duration in webp should be int(1000 * clip_stride / 30) = int(1000 * 4 / 30) = 133 ms
    assert len(durations) > 0
    assert durations[0] == int(1000 * 4 / 30)


def test_wrist_orientation_and_normals_passed_to_expert(tmp_path: Path, monkeypatch) -> None:
    mock_env = MagicMock()
    mock_base = MagicMock()
    mock_env.unwrapped = mock_base
    mock_base.device = "cpu"
    mock_base.substrate.name = "paper"
    mock_base.pad_sheets = MagicMock()
    mock_base.surface = MagicMock()

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]
    mock_qpos = MagicMock()
    mock_qpos.clone.return_value = mock_qpos
    mock_qpos.__getitem__.return_value = np.zeros((1, 1))
    mock_robot.get_qpos.return_value = mock_qpos
    mock_base.agent.robot = mock_robot

    mock_expert = MagicMock()
    mock_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    mock_expert.solve_pose.return_value = np.zeros((1, 1))
    mock_expert.act.return_value = {}
    sys.modules["tatbot_sim.expert"].StrokeExpert.return_value = mock_expert

    expected_normals = np.ones((1, 5, 3))
    expected_surf_pts = np.ones((1, 3))
    expected_surf_normals = np.array([[0.0, 0.0, 1.0]])

    fake_plan = MagicMock()
    fake_plan.preink = None
    fake_plan.targets = np.zeros((1, 5, 3))
    fake_plan.pen_normals = expected_normals
    fake_plan.q_raised = np.zeros((1, 1))
    fake_plan.n_app = 10
    fake_plan.surface_points = expected_surf_pts
    fake_plan.surface_normals = expected_surf_normals
    fake_plan.episode_steps = 1
    fake_plan.tasks = ["wrist task"]

    plan_batch_mock = sys.modules["tatbot_sim.planning"].plan_batch
    plan_batch_mock.reset_mock()
    plan_batch_mock.return_value = fake_plan

    monkeypatch.setattr(sim_preview.gym, "make", MagicMock(return_value=mock_env))

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    fake_obs = {
        "sensor_data": {
            "wrist_upper": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8)), "depth": FakeTensor(np.ones((1, 480, 640, 1), dtype=np.float32))},
            "wrist_lower": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
            "thirdperson": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
            "topdown": {"rgb": FakeTensor(np.zeros((1, 480, 640, 3), dtype=np.uint8))},
        }
    }
    mock_env.reset.return_value = (fake_obs, {})
    mock_env.step.return_value = (fake_obs, 0, False, False, {})

    mock_pil = sys.modules["PIL"]
    mock_pil.Image.fromarray = lambda arr: MagicMock()

    args = sim_preview.Args(out=str(tmp_path), num_envs=1, horizon=100, clip_stride=1)
    sim_preview.main(args)

    # Check that expert.solve_pose was passed pen_normals[:, 0] for wrist pose solving
    mock_expert.solve_pose.assert_called_once()
    _, kwargs_solve = mock_expert.solve_pose.call_args
    np.testing.assert_array_equal(kwargs_solve["normals"], expected_normals[:, 0])

    # Check that expert.reset was called with pen_normals and surface floor_plane
    mock_expert.reset.assert_called_once()
    _, kwargs_reset = mock_expert.reset.call_args
    np.testing.assert_array_equal(kwargs_reset["pen_normals"], expected_normals)
    assert kwargs_reset["floor_plane"][0] is expected_surf_pts
    assert kwargs_reset["floor_plane"][1] is expected_surf_normals


def test_main_runs_simulation_and_generates_outputs(tmp_path: Path, monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
    monkeypatch.setattr(mock_tools, "active_tool", MagicMock(return_value=mock_tool))

    mock_env = MagicMock()
    mock_base = MagicMock()
    mock_env.unwrapped = mock_base
    mock_base.device = "cpu"
    mock_base.substrate.name = "paper"
    mock_base.pad_sheets = MagicMock()
    mock_base.surface = MagicMock()

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]

    mock_qpos = MagicMock()
    mock_qpos.clone.return_value = mock_qpos
    mock_qpos.__getitem__.return_value = np.zeros((2, 1))
    mock_robot.get_qpos.return_value = mock_qpos
    mock_base.agent.robot = mock_robot

    mock_expert = MagicMock()
    mock_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    mock_expert.solve_pose.return_value = np.zeros((2, 1))
    mock_expert.act.return_value = {}
    sys.modules["tatbot_sim.expert"].StrokeExpert.return_value = mock_expert

    fake_plan = MagicMock()
    fake_plan.preink = "some_preink_pattern"
    fake_plan.targets = np.zeros((2, 5, 3))
    fake_plan.pen_normals = np.zeros((2, 5, 3))
    fake_plan.q_raised = np.zeros((2, 1))
    fake_plan.n_app = 10
    fake_plan.surface_points = np.zeros((2, 3))
    fake_plan.surface_normals = np.zeros((2, 3))
    fake_plan.episode_steps = 2
    fake_plan.tasks = ["task 0", "task 1"]

    plan_batch_mock = sys.modules["tatbot_sim.planning"].plan_batch
    plan_batch_mock.reset_mock()
    plan_batch_mock.return_value = fake_plan

    monkeypatch.setattr(sim_preview.gym, "make", MagicMock(return_value=mock_env))

    # Mock step observations with Tensor-like or numpy object with .cpu().numpy()
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    fake_obs = {
        "sensor_data": {
            "wrist_upper": {
                "rgb": FakeTensor(np.zeros((2, 480, 640, 3), dtype=np.uint8)),
                "depth": FakeTensor(np.ones((2, 480, 640, 1), dtype=np.float32) * 100.0),
            },
            "wrist_lower": {
                "rgb": FakeTensor(np.zeros((2, 480, 640, 3), dtype=np.uint8)),
            },
            "thirdperson": {
                "rgb": FakeTensor(np.zeros((2, 480, 640, 3), dtype=np.uint8)),
            },
            "topdown": {
                "rgb": FakeTensor(np.zeros((2, 480, 640, 3), dtype=np.uint8)),
            },
        }
    }
    mock_env.reset.return_value = (fake_obs, {})
    mock_env.step.return_value = (fake_obs, 0, False, False, {})

    # Mock PIL Image save behavior
    saved_images = {}

    class FakePILImage:
        def __init__(self, arr):
            self.arr = arr

        def save(self, path, **kwargs):
            saved_images[str(path)] = kwargs

    mock_pil = sys.modules["PIL"]
    mock_pil.Image.fromarray = lambda arr: FakePILImage(arr)

    args = sim_preview.Args(
        out=str(tmp_path),
        num_envs=2,
        horizon=100,
        clip_stride=1,
    )

    sim_preview.main(args)

    # Verify base.preink was called with plan.preink
    mock_base.preink.assert_called_once_with("some_preink_pattern")

    # Verify plan_batch was called with active tool prompt phrase task names
    plan_batch_mock.assert_called_once()
    _, kwargs = plan_batch_mock.call_args
    assert "using 3RL bugpin cartridge" in kwargs["task_name"]
    assert "using 3RL bugpin cartridge" in kwargs["maze_task_name"]

    # Verify sensor config was patched with extra views (thirdperson, topdown)
    sensor_configs = sim_preview.TatbotDrawEnv._default_sensor_configs.fget(None)
    uids = [cfg.uid for cfg in sensor_configs]
    assert "thirdperson" in uids
    assert "topdown" in uids

    # Verify env was closed
    mock_env.close.assert_called_once()

    # Verify files were saved
    assert len(saved_images) > 0
