"""Tests for sim_preview.py CLI and batch preview rendering."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from sim_stubs import SimStubs, install_episode_stubs


class CameraConfig:
    def __init__(self, uid, pose, width, height, fov, near, far):
        self.uid = uid
        self.pose = pose
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far


class FakeDRConfig:
    noise = MagicMock()
    pen_lean = MagicMock(max_off_base_rad=0.1)
    depth_noise = MagicMock()
    corrupt_depth = True
    rgb = MagicMock()


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


class FakeTatbotDrawEnv:
    _default_sensor_configs = property(lambda self: [])


def _build_stubs() -> SimStubs:
    stubs = SimStubs()
    stubs.module("gymnasium", make=MagicMock())
    stubs.module("tyro", cli=MagicMock())
    stubs.module("PIL", Image=MagicMock())

    stubs.package("mani_skill")
    stubs.package("mani_skill.sensors")
    stubs.module("mani_skill.sensors.camera", CameraConfig=CameraConfig)
    stubs.package("mani_skill.utils")
    stubs.module("mani_skill.utils.sapien_utils", look_at=MagicMock())

    stubs.package("tatbot_sim")
    install_episode_stubs(stubs)
    stubs.module("tatbot_sim.config", DRConfig=FakeDRConfig)
    stubs.module("tatbot_sim.depth_noise", DepthCorruptor=FakeDepthCorruptor, RGBJitter=FakeRGBJitter)
    stubs.module("tatbot_sim.env", TatbotDrawEnv=FakeTatbotDrawEnv)
    stubs.module(
        "tatbot_sim.expert",
        StrokeExpert=MagicMock(),
        reachable_canvas_masks=MagicMock(return_value=[MagicMock(fraction=0.8)]),
        reachable_height_ceiling=MagicMock(return_value=0.05),
    )
    stubs.module("tatbot_sim.planning", plan_batch=MagicMock())
    stubs.module("tatbot_sim.interaction", WORKING_OFFSET_M=0.0)
    substrate = MagicMock()
    substrate.name = "skin"
    stubs.module(
        "tatbot_sim.tools",
        active_tool=MagicMock(
            return_value=MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
        ),
        active_substrate=MagicMock(return_value=substrate),
        set_supply=MagicMock(),
        supply=MagicMock(return_value=("wet", "nighthawk_black")),
    )
    # No design asked for, no sampler handed to the planner.
    stubs.module(
        "tatbot_sim.design_scene",
        from_args=MagicMock(return_value=None),
        sampler=MagicMock(return_value=None),
    )
    def resolve_config(**kwargs):
        if kwargs.get('supply', ('bench',))[0] == 'invalid':
            raise ValueError("unknown supply kind 'invalid'")
        value = MagicMock()
        value.tool = MagicMock(tool_id=kwargs.get('tool_id'), prompt_phrase='using 3RL bugpin cartridge')
        value.substrate = substrate
        value.supply = kwargs.get('supply', ('wet', 'nighthawk_black'))
        value.cameras = (MagicMock(role='wrist_upper'),)
        value.metadata.return_value = {}
        value.with_dr.return_value = value
        return value
    stubs.module('tatbot_sim.resolved', resolve=MagicMock(side_effect=resolve_config))
    return stubs


REPO = Path(__file__).resolve().parents[2]
STUBS = _build_stubs()
sim_preview = STUBS.load_script("sim_preview", REPO / "scripts" / "sim_preview.py")
sim_stubs_installed = STUBS.fixture()


def test_args_defaults_include_active_tool_phrase(monkeypatch) -> None:
    mock_tools = sys.modules["tatbot_sim.tools"]
    mock_tool = MagicMock(tool_id="lutin-3rl-bugpin", prompt_phrase="using 3RL bugpin cartridge")
    monkeypatch.setattr(mock_tools, "active_tool", MagicMock(return_value=mock_tool))

    args = sim_preview.Args()
    assert "{tool}" in args.task_name
    assert "{tool}" in args.maze_task_name


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
    mock_base.agent.camera_descriptions = [MagicMock(role="wrist_upper")]

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

    monkeypatch.setattr(sim_preview, "TatbotDrawEnv", MagicMock(return_value=mock_env))

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


def test_planned_orientation_and_floor_are_handed_to_runtime(tmp_path: Path, monkeypatch) -> None:
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
    mock_base.agent.camera_descriptions = [MagicMock(role="wrist_upper")]

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

    monkeypatch.setattr(sim_preview, "TatbotDrawEnv", MagicMock(return_value=mock_env))

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

    # Presentation passes the planned scene to the runtime and owns no IK setup.
    mock_expert.solve_pose.assert_not_called()
    mock_expert.reset.assert_not_called()
    assert sys.modules["tatbot_sim.episode"].Episode.instances[-1].plan is fake_plan


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
    mock_base.agent.camera_descriptions = [MagicMock(role="wrist_upper")]

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

    monkeypatch.setattr(sim_preview, "TatbotDrawEnv", MagicMock(return_value=mock_env))

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

    # Material initialization belongs to the shared runtime.
    assert sys.modules["tatbot_sim.episode"].Episode.instances[-1].plan is fake_plan
    mock_base.preink.assert_not_called()

    # Verify plan_batch was called with active tool prompt phrase task names
    plan_batch_mock.assert_called_once()
    _, kwargs = plan_batch_mock.call_args
    assert "{tool}" in kwargs["task_name"]
    assert "{tool}" in kwargs["maze_task_name"]

    # Inspection views are explicit instance options.
    sensor_configs = sim_preview.TatbotDrawEnv.call_args.kwargs["presentation_cameras"](None)
    uids = [cfg.uid for cfg in sensor_configs]
    assert "thirdperson" in uids
    assert "topdown" in uids

    # Verify env was closed
    mock_env.close.assert_called_once()

    # Verify files were saved
    assert len(saved_images) > 0
