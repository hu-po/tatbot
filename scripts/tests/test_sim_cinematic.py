"""Tests for sim_cinematic.py CLI and supply resolution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sim_stubs import SimStubs, install_episode_stubs


class FakeTatbotWXAI:
    CAM_FOV = 0.96
    CAM_WIDTH = 640
    CAM_HEIGHT = 480


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


class FakeTatbotDrawEnv:
    _load_lighting = None
    _default_sensor_configs = None


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


def _build_stubs() -> SimStubs:
    stubs = SimStubs()
    stubs.module("gymnasium", make=MagicMock())
    stubs.module("sapien", Pose=MagicMock())
    stubs.module("tyro", cli=MagicMock())
    stubs.module("cv2", imwrite=MagicMock())
    stubs.module("PIL", Image=MagicMock())

    stubs.package("mani_skill")
    stubs.package("mani_skill.sensors")
    stubs.module("mani_skill.sensors.camera", CameraConfig=MagicMock())
    stubs.package("mani_skill.utils")
    stubs.module("mani_skill.utils.sapien_utils", look_at=MagicMock())

    stubs.package("tatbot_sim")
    install_episode_stubs(stubs)
    stubs.module("tatbot_sim.agent", TatbotWXAI=FakeTatbotWXAI)
    stubs.module(
        "tatbot_sim.distributions",
        DISTRIBUTIONS={
            "skin-tattoo": FakeDist("skin-tattoo", "lutin-3rl-bugpin"),
            "paper-draw": FakeDist("paper-draw", "lutin-ballpoint-dot"),
        },
    )
    stubs.module("tatbot_sim.env", TatbotDrawEnv=FakeTatbotDrawEnv)
    stubs.module("tatbot_sim.textures", TEX_DIR=Path("/tmp"))
    stubs.module("tatbot_sim.tasks", validate_task=MagicMock(), validate_supply=MagicMock())
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
    stubs.module(
        "tatbot_sim.expert",
        StrokeExpert=MagicMock(),
        reachable_canvas_masks=MagicMock(return_value=[MagicMock(fraction=0.8)]),
        reachable_height_ceiling=MagicMock(return_value=0.05),
    )
    stubs.module("tatbot_sim.planning", plan_batch=MagicMock())
    stubs.module("tatbot_sim.judge", strokes_from_plan_paths=lambda path: [MagicMock(points=np.asarray(stroke)) for stroke in path])
    stubs.module(
        "tatbot_sim.language",
        CLEAN_STYLE="clean",
        DEFAULT_STYLE="full",
        MOTIFS={"flower_of_life": None},
        SceneStyle=MagicMock(),
    )
    stubs.module("tatbot_sim.config", DRConfig=FakeDRConfig)
    stubs.module("tatbot_sim.depth_noise", DepthCorruptor=FakeDepthCorruptor, RGBJitter=FakeRGBJitter)
    # These presentation tests do not ask for a portable design.
    stubs.module(
        "tatbot_sim.design_scene",
        from_args=MagicMock(return_value=None),
        sampler=MagicMock(return_value=None),
    )
    def resolve_config(**kwargs):
        value = MagicMock()
        value.tool = MagicMock(tool_id=kwargs.get('tool_id'), prompt_phrase='using 3RL bugpin cartridge')
        value.substrate = substrate
        value.supply = kwargs.get('supply', ('wet', 'nighthawk_black'))
        value.cameras = (MagicMock(role='wrist_upper'),)
        value.metadata.return_value = {}
        value.with_dr.return_value = value
        return value
    stubs.module('tatbot_sim.resolved', resolve=MagicMock(side_effect=resolve_config))
    stubs.module('tatbot_sim.factory', select_tool=lambda dist: dist.tool_id)
    return stubs


REPO = Path(__file__).resolve().parents[2]
STUBS = _build_stubs()
sim_cinematic = STUBS.load_script("sim_cinematic", REPO / "scripts" / "sim_cinematic.py")
sim_stubs_installed = STUBS.fixture()


def test_main_supply_resolution_bench(monkeypatch) -> None:
    mock_tasks = sys.modules["tatbot_sim.tasks"]
    sys.modules["tatbot_sim.resolved"].resolve.reset_mock()
    mock_tasks.validate_supply.reset_mock()

    args = sim_cinematic.Args(out="/tmp/out", supply="bench", wet="", look="bench")
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    monkeypatch.setattr(sim_cinematic, "TatbotDrawEnv", MagicMock(side_effect=RuntimeError("stop_after_setup")))

    with pytest.raises(RuntimeError, match="stop_after_setup"):
        sim_cinematic.main(args, dist)

    assert sys.modules["tatbot_sim.resolved"].resolve.call_args.kwargs["supply"] == ("bench", "nighthawk_black")
    mock_tasks.validate_supply.assert_called_once()


def test_main_supply_resolution_wet_override(monkeypatch) -> None:
    mock_tasks = sys.modules["tatbot_sim.tasks"]
    sys.modules["tatbot_sim.resolved"].resolve.reset_mock()
    mock_tasks.validate_supply.reset_mock()

    args = sim_cinematic.Args(out="/tmp/out", supply="bench", wet="triple_black", look="bench")
    dist = sys.modules["tatbot_sim.distributions"].DISTRIBUTIONS["skin-tattoo"]

    monkeypatch.setattr(sim_cinematic, "TatbotDrawEnv", MagicMock(side_effect=RuntimeError("stop_after_setup")))

    with pytest.raises(RuntimeError, match="stop_after_setup"):
        sim_cinematic.main(args, dist)

    # When wet is set to "triple_black", set_supply should receive ("wet", "triple_black")
    assert sys.modules["tatbot_sim.resolved"].resolve.call_args.kwargs["supply"] == ("wet", "triple_black")
    mock_tasks.validate_supply.assert_called_once()


def test_cli_constructs_in_this_process(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['sim_cinematic.py', 'skin-tattoo', '--out', '/tmp/out'])
    monkeypatch.setenv('TATBOT_TOOL_ID', 'lutin-3rl-bugpin')
    main = MagicMock()
    monkeypatch.setattr(sim_cinematic, 'main', main)
    sim_cinematic.cli()
    main.assert_called_once()
    assert main.call_args.args[1].tool_id == 'lutin-3rl-bugpin'


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
    mock_base.pad_sheets = MagicMock()
    mock_base.surface = MagicMock()
    # (points, normals) for one canvas point: where the cameras get aimed.
    mock_base.surface.frame_np.return_value = (np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]))
    mock_base.cap_rims_np.return_value = None
    mock_base.ink_field.coverage.return_value = [0.15]
    mock_base.ink_policy.mode = "dip"
    mock_base.ink_episode_stats.return_value = {"dips": [2.0], "capacity": [1.0]}

    mock_robot = MagicMock()
    joint1 = MagicMock()
    joint1.name = "joint1"
    mock_robot.active_joints = [joint1]
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
    # One stroke on the canvas: the staged cameras are aimed at its centre.
    fake_plan.paths = [[[[0.0, 0.0], [0.01, 0.0]]]]
    sys.modules["tatbot_sim.planning"].plan_batch.return_value = fake_plan

    fake_expert = MagicMock()
    fake_expert.ik.chain.get_joint_parameter_names.return_value = ["joint1"]
    fake_expert.solve_pose.return_value = np.array([0.0])
    fake_expert.act.return_value = {}
    sys.modules["tatbot_sim.expert"].StrokeExpert.return_value = fake_expert

    monkeypatch.setattr(sim_cinematic, "TatbotDrawEnv", MagicMock(return_value=mock_env))

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

    mock_base.set_dip_schedule.assert_not_called()
    assert sys.modules["tatbot_sim.episode"].Episode.instances[-1].plan is fake_plan
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
