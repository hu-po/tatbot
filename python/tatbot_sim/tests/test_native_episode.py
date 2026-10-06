"""Production commands run on their own clock through the shared episode."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tatbot_contracts.timing import SampleCadence


def test_sensor_sampling_keeps_every_command_and_bounds_capture_lateness():
    cadence = SampleCadence(400, 30)
    ticks = np.array([tick for tick in range(1, 40001) if cadence.due(tick)])
    assert len(ticks) == 3000
    np.testing.assert_array_equal(ticks[:3], [14, 27, 40])
    late = ticks / 400 - np.arange(1, len(ticks) + 1) / 30
    assert late.min() >= -1e-12 and late.max() < 1 / 400
    assert ticks[-1] == 40000
    assert all(SampleCadence(30, 30).due(tick) for tick in range(1, 100))


def test_session_velocity_references_reach_the_physics_controller():
    from tatbot_sim.backends.maniskill import ManiSkillWorld
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.expert import StrokeExpert
    from tatbot_sim.resolved import Timing, resolve

    config = resolve(tool_id='lutin-ballpoint-dot', seed=11,
                     timing=Timing(physics_hz=1200, control_hz=400))
    env = TatbotDrawEnv(config=config, num_envs=1, obs_mode='state',
                       control_mode='pd_joint_pos_vel', sim_backend='cpu', num_textures=1)
    world = ManiSkillWorld(env, config, StrokeExpert(1, env.device, config=config))
    try:
        world.reset(seed=11)
        positions = world.robot.get_qpos()[:, world.idx7].clone()
        velocities = torch.tensor([[.012, -.013, .014, -.015, .016, -.017, .001]])
        world.advance_joints(positions, velocities, capture=False)
        controller = env.agent.controller.controllers['arm']
        torch.testing.assert_close(controller._target_qpos, positions, rtol=0, atol=0)
        torch.testing.assert_close(controller._target_qvel, velocities, rtol=0, atol=0)
        assert torch.isfinite(world.robot.get_qvel()).all()
        assert world.metadata()['command_channels'] == ['position', 'velocity']
        before = env._step_count
        with pytest.raises(ValueError, match='finite follower'):
            world.advance_joints(positions, torch.full_like(positions, float('nan')))
        assert env._step_count == before
    finally:
        world.close()


def _short_intent(surface):
    from tatbot_sim.planning import BatchPlan

    paths = [np.array([[-.001, 0], [-.0005, 0]]), np.array([[.0005, 0], [.001, 0]])]
    points, normals = surface.frame_np(0, np.concatenate(paths))
    return BatchPlan(0, None, 4, points[None], normals[None], points[None], normals[None],
                     [np.zeros((4, 2))], ['line'], ['draw two short lines'],
                     [[path.tolist() for path in paths]], [None], np.array([4]))


@pytest.fixture
def nominal_ballpoint_workspace(monkeypatch):
    from tatbot_sim import tools

    workspace = tools.workspace()
    side = {**workspace['right'], 'tool_id': 'lutin-ballpoint-dot'}
    # A preview of the ballpoint cannot borrow the fitted tool's touch-off.
    for axis in 'xyz':
        side[f'pen_tip_offset_{axis}'] = None
    monkeypatch.setattr(tools, 'workspace', lambda: {**workspace, 'right': side})


def test_production_config_keeps_strict_resolved_tcp_parity(tmp_path, monkeypatch):
    from dataclasses import replace

    from tatbot_sim import tools
    from tatbot_sim.production_plan import read_config
    from tatbot_sim.resolved import resolve

    live = tools.workspace
    # production-flat models the ballpoint, with its tip the measured one as when it is the fitted tool
    monkeypatch.setattr(tools, 'workspace', lambda: {**live(), 'right': {**live()['right'], 'tool_id': 'lutin-ballpoint-dot'}})
    config = resolve(tool_id='lutin-ballpoint-dot', seed=11)
    settings = {'schema': 'tatbot.draw-config/1', 'tool': config.tool.tool_id, 'draw_speed_mm_s': 3.5}
    path = tmp_path/'draw.json'
    path.write_text(json.dumps(settings))
    assert read_config(str(path), config) == settings
    changed = replace(config.geometry, tcp_offset_m=tuple(np.asarray(config.geometry.tcp_offset_m) + [1e-6, 0., 0.]))
    with pytest.raises(ValueError, match='TCP differs'):
        read_config(str(path), replace(config, geometry=changed))


@pytest.mark.slow
@pytest.mark.timeout(240)
def test_native_reference_runs_without_synthetic_resolve(tmp_path, monkeypatch, nominal_ballpoint_workspace):
    from tatbot_sim.backends.maniskill import ManiSkillWorld
    from tatbot_sim.config import DRConfig
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.episode import Episode
    from tatbot_sim.expert import StrokeExpert
    from tatbot_sim.observations import ObservationBuilder
    from tatbot_sim.production_plan import from_intent, read_config
    from tatbot_sim.resolved import Timing, resolve

    dr = DRConfig()
    dr.noise.prob = (0, 0)
    dr.noise.scale = (0, 0)
    config = resolve(tool_id='lutin-ballpoint-dot', seed=11, dr=dr,
                     timing=Timing(physics_hz=1200, control_hz=400))
    assert not config.geometry.measured
    settings = {'schema': 'tatbot.draw-config/1', 'tool': config.tool.tool_id,
                'draw_speed_mm_s': 3.5, 'ease_s': .05, 'path': {'approach_mm': 8}}
    path = tmp_path / 'draw.json'
    path.write_text(json.dumps(settings))
    settings = read_config(str(path), config)
    env = TatbotDrawEnv(config=config, num_envs=1, obs_mode='rgbd',
                       control_mode='pd_joint_pos', sim_backend='cpu', num_textures=1)
    expert = StrokeExpert(1, env.device, config=config)
    world = ManiSkillWorld(env, config, expert)
    episode = Episode(world, ObservationBuilder(config, 1, env.device))
    try:
        episode.reset(seed=11)
        intent = _short_intent(env.surface)
        plan = from_intent(intent, world, settings, tmp_path / 'references', max_ticks=20000)
        assert plan.native_reference is not None
        assert len(plan.native_reference.provenance[0]['chunks']) == 2
        tool = plan.native_reference.provenance[0]['tool_model']
        assert tool['hardware_authority'] is False
        assert tool['tip_source'] == ('touch-off' if config.geometry.measured else 'datasheet nominal')
        for chunk in plan.native_reference.provenance[0]['chunks']:
            native = json.loads((tmp_path/'references/env-0'/chunk['plan_file']).read_text())
            assert native['tool_model']['source'] == 'bound-input'
            np.testing.assert_allclose(native['tool_model']['tip_in_link6'], tool['tip_in_link6'], atol=1e-12, rtol=0)
        source = plan.native_reference.positions.copy()

        def no_resolve(*args, **kwargs):
            raise AssertionError('native reference must not be solved or refined again')

        monkeypatch.setattr(expert, 'solve_pose', no_resolve)
        monkeypatch.setattr(expert, 'reset', no_resolve)
        episode.install(plan)
        cadence = SampleCadence(400, 30)
        ticks = []
        while not episode.done:
            capture = cadence.due(episode.step_index + 1)
            action, observation, _ = episode.step(capture=capture)
            torch.testing.assert_close(action.cpu(), torch.tensor(source[:, episode.step_index - 1], dtype=torch.float32),
                                       rtol=0, atol=0)
            assert observation.time_s == episode.step_index / 400
            assert bool(observation.rgb) == capture
            if capture:
                ticks.append(episode.step_index)
                assert set(observation.rgb) == {'wrist_upper'}
        assert len(ticks) == episode.horizon * 30 // 400
        assert episode.metadata()['motion']['reference_modified'] is False
        assert episode.metadata()['motion']['planner'] == 'production-cartesian-cpp'
        assert episode.step_index == len(source[0])
        assert world.coverage().item() > 0
        np.testing.assert_array_equal(plan.native_reference.positions, source)
    finally:
        episode.close()


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_factory_exports_native_actions_at_recorded_capture_ticks(tmp_path, monkeypatch, nominal_ballpoint_workspace):
    import pandas as pd
    from tatbot_sim import generate
    from tatbot_sim.distributions import DISTRIBUTIONS

    settings = {'schema': 'tatbot.draw-config/1', 'tool': 'lutin-ballpoint-dot',
                'draw_speed_mm_s': 3.5, 'ease_s': .05, 'path': {'approach_mm': 8}}
    path = tmp_path / 'draw.json'
    path.write_text(json.dumps(settings))
    args = DISTRIBUTIONS['paper-draw'].build_args()
    args.out_dir = str(tmp_path / 'dataset')
    args.num_episodes = args.num_envs = 1
    args.seed = 11
    args.task = 'maze'
    args.maze_horizon = 900
    args.sim_backend = 'cpu'
    args.tool_calibration_jitter = False
    args.production_draw_config = str(path)
    args.dr.latency.obs_delay_steps = (0, 0)
    args.save_privileged_labels = True
    args.texture_refresh_steps = 1
    monkeypatch.setattr(generate, 'plan_batch', lambda rng, sheets, surface, **kwargs: _short_intent(surface))
    generate.main(args)
    root = tmp_path / 'dataset'
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['fps'] == 30 and info['total_episodes'] == 1
    metadata = json.loads((root / 'meta/run_meta.json').read_text())
    episode = metadata['episodes'][0]
    assert episode['batch_runtime']['world']['backend'] == 'maniskill'
    motion = episode['batch_runtime']['motion']
    assert motion['planner'] == 'production-cartesian-cpp' and motion['period_s'] == .0025
    source = motion['episodes'][0]
    native = np.concatenate([json.loads((root / source['reference_directory'] / chunk['plan_file']).read_text())['positions']
                             for chunk in source['chunks']])
    ticks = np.array(episode['capture_control_ticks'])
    assert len(ticks) == len(native) * 30 // 400
    data = pd.concat([pd.read_parquet(file) for file in sorted((root / 'data').rglob('*.parquet'))])
    np.testing.assert_array_equal(np.stack(data.action), native[ticks - 1].astype(np.float32))
    np.testing.assert_allclose(data.timestamp, np.arange(len(ticks)) / 30, atol=1e-6, rtol=0)
    assert episode['batch_runtime']['time_s'] == len(native) / 400
    assert episode['steps_executed'] == len(native) and episode['frames_recorded'] == len(ticks)
    assert episode['ink_coverage_end'] > episode['ink_coverage_start']
    from tatbot_sim.temporal_labels import audit_timeline
    timeline = json.loads((root / 'meta/privileged/episode_000000.json').read_text())
    assert timeline['steps'] == len(ticks)
    problems = audit_timeline(root / 'meta/privileged/episode_000000.npz', timeline)
    # Position-controller lag can leave measured contact during a commanded
    # lift. Preserve that quality finding; it is not a malformed timeline or
    # permission to relabel measured contact as pen-up.
    assert all(problem.endswith('pen-down frame has no intended target') for problem in problems)
    with np.load(root / 'meta/privileged/episode_000000.npz') as labels:
        assert bool(problems) == bool(np.any(labels['pen_down'] & ~labels['target_valid']))


@pytest.mark.parametrize('carriage', [-.007, .035])
def test_planned_carriage_setup_refuses_invalid_measured_rest(carriage):
    from tatbot_sim.production_plan import planned_drawing_start

    with pytest.raises(ValueError, match='pen-up envelope'):
        planned_drawing_start({'carriage_m': carriage})
