"""A selected video cadence must preserve the simulated episode and samples."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _plan(surface):
    from tatbot_sim.planning import BatchPlan

    # Two short contacts with a deliberate pen-up gap between them.
    uv = np.column_stack((np.linspace(-.004, .004, 30), np.zeros(30))).astype(np.float32)
    points, normals = surface.frame_np(0, uv)
    targets = points.copy()
    targets[10:20] += normals[10:20] * .003
    return BatchPlan(
        n_app=0, q_raised=None, draw_horizon=30, targets=targets[None],
        pen_normals=normals[None], surface_points=points[None], surface_normals=normals[None],
        lean_profiles=[np.zeros((30, 2))], kinds=['line'], tasks=['draw two short lines'],
        paths=[[uv[:10], uv[20:]]], programs=[None], lengths=np.array([30]),
    )


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_world_state_marks_and_sample_time_agree_across_capture_cadences():
    from tatbot_sim.backends.maniskill import ManiSkillWorld
    from tatbot_sim.config import DRConfig
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.episode import Episode
    from tatbot_sim.expert import StrokeExpert
    from tatbot_sim.observations import ObservationBuilder
    from tatbot_sim.resolved import resolve

    dr = DRConfig()
    dr.noise.prob = (0, 0)
    dr.noise.scale = (0, 0)
    config = resolve(tool_id='lutin-ballpoint-dot', seed=11, dr=dr)
    runs = []
    for stride in (1, 3):
        env = TatbotDrawEnv(config=config, num_envs=1, obs_mode='rgbd',
                           control_mode='pd_joint_pos', sim_backend='cpu', num_textures=1)
        expert = StrokeExpert(1, env.device, config=config, noise=dr.noise)
        world = ManiSkillWorld(env, config, expert)
        episode = Episode(world, ObservationBuilder(config, 1, env.device))
        try:
            episode.reset(seed=11)
            episode.install(_plan(env.surface))
            states, frames = [], {}
            while not episode.done:
                capture = (episode.step_index + 1) % stride == 0
                _, observation, _ = episode.step(capture=capture)
                states.append(observation.state.cpu().numpy())
                assert observation.time_s == episode.step_index / 30
                assert episode.measure() is observation
                if capture:
                    frames[episode.step_index] = (
                        observation.rgb['wrist_upper'].cpu().clone(),
                        observation.depth_mm['wrist_upper'].cpu().clone(),
                    )
            assert episode.step_index == 30
            with pytest.raises(RuntimeError, match='unfinished'):
                episode.step()
            runs.append((np.asarray(states), frames, env.ink_field.field.cpu().numpy().copy()))
        finally:
            episode.close()
    np.testing.assert_allclose(runs[0][0], runs[1][0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(runs[0][2], runs[1][2], atol=1e-6, rtol=0)
    assert runs[0][2].sum() > 0, 'the comparison must include actual deposited marks'
    for step, samples in runs[1][1].items():
        assert all(torch.equal(left, right) for left, right in zip(runs[0][1][step], samples, strict=True))


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_worker_uses_shared_placement_samples_and_completion(tmp_path, monkeypatch):
    from tatbot_sim import policy_worker

    def fixed_plan(rng, sheets, surface, **kwargs):
        plan = _plan(surface)
        plan.q_raised = np.asarray(kwargs['config'].staged_pose)
        plan.n_app = 10
        plan.lengths += 10
        return plan

    monkeypatch.setattr(policy_worker, 'plan_batch', fixed_plan)
    worker = policy_worker.PolicyEpisode(
        distribution='paper-draw', scenario_path=None, seed=11, output_dir=tmp_path / 'worker',
        surface_profile='flat', record_video=False, allow_expert_actions=True)
    try:
        assert worker.runtime.horizon == worker.horizon == 40
        assert worker.header()['time_s'] == 0
        assert worker.expert_actions.shape == (40, 7)
        arrays = worker.observation_arrays()
        assert set(arrays) == {'qpos', 'external_effort', 'wrist_upper_rgb', 'wrist_upper_depth'}
        assert not arrays['external_effort'].any()
        assert not any(worker.runtime.measure().effort_available)
        assert arrays['wrist_upper_depth'].dtype == np.uint16
        for action in worker.expert_actions:
            header, arrays = worker.step(action)
            np.testing.assert_array_equal(arrays['qpos'], worker.runtime.measure().qpos[0].cpu().numpy())
        assert header['done'] and header['time_s'] == 40 / 30
        assert worker.runtime.metadata()['world']['episode_seeds'] == (11,)
        with pytest.raises(RuntimeError, match='complete'):
            worker.step(worker.expert_actions[-1])
    finally:
        worker.close()


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_rebuilding_or_retaining_same_seed_scene_preserves_mounts_and_placement():
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.resolved import resolve

    config = resolve(tool_id='lutin-ballpoint-dot', seed=23)
    env = TatbotDrawEnv(config=config, num_envs=1, obs_mode='rgbd', sim_backend='cpu', num_textures=1)
    try:
        scenes = []
        for reconfigure in (False, True, False):
            env.reset(seed=23, options={'reconfigure': reconfigure})
            scenes.append((env.agent.camera_mounts.copy(), env.pad.pose.raw_pose.cpu().numpy().copy(),
                           env.get_obs()['sensor_data']['wrist_upper']['rgb'].cpu().clone()))
        for scene in scenes[1:]:
            assert scenes[0][0] == scene[0]
            np.testing.assert_allclose(scenes[0][1], scene[1], atol=1e-7, rtol=0)
            assert torch.equal(scenes[0][2], scene[2])
        env.reset(seed=24, options={'reconfigure': True})
        assert env.agent.camera_mounts != scenes[0][0]
    finally:
        env.close()
