"""Sequential backend construction and rendering with explicit tool selection."""
import pytest
from tatbot_sim.resolved import resolve


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_two_tools_construct_and_render_in_one_process(monkeypatch):
    import numpy as np
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.expert import StrokeExpert

    # A conflicting default remains in place throughout explicit construction.
    monkeypatch.setenv('TATBOT_TOOL_ID', 'lutin-3rl-bugpin')
    models = []
    for tool, substrate in [('lutin-ballpoint-dot', 'paper_pad'), ('picosecond-laser-pen', 'silicon_skin')]:
        config = resolve(tool_id=tool, seed=7)
        world = TatbotDrawEnv(config=config, num_envs=1, obs_mode='rgbd',
                             control_mode='pd_joint_pos', sim_backend='cpu', num_textures=1)
        try:
            obs, _ = world.reset(seed=7)
            expert = StrokeExpert(1, world.device, config=config)
            assert world.tool.tool_id == tool
            assert world.substrate.name == substrate
            assert expert.ik.urdf_path == world.agent.urdf_path
            models.append(world.agent.urdf_path)
            assert set(obs['sensor_data']) == {'wrist_upper'}
            image = obs['sensor_data']['wrist_upper']['rgb'].cpu().numpy()
            assert image.shape == (1, 480, 640, 3)
            assert image.std() > 1
            names = [joint.name for joint in world.agent.robot.active_joints]
            indices = [names.index(name) for name in expert.ik.chain.get_joint_parameter_names()]
            q = world.agent.robot.get_qpos()[:, indices]
            np.testing.assert_allclose(expert.ik.fk(q)[:, :3, 3].cpu(),
                                       world.agent.tcp.pose.p.cpu(), atol=1e-5)
        finally:
            world.close()
    assert models[0] != models[1]

