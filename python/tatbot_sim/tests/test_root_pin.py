"""The GPU backend must not let the fixed robot base creep.

2026-09-03: an idle arm's root drifted 3.1 mm in +x/+z over 30 s on the PhysX
GPU pipeline (quadratic in time, zero on the CPU backend). The tool rode
that drift out of the 0.5 mm contact band on every later stroke, so paper
and silicone episodes lost their second stroke while the joint state said
the pen was on the sheet. The env re-pins the root every control step; this
holds it to that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the drift only exists on the GPU backend")
def test_idle_arm_base_does_not_drift_on_gpu(monkeypatch):
    monkeypatch.setenv("TATBOT_TOOL_ID", "lutin-ballpoint-dot")
    from tatbot_sim.config import DRConfig
    from tatbot_sim.env import TatbotDrawEnv

    env = TatbotDrawEnv( num_envs=1, obs_mode="rgb", control_mode="pd_joint_pos",
        sim_backend="gpu", dr=DRConfig(),
    )
    try:
        env.reset(seed=0)
        base = env.unwrapped
        robot = base.agent.robot
        action = robot.get_qpos()[:, :7].clone()
        start = robot.pose.p.clone()
        for _ in range(450):
            env.step(action)
        base.scene._gpu_fetch_all()
        drift_mm = float(torch.linalg.norm(robot.pose.p - start)) * 1000.0
    finally:
        env.close()
    assert drift_mm < 0.02, f"root drifted {drift_mm:.3f} mm in 15 s of idling"
    assert np.isfinite(drift_mm)
