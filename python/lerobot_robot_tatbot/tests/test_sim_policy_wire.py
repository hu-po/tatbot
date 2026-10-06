from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.async_inference.helpers import TimedAction, map_robot_keys_to_lerobot_features

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from wire_client import (  # noqa: E402
    SCENARIOS,
    ActionQueue,
    ExecutionFilter,
    build_feature_robot,
    observation_from_worker,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "qpos": np.arange(7, dtype=np.float32),
        "external_effort": np.arange(7, dtype=np.float32) + 10,
        "wrist_upper_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_lower_rgb": np.ones((480, 640, 3), dtype=np.uint8),
        "wrist_upper_depth": np.full((480, 640), 150, dtype=np.uint16),
        "wrist_lower_depth": np.full((480, 640), 160, dtype=np.uint16),
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_feature_only_follower_maps_worker_observation_without_hardware(name, monkeypatch):
    import wrist_cameras

    def forbid_registry(*args, **kwargs):
        raise AssertionError("feature declarations must not require physical camera identities")

    monkeypatch.setattr(wrist_cameras, "for_arm", forbid_registry)
    scenario = SCENARIOS[name]
    robot = build_feature_robot(scenario)
    observation = observation_from_worker(robot, _arrays(), "draw a generated design")
    assert set(observation) == set(robot.observation_features) | {"task"}
    assert set(map_robot_keys_to_lerobot_features(robot))
    assert observation["joint_3.pos"] == 3.0
    if scenario.external_effort:
        expected = 0.0 if scenario.mask_external_effort else 13.0
        assert observation["joint_3.ext_eff"] == expected
    for key, shape in robot.observation_features.items():
        if isinstance(shape, tuple):
            assert observation[key].shape == shape


def _timed(timestep: int, value: float) -> TimedAction:
    return TimedAction(timestamp=0.0, timestep=timestep, action=torch.full((7,), value))


def test_action_queue_matches_rollout_weighted_overlap_and_drops_the_past():
    queue = ActionQueue()
    assert queue.merge([_timed(0, 1), _timed(1, 1)]) == {
        "accepted": 2, "skipped_old": 0, "overlapped": 0,
    }
    _, first = queue.pop()
    np.testing.assert_allclose(first, 1)
    result = queue.merge([_timed(0, 9), _timed(1, 2), _timed(2, 2)])
    assert result == {"accepted": 2, "skipped_old": 1, "overlapped": 1}
    _, second = queue.pop()
    np.testing.assert_allclose(second, 1.7)


def test_execution_filter_holds_carriage_and_bounds_every_arm_step():
    filt = ExecutionFilter(
        np.zeros(7, dtype=np.float32),
        fps=30,
        target_filter_tau_s=0.3,
        max_joint_velocity_rad_s=0.25,
    )
    previous = np.zeros(7, dtype=np.float32)
    for _ in range(20):
        sent = filt.apply(np.ones(7, dtype=np.float32))
        assert np.max(np.abs(sent[:6] - previous[:6])) <= 0.25 / 30 + 1e-6
        assert sent[6] == 0.0
        previous = sent
    assert filt.stats()["saturated_fraction"] > 0
