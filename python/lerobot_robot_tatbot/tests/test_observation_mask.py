from __future__ import annotations

import pytest
from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
from lerobot_robot_tatbot.tatbot_follower import TatbotFollower
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower


def _config(**overrides) -> TatbotFollowerConfig:
    values = {
        "id": "masked-observation-test",
        "use_tatbot_yaml": False,
        "use_tool_registry": False,
        "coordinated_arms": False,
        "tuning_enabled": False,
        "estop_device": "",
        "estop_required": False,
    }
    values.update(overrides)
    return TatbotFollowerConfig(**values)


def test_masked_external_effort_preserves_width_and_zeroes_only_policy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(include_external_effort=True, mask_external_effort=True)
    source = {}
    for index, joint in enumerate(config.joint_names):
        source[f"{joint}.pos"] = float(index) / 10
        source[f"{joint}.ext_eff"] = float(index + 1)
    monkeypatch.setattr(
        WidowXAIFollower,
        "get_observation",
        lambda _self: dict(source),
    )

    observation = TatbotFollower(config).get_observation()

    assert [key for key in observation if key.endswith(".ext_eff")] == [
        f"{joint}.ext_eff" for joint in config.joint_names
    ]
    assert all(observation[f"{joint}.ext_eff"] == 0.0 for joint in config.joint_names)
    assert [observation[f"{joint}.pos"] for joint in config.joint_names] == [
        float(index) / 10 for index in range(7)
    ]


def test_mask_requires_external_effort_features() -> None:
    with pytest.raises(ValueError, match="state remains 14-wide"):
        _config(include_external_effort=False, mask_external_effort=True)
