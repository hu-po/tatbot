from __future__ import annotations

from types import SimpleNamespace

from tatbot_sim.generate import _current_robot_and_joint_indices


class _Joint:
    def __init__(self, name: str) -> None:
        self.name = name


class _Robot:
    def __init__(self, names: list[str]) -> None:
        self.active_joints = [_Joint(name) for name in names]


def test_live_robot_is_resolved_again_after_agent_reconfiguration() -> None:
    names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
             "left_carriage_joint"]
    old_robot = _Robot(names)
    new_robot = _Robot(names)
    env = SimpleNamespace(agent=SimpleNamespace(robot=old_robot))

    first, idx7, idx_ik = _current_robot_and_joint_indices(env, names[:6])
    env.agent = SimpleNamespace(robot=new_robot)
    second, new_idx7, new_idx_ik = _current_robot_and_joint_indices(env, names[:6])

    assert first is old_robot
    assert second is new_robot
    assert idx7 == new_idx7 == list(range(7))
    assert idx_ik == new_idx_ik == list(range(6))
