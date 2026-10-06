"""The production executor's offline joint planner, without an engine import.

Simulation reference consumers call the same C++ parser/planner
(path_plan_check) that the ROS CLIK parity test checks its port against. Its existing tatbot.joint-plan/1 output remains authoritative;
this adapter neither solves IK nor grants hardware authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pen_path as dp
import stroke_operation as operation
from fleet_release import planner_binary

from tatbot_sim.repo import repo_root


@dataclass(frozen=True)
class ReferenceBatch:
    """Native commands in canonical follower order, including the carriage.

    Each row is one controller tick. Shorter batch members hold their last
    command; their natural lengths remain in the enclosing episode plan.
    """

    seed: np.ndarray
    positions: np.ndarray
    period_s: float
    provenance: tuple[dict, ...]

    def __post_init__(self):
        seed = np.array(self.seed, dtype=np.float64, copy=True)
        positions = np.array(self.positions, dtype=np.float64, copy=True)
        if (positions.ndim != 3 or positions.shape[2] != 7 or positions.shape[1] == 0
                or seed.shape != (positions.shape[0], 7)
                or not np.isfinite(seed).all() or not np.isfinite(positions).all()
                or self.period_s != dp.C.period_s
                or len(self.provenance) != positions.shape[0]):
            raise ValueError('invalid native joint-reference batch')
        seed.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'positions', positions)

    def metadata(self):
        return {'planner': 'production-cartesian-cpp', 'schema': 'tatbot.joint-plan/1',
                'period_s': self.period_s, 'constants_sha': dp.SHA,
                'reference_modified': False, 'hardware_authority': False,
                'action_noise_applied': False, 'command_dtype': 'float32',
                'episodes': list(self.provenance)}


def compile_joint_plan(samples: Path, seed: np.ndarray, period: float, *, model) -> dict:
    model.assert_cpp_wxai_compatible()
    binary = planner_binary(repo_root())
    if not binary.is_file():
        raise ValueError("offline executor planner is not built; no skipped preflight is accepted")
    return operation.executor_plan(binary, samples, seed, period, dp.SHA,
                                   arm_prefix=model.prefix, tool_tip_in_link6=model.tcp_in_link6()).plan
