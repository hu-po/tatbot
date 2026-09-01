"""The generate pre-flight's reach probe must not be basin-bound.

2026-08-31: the fixed-EE staged pose (wrist +pi/2) seeded the probe's single
DLS solve into a bad basin at scattered pad targets — 73-152 mm "residuals"
at points that solve to 0.0 mm from a nudged seed — and the factory refused
paper-draw and skin-tattoo outright. These pin the retry behaviour so the
gate can never again mistake a solver basin for the arm's workspace.
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.expert import StrokeExpert, reach_residual_at
from tatbot_sim.tools import carriage_rest_m, staged_pose


class _BasinTrapIK:
    """Pretends the exact rest seed converges 100 mm short; any other seed
    lands on the target. ``q`` doubles as the EE position: fk is identity."""

    def __init__(self, rest: torch.Tensor):
        self.rest = rest
        self.solves = 0

    def step(self, q, target_pos, target_rot, iters):
        self.solves += 1
        if torch.equal(q, self.rest):
            return target_pos + torch.tensor([[0.1, 0.0, 0.0]])
        return target_pos.clone()

    def fk(self, q):
        mat = torch.eye(4).repeat(len(q), 1, 1)
        mat[:, :3, 3] = q
        return mat


class _StubExpert:
    device = "cpu"

    def __init__(self):
        self.rest = torch.zeros((1, 3))
        self.ik = _BasinTrapIK(self.rest)

    def target_rotations(self, normals, n):
        return None


def test_a_bad_first_basin_does_not_fail_the_gate():
    ex = _StubExpert()
    res = reach_residual_at(ex, ex.rest, np.array([0.29, 0.0]), 0.0, 0.004)
    assert res < 1e-6, "a perturbed retry solves this target; the gate must see it"
    assert ex.ik.solves > 2, "the retry path never ran"

    ex = _StubExpert()
    res = reach_residual_at(ex, ex.rest, np.array([0.29, 0.0]), 0.0, 0.004, retries=0)
    assert res > 0.09, "with retries off the trap seed must miss — else this stub tests nothing"


def test_the_probe_verdict_is_deterministic():
    a = reach_residual_at(_StubExpert(), torch.zeros((1, 3)), np.array([0.29, 0.0]), 0.0, 0.004)
    b = reach_residual_at(_StubExpert(), torch.zeros((1, 3)), np.array([0.29, 0.0]), 0.0, 0.004)
    assert a == b


def test_the_fitted_tool_reaches_the_pad_centre_from_the_staged_pose():
    """The exact corner the factory refused: PAD_CENTER, pad top on the table,
    seeded from the staged pose. Runs on whichever tool is fitted; before the
    retry fix this failed by 73 mm (ballpoint) / 76 mm (3RL) and passed only
    for the laser, whose longer body happens to keep the solve in a good basin.

    The hard assertion holds only for a tool welded at its datasheet nominal.
    A tool carrying its own measured touch-off is hardware truth — the 8/31
    ballpoint seat leans ~9 deg and genuinely cannot hold its tip axis
    vertical over the default envelope (5.2 mm here, multi-seed) — and a test
    must not fail the suite for reporting the bench honestly. That case skips
    with the number, and the factory pre-flight remains the operative gate.
    """
    import pytest
    from tatbot_sim.tools import active_tool, registry
    from tatbot_sim.urdf import REPO

    ex = StrokeExpert(1, "cpu", noise=None, seed=0)
    names = ex.ik.chain.get_joint_parameter_names()
    staged = dict(zip([f"joint_{i}" for i in range(6)], staged_pose()[:6], strict=True))
    q_rest = torch.tensor(
        [[staged.get(n, carriage_rest_m()) for n in names]], dtype=torch.float32
    )
    res = reach_residual_at(ex, q_rest, np.array([0.29, 0.0]), 0.0, 0.004)
    reg, spec = registry(), active_tool()
    ws = reg.read_workspace(REPO)
    own_touchoff = (reg.active_tool_id(REPO, "right", ws) == spec.tool_id
                    and reg.tip_offset_m(ws, "right") is not None)
    if own_touchoff and res >= 1e-3:
        pytest.skip(
            f"measured tip cannot hold vertical at the probe corner "
            f"({res * 1000:.1f} mm) — hardware state, not a solver bug; "
            "the factory pre-flight is the gate for this")
    assert res < 1e-3, f"pre-flight would refuse the fitted tool: {res * 1000:.1f} mm"
