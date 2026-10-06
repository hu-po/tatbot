"""The contact settle must report the reference it leaves behind.

2026-09-08: the loop measured the error, corrected the targets and re-solved,
then returned the measurement taken BEFORE that final correction. With the
default three rounds only two were ever credited, so a batch that the third
round fixed was still dropped as `ik_reference`, and the `reference` block
written into every retained episode's run_meta described a joint reference the
solver had already replaced.
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.reference import CONTACT_REFERENCE_TOLERANCE_M, _settle_contact_reference


class _FakeIK:
    """FK that reports whatever height the fake solver currently reaches."""

    def __init__(self, owner: "_FakeExpert") -> None:
        self.owner = owner
        self.n_joints = 6

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        poses = torch.eye(4).repeat(q.shape[0], 1, 1)
        poses[:, 2, 3] = self.owner.height_m
        return poses


class _FakeExpert:
    """Converges one step per re-solve, through the heights it is given."""

    def __init__(self, heights_m: list[float], steps: int) -> None:
        self.heights_m = heights_m
        self.solves = 0
        self.reset_kwargs: list[dict] = []
        self.q_ref = torch.zeros(1, steps, 6)
        self.ik = _FakeIK(self)

    @property
    def height_m(self) -> float:
        return self.heights_m[min(self.solves, len(self.heights_m) - 1)]

    def reset(self, targets, q_start, **kwargs) -> None:
        self.solves += 1
        self.reset_kwargs.append(kwargs)


class _Plan:
    """One env, every step planted on the surface, so every step is pen-down."""

    def __init__(self, steps: int) -> None:
        self.targets = np.zeros((1, steps, 3), dtype=np.float32)
        self.surface_points = np.zeros((1, steps, 3), dtype=np.float32)
        self.surface_normals = np.zeros((1, steps, 3), dtype=np.float32)
        self.surface_normals[..., 2] = 1.0
        self.q_raised = None
        self.n_app = 0


def test_the_final_round_is_measured_rather_than_the_error_it_corrected() -> None:
    # Still 0.8 mm out when the third correction is applied, and exact after it.
    expert = _FakeExpert([0.002, 0.001, 0.0008, 0.0], steps=2)

    worst_mm = _settle_contact_reference(expert, _Plan(2), torch.zeros(1, 6), {}, num_envs=1)

    assert expert.solves == 3, "all three correction rounds should have been spent"
    # The old code returned 0.8 mm here -- the error the third round was asked
    # to correct -- and the batch was dropped despite having converged.
    assert worst_mm[0] == 0.0
    assert worst_mm[0] <= CONTACT_REFERENCE_TOLERANCE_M * 1000


def test_an_early_convergence_still_reports_its_own_reference() -> None:
    expert = _FakeExpert([0.0], steps=2)

    worst_mm = _settle_contact_reference(expert, _Plan(2), torch.zeros(1, 6), {}, num_envs=1)

    assert expert.solves == 0, "a converged reference should not be re-solved"
    assert worst_mm[0] == 0.0


def test_the_callers_solve_budget_reaches_every_internal_resolve() -> None:
    """The escalated budget is the caller's; the settle must not drop it."""
    expert = _FakeExpert([0.002, 0.0], steps=2)
    budget = {"batch_iters": 240, "sweeps": 2, "sweep_iters": 12}

    _settle_contact_reference(expert, _Plan(2), torch.zeros(1, 6), dict(budget), num_envs=1)

    assert expert.reset_kwargs, "the settle should have re-solved at least once"
    for kwargs in expert.reset_kwargs:
        assert {k: kwargs[k] for k in budget} == budget
        # A settle round re-solves the SAME batch, so it must keep that batch's
        # DART bursts rather than advancing the noise stream -- otherwise the
        # episode's noise depends on how hard its solve turned out to be.
        assert kwargs["resolve"] is True
