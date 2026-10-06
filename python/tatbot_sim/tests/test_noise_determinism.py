"""DART bursts belong to a batch, not to how hard its solve turned out to be.

2026-09-08: every ``StrokeExpert.reset`` drew from one running generator, and
the contact settle re-solves a batch up to three times while the residual gate
can add a fourth. So the noise depended on solver effort: two runs differing
only in a solver flag got different bursts, which makes any solver A/B in this
factory incomparable and means a seed does not reproduce a run whose solve path
changed. Measured before the fix: burst magnitude 30.11 against 26.74 for one
extra re-solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tatbot_sim import tools
from tatbot_sim.config import NoiseDR
from tatbot_sim.expert import StrokeExpert

STEPS = 40
NOISE = NoiseDR(prob=(1.0, 1.0), scale=(0.05, 0.05))


def _path():
    points = np.stack(
        [np.linspace(0.27, 0.33, STEPS), np.zeros(STEPS), np.full(STEPS, 0.14)], axis=1
    )[None].astype(np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (STEPS, 1))[None].astype(np.float32)
    return points, normals


def _staged() -> torch.Tensor:
    return torch.tensor([tools.staged_pose()[:6]], dtype=torch.float32)


def _bursts(resolves: int) -> np.ndarray:
    """The bursts a batch ends up with after ``resolves`` extra re-solves."""
    points, normals = _path()
    expert = StrokeExpert(1, torch.device("cpu"), noise=NOISE, seed=7)
    expert.reset(points, _staged(), pen_normals=normals)
    for _ in range(resolves):
        expert.reset(points, _staged(), pen_normals=normals, resolve=True)
    return (expert.actions - expert.q_ref)[0].numpy()


@pytest.mark.slow
def test_a_batchs_bursts_do_not_depend_on_how_often_it_re_solved() -> None:
    baseline = _bursts(0)
    assert np.abs(baseline).sum() > 0, "the fixture must actually produce bursts"
    for resolves in (1, 2, 3, 4):
        assert np.array_equal(_bursts(resolves), baseline), (
            f"{resolves} re-solves changed the batch's DART bursts"
        )


def test_a_new_batch_still_gets_its_own_bursts() -> None:
    """The fix must not turn every batch into a replay of the first one.

    Re-seeding per batch was itself a bug once: it replayed identical burst
    timing in every batch of a dataset.
    """
    points, normals = _path()
    expert = StrokeExpert(1, torch.device("cpu"), noise=NOISE, seed=7)
    expert.reset(points, _staged(), pen_normals=normals)
    first = (expert.actions - expert.q_ref)[0].numpy().copy()
    expert.reset(points, _staged(), pen_normals=normals)
    second = (expert.actions - expert.q_ref)[0].numpy()
    assert not np.array_equal(first, second)


def test_a_run_that_never_re_solves_keeps_the_stream_it_always_had() -> None:
    """One draw per batch, so an untouched seed reproduces its old bursts."""
    points, normals = _path()

    def two_batches() -> list[np.ndarray]:
        expert = StrokeExpert(1, torch.device("cpu"), noise=NOISE, seed=11)
        out = []
        for _ in range(2):
            expert.reset(points, _staged(), pen_normals=normals)
            out.append((expert.actions - expert.q_ref)[0].numpy().copy())
        return out

    a, b = two_batches(), two_batches()
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_the_carriage_is_never_given_a_burst() -> None:
    expert = StrokeExpert(1, torch.device("cpu"), noise=NOISE, seed=7, carriage_ik=True)
    points, normals = _path()
    expert.reset(points, _staged(), floor_plane=(points, normals), pen_normals=normals)
    carriage = (expert.actions - expert.q_ref)[0, :, expert.ik.carriage_index].numpy()
    assert np.allclose(carriage, 0.0)


def test_explicit_episode_seed_replays_after_another_episode():
    points, normals = _path()
    expert = StrokeExpert(1, torch.device('cpu'), noise=NOISE, seed=7)
    noise = []
    for seed in (11, 12, 11):
        expert.begin_episode(seed)
        expert.reset(points, _staged(), pen_normals=normals)
        noise.append(expert._noise.clone())
    assert torch.equal(noise[0], noise[2])
    assert not torch.equal(noise[0], noise[1])
