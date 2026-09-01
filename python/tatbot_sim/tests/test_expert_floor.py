"""The floor the expert clamps against, when the surface is not flat.

The clamp does two things with its plane: it holds the needle above it, and it
decides from the same plane whether a step is DRAWING or TRAVELLING — a step
more than 6 mm above the floor is treated as travel and held a further 7.5 mm
clear, so a burst cannot stamp a stray mark mid-flight.

Both of those are only correct while the plane IS the drawing surface. Lift the
skin off the pad plane and the second one inverts: drawing steps read as
travel, get held clear of a floor they were never near, and the noise that
makes a demonstration worth learning from is suppressed exactly where the
drawing happens. Nothing looks broken — the clamp falls back to the reference,
which never inks — which is why this is worth a test rather than an eyeball.

Needs the URDF and pytorch_kinematics, no render device:

    cd python/tatbot_sim && uv run python tests/test_expert_floor.py
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.config import NoiseDR
from tatbot_sim.expert import StrokeExpert, _per_step
from tatbot_sim.planning import canvas_to_world
from tatbot_sim.surface import DisplacedSurface, PlaneChart

B, T = 2, 24
LIFT_M = 0.003  # a skin sitting 3 mm proud of the pad plane


def _quiet_expert():
    """No bursts: this is about the floor's classification, not about noise."""
    return StrokeExpert(B, torch.device("cpu"),
                        noise=NoiseDR(prob=(0.0, 0.0), scale=(0.0, 0.0)), seed=0)


def _scene(lift_m: float):
    """A surface a constant ``lift_m`` above its chart, over the pad's spot."""
    center = torch.tensor([[0.29, 0.0, 0.03]] * B, dtype=torch.float32)
    rot = torch.eye(3).expand(B, 3, 3).contiguous()
    rows, cols = 41, 33
    height = torch.full((B, rows, cols), lift_m)
    surf = DisplacedSurface(PlaneChart(center, rot), height)
    xs = np.linspace(-0.02, 0.02, T)
    traj = np.stack([xs, np.zeros(T), np.zeros(T)], axis=1).astype(np.float32)
    targets = np.zeros((B, T, 3), dtype=np.float32)
    pts = np.zeros((B, T, 3), dtype=np.float32)
    nrm = np.zeros((B, T, 3), dtype=np.float32)
    for i in range(B):
        targets[i], pts[i], nrm[i] = canvas_to_world(traj, surf, i, 0.004)
    return center.numpy(), rot[:, :, 2].numpy(), targets, pts, nrm


def _clamped(floor, targets):
    expert = _quiet_expert()
    q0 = torch.zeros(B, 6)
    expert.reset(targets, q0, floor_plane=floor, batch_iters=30, sweep_iters=2)
    return expert.clamped_fraction


def test_one_plane_per_episode_fights_a_surface_that_sits_above_it():
    """The bug, stated as a measurement: with the chart plane as the floor,
    every drawing step of a 3 mm-proud skin reads as travel (3 + 4 = 7 mm > 6)
    and is then held 7.5 mm clear of a floor it is only 7 mm above."""
    center, normal, targets, _, _ = _scene(LIFT_M)
    assert _clamped((center, normal), targets) > 0.9


def test_a_plane_per_step_leaves_the_drawing_alone():
    """With the surface itself as the floor, a drawing step is 4 mm above it —
    on-stroke, offset 0 — and nothing is clamped."""
    _, _, targets, pts, nrm = _scene(LIFT_M)
    assert _clamped((pts, nrm), targets) == 0.0


def test_the_flat_case_is_unchanged_either_way():
    """With no lift the chart plane IS the surface, so both spellings of the
    floor have to agree — that is what makes this safe for every planar run."""
    center, normal, targets, pts, nrm = _scene(0.0)
    assert _clamped((center, normal), targets) == _clamped((pts, nrm), targets) == 0.0


def test_per_step_padding_puts_the_approach_on_the_first_plane():
    """An approach is prepended to the trajectory and descends toward the first
    drawing pose, so it belongs on that pose's plane."""
    v = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)
    out = _per_step(v, 2, 5)
    assert out.shape == (2, 5, 3)
    assert torch.equal(out[:, 2:], v)              # the draw steps, in place
    assert torch.equal(out[:, 0], v[:, 0])          # approach on the first plane
    assert torch.equal(out[:, 1], v[:, 0])


def test_one_plane_still_broadcasts():
    v = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    out = _per_step(v, 2, 4)
    assert out.shape == (2, 4, 3)
    assert torch.equal(out[:, 0], v) and torch.equal(out[:, 3], v)


def test_a_floor_longer_than_its_trajectory_is_refused():
    v = torch.zeros(2, 9, 3)
    try:
        _per_step(v, 2, 4)
    except ValueError as exc:
        assert "9 steps" in str(exc)
        return
    raise AssertionError("a mismatched floor must raise")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
