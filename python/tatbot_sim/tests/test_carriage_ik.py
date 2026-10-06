"""The carriage as a seventh IK axis, on flat and curved surfaces alike.

The executor solves this axis but runs `carriage_ik 0` in every real draw
session: on a curved surface its weighted solve read the wrist's normal-tracking
rotation, swung through a ~200 mm lever arm, as tool-axis motion and walked the
carriage out of its envelope within 25 s (square_probe.cpp, the pen-up branch).

Driving it from surface-normal error instead is what these tests hold: the
envelope has to survive a cylinder, not just a sheet, and no shape may be a
special case.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tatbot_sim import tools
from tatbot_sim.carriage import CARRIAGE_JOINT, CarriagePolicy
from tatbot_sim.expert import StrokeExpert

STEPS = 240
CONTROL_HZ = 30.0


def _staged(batch: int = 1) -> torch.Tensor:
    return torch.tensor([tools.staged_pose()[:6]] * batch, dtype=torch.float32)


def _plane():
    x = np.linspace(0.27, 0.33, STEPS)
    points = np.stack([x, np.zeros(STEPS), np.full(STEPS, 0.14)], axis=1)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (STEPS, 1))
    return points[None].astype(np.float32), normals[None].astype(np.float32)


def _cylinder(radius_m: float, undulation_m: float = 0.0):
    """A stroke wrapping a cylinder, optionally with paper that is not smooth."""
    phi = np.linspace(-0.35, 0.35, STEPS)
    normals = np.stack([np.sin(phi), np.zeros(STEPS), np.cos(phi)], axis=1)
    surface = np.stack(
        [0.30 + radius_m * np.sin(phi), np.zeros(STEPS), 0.14 - radius_m + radius_m * np.cos(phi)],
        axis=1,
    )
    if undulation_m:
        bump = undulation_m * np.sin(3.0 * 2 * np.pi * np.linspace(0, 1, STEPS))
        surface = surface + normals * bump[:, None]
    return surface[None].astype(np.float32), normals[None].astype(np.float32)


SURFACES = {
    "plane": _plane(),
    "cylinder-90mm": _cylinder(0.090),
    "cylinder-40mm": _cylinder(0.040),
    "cylinder-90mm-deformed": _cylinder(0.090, undulation_m=0.0008),
}


def _solve(points, normals, *, carriage_ik: bool) -> StrokeExpert:
    expert = StrokeExpert(1, torch.device("cpu"), noise=None, seed=0,
                          carriage_ik=carriage_ik, control_hz=CONTROL_HZ)
    expert.reset(points, _staged(), floor_plane=(points, normals), pen_normals=normals)
    return expert


def test_the_tool_rides_the_carriage_so_the_chain_has_seven_joints() -> None:
    expert = StrokeExpert(1, torch.device("cpu"), noise=None, seed=0)
    names = expert.ik.chain.get_joint_parameter_names()
    assert names[-1] == CARRIAGE_JOINT
    assert expert.ik.n_joints == 7
    assert expert.ik.carriage_index == 6


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_the_carriage_stays_inside_its_guarded_envelope_on_every_surface(surface) -> None:
    policy = CarriagePolicy.from_repo()
    points, normals = SURFACES[surface]
    carriage = _solve(points, normals, carriage_ik=True).q_ref[0][:, 6].numpy()

    assert carriage.min() >= policy.min_m, f"{surface} drove the carriage below its envelope"
    assert carriage.max() <= policy.max_m, f"{surface} walked the carriage out of its envelope"


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_the_carriage_respects_the_executors_rate_cap(surface) -> None:
    """A reference the hardware could not track is not a demonstration."""
    policy = CarriagePolicy.from_repo()
    points, normals = SURFACES[surface]
    carriage = _solve(points, normals, carriage_ik=True).q_ref[0][:, 6].numpy()

    step = np.abs(np.diff(carriage)).max()
    assert step <= policy.max_step_m(CONTROL_HZ) + 1e-9, (
        f"{surface} moved the carriage {step * 1e6:.1f} um in one control frame"
    )


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_a_locked_carriage_holds_rest_and_never_becomes_a_reaching_axis(surface) -> None:
    points, normals = SURFACES[surface]
    expert = _solve(points, normals, carriage_ik=False)
    carriage = expert.q_ref[0][:, 6].numpy()

    assert np.allclose(carriage, expert.carriage_rest_m, atol=1e-9), (
        "with carriage_ik off the axis must stay exactly where the follower holds it"
    )
    assert np.allclose(expert.actions[0][:, 6].numpy(), expert.carriage_rest_m, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_solving_the_carriage_never_costs_tip_accuracy(surface) -> None:
    points, normals = SURFACES[surface]
    errors = {}
    for enabled in (True, False):
        expert = _solve(points, normals, carriage_ik=enabled)
        tip = expert.ik.fk(expert.q_ref[0]).detach().numpy()[:, :3, 3]
        errors[enabled] = np.abs(((tip - points[0]) * normals[0]).sum(-1)).max()
    assert errors[True] <= errors[False] + 5e-6, (
        f"{surface}: solving the carriage worsened normal tracking "
        f"({errors[True] * 1e6:.2f} um vs {errors[False] * 1e6:.2f} um)"
    )


def test_the_dart_bursts_leave_the_safety_owned_axis_alone() -> None:
    from tatbot_sim.config import NoiseDR

    points, normals = SURFACES["cylinder-90mm"]
    expert = StrokeExpert(1, torch.device("cpu"), noise=NoiseDR(prob=(1.0, 1.0), scale=(0.05, 0.05)),
                          seed=0, carriage_ik=True, control_hz=CONTROL_HZ)
    expert.reset(points, _staged(), floor_plane=(points, normals), pen_normals=normals)

    reference = expert.q_ref[0][:, 6].numpy()
    commanded = expert.actions[0][:, 6].numpy()
    assert np.allclose(reference, commanded, atol=1e-9), (
        "noise reached the carriage; the real follower's safety layer owns that axis"
    )


def test_the_envelope_comes_from_the_executors_own_constants() -> None:
    """No second copy of these numbers: they render motion_constants.hpp too."""
    import json

    from tatbot_sim.repo import repo_root

    raw = json.loads((repo_root() / "config" / "motion_constants.json").read_text())
    policy = CarriagePolicy.from_repo()
    assert policy.bias_m == raw["carriage_ik"]["bias_m"]
    assert policy.min_m == raw["carriage_ik"]["min_m"]
    assert policy.max_m == raw["carriage_ik"]["max_m"]
    assert policy.max_velocity_m_s == raw["planner"]["max_carriage_velocity_m_s"]


def test_the_reach_mask_that_chooses_a_scene_ignores_the_carriage_flag() -> None:
    """carriage_ik picks how a plan is executed, never which plan is accepted.

    Two runs differing only in this flag drew different designs in 2 of 4 flat
    pairs (2026-09-08) because the mask was solved from a seed pose that moved
    with it, silently re-qualifying the basis every reach and clearance gate
    here was accepted against.
    """
    poses = {}
    for enabled in (True, False):
        expert = StrokeExpert(1, torch.device("cpu"), noise=None, seed=0, carriage_ik=enabled)
        poses[enabled] = expert.seed_pose(
            _staged(), 1, carriage_m=expert.carriage_rest_m
        ).numpy()
    assert np.array_equal(poses[True], poses[False])


def test_a_reach_retry_nudges_the_arm_not_the_carriage() -> None:
    """0.3 is radians; on a metre-scale axis it is 300 mm of jitter."""
    from tatbot_sim.expert import reach_residual_at

    residuals = []
    for enabled in (True, False):
        expert = StrokeExpert(1, torch.device("cpu"), noise=None, seed=0, carriage_ik=enabled)
        q_rest = expert.seed_pose(_staged(), 1, carriage_m=expert.carriage_rest_m)
        residuals.append(reach_residual_at(
            expert, q_rest, np.array([0.30, 0.0, 0.0]), 0.14, 0.0,
        ))
    assert residuals[0] == residuals[1], (
        "the pre-flight reach verdict moved with the carriage solver flag"
    )
