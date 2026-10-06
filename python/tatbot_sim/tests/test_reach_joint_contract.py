"""The reach audits must seed the IK chain the robot actually has.

2026-09-08: making the tool carriage a seventh solved axis took the chain from
six joints to seven. Three ``reshape(-1, 6)`` seeds in ``inkmap/reach.py`` and a
``qpos[:6]`` in ``audit_reach.py`` kept the old width, so ``tatbot sim sample``,
``sim resolve`` and ``sim reach`` all died on their first candidate solve —

    RuntimeError: shape '[-1, 6]' is invalid for input of size 6272

— while ``scripts/check sim`` stayed green. Nothing covered them: the only test
that reaches ``optimize_body_placement`` monkeypatches it away
(``test_scenario_resolver``), and ``audit_reach`` had no test at all. These run
the real solvers over the real chain, so the next change to it fails here
rather than in the factory.

These are SHAPE-CONTRACT tests. They assert that the audits run and produce a
verdict, not that any particular placement is reachable — reachability is
tool-geometry truth that moves with the bench, and the factory pre-flight and
``optimize_body_placement``'s own ledger are the gates for that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tatbot_sim import tools
from tatbot_sim.expert import StrokeExpert
from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.reach import ReachAuditError, optimize_body_placement
from tatbot_sim.repo import repo_root
from tatbot_sim.tools import active_tool

EXAMPLE = repo_root() / "config/inkmap/examples/forearm-placement-v6.json"


def test_the_ik_chain_carries_the_six_arm_joints_and_the_carriage():
    """Pins the width every seed builder in this repo has to match."""
    names = StrokeExpert(1, "cpu", noise=None, seed=0).ik.chain.get_joint_parameter_names()
    assert [n for n in names if n.startswith("joint_")] == [f"joint_{i}" for i in range(6)]
    assert "left_carriage_joint" in names, "the carriage is an EE ancestor and is solved"
    assert len(names) == 7


@pytest.mark.parametrize('height_m', [.028, .034])
def test_pad_ik_keeps_tip_and_axis_when_camera_roll_conflicts(height_m):
    """A camera-frame preference cannot displace a reachable contact pose."""
    import arm_kinematics as kin
    from tatbot_contracts.observations import FOLLOWER_JOINTS

    expert = StrokeExpert(1, 'cpu', noise=None, seed=0)
    if not expert.config.geometry.measured:
        pytest.skip('the measured ballpoint roll and tip are not in this fixture')
    target = np.array([[.29, 0., height_m]], dtype=np.float32)
    q0 = torch.tensor([tools.staged_pose()[:6]], dtype=torch.float32)
    q = expert.solve_pose(target, q0, normals=np.array([[0., 0., 1.]]))
    pose = expert.ik.fk(q)[0].numpy()
    tip = pose[:3, 3]
    np.testing.assert_allclose(tip, target[0], atol=.0003, rtol=0)
    axis = pose[:3, :3] @ expert.ik.tool_axis.numpy()
    assert np.degrees(np.arccos(np.clip(axis[2], -1., 1.))) < 1.
    margin = torch.minimum(q[:, :6] - expert.ik.q_lo[:6], expert.ik.q_hi[:6] - q[:, :6])
    assert float(margin.min()) > kin.JOINT_LIMIT_MARGIN_RAD

    order = [expert.ik.chain.get_joint_parameter_names().index(name) for name in FOLLOWER_JOINTS]
    joints = q[0, order].numpy().astype(float)
    production_tip = kin.ArmModel('right').fk_tcp(joints[:6], joints[6])[0]
    np.testing.assert_allclose(production_tip, tip, atol=2e-6, rtol=0)


def test_the_envelope_audit_solves_over_the_whole_chain():
    """``sim reach``'s sampler, on the fitted tool. Eight draws is enough: the
    break was in the seed's width, which every sample shares."""
    from tatbot_sim import audit_reach

    args = audit_reach.Args(samples=8)
    expert = StrokeExpert(1, "cpu", noise=None, seed=0)
    res, ok, top_z, kind = audit_reach.pass_rate(
        expert, np.random.default_rng(0), args, args.tolerance_mm / 1000
    )
    assert res.shape == (8,) and np.isfinite(res).all()
    assert ok.shape == (8,) and top_z.shape == (8,) and kind.shape == (8,)


@pytest.mark.slow
def test_the_body_placement_optimizer_solves_over_the_whole_chain():
    """``sim sample`` / ``sim resolve``'s search, over a real compiled scenario.

    One yaw and one offset candidate at the 16-probe floor: this is the batched
    ``q_start`` expand/reshape that raised the shape error, and it runs before
    any reach verdict. A refusal is a legitimate outcome and passes — a shape
    error is not, and would escape as RuntimeError.
    """
    tool = active_tool()
    scenario = compile_scenario(
        json.loads(EXAMPLE.read_text()),
        pose_id="reclined-left-arm-supported",
        seed=42,
        tool_id=tool.tool_id,
        created_at="2026-09-08T00:00:00Z",
        git_sha="0000000",
    )
    try:
        selection = optimize_body_placement(
            scenario,
            trajectory_seed=0,
            yaw_candidates=(0.0,),
            offset_candidates=(((0.0, 0.0), (0.0, 0.0, 0.0)),),
            probe_points=16,
        )
    except ReachAuditError:
        return  # a reach/clearance verdict — the solve ran, which is the point
    assert len(selection.candidates) == 1
    record = selection.candidates[0]
    assert np.isfinite(record["max_residual_m"])
    assert record["targets_over_tolerance"] <= 16
    assert np.isfinite(selection.probe_max_residual_m)


def test_the_palette_audit_scores_position_and_axis_over_the_whole_chain():
    """``sim reach``'s dip audit. The axis metric has to read zero where the
    arm demonstrably holds the pose asked for — a metric that always reports a
    tilt would condemn every placement, and one that always reports none would
    have missed the 23 degrees measured at the caps on 2026-09-09."""
    from tatbot_sim import audit_reach

    args = audit_reach.Args(samples=4, iters=200)
    args.dr.resolve_for(tools.active_substrate())
    expert = StrokeExpert(1, "cpu", noise=None, seed=0)
    n = 4
    q_seed = audit_reach._warm_seed(expert, n, args)

    # the warm seed's own pose: reached, and held at the axis it was given
    pad = np.repeat(np.array([[args.pad_center_x, 0.0, 0.034]]), n, axis=0)
    up = np.repeat(np.array([[0.0, 0.0, 1.0]]), n, axis=0)
    res, ang, tip = audit_reach._solve_and_score(expert, args, pad, up, q_seed)
    assert res.shape == (n,) and ang.shape == (n,) and tip.shape == (n, 3)
    assert res.max() < 1e-3, "the pad centre is reachable"
    assert np.degrees(ang).max() < 1.0, "and the commanded axis is achieved there"


def test_the_palette_audit_runs_for_the_fitted_tool(capsys):
    """Shape contract only: it produces a per-cap verdict. Whether this bench's
    palette placement passes is truth that moves with the rack, not a test."""
    from tatbot_sim import audit_reach

    args = audit_reach.Args(samples=4, palette_samples=4, iters=200)
    args.dr.resolve_for(tools.active_substrate())
    expert = StrokeExpert(1, "cpu", noise=None, seed=0)
    audit_reach.run_palette(expert, args, args.tolerance_mm / 1000)
    out = capsys.readouterr().out
    if tools.active_ink_policy().dips:
        assert "palette dip approach" in out
        for slot in tools.palette():
            assert slot in out
    else:
        assert "never dips" in out
