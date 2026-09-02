"""Pin scripts/lib/draw_kinematics.py against the URDF and the C++ spiral.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_draw_kinematics.py

What the draw stages depend on: the numpy FK is the URDF's FK (rotation and
tip, random joints); the ballpoint tip constant the C++ refuses against is
what config/workspace.yaml + the URDF actually say; align_rotation is a proper
minimal rotation; the advisory carriage-IK loop reproduces the C++ test's
6 mm / 3-turn / 120 s spiral to the C++ plan's own statistics, and refuses a
jump. The C++ numbers came from a throwaway driver compiled against
square_probe.cpp:

    g++ -std=c++17 -O2 -I cpp/teleop -o /tmp/x x.cpp cpp/teleop/square_probe.cpp

calling plan_joint_spiral_with_carriage(witness, 0.002, 0.006, 3.0, 120.0, 2.0, 0.0025).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import draw_kinematics as dk  # noqa: E402
import draw_path as dp  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

# The C++ test's carriage witness pose (square_probe_test.cpp).
WITNESS_JOINTS = np.array([
    0.173762112856, 1.544403791428, 0.826848268509,
    -0.061226826161, 0.121118485928, 1.642061471939])


@pytest.fixture(scope="module")
def chain():
    return UrdfChain(dk.URDF_PATH)


def _random_joints(rng, n):
    lower = dk.JOINT_LOWER + dk.JOINT_LIMIT_MARGIN_RAD
    upper = dk.JOINT_UPPER - dk.JOINT_LIMIT_MARGIN_RAD
    return rng.uniform(lower, upper, size=(n, 6))


def test_fk_matches_urdf_link6(chain):
    rng = np.random.default_rng(1)
    for joints in _random_joints(rng, 12):
        position, rotation, _ = dk.fk_link6(joints)
        pose = chain.link_pose(dk.LINK6_NAME, dk.joint_map(joints))
        assert np.allclose(pose[:3, :3], rotation, atol=1e-12)
        assert np.allclose(pose[:3, 3], dk.root_from_base(position), atol=1e-12)


def test_ballpoint_tip_matches_urdf_tool_mount_plus_pen_offset(chain):
    """FK tip == UrdfChain(right/tool_mount) @ workspace pen offset.

    Against the derived tip the agreement is < 1e-9 m. Against the published
    8-decimal constant it is ~5e-9 m (the rounding of the constant itself), so
    the constant path is pinned at 1e-8.
    """
    import tool_spec

    offset = np.asarray(tool_spec.tip_offset_m(tool_spec.read_workspace(REPO)), float)
    derived = dk.ballpoint_tip_in_link6_from_config(chain)
    rng = np.random.default_rng(2)
    for joints in _random_joints(rng, 8):
        carriage = float(rng.uniform(0.0, 0.0035))
        mount = chain.link_pose(dk.TOOL_MOUNT_NAME, dk.joint_map(joints, carriage))
        expected = mount[:3, :3] @ offset + mount[:3, 3]
        constant_tip, rotation, jacobian = dk.fk_ballpoint(joints, carriage)
        assert np.linalg.norm(dk.root_from_base(constant_tip) - expected) < 1e-8
        exact_tip, _, _ = dk.fk(joints, derived + carriage * dk.CARRIAGE_AXIS_IN_LINK6)
        assert np.linalg.norm(dk.root_from_base(exact_tip) - expected) < 1e-9
        assert jacobian.shape == (6, 7)
        assert np.allclose(jacobian[:3, 6], rotation @ dk.CARRIAGE_AXIS_IN_LINK6)


def test_tip_constant_matches_workspace_derivation(chain):
    derived = dk.ballpoint_tip_in_link6_from_config(chain)
    gap = np.linalg.norm(derived - dk.BALLPOINT_TIP_IN_LINK6)
    assert gap < 1e-6, f"tip from workspace.yaml {derived} differs from the C++ constant by {gap * 1e3:.4f} mm"


def test_tool_axis_is_the_urdf_mount_bore(chain):
    axis = dk.tool_axis_in_link6(chain)
    assert np.allclose(axis, [math.sqrt(0.5), -math.sqrt(0.5), 0.0], atol=1e-6)
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-12


def test_jacobian_matches_finite_differences():
    joints = WITNESS_JOINTS.copy()
    carriage = 0.002
    position, rotation, jacobian = dk.fk_ballpoint(joints, carriage)
    eps = 1e-7
    for j in range(6):
        bumped = joints.copy()
        bumped[j] += eps
        p2, r2, _ = dk.fk_ballpoint(bumped, carriage)
        assert np.allclose((p2 - position) / eps, jacobian[:3, j], atol=1e-5)
        assert np.allclose(dk.orientation_error(rotation, r2) / eps, jacobian[3:, j], atol=1e-5)
    p2, _, _ = dk.fk_ballpoint(joints, carriage + eps)
    assert np.allclose((p2 - position) / eps, jacobian[:3, 6], atol=1e-6)


def test_orientation_error_is_sin_theta_times_axis():
    rng = np.random.default_rng(3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    other = rng.normal(size=3)
    base = dk.axis_rotation(other / np.linalg.norm(other), 0.7)
    target = dk.axis_rotation(axis, 0.2) @ base
    error = dk.orientation_error(base, target)
    assert np.allclose(error, math.sin(0.2) * axis, atol=1e-12)
    assert np.allclose(dk.orientation_error(base, base), 0.0)


def test_align_rotation_properties():
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        r = dk.align_rotation(a, b)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(r) - 1.0) < 1e-12
        assert np.allclose(r @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-12)
        # minimal: the axis is perpendicular to both vectors
        axis, angle = dk.rotation_log(r)
        assert abs(np.dot(axis, a)) < 1e-9 * np.linalg.norm(a) + 1e-9
        assert abs(angle - math.acos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))) < 1e-9
    assert np.allclose(dk.align_rotation([0, 0, 1], [0, 0, 1]), np.eye(3))
    flip = dk.align_rotation([0, 0, 1], [0, 0, -1])
    assert np.allclose(flip @ [0, 0, 1], [0, 0, -1], atol=1e-12)
    assert np.allclose(flip @ flip.T, np.eye(3), atol=1e-12)


def test_root_base_helpers_are_the_urdf_offset(chain):
    base = chain.link_pose("right/base_link", {})
    assert np.allclose(base[:3, 3], dk.BASE_IN_ROOT)
    assert np.allclose(base[:3, :3], np.eye(3))
    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(dk.base_from_root(dk.root_from_base(p)), p)


def _flat_spiral_samples(joints, carriage, radius, turns, duration, ease, period):
    """The C++ spiral exactly: analytic reference and feedforward at t = tick * period, constant R."""
    center, rotation, _ = dk.fk_ballpoint(joints, carriage)
    total_angle = 2.0 * math.pi * turns
    scale = radius / total_angle
    length = dp.spiral_path_length(radius, turns)
    _, s, sdot = dp.time_law(length, duration, ease, period)
    angle = dp._spiral_angle(s, scale, total_angle, length)
    r = scale * angle
    cos, sin = np.cos(angle), np.sin(angle)
    root = np.sqrt(1.0 + angle * angle)
    p = center[None, :] + np.stack([r * cos, r * sin, np.zeros_like(r)], axis=1)
    v = np.stack([sdot * (cos - angle * sin) / root, sdot * (sin + angle * cos) / root, np.zeros_like(r)], axis=1)
    return {"p": p, "v": v, "R": np.repeat(rotation[None], len(p), axis=0)}, center


# What cpp/teleop/square_probe.cpp::plan_joint_spiral_with_carriage returns for this
# case (built and run against the C++ on 2026-09-01; see the module docstring).
CPP_STATS = {
    "max_model_error_mm": 0.000683979946, "max_orientation_error_rad": 9.79191664e-07,
    "max_joint_velocity_rad_s": 0.00181787784, "max_carriage_velocity_m_s": 0.000255973929,
    "max_carriage_acceleration_m_s2": 0.000460534724,
    "min_carriage_m": 0.000861911849, "max_carriage_m": 0.00317286283,
}
CPP_TICK_48000 = (0.170318975851, 1.56632578342, 0.846680908569, -0.0588458157614, 0.11866728182,
                  1.63962575785, 0.00250857038298)
CPP_TICK_10000 = (0.182391322299, 1.54228652821, 0.827643370486, -0.0648964496386, 0.127258371517,
                  1.64817219568, 0.00116030999758)


def test_plan_joints_reproduces_the_cpp_carriage_spiral():
    """Bit-for-bit parity with the executor's plan, not just 'inside the caps'.

    The hardware A/B measured the carriage inside 1.08-2.94 mm; the *plan*
    itself dips to 0.862 mm (still inside the 0.5-3.5 mm envelope), and that is
    what both the C++ and this port compute.
    """
    period = 0.0025
    samples, center = _flat_spiral_samples(WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, 0.006, 3.0, 120.0, 2.0, period)
    assert len(samples["p"]) == 48000
    plan = dk.plan_joints(samples, WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, period)
    stats = plan["stats"]
    assert plan["positions"].shape == (48000, 7)
    assert plan["velocities"].shape == (48000, 7)
    for key, value in CPP_STATS.items():
        assert stats[key] == pytest.approx(value, rel=1e-6, abs=1e-12), key
    assert np.allclose(plan["positions"][47999], CPP_TICK_48000, atol=1e-9)
    assert np.allclose(plan["positions"][9999], CPP_TICK_10000, atol=1e-9)
    assert stats["max_model_error_mm"] < 0.01
    assert stats["max_orientation_error_rad"] < 1e-5
    assert stats["max_joint_velocity_rad_s"] < 0.01
    assert stats["min_carriage_m"] >= dk.CARRIAGE_IK_MIN_M and stats["max_carriage_m"] <= dk.CARRIAGE_IK_MAX_M
    assert stats["max_carriage_m"] - stats["min_carriage_m"] > 0.00025
    end_tip, _, _ = dk.fk_ballpoint(plan["positions"][-1, :6], plan["positions"][-1, 6])
    assert np.allclose(end_tip, center + [0.006, 0.0, 0.0], atol=1e-6)


def test_plan_joints_refuses_a_jump():
    period = 0.0025
    samples, _ = _flat_spiral_samples(WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, 0.006, 3.0, 12.0, 2.0, period)
    samples["p"][2000:] += np.array([0.05, 0.0, 0.0])
    with pytest.raises(dk.PlanRefusal) as info:
        dk.plan_joints(samples, WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, period)
    assert info.value.reason in ("joint_velocity", "model_error", "carriage_velocity", "carriage_acceleration")


def test_plan_joints_refuses_bad_starts():
    samples, _ = _flat_spiral_samples(WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, 0.006, 3.0, 12.0, 2.0, 0.0025)
    with pytest.raises(dk.PlanRefusal, match="carriage_envelope"):
        dk.plan_joints(samples, WITNESS_JOINTS, 0.004, 0.0025)
    with pytest.raises(dk.PlanRefusal, match="joint_limit"):
        dk.plan_joints(samples, np.zeros(6), dk.CARRIAGE_IK_BIAS_M, 0.0025)
    samples["p"][10, 0] = np.nan
    with pytest.raises(dk.PlanRefusal, match="nan"):
        dk.plan_joints(samples, WITNESS_JOINTS, dk.CARRIAGE_IK_BIAS_M, 0.0025)
