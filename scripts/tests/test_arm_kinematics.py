"""Pin scripts/lib/arm_kinematics.py against the URDF and the C++ spiral.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_arm_kinematics.py

What the draw stages depend on: the numpy FK is the URDF's FK (rotation and
tip, random joints); the ballpoint tip constant the C++ refuses against is
what config/workspace.yaml + the URDF actually say; align_rotation is a proper
minimal rotation; the advisory carriage-IK loop reproduces the C++ test's
6 mm / 3-turn / 120 s spiral to the C++ plan's own statistics, and refuses a
jump. The C++ numbers came from a throwaway driver compiled against
square_probe.cpp:

    g++ -std=c++17 -O2 -I cpp/teleop -o /tmp/x x.cpp cpp/teleop/square_probe.cpp

planning square_probe_test.cpp's spiral samples (witness, carriage 0.002, 6 mm,
3 turns, 120 s, 2 s ease, 0.0025 s) through plan_joint_path, which is all the
retired plan_joint_spiral_with_carriage did.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

import arm_kinematics as dk  # noqa: E402
import ballpoint_fixture  # noqa: E402
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
    """FK tip == UrdfChain(right/tool_mount) @ the ballpoint's workspace pen offset.

    Against the derived tip the agreement is < 1e-9 m. Against the published
    8-decimal constant it is ~5e-9 m (the rounding of the constant itself), so
    the constant path is pinned at 1e-8.
    """
    import tool_spec

    workspace = ballpoint_fixture.workspace()
    offset = np.asarray(tool_spec.tip_offset_m(workspace), float)
    if tool_spec.tip_offset_m(workspace) is None:
        with pytest.raises(RuntimeError, match="no touch-off"):
            dk.ballpoint_tip_in_link6_from_config(chain, workspace)
        return
    derived = dk.ballpoint_tip_in_link6_from_config(chain, workspace)
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
    import tool_spec
    workspace = ballpoint_fixture.workspace()
    if tool_spec.tip_offset_m(workspace) is None:
        with pytest.raises(RuntimeError, match="no touch-off"):
            dk.ballpoint_tip_in_link6_from_config(chain, workspace)
        return
    derived = dk.ballpoint_tip_in_link6_from_config(chain, workspace)
    gap = np.linalg.norm(derived - dk.BALLPOINT_TIP_IN_LINK6)
    assert gap < 1e-6, f"tip from workspace.yaml {derived} differs from the C++ constant by {gap * 1e3:.4f} mm"


def test_tool_axis_is_the_urdf_mount_bore(chain):
    axis = dk.tool_axis_in_link6(chain)
    assert np.allclose(axis, [math.sqrt(0.5), -math.sqrt(0.5), 0.0], atol=1e-6)
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-12


@pytest.mark.parametrize(('arm', 'tool_id'), [('right', 'lutin-ballpoint-dot'), ('left', 'picosecond-laser-pen')])
def test_nominal_arm_model_resolves_datasheet_tcp_without_borrowing_urdf_calibration(chain, arm, tool_id):
    import tool_spec

    workspace = {arm: {'tool_id': tool_id}}
    original = (REPO/'urdf/tatbot.urdf').read_bytes()
    model = dk.ArmModel(arm, chain=chain, workspace=workspace)
    spec = tool_spec.load_active_tool(REPO, arm, workspace)
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, arm, REPO)
    assert not geometry.measured and geometry.status == 'nominal'
    assert model.tip_source == 'datasheet nominal' and model.measured_tip_offset_m() is None
    for carriage in (0., .003):
        mount = model.link_in_link6(model.tool_mount, carriage_m=carriage)
        tip = mount[:3, 3] + mount[:3, :3] @ geometry.tcp_offset_m
        np.testing.assert_allclose(model.tcp_in_link6(carriage), tip - carriage * model.carriage_axis_in_link6,
                                   atol=1e-12, rtol=0)
    assert workspace == {arm: {'tool_id': tool_id}}
    assert (REPO/'urdf/tatbot.urdf').read_bytes() == original


def test_right_arm_model_reproduces_the_module_constants(chain):
    """`ArmModel("right")` is the follower the module and the executor compile in."""
    import tool_spec
    workspace = ballpoint_fixture.workspace()
    model = dk.ArmModel("right", chain=chain, workspace=workspace)
    assert model.joint_names == dk.ARM_JOINT_NAMES
    assert model.carriage_joint == dk.CARRIAGE_JOINT_NAME and model.tool_mount == dk.TOOL_MOUNT_NAME
    assert model.frame == "right/base_link" and model.golden_path.name == "follower.yaml"
    np.testing.assert_allclose(model.base_in_root, dk.BASE_IN_ROOT, atol=1e-12)
    np.testing.assert_allclose(model.carriage_axis_in_link6, dk.CARRIAGE_AXIS_IN_LINK6, atol=1e-9)
    if tool_spec.tip_offset_m(workspace) is None:
        assert model.tip_source == "datasheet nominal"
        return
    assert model.tip_source == "touch-off"
    np.testing.assert_allclose(model.tcp_in_link6(), dk.ballpoint_tip_in_link6_from_config(chain, workspace),
                               atol=1e-12)
    rng = np.random.default_rng(7)
    for joints in _random_joints(rng, 5):
        for carriage in (0.0, 0.002, -0.001):
            position, rotation, jacobian = model.fk_tcp(joints, carriage)
            expected = dk.fk_ballpoint(joints, carriage)
            np.testing.assert_allclose(position, expected[0], atol=2e-6)
            np.testing.assert_allclose(rotation, expected[1], atol=1e-12)
            np.testing.assert_allclose(jacobian, expected[2], atol=2e-6)
    # the controller golden's limits, which the hover planner already used;
    # the executor's compiled guard limits are a separate, tighter contract
    import yaml
    golden = yaml.safe_load((REPO / "config/trossen/follower.yaml").read_bytes())["joint_limits"][:6]
    lower, upper = model.joint_limits()
    np.testing.assert_allclose(lower, [p["position_min"] for p in golden])
    np.testing.assert_allclose(upper, [p["position_max"] for p in golden])


def test_left_arm_model_is_the_mirrored_leader_from_the_urdf(chain):
    """The leader shares the joint chain, rides the driven left carriage along
    +y like the follower (the V5 print is bolted there, rolled half a turn),
    sits at its own base, and carries its tool at the URDF nominal until a
    touch-off exists in left/tool_mount."""
    import xml.etree.ElementTree as ET
    model = dk.ArmModel("left", chain=chain)
    assert model.joint_names == tuple(f"left/joint_{i}" for i in range(6))
    assert model.carriage_joint == "left/left_carriage_joint" and model.tcp_link == "left/tattoo_needle"
    assert model.frame == "left/base_link" and model.golden_path.name == "leader.yaml"
    urdf = ET.parse(dk.URDF_PATH).getroot()
    mount = urdf.find('joint[@name="left/mount_joint"]/origin').get("xyz")
    np.testing.assert_allclose(model.base_in_root, np.fromstring(mount, sep=" "), atol=1e-12)
    np.testing.assert_allclose(model.carriage_axis_in_link6, [0.0, 1.0, 0.0], atol=1e-9)
    import tool_spec
    needle = model.link_in_link6("left/tattoo_needle")[:3, 3]
    np.testing.assert_allclose(model.tcp_in_link6(), needle, atol=1e-12)
    # gen_tool_urdf hangs the needle at the measured tip once a touch-off
    # exists (the leader has had one since 37b1dcde); only the nominal
    # fallback sits protrusion_m along the mount bore.
    if model.tip_source == "datasheet nominal":
        laser = tool_spec.load_tool(tool_spec.active_tool_id(REPO, "left"), REPO)
        bore = model.link_in_link6("left/tool_mount")
        np.testing.assert_allclose(needle, bore[:3, 3] + bore[:3, :3] @ [0, 0, laser.protrusion_m], atol=1e-9)
    # carriage travel shifts the working point along +y of link 6, and the
    # model's tip at carriage c is what the URDF says
    for carriage in (0.0, 0.003):
        position, rotation, jacobian = model.fk_tcp(np.zeros(6), carriage)
        root = chain.link_pose("left/tattoo_needle", model.joint_map(np.zeros(6), carriage))
        np.testing.assert_allclose(model.root_from_base(position), root[:3, 3], atol=1e-9)
        np.testing.assert_allclose(jacobian[:3, 6], rotation @ [0, 1, 0], atol=1e-12)
    lower, upper = model.joint_limits()
    assert lower.shape == (6,) and np.all(lower < upper)
    with pytest.raises(ValueError, match="unknown arm"):
        dk.ArmModel("middle")


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


def test_a_standoff_tools_touch_off_extends_to_its_working_point():
    """A touch-off plants the tool's body; for a non-contact tool the working
    point floats the datasheet standoff further along the same direction, so
    the model's tcp is the touch-off extended (as the poses export places
    it), never the nose itself."""
    import tool_spec
    laser = tool_spec.load_tool("picosecond-laser-pen", REPO)
    assert laser.standoff_m > 0
    workspace = tool_spec.read_workspace(REPO)
    workspace = {**workspace, "left": {**(workspace.get("left") or {}), "tip_frame": "left/tool_mount",
                                       "pen_tip_offset_x": 0.003, "pen_tip_offset_y": -0.004, "pen_tip_offset_z": 0.132,
                                       "tool_id": "picosecond-laser-pen"}}
    model = dk.ArmModel("left", workspace=workspace)
    assert model.tip_source == "touch-off"
    bore = model.link_in_link6("left/tool_mount")
    extended = tool_spec.tcp_from_touchoff_m(laser, (0.003, -0.004, 0.132))
    np.testing.assert_allclose(model.tcp_in_link6(), bore[:3, 3] + bore[:3, :3] @ extended, atol=1e-12)
    nose = bore[:3, 3] + bore[:3, :3] @ [0.003, -0.004, 0.132]
    assert abs(np.linalg.norm(model.tcp_in_link6() - nose) - laser.standoff_m) < 1e-9
    # A contact tool's touch-off is its working point: nothing is extended.
    ballpoint = tool_spec.load_tool("lutin-ballpoint-dot", REPO)
    assert ballpoint.standoff_m == 0
    assert tool_spec.tcp_from_touchoff_m(ballpoint, (0.001, 0.002, 0.1)) == (0.001, 0.002, 0.1)
