"""The generated description: one root, the carriage travel, the TCP from workspace.yaml, and the
<ros2_control> systems for mock, fake and real hardware."""
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml
from tatbot_description import hardware_params, load_stack, names, repo_root, robot_description

# `real` needs the private arm addresses (config/profiles/<profile>.json); a public clone has none.
HAS_PROFILE = (repo_root() / "config" / "profiles" / f"{load_stack()['profile']}.json").is_file()
VARIANTS = [pytest.param(hw, arms, marks=pytest.mark.skipif(hw == "real" and not HAS_PROFILE, reason="no profile"))
            for hw in (None, "mock", "fake", "real") for arms in (("right",), ("left", "right"))]


def _joints(robot):
    return {joint.attrib["name"]: joint for joint in robot.findall("joint") if "name" in joint.attrib}


def test_right_arm_kinematic_description():
    robot = ET.fromstring(robot_description(arms=("right",)))
    links = {link.attrib["name"] for link in robot.findall("link") if "name" in link.attrib}
    children = {child.attrib["link"] for joint in robot.findall("joint") if (child := joint.find("child")) is not None and "link" in child.attrib}
    assert links - children == {names.WORLD}
    assert {names.base_frame("right"), names.tool_mount_frame("right"), names.tcp_frame("right")} <= links
    joints = _joints(robot)
    for name in names.joint_names("right"):
        assert joints[name].attrib.get("type") in ("revolute", "prismatic")
    carriage = joints[names.joint_names("right")[-1]].find("limit")
    assert carriage is not None and "lower" in carriage.attrib and "upper" in carriage.attrib
    assert (float(carriage.attrib["lower"]), float(carriage.attrib["upper"])) == (-0.006, 0.040)
    assert not any(name.startswith("left/") for name in links)
    defined = {material.attrib["name"] for material in robot.findall("material") if "name" in material.attrib}
    used = {m.attrib["name"] for m in robot.iter("material") if "name" in m.attrib and m.find("color") is None and m.find("texture") is None}
    assert used <= defined


@pytest.mark.parametrize("arm", names.ARMS)
def test_tcp_is_the_workspace_calibration(arm):
    """<arm>/tcp sits at workspace.yaml pen_tip_offset_* in its tip_frame, identity rotation."""
    section = yaml.safe_load((repo_root() / "config" / "workspace.yaml").read_text())[arm]
    tcp = _joints(ET.fromstring(robot_description(arms=(arm,))))[f"{arm}/tcp_joint"]
    assert tcp.attrib.get("type") == "fixed"
    parent, child, origin = tcp.find("parent"), tcp.find("child"), tcp.find("origin")
    assert parent is not None and child is not None and origin is not None
    assert parent.attrib.get("link") == section["tip_frame"] == names.tool_mount_frame(arm)
    assert child.attrib.get("link") == names.tcp_frame(arm)
    xyz_str, rpy_str = origin.attrib.get("xyz"), origin.attrib.get("rpy")
    assert xyz_str is not None and rpy_str is not None
    xyz = [float(v) for v in xyz_str.split()]
    assert xyz == [float(section[f"pen_tip_offset_{axis}"]) for axis in "xyz"]
    assert [float(v) for v in rpy_str.split()] == [0.0, 0.0, 0.0]


def test_the_source_urdf_is_untouched():
    """The ROS description overrides the carriage travel; urdf/tatbot.urdf keeps the CAD 0..44 mm."""
    source = ET.parse(repo_root() / "urdf" / "tatbot.urdf").getroot()
    limit = _joints(source)["right/left_carriage_joint"].find("limit")
    assert limit is not None and "lower" in limit.attrib and "upper" in limit.attrib
    assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == (0.0, 0.044)


def test_joint_corrections_change_geometry_at_raw_encoders_without_changing_limits(tmp_path):
    import sys

    import numpy as np
    sys.path.insert(0, str(repo_root()/'scripts/vision'))
    from urdf_kinematics import UrdfChain, driver_joint_names

    repo = repo_root()
    (tmp_path/'config').mkdir()
    (tmp_path/'urdf').symlink_to(repo/'urdf', target_is_directory=True)
    for source in ('arms.json', 'trossen'):
        (tmp_path/'config'/source).symlink_to(repo/'config'/source)
    # Both models set their own offsets: the committed workspace carries adopted ones.
    workspace = yaml.safe_load((repo/'config/workspace.yaml').read_text())

    def describe(joint_offsets):
        workspace['right']['joint_offsets_rad'] = joint_offsets
        (tmp_path/'config/workspace.yaml').write_text(yaml.safe_dump(workspace))
        return robot_description(tmp_path, arms=('right',))

    offsets = [0., .012, -.018, .009, -.007, 0., 0.]
    nominal, corrected = describe([0.]*7), describe(offsets)
    models = []
    for name, xml in (('nominal', nominal), ('corrected', corrected)):
        path = tmp_path/f'{name}.urdf'
        path.write_text(xml)
        models.append(UrdfChain(path))
    for q in ([0., -.3, .5, -.8, .4, .2, 0.], [.4, -.8, 1., -.5, .7, -.4, .02],
              [-.4, -.5, .8, -.9, -.3, .9, -.003]):
        raw = dict(zip(driver_joint_names('right'), q, strict=True))
        shifted = dict(zip(driver_joint_names('right'), np.array(q)+offsets, strict=True))
        for frame in ('right/tcp', 'right/realsense_color_optical_frame'):
            np.testing.assert_allclose(models[1].link_pose(frame, raw), models[0].link_pose(frame, shifted), atol=1e-12)
    before, after = _joints(ET.fromstring(nominal)), _joints(ET.fromstring(corrected))
    for name in names.joint_names('right'):
        limit_before, limit_after = before[name].find('limit'), after[name].find('limit')
        assert limit_before is not None and limit_after is not None
        assert ET.tostring(limit_before) == ET.tostring(limit_after)


def test_registration_places_the_base(tmp_path):
    matrix = [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    path = tmp_path / "arm-registration-right-current.json"
    path.write_text('{"world_from_arm_base": %s}' % matrix)
    joint = _joints(ET.fromstring(robot_description(arms=("right",), registrations={"right": path})))["right/world_joint"]
    origin = joint.find("origin")
    assert origin is not None
    xyz_str, rpy_str = origin.attrib.get("xyz"), origin.attrib.get("rpy")
    assert xyz_str is not None and rpy_str is not None
    assert [float(v) for v in xyz_str.split()] == [0.1, -0.2, 0.3]
    assert [float(v) for v in rpy_str.split()] == pytest.approx([0.0, 0.0, 1.5707963])


@pytest.mark.parametrize("hardware,arms", [v for v in VARIANTS if v.values[0]])
def test_ros2_control_systems(hardware, arms):
    robot = ET.fromstring(robot_description(arms=arms, hardware=hardware))
    systems = robot.findall("ros2_control")
    assert [s.get("name") for s in systems] == [names.hardware_name(arm) for arm in arms]
    for arm, system in zip(arms, systems, strict=True):
        plugin_elem = system.find("hardware/plugin")
        assert plugin_elem is not None
        plugin = plugin_elem.text
        params = {p.get("name"): p.text or "" for p in system.findall("hardware/param")}
        if hardware == "mock":
            assert plugin == "mock_components/GenericSystem" and params == {}
        else:
            assert plugin == "tatbot_hardware/TatbotArm"
            extra = ("fake_page_z", "fake_page_stiffness_n_m") if hardware == "fake" else ()  # stack.yaml fake.* is set
            assert tuple(params) == names.HARDWARE_PARAMETERS + extra
            assert (params["arm"], params["sdk"]) == (arm, hardware)
            assert (params["base_frame"], params["tcp_frame"]) == (names.base_frame(arm), names.tcp_frame(arm))
        joints = system.findall("joint")
        assert tuple(j.get("name") for j in joints) == names.joint_names(arm)
        for joint in joints:
            assert tuple(i.get("name") for i in joint.findall("command_interface")) == names.JOINT_COMMAND_INTERFACES
            assert tuple(i.get("name") for i in joint.findall("state_interface")) == names.JOINT_STATE_INTERFACES
        (gpio,) = system.findall("gpio")
        assert gpio.get("name") == names.safety_gpio(arm)
        states = gpio.findall("state_interface")
        assert tuple(i.get("name") for i in gpio.findall("command_interface")) == names.SAFETY_COMMAND_INTERFACES
        assert tuple(i.get("name") for i in states) == names.SAFETY_STATE_INTERFACES
        initial = {i.get("name"): i.find("param") for i in states}
        if hardware == "mock":
            assert {k: float(v.text) for k, v in initial.items() if v is not None and v.text is not None} == {
                k: (1.0 if k == "estop_ok" else 0.0) for k in names.SAFETY_STATE_INTERFACES}
            staged = yaml.safe_load((repo_root() / "config" / "trossen" / "tatbot.yaml").read_text())
            positions = []
            for j in joints:
                p_elem = j.find("state_interface/param")
                assert p_elem is not None and p_elem.text is not None
                positions.append(float(p_elem.text))
            assert positions == staged["follower"]["staged_positions"]
        else:
            assert all(v is None for v in initial.values())


def test_hardware_params_follow_the_stack():
    stack = load_stack()
    stack["estop"].update(source="udp", relay_addr="192.0.2.7")
    stack["rt"]["cpus"] = [4, 5]
    stack["flight_path"] = "/tmp/run"
    params = hardware_params(None, "right", stack, "fake")
    assert params["estop_timeout_s"] == repr(stack["estop"]["udp_timeout_s"])
    assert params["estop_relay_addr"] == "192.0.2.7"
    assert params["rt_cpus"] == "4,5"
    assert params["tip_lag_trip_m"] == repr(stack["safety"]["tip_lag"]["trip_m"])
    assert params["landing_budget_s"] == repr(stack["safety"]["landing"]["budget_s"])
    assert params["flight_path"] == "/tmp/run"
    assert len(params["staged_positions"].split(",")) == 7
    stack["estop"]["source"] = "serial"
    assert hardware_params(None, "right", stack, "fake")["estop_timeout_s"] == repr(stack["estop"]["serial_timeout_s"])


def test_real_needs_an_arm_address():
    stack = load_stack()
    stack["profile"] = "no-such-profile"
    assert hardware_params(None, "right", stack, "fake")["ip"] == ""
    with pytest.raises(ValueError, match="no address"):
        hardware_params(None, "right", stack, "real")
    with pytest.raises(ValueError):
        hardware_params(None, "right", stack, "sim")


@pytest.mark.skipif(not shutil.which("check_urdf"), reason="check_urdf (liburdfdom-tools) not installed")
@pytest.mark.parametrize("hardware,arms", VARIANTS)
def test_check_urdf(tmp_path, hardware, arms):
    path = tmp_path / "tatbot.urdf"
    path.write_text(robot_description(arms=arms, hardware=hardware))
    out = subprocess.run(["check_urdf", str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"root Link: world has {len(arms)} child(ren)" in out.stdout


def test_tcp_through_pinocchio():
    """What tatbot_motion sees: FK of right/tcp in right/tool_mount is the calibration at any q."""
    pin = pytest.importorskip("pinocchio")
    import numpy as np

    section = yaml.safe_load((repo_root() / "config" / "workspace.yaml").read_text())["right"]
    model = pin.buildModelFromXML(robot_description(arms=("right",)))
    data = model.createData()
    q = np.array([0.3, 0.8, 0.9, -0.2, 0.4, 1.1, 0.012])
    pin.framesForwardKinematics(model, data, q)
    mount = data.oMf[model.getFrameId(names.tool_mount_frame("right"))]
    tcp = data.oMf[model.getFrameId(names.tcp_frame("right"))]
    offset = mount.actInv(tcp)
    assert offset.translation == pytest.approx([float(section[f"pen_tip_offset_{a}"]) for a in "xyz"], abs=1e-12)
    assert offset.rotation == pytest.approx(np.eye(3), abs=1e-12)


def test_fake_gets_no_arm_address_and_an_optional_page():
    stack = load_stack()
    params = hardware_params(None, "right", stack, "fake")
    # stack.yaml puts the fake paper at page.fixed.right's z, so `up --hardware fake --page fixed` touches it
    assert params["ip"] == "" and float(params["fake_page_z"]) == stack["page"]["fixed"]["right"]["xyz"][2]
    stack["fake"] = {"page_z": None}
    assert "fake_page_z" not in hardware_params(None, "right", stack, "fake")
    if HAS_PROFILE:
        assert "fake_page_z" not in hardware_params(None, "right", stack, "real")


@pytest.mark.parametrize("arm", names.ARMS)
def test_carriage_limits_are_the_arms_controller_config(arm):
    import json

    config = json.loads((repo_root() / "config" / "arms.json").read_text())["arms"][arm]["controller_config"]
    limit = yaml.safe_load((repo_root() / config).read_text())["joint_limits"][6]
    carriage = _joints(ET.fromstring(robot_description(arms=(arm,))))[names.joint_names(arm)[-1]].find("limit")
    assert carriage is not None and "lower" in carriage.attrib and "upper" in carriage.attrib
    assert (float(carriage.attrib["lower"]), float(carriage.attrib["upper"])) == pytest.approx(
        (limit["position_min"], limit["position_max"]), abs=1e-6)
