"""The generated description: one root, the carriage travel, the TCP from workspace.yaml, and the
<ros2_control> systems for mock, fake and real hardware."""
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from tatbot_description import hardware_params, load_stack, names, repo_root, robot_description

# `real` needs the private arm addresses (config/profiles/<profile>.json); a public clone has none.
HAS_PROFILE = (repo_root() / "config" / "profiles" / f"{load_stack()['profile']}.json").is_file()
VARIANTS = [pytest.param(hw, ("right",), id=f"{hw}-right",
                         marks=pytest.mark.skipif(hw == "real" and not HAS_PROFILE, reason="no profile"))
            for hw in (None, "mock", "fake", "real")] + [
    pytest.param(hw, ("xarm",), id=f"{hw}-xarm") for hw in (None, "mock", "fake")]   # no real backend for it yet
ARMS_JSON = json.loads((repo_root() / "config" / "arms.json").read_text())["arms"]
TROSSEN_ARMS = tuple(arm for arm in names.ARMS if ARMS_JSON[arm]["controller_config"] is not None)


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
    from urdf_kinematics import UrdfChain

    repo = repo_root()
    (tmp_path/'config').mkdir()
    (tmp_path/'urdf').symlink_to(repo/'urdf', target_is_directory=True)
    for source in ('arms.json', 'arm-models', 'trossen', 'tools'):
        (tmp_path/'config'/source).symlink_to(repo/'config'/source)
    # Both models set their own offsets: the committed workspace carries adopted ones. The file is edited in place,
    # as its adopt steps edit it: tool_spec reads a YAML subset (inline arrays), not every YAML a dump can write.
    workspace = (repo/'config/workspace.yaml').read_text()
    right = workspace.index('\nright:')
    line = re.compile(r'(?m)^  joint_offsets_rad: .*$')

    def describe(joint_offsets):
        edited = workspace[:right] + line.sub(f'  joint_offsets_rad: {json.dumps(joint_offsets)}', workspace[right:], 1)
        (tmp_path/'config/workspace.yaml').write_text(edited)
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
        raw = dict(zip(names.joint_names('right'), q, strict=True))
        shifted = dict(zip(names.joint_names('right'), np.array(q)+offsets, strict=True))
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
            trossen = arm in TROSSEN_ARMS
            plant = () if trossen else ("fake_delay_s", "fake_lag_s")   # its model's fake_plant
            extra = (*plant, "fake_page_z", "fake_page_stiffness_n_m") if hardware == "fake" else ()   # stack.yaml fake.*
            other = names.XARM_HARDWARE_PARAMETERS if trossen else names.TROSSEN_HARDWARE_PARAMETERS
            expected = tuple(k for k in names.HARDWARE_PARAMETERS if k not in other)
            assert tuple(params) == expected + extra
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
            from tatbot_description import rest_positions

            positions = []
            for j in joints:
                p_elem = j.find("state_interface/param")
                assert p_elem is not None and p_elem.text is not None
                positions.append(float(p_elem.text))
            assert positions == rest_positions(None, arm)   # the WXAI's golden staged pose, the xArm's model rest
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
    assert params["snag_off_m"] == repr(stack["safety"]["snag"]["off_m"])
    assert params["landing_budget_s"] == repr(stack["safety"]["landing"]["budget_s"])
    assert params["flight_path"] == "/tmp/run"
    assert len(params["staged_positions"].split(",")) == 7
    stack["estop"]["source"] = "serial"
    assert hardware_params(None, "right", stack, "fake")["estop_timeout_s"] == repr(stack["estop"]["serial_timeout_s"])


@pytest.mark.skipif(not (repo_root() / "config" / "workspace.yaml").is_file(), reason="no rig workspace")
def test_a_stack_started_on_the_calibration_profile_connects_with_its_velocity_integral():
    """stack.yaml tuning (tatbot ros up --tuning) lays ros_tuning.profiles.<name>.<role> over the role's tuning at
    connect: calibration's velocity integral x1, the role's position gain kept (bench 2026-10-22)."""
    drawing = hardware_params(None, "right", load_stack(), "fake")
    calibration = hardware_params(None, "right", {**load_stack(), "tuning": "calibration"}, "fake")
    assert [float(v) for v in calibration["velocity_ki_scale"].split(",")] == [1.0] * 6
    assert drawing["velocity_ki_scale"] != calibration["velocity_ki_scale"]
    assert calibration["position_kp_scale"] == drawing["position_kp_scale"]
    with pytest.raises(ValueError, match="no profile 'nonesuch'"):
        hardware_params(None, "right", {**load_stack(), "tuning": "nonesuch"}, "fake")


@pytest.mark.skipif(not (repo_root() / "config" / "workspace.yaml").is_file(), reason="no rig workspace")
def test_the_palm_and_carriage_offset_come_from_the_checkout(tmp_path):
    root = repo_root()
    workspace = yaml.safe_load((root / "config" / "workspace.yaml").read_text())
    for arm in TROSSEN_ARMS:
        params = hardware_params(None, arm, load_stack(), "fake")
        payload = workspace[arm].get("payload") or {}
        if payload.get("identified_with") == workspace[arm]["tool_id"]:
            assert params["palm_mass_kg"] == repr(float(payload["palm_mass_kg"]))
            assert params["palm_com_m"] == ",".join(repr(float(v)) for v in payload["palm_com_m"])
        assert params["motor_parameters"] == "wxai_v0_20250509"
        assert [float(v) for v in params["idle_velocity_ki_scale"].split(",")] == [8.0, 8.0, 8.0, 3.0, 3.0, 3.0]
        assert float(params["carriage_position_offset_m"]) == pytest.approx(0.0007)
        role = Path(json.loads((root / "config" / "arms.json").read_text())["arms"][arm]["controller_config"]).stem
        tuning = yaml.safe_load((root / "config" / "trossen" / "tatbot.yaml").read_text())["ros_tuning"].get(role)
        for key in ("position_kp_scale", "velocity_kp_scale", "velocity_ki_scale"):
            assert params[key] == ("" if tuning is None else ",".join(repr(float(v)) for v in tuning[key]))
    # a payload identified with another tool, or on another preset, leaves the preset's palm
    for field, value in (("identified_with", "another-tool"), ("preset", "wxai_v0_follower")):
        copy = tmp_path / field
        for rel in ("config/arms.json", "config/trossen/follower.yaml", "config/trossen/leader.yaml"):
            (copy / rel).parent.mkdir(parents=True, exist_ok=True)
            (copy / rel).write_text((root / rel).read_text())
        text = (root / "config" / "workspace.yaml").read_text()   # edited in place: tool_spec reads a YAML subset
        right = text.index("\nright:")
        text = text[:right] + re.sub(rf"(?m)^    {field}: .*$", f"    {field}: {value}", text[right:], count=1)
        (copy / "config" / "workspace.yaml").write_text(text)
        from tatbot_description import _palm
        arm_cfg = json.loads((copy / "config" / "arms.json").read_text())["arms"]["right"]
        assert _palm(copy, arm_cfg) == {"palm_mass_kg": "", "palm_com_m": ""}


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
    # the fake paper is at page.fixed.<arm>'s z, so `up --hardware fake --page fixed` touches it
    assert params["ip"] == "" and float(params["fake_page_z"]) == stack["page"]["fixed"]["right"]["xyz"][2]
    assert float(hardware_params(None, "xarm", stack, "fake")["fake_page_z"]) == stack["page"]["fixed"]["xarm"]["xyz"][2]
    stack["fake"] = {"page": False}
    assert "fake_page_z" not in hardware_params(None, "right", stack, "fake")
    if HAS_PROFILE:
        assert "fake_page_z" not in hardware_params(None, "right", stack, "real")


@pytest.mark.parametrize("arm", TROSSEN_ARMS)
def test_carriage_limits_are_the_arms_controller_config(arm):
    config = ARMS_JSON[arm]["controller_config"]
    limit = yaml.safe_load((repo_root() / config).read_text())["joint_limits"][6]
    carriage = _joints(ET.fromstring(robot_description(arms=(arm,))))[names.joint_names(arm)[-1]].find("limit")
    assert carriage is not None and "lower" in carriage.attrib and "upper" in carriage.attrib
    assert (float(carriage.attrib["lower"]), float(carriage.attrib["upper"])) == pytest.approx(
        (limit["position_min"], limit["position_max"]), abs=1e-6)


def test_the_xarm_comes_from_its_own_description_with_its_own_kinematics():
    """urdf/xarm7.urdf (scripts/gen_arm_urdf.py): at zero the flange stands where config/xarm-kinematics.yaml's joint
    origins chain it, this arm's factory calibration, within 3 mm of where the xArm 7's DH parameters put it (206 mm
    out, 52.5 + 77.5 + 76, and 120.5 mm up, 267 + 293 - 342.5 - 97), pointing down; at its model's rest the pen is
    straight down 400 mm out, as the arm stood at that home (2026-11-03)."""
    import numpy as np
    import yaml
    from tatbot_description import arm_model, repo_root
    from tatbot_description.transforms import rpy_matrix
    from tatbot_motion import Kinematics

    text = robot_description(arms=("xarm",))
    robot = ET.fromstring(text)
    links = {link.attrib["name"] for link in robot.findall("link")}
    assert {"xarm/base_link", "xarm/link7", "xarm/tool_mount", "xarm/tattoo_needle", "xarm/tcp"} <= links
    assert not any(n.startswith("right/") for n in links)
    meshes = [m.get("filename") for m in robot.iter("mesh")]
    assert meshes and all(Path(m.removeprefix("file://")).is_file() for m in meshes)
    kin = Kinematics(text, "xarm")
    assert kin.joint_names == names.joint_names("xarm") and kin.carriage is None
    flange = np.linalg.inv(kin.frame(np.zeros(7), "xarm/base_link")) @ kin.frame(np.zeros(7), "xarm/link7")
    chained = np.eye(4)
    for joint in yaml.safe_load((repo_root(None) / "config" / "xarm-kinematics.yaml").read_text())["kinematics"].values():
        chained = chained @ rpy_matrix([joint["x"], joint["y"], joint["z"]],
                                       [joint["roll"], joint["pitch"], joint["yaw"]])
    np.testing.assert_allclose(flange, chained, atol=1e-6)
    np.testing.assert_allclose(flange[:3, 3], [0.206, 0.0, 0.1205], atol=0.003)
    np.testing.assert_allclose(flange[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
    tcp = kin.fk(np.array(arm_model.for_arm("xarm").rest))
    np.testing.assert_allclose(tcp[:3, 2], [0.0, 0.0, -1.0], atol=0.01)
    assert tcp[0, 3] == pytest.approx(0.400, abs=0.002)


def test_both_models_in_one_description_share_its_materials_once():
    robot = ET.fromstring(robot_description(arms=("right", "xarm")))
    materials = [m.get("name") for m in robot.findall("material")]
    assert len(materials) == len(set(materials)) and "xarm/White" in materials
    assert {names.tcp_frame("right"), names.tcp_frame("xarm")} <= {link.get("name") for link in robot.findall("link")}


def test_the_xarm_takes_its_own_parameters_and_rests_at_its_models_rest():
    from tatbot_description import arm_model, rest_positions

    stack = load_stack()
    params = hardware_params(None, "xarm", stack, "fake")
    assert not set(params) & set(names.TROSSEN_HARDWARE_PARAMETERS)
    assert params["controller"] == "xarm" and params["collision_sensitivity"] == "3"
    payload = yaml.safe_load((repo_root() / "config" / "workspace.yaml").read_text())["xarm"]["payload"]
    assert float(params["payload_mass_kg"]) == payload["mass_kg"]
    assert [float(v) for v in params["payload_com_m"].split(",")] == payload["com_m"]
    assert not set(hardware_params(None, "right", stack, "fake")) & set(names.XARM_HARDWARE_PARAMETERS)
    assert params["joints"] == " ".join(names.joint_names("xarm")) and params["carriage_joint"] == ""
    assert float(params["command_delay_s"]) == pytest.approx(0.095)
    assert (float(params["fake_delay_s"]), float(params["fake_lag_s"])) == pytest.approx((0.083, 0.007))
    assert rest_positions(None, "xarm") == list(arm_model.for_arm("xarm").rest)
    assert [float(v) for v in params["staged_positions"].split(",")] == pytest.approx(rest_positions(None, "xarm"))
    if HAS_PROFILE:   # real takes its control box's address from the profile
        assert hardware_params(None, "xarm", stack, "real")["ip"]


def test_a_ros_owners_fitted_cartridge_is_the_tcp_and_the_tool_it_models(tmp_path):
    """On a ROS owner (calib/ beside its deployed repo/) the cartridge fitted there, not git's tool, hangs off the mount:
    its tip is <arm>/tcp and the tool block is its datasheet's, re-rendered at every start; no record, git's as is."""
    import cartridge_fit
    import tool_spec

    repo, source = tmp_path / "repo", repo_root()
    (repo / "config").mkdir(parents=True)
    (repo / "urdf").symlink_to(source / "urdf", target_is_directory=True)
    for name in ("arms.json", "arm-models", "trossen", "tools"):
        (repo / "config" / name).symlink_to(source / "config" / name)
    shutil.copy(source / "config" / "workspace.yaml", repo / "config" / "workspace.yaml")
    committed = robot_description(repo, arms=("xarm",))
    assert committed == robot_description(arms=("xarm",)).replace(str(source), str(repo))

    other = "lutin-5rl-mast"
    seed = cartridge_fit.seeded(tool_spec.read_workspace(repo)["xarm"], "xarm")
    cartridge_fit.write(repo, cartridge_fit.fit(seed, other, tool_spec.load_tool(other, repo).touchoff_nominal_m,
                                                instance=None, utc="2026-11-10T18:00:00Z"))
    fitted = ET.fromstring(robot_description(repo, arms=("xarm",)))
    tcp = _joints(fitted)["xarm/tcp_joint"].find("origin")
    assert tcp is not None
    assert [float(v) for v in tcp.attrib["xyz"].split()] == list(tool_spec.load_tool(other, repo).touchoff_nominal_m)
    import tool_urdf

    model = ET.fromstring(tool_urdf.build(repo, repo / "urdf" / "xarm7.urdf", ["xarm"], tool_spec.read_workspace(repo)))
    pen, before, its = (_joints(robot)["xarm/tattoo_pen_joint"].find("origin")
                        for robot in (fitted, ET.fromstring(committed), model))
    assert pen is not None and before is not None and its is not None
    assert pen.attrib == its.attrib != before.attrib
