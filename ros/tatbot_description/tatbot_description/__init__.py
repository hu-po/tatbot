"""The robot description the whole stack uses, generated from the repo's own geometry.

`robot_description()` is importable without ROS: the kinematic part (arm subtrees of
urdf/tatbot.urdf, the corrected carriage limits, `<arm>/tcp` from config/workspace.yaml, and the
fixed `world -> <arm>/base_link` joints from the adopted registrations) is plain ElementTree. Only
the `<ros2_control>` blocks (hardware != None) run xacro, through the `xacro` Python module, on
urdf/ros2_control.xacro. tatbot_motion calls it with hardware=None; stack.launch.py with a hardware.
"""
from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tatbot_description import names

XACRO = Path("ros") / "tatbot_description" / "urdf" / "ros2_control.xacro"  # both read in the checkout
STACK_YAML = Path("ros") / "tatbot_bringup" / "config" / "stack.yaml"
# config/trossen/tatbot.yaml `follower` keys the driver takes as parameters of the same name. The
# follower section describes the pink (right) arm's qualified carriage; every TatbotArm reads it.
FOLLOWER_KEYS = ("carriage_contact_cap_n", "carriage_contact_deflect_m", "carriage_retract_m", "staged_positions")


def repo_root(repo: str | Path | None = None) -> Path:
    """The repo checkout the stack reads config from: the argument, else $TATBOT_REPO, else this file's tree."""
    if repo:
        return Path(repo).expanduser().resolve()
    if os.environ.get("TATBOT_REPO"):
        return Path(os.environ["TATBOT_REPO"]).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "urdf" / "tatbot.urdf").is_file() and (parent / "config" / "workspace.yaml").is_file():
            return parent
    raise FileNotFoundError("no repo: pass repo= or set TATBOT_REPO (the checkout holding urdf/ and config/)")


def tcp_offset(repo: Path, arm: str) -> tuple[str, tuple[float, float, float]]:
    """(parent frame, xyz metres) of the fitted tool's working point from config/workspace.yaml."""
    import yaml

    section = yaml.safe_load((repo / "config" / "workspace.yaml").read_text())[arm]
    xyz = tuple(float(section[f"pen_tip_offset_{axis}"]) for axis in "xyz")
    return section.get("tip_frame") or names.tool_mount_frame(arm), xyz


def joint_calibration(repo=None):
    """The shared model correction helper, also used by the wrist observer."""
    sys.path.insert(0, str(repo_root(repo) / 'scripts/lib'))
    import kinematic_calibration
    return kinematic_calibration


def registration(path: str | Path) -> list[list[float]]:
    """world_from_arm_base (4x4) from a `tatbot.arm-registration/1` record."""
    return json.loads(Path(path).expanduser().read_text())["world_from_arm_base"]


def carriage_limits(repo: Path, arm: str) -> tuple[float, float]:
    """(lower, upper) metres: the carriage travel of the arm's controller config (config/arms.json
    `controller_config` joint_limits[6]), not the CAD 0-44 mm in urdf/tatbot.urdf. Rounded to 1 um."""
    import yaml

    config = json.loads((repo / "config" / "arms.json").read_text())["arms"][arm]["controller_config"]
    limit = yaml.safe_load((repo / config).read_text())["joint_limits"][6]
    return round(float(limit["position_min"]), 6), round(float(limit["position_max"]), 6)


def _rpy(matrix) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) of a 3x3 rotation."""
    r = matrix
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        return math.atan2(r[2][1], r[2][2]), pitch, math.atan2(r[1][0], r[0][0])
    return math.atan2(-r[1][2], r[1][1]), pitch, 0.0


def _fmt(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _arm_subtree(robot: ET.Element, arm: str, repo: Path) -> tuple[list[ET.Element], ET.Element]:
    import yaml

    links = {e.get("name"): e for e in robot.findall("link")}
    joints = robot.findall("joint")
    section = yaml.safe_load((repo / 'config/workspace.yaml').read_text())[arm]
    calibration = joint_calibration(repo)
    calibration.apply_origins(joints, arm, section.get(calibration.KEY))
    children: dict[str, list[ET.Element]] = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    mount = next(j for j in joints if j.find("child").get("link") == names.base_frame(arm))
    keep, stack = [], [names.base_frame(arm)]
    while stack:
        link = stack.pop()
        keep.append(links[link])
        for joint in children.get(link, []):
            if joint.get("name") == f"{arm}/{names.CARRIAGE}":
                limit = joint.find("limit")
                lower, upper = carriage_limits(repo, arm)
                limit.set("lower", repr(lower))
                limit.set("upper", repr(upper))
            keep.append(joint)
            stack.append(joint.find("child").get("link"))
    for mesh in (m for e in keep for m in e.iter("mesh")):
        filename = mesh.get("filename", "")
        if filename and "://" not in filename:
            mesh.set("filename", "file://" + str(repo / "urdf" / filename))
    return keep, mount


def robot_description(repo: str | Path | None = None, *, arms=names.DEFAULT_ARMS, hardware: str | None = None,
                      registrations: dict | None = None, ros2_control: dict | None = None) -> str:
    """URDF XML text rooted at `world`, one arm subtree per entry of `arms`.

    registrations: {arm: path to arm-registration-<arm>-current.json, or a 4x4 world_from_arm_base}.
        An arm without one hangs at its nominal urdf/tatbot.urdf mount (world = the URDF root); the
        launch allows that only when the page is not the camera-tracked stencil (or the arm not real).
    hardware: None (kinematics only) | "mock" | "fake" | "real" -> <ros2_control> per arm via xacro.
    ros2_control: the effective stack.yaml dict (launch overrides applied, plus "flight_path") that
        hardware_params() reads; None = the checkout's stack.yaml.
    """
    root = repo_root(repo)
    source = ET.parse(root / "urdf" / "tatbot.urdf").getroot()
    out = ET.Element("robot", {"name": "tatbot"})
    out.extend(source.findall("material"))  # the arm links reference the file's named materials
    ET.SubElement(out, "link", {"name": names.WORLD})
    for arm in arms:
        elements, mount = _arm_subtree(source, arm, root)
        value = (registrations or {}).get(arm)
        if value is not None:
            matrix = registration(value) if isinstance(value, (str, Path)) else value
            xyz, rpy = [row[3] for row in matrix[:3]], _rpy([row[:3] for row in matrix[:3]])
        else:
            origin = mount.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        joint = ET.SubElement(out, "joint", {"name": f"{arm}/world_joint", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
        ET.SubElement(joint, "parent", {"link": names.WORLD})
        ET.SubElement(joint, "child", {"link": names.base_frame(arm)})
        out.extend(elements)
        parent, tip = tcp_offset(root, arm)
        ET.SubElement(out, "link", {"name": names.tcp_frame(arm)})
        joint = ET.SubElement(out, "joint", {"name": f"{arm}/tcp_joint", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": _fmt(tip), "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": names.tcp_frame(arm)})
    text = ET.tostring(out, encoding="unicode")
    if hardware is None:
        return text
    return _with_ros2_control(text, root, arms, hardware, ros2_control)


def load_stack(repo: str | Path | None = None, path: str | Path | None = None) -> dict:
    """stack.yaml as a dict: `path`, else the checkout's ros/tatbot_bringup/config/stack.yaml."""
    import yaml

    path = Path(path).expanduser() if path else repo_root(repo) / STACK_YAML
    return yaml.safe_load(path.read_text())


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}_"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _param_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_param_text(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    return "" if value is None else str(value)


def hardware_params(repo: str | Path | None, arm: str, stack: dict, hardware: str) -> dict[str, str]:
    """The `<param>`s of one arm's `<ros2_control>` system (README "ros2_control"). mock takes none.

    Values resolve from the checkout, never from literals here: the arm address (real only; fake gets
    none) and SDK end effector through config/arms.json and config/profiles/<profile>.json, the
    carriage cap, deflection, retract and staged pose from config/trossen/tatbot.yaml `follower`,
    everything else from `stack`. `stack["flight_path"]` (set by the launch) is where the driver
    writes its flight rings. fake alone may add `fake_page_z` (stack.yaml `fake.page_z`, base-frame z of
    a simulated paper plane the fake tip cannot pass) after the HARDWARE_PARAMETERS.
    """
    if hardware == "mock":
        return {}
    if hardware not in ("fake", "real"):
        raise ValueError(f"hardware {hardware!r}: mock | fake | real")
    import yaml

    root = repo_root(repo)
    arm_cfg = json.loads((root / "config" / "arms.json").read_text())["arms"][arm]
    ip = ""  # the fake SDK is never handed an arm address
    if hardware == "real":
        profile = root / "config" / "profiles" / f"{stack['profile']}.json"
        ip = json.loads(profile.read_text())["driver"].get(arm_cfg["profile_ip_field"], "") if profile.is_file() else ""
        if not ip:
            raise ValueError(f"no address for the {arm} arm: {profile} driver.{arm_cfg['profile_ip_field']}")
    trossen = yaml.safe_load((root / "config" / "trossen" / "tatbot.yaml").read_text())
    follower = trossen["follower"]
    # The arm's controller role (config/trossen/<role>.yaml) keys its section of tatbot.yaml.
    role = Path(arm_cfg["controller_config"]).stem
    estop, rt, probe = stack["estop"], stack["rt"], stack.get("probe") or {}
    params = {
        "arm": arm,
        "sdk": hardware,
        "ip": ip,
        "end_effector": arm_cfg["sdk_end_effector"],
        "estop_source": estop["source"],
        "estop_device": estop["serial_device"],
        "estop_timeout_s": estop["udp_timeout_s" if estop["source"] == "udp" else "serial_timeout_s"],
        "estop_udp_port": estop["udp_port"],
        "estop_relay_addr": estop["relay_addr"],
        "estop_debounce_frames": estop["debounce_frames"],
        "probe_enabled": bool(probe.get("enabled", False)),
        "probe_udp_port": probe.get("udp_port", 7641),
        "probe_timeout_s": probe.get("timeout_s", 0.2),
        "probe_relay_addr": probe.get("relay_addr", ""),
        "rt_priority": rt["priority"],
        "rt_cpus": rt["cpus"],
        "base_frame": names.base_frame(arm),
        "tcp_frame": names.tcp_frame(arm),
        **_flatten(stack["safety"]),
        **{key: follower[key] for key in FOLLOWER_KEYS},
        "carriage_qualified": bool((trossen.get(role) or {}).get("carriage_qualified", True)),
        "flight_path": stack.get("flight_path", ""),
    }
    page_z = (stack.get("fake") or {}).get("page_z")
    if hardware == "fake" and page_z is not None:
        params["fake_page_z"] = float(page_z)
        stiffness = (stack.get("fake") or {}).get("page_stiffness_n_m")
        if stiffness:
            params["fake_page_stiffness_n_m"] = float(stiffness)
    text = {key: _param_text(value) for key, value in params.items()}
    bad = [f"{k}={v}" for k, v in text.items() if ";" in v or "=" in v]
    if bad:
        raise ValueError(f"hardware parameter values may not hold ';' or '=': {bad}")
    return text


def ros2_control_xml(repo: str | Path | None, arm: str, hardware: str, stack: dict) -> str:
    """One arm's `<ros2_control>` element, rendered from urdf/ros2_control.xacro by the xacro module."""
    import xacro

    initial = []
    if hardware == "mock":  # GenericSystem starts where the real arm rests: the staged pose
        import yaml

        follower = yaml.safe_load((repo_root(repo) / "config" / "trossen" / "tatbot.yaml").read_text())["follower"]
        initial = [repr(float(v)) for v in follower["staged_positions"]]
    mappings = {
        "arm": arm,
        "hardware": hardware,
        "joints": " ".join(names.joint_names(arm)),
        "initial": " ".join(initial),
        "safety_states": " ".join(names.SAFETY_STATE_INTERFACES),
        "safety_commands": " ".join(names.SAFETY_COMMAND_INTERFACES),
        "params": ";".join(f"{k}={v}" for k, v in hardware_params(repo, arm, stack, hardware).items()),
    }
    doc = xacro.process_file(str(repo_root(repo) / XACRO), mappings=mappings)
    return doc.getElementsByTagName("ros2_control")[0].toxml()


def _with_ros2_control(urdf: str, repo, arms, hardware: str, stack: dict | None) -> str:
    """Append one <ros2_control> system per arm."""
    stack = stack if stack is not None else load_stack(repo)
    robot = ET.fromstring(urdf)
    for arm in arms:
        robot.append(ET.fromstring(ros2_control_xml(repo, arm, hardware, stack)))
    return ET.tostring(robot, encoding="unicode")
