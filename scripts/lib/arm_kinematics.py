"""Arm kinematics in numpy — a mirror of square_probe.cpp.

Contract: docs/surface-formats.md. The C++ executor (`cpp/teleop/square_probe.cpp`) is the
authority on what the arm does; this module exists so the Python stages can
(a) compute the same FK the executor uses when they turn a surface into tip
samples, and (b) derive the tool axis and the ballpoint tip from the URDF and
the touch-off so the samples file can carry them and the executor can refuse a
mismatch.

The numbers the executor and this module must agree on (carriage-IK window,
damping) come from `config/motion_constants.json` via
`motion_constants` — the same file renders `cpp/teleop/motion_constants.hpp`, and
every samples file carries its SHA so the executor refuses a planner built
from different values. The kinematic model itself (joint origins, axes, limits,
tip offset) is still restated here; the tip-constant test is what catches
drift in those.

Frames: the C++ FK works in `right/base_link` ("base"). The URDF root is
`root` = base + BASE_IN_ROOT with no rotation. Only translations differ.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from motion_constants import C
from tatbot_cli import arms as arm_registry

REPO = Path(__file__).resolve().parents[2]
URDF_PATH = REPO / "urdf" / "tatbot.urdf"

# --- kinematic model (restated from the URDF; parity-tested against the executor) ---

JOINT_ORIGINS = np.array([
    [0.0, 0.0, 0.05725],
    [0.02, 0.0, 0.04625],
    [-0.264, 0.0, 0.0],
    [0.245, 0.0, 0.06],
    [0.06775, 0.0, 0.0455],
    [0.02895, 0.0, -0.0455],
])
JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])
JOINT_LOWER = np.array([
    -3.0543261909900767, 0.0, 0.0, -1.5707963267948966,
    -1.5707963267948966, -3.141592653589793])
JOINT_UPPER = np.array([
    3.0543261909900767, 3.141592653589793, 2.356194490192345,
    1.5707963267948966, 1.5707963267948966, 3.141592653589793])
JOINT_LIMIT_MARGIN_RAD = C.planner.joint_limit_margin_rad

BALLPOINT_TIP_IN_LINK6 = np.array([0.20498692817078468, 0.012312678895000949, -0.0005439999999999881])
CARRIAGE_AXIS_IN_LINK6 = np.array([0.0, 1.0, 0.0])

# --- planner caps and gains: config/motion_constants.json (shared with the executor) ---

DLS_DAMPING = C.planner.dls_damping
CARRIAGE_IK_MIN_M = C.carriage_ik.min_m
CARRIAGE_IK_MAX_M = C.carriage_ik.max_m
PLAN_MAX_TICKS = C.planner.max_ticks


def _base_translation_in_root(repo):
    """The translation-only planner must refuse an unhandled rotated mount."""
    from ink_spec import base_from_root_matrix
    transform = np.linalg.inv(base_from_root_matrix(repo, 'right'))
    if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-10, rtol=0):
        raise ValueError('draw planner requires parallel URDF-root and arm-base axes')
    return transform[:3, 3]


BASE_IN_ROOT = _base_translation_in_root(REPO)

ARM_JOINT_NAMES = tuple(f"right/joint_{i}" for i in range(6))
CARRIAGE_JOINT_NAME = "right/left_carriage_joint"
LINK6_NAME = "right/link_6"
TOOL_MOUNT_NAME = "right/tool_mount"


class PlanRefusal(RuntimeError):  # noqa: N818 - the contract's name
    """A joint limit or an IK solve refused the request."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- small algebra -----------------------------------------------------------

def axis_rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis — same closed form as the C++."""
    x, y, z = (float(v) for v in axis)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [x * x * one_c + c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, y * y * one_c + c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, z * z * one_c + c],
    ])


def skew(v) -> np.ndarray:
    x, y, z = (float(c) for c in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def orientation_error(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """0.5 * sum_c cross(R_cur[:, c], R_tgt[:, c]) — exactly the C++ term."""
    r_cur = np.asarray(r_cur, float)
    r_tgt = np.asarray(r_tgt, float)
    return 0.5 * np.cross(r_cur.T, r_tgt.T).sum(axis=0)


def rotation_angle(rotation: np.ndarray) -> float:
    """Angle of a rotation matrix, radians in [0, pi]."""
    trace = float(np.trace(rotation))
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) * 0.5)))


def rotation_log(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """(unit axis, angle) of a rotation matrix; axis is arbitrary at angle 0."""
    angle = rotation_angle(rotation)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-6:
        # Near pi the antisymmetric part vanishes; take the dominant column of R + I.
        sym = rotation + np.eye(3)
        col = int(np.argmax(np.linalg.norm(sym, axis=0)))
        axis = sym[:, col] / np.linalg.norm(sym[:, col])
        return axis, angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * math.sin(angle))
    return axis, angle


def align_rotation(a, b) -> np.ndarray:
    """Minimal rotation carrying unit vector a onto unit vector b.

    Identity when a ~ b; for antiparallel vectors a rotation of pi about an
    axis perpendicular to a (the choice is arbitrary, and made deterministic).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s2 = float(np.dot(v, v))
    if s2 < 1e-24:
        if c > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return axis_rotation(axis, math.pi)
    k = skew(v)
    return np.eye(3) + k + k @ k * ((1.0 - c) / s2)


def root_from_base(p) -> np.ndarray:
    return np.asarray(p, float) + BASE_IN_ROOT


def base_from_root(p) -> np.ndarray:
    return np.asarray(p, float) - BASE_IN_ROOT


# --- forward kinematics (right arm, base frame) ------------------------------

def fk(joints, tcp_in_link6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C++ `evaluate_at`: (position (3,), rotation (3,3), jacobian (6,6)) in base."""
    joints = np.asarray(joints, float)
    if joints.shape != (6,):
        raise ValueError(f"expected 6 joints, got shape {joints.shape}")
    rotation = np.eye(3)
    position = np.zeros(3)
    joint_positions = np.zeros((6, 3))
    joint_axes = np.zeros((6, 3))
    for j in range(6):
        position = position + rotation @ JOINT_ORIGINS[j]
        joint_positions[j] = position
        joint_axes[j] = rotation @ JOINT_AXES[j]
        rotation = rotation @ axis_rotation(JOINT_AXES[j], float(joints[j]))
    position = position + rotation @ np.asarray(tcp_in_link6, float)
    jacobian = np.empty((6, 6))
    jacobian[:3, :] = np.cross(joint_axes, position[None, :] - joint_positions).T
    jacobian[3:, :] = joint_axes.T
    return position, rotation, jacobian


def fk_link6(joints) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Link-6 origin (TCP zero): (position, rotation, jacobian 6x6)."""
    return fk(joints, np.zeros(3))


def fk_ballpoint(joints, carriage_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C++ `evaluate_ballpoint`: tip in base, link-6 rotation, jacobian (6,7)."""
    tip = BALLPOINT_TIP_IN_LINK6 + float(carriage_m) * CARRIAGE_AXIS_IN_LINK6
    position, rotation, arm_jacobian = fk(joints, tip)
    jacobian = np.zeros((6, 7))
    jacobian[:, :6] = arm_jacobian
    jacobian[:3, 6] = rotation @ CARRIAGE_AXIS_IN_LINK6
    return position, rotation, jacobian


def check_joint_limits(joints) -> None:
    joints = np.asarray(joints, float)
    lower = JOINT_LOWER + JOINT_LIMIT_MARGIN_RAD
    upper = JOINT_UPPER - JOINT_LIMIT_MARGIN_RAD
    bad = ~np.isfinite(joints) | (joints < lower) | (joints > upper)
    if bad.any():
        joint = int(np.argmax(bad))
        raise PlanRefusal("joint_limit", f"joint {joint} reaches the guarded limit ({joints[joint]:.4f} rad)")


IK_ITERATIONS = 400
IK_POSITION_TOL_M = 1e-4
IK_ORIENTATION_TOL_RAD = 1e-3
IK_STEP_CAP_RAD = 0.15


def solve_ik(tip_target, rot_target, seed_joints, carriage_m: float = 0.0, label: str = "target") -> np.ndarray:
    """Six joints placing the ballpoint tip at `tip_target` (base) with link-6
    rotation `rot_target`, by the executor's damped least squares iterated from
    `seed_joints`. Seed from a non-singular pose: the sleep pose is the arm at
    full reach, where DLS cannot pull the tip inward (2026-09-03)."""
    q = np.array(seed_joints, float)[:6].copy()
    tip_target = np.asarray(tip_target, float)
    rot_target = np.asarray(rot_target, float)
    lower = JOINT_LOWER + JOINT_LIMIT_MARGIN_RAD
    upper = JOINT_UPPER - JOINT_LIMIT_MARGIN_RAD
    for _ in range(IK_ITERATIONS):
        pos, rot, jac = fk_ballpoint(q, carriage_m)
        e_p = tip_target - pos
        e_r = orientation_error(rot, rot_target)
        if np.linalg.norm(e_p) < IK_POSITION_TOL_M and np.linalg.norm(e_r) < IK_ORIENTATION_TOL_RAD:
            check_joint_limits(q)
            return q
        dq = damped_least_squares(jac, np.concatenate([e_p, e_r]))
        worst = float(np.max(np.abs(dq)))
        if worst > IK_STEP_CAP_RAD:
            dq *= IK_STEP_CAP_RAD / worst
        q = np.clip(q + dq, lower, upper)
    pos, rot, _ = fk_ballpoint(q, carriage_m)
    raise PlanRefusal("ik", f"{label}: no joint solution — tip residual {np.linalg.norm(tip_target - pos) * 1000:.1f} mm, "
                            f"orientation residual {np.linalg.norm(orientation_error(rot, rot_target)):.3f} rad")


# --- URDF-derived tool geometry ----------------------------------------------

_CHAIN = None


def urdf_chain(urdf_path: Path | str | None = None):
    """The shared UrdfChain (scripts/vision/urdf_kinematics.py), loaded once."""
    global _CHAIN
    if urdf_path is not None:
        from urdf_kinematics import UrdfChain  # scripts/vision on sys.path
        return UrdfChain(urdf_path)
    if _CHAIN is None:
        from urdf_kinematics import UrdfChain
        _CHAIN = UrdfChain(URDF_PATH)
    return _CHAIN


def joint_map(joints, carriage_m: float = 0.0) -> dict[str, float]:
    """{urdf joint name: value} for the six arm joints plus the carriage."""
    values = {name: float(v) for name, v in zip(ARM_JOINT_NAMES, np.asarray(joints, float), strict=True)}
    values[CARRIAGE_JOINT_NAME] = float(carriage_m)
    return values


def link_in_link6(link: str, joints=None, carriage_m: float = 0.0, chain=None) -> np.ndarray:
    """4x4 pose of `link` expressed in right/link_6 (URDF, given carriage)."""
    chain = chain or urdf_chain()
    values = joint_map(np.zeros(6) if joints is None else joints, carriage_m)
    root_from_link6 = chain.link_pose(LINK6_NAME, values)
    root_from_link = chain.link_pose(link, values)
    return np.linalg.inv(root_from_link6) @ root_from_link


def tool_axis_in_link6(chain=None) -> np.ndarray:
    """+z of right/tool_mount expressed in right/link_6 at carriage 0 (unit)."""
    axis = link_in_link6(TOOL_MOUNT_NAME, chain=chain)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return axis / np.linalg.norm(axis)


def tool_tip_in_link6(offset_m, carriage_m: float = 0.0, chain=None) -> np.ndarray:
    """A mount-frame tool offset expressed in link 6 at carriage zero."""
    offset = np.asarray(offset_m, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("tool tip offset must be a finite 3-vector")
    pose = link_in_link6(TOOL_MOUNT_NAME, carriage_m=carriage_m, chain=chain)
    tip = pose[:3, :3] @ offset + pose[:3, 3]
    return tip - carriage_m * CARRIAGE_AXIS_IN_LINK6


def ballpoint_tip_in_link6_from_config(chain=None, workspace: dict | None = None) -> np.ndarray:
    """The touch-off tip (config/workspace.yaml, right/tool_mount) carried to link 6 at carriage 0."""
    import tool_spec  # scripts/lib

    ws = tool_spec.read_workspace(REPO) if workspace is None else workspace
    offset = tool_spec.tip_offset_m(ws, "right")
    if offset is None:
        raise RuntimeError("config/workspace.yaml has no tip offset in right/tool_mount (no touch-off)")
    side = ws.get("right") or {}
    carriage = float(side.get("carriage_m") or 0.0)
    return tool_tip_in_link6(offset, carriage, chain)


# --- per-arm model: the same joint chain, this arm's frames and tool ---------

ARM_IDS = tuple(arm_registry.load(REPO))


class ArmModel:
    """One arm's names, frames and tool for the shared seven-axis kinematics.

    Both WXAI arms carry the same joint chain (the URDF pins the left chain to
    the right's joint origins, axes and limits), so `fk` above serves either;
    what differs is where the base sits in root, which carriage carries the
    tool mount, and where the tool's working point is. The follower keeps the
    module-level right-arm constants and the executor's compiled ballpoint
    constant: `ArmModel("right")` reproduces them. The leader resolves the same
    quantities from the URDF and its own `config/workspace.yaml` section.
    Without a touch-off, either arm derives its TCP from the fitted datasheet;
    the URDF tool visual may still represent a different calibrated workspace.

    The driver's seventh axis is `<arm>/left_carriage_joint` on both arms, and
    both tool mounts ride that driven carriage (the leader's V5 print is
    bolted to it, rolled half a turn — see urdf/tatbot.urdf), so either tool
    moves along +y of link 6 per metre of carriage. The URDF is the authority:
    `carriage_axis_in_link6` is derived from it, never assumed.
    """

    def __init__(self, arm: str = "right", chain=None, workspace: dict | None = None,
                 repo: Path = REPO):
        physical = arm_registry.load(repo).get(arm)
        if physical is None:
            raise ValueError(f"unknown arm {arm!r}; expected a configured arm ID")
        if not physical.sdk_end_effector.startswith("wxai_v0_"):
            raise ValueError(f"{arm}: the WXAI kinematics adapter does not support {physical.sdk_end_effector}")
        self.arm = arm
        self.repo = repo
        self.prefix = physical.urdf_prefix
        self.workspace_section = physical.workspace_section
        self._chain = chain
        self._workspace = workspace
        self._base_in_root = None
        self._carriage_axis = None
        self.joint_names = tuple(f"{self.prefix}/joint_{i}" for i in range(6))
        self.carriage_joint = f"{self.prefix}/left_carriage_joint"
        self.link6 = f"{self.prefix}/link_6"
        self.tool_mount = f"{self.prefix}/tool_mount"
        self.tcp_link = f"{self.prefix}/tattoo_needle"
        self.frame = f"{self.prefix}/base_link"
        self.controller_role = arm_registry.CONTROLLER_ROLES.get(
            physical.controller_config, Path(physical.controller_config).stem)
        self.golden_path = repo / physical.controller_config

    @property
    def chain(self):
        if self._chain is None:
            self._chain = urdf_chain(self.repo / "urdf/tatbot.urdf")
        return self._chain

    @property
    def workspace(self) -> dict:
        if self._workspace is None:
            import tool_spec  # scripts/lib
            self._workspace = tool_spec.read_workspace(self.repo)
        return self._workspace

    @property
    def base_in_root(self) -> np.ndarray:
        """Translation root -> this arm's base; the mount must not be rotated."""
        if self._base_in_root is None:
            from ink_spec import base_from_root_matrix
            transform = np.linalg.inv(base_from_root_matrix(self.repo, self.prefix))
            if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-10, rtol=0):
                raise ValueError(f"draw planner requires parallel URDF-root and {self.arm} arm-base axes")
            self._base_in_root = transform[:3, 3].copy()
        return self._base_in_root

    def root_from_base(self, p) -> np.ndarray:
        return np.asarray(p, float) + self.base_in_root

    def base_from_root(self, p) -> np.ndarray:
        return np.asarray(p, float) - self.base_in_root

    def joint_map(self, joints, carriage_m: float = 0.0) -> dict[str, float]:
        values = {name: float(v) for name, v in zip(self.joint_names, np.asarray(joints, float), strict=True)}
        values[self.carriage_joint] = float(carriage_m)
        return values

    def link_in_link6(self, link: str, joints=None, carriage_m: float = 0.0) -> np.ndarray:
        """4x4 pose of `link` expressed in this arm's link 6 (URDF, given carriage)."""
        values = self.joint_map(np.zeros(6) if joints is None else joints, carriage_m)
        root_from_link6 = self.chain.link_pose(self.link6, values)
        return np.linalg.inv(root_from_link6) @ self.chain.link_pose(link, values)

    @property
    def carriage_axis_in_link6(self) -> np.ndarray:
        """Unit direction the tool mount moves in link 6 per metre of carriage."""
        if self._carriage_axis is None:
            step = 0.01
            moved = self.link_in_link6(self.tool_mount, carriage_m=step)[:3, 3]
            rest = self.link_in_link6(self.tool_mount, carriage_m=0.0)[:3, 3]
            axis = (moved - rest) / step
            if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-9):
                raise ValueError(f"{self.arm}: the tool mount does not ride a unit prismatic carriage")
            self._carriage_axis = axis / np.linalg.norm(axis)
        return self._carriage_axis

    @property
    def tool_axis_in_link6(self) -> np.ndarray:
        """+z of this arm's tool mount expressed in its link 6 at carriage 0 (unit)."""
        axis = self.link_in_link6(self.tool_mount)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return axis / np.linalg.norm(axis)

    def measured_tip_offset_m(self):
        """The touch-off tip in this arm's mount frame, or None before one."""
        import tool_spec  # scripts/lib
        return tool_spec.tip_offset_m(self.workspace, self.workspace_section)

    @property
    def tip_source(self) -> str:
        return "touch-off" if self.measured_tip_offset_m() is not None else "datasheet nominal"

    def tcp_in_link6(self, carriage_m: float = 0.0) -> np.ndarray:
        """The working point in link 6 at carriage zero: the touch-off tip when
        this arm has one — extended by the fitted tool's standoff for a
        non-contact tool, whose touch-off plants its nose while its working
        point floats past it (`tool_spec.tcp_from_touchoff_m`, as the poses
        export places it; the laser's first trace flew its nose at the lift
        meant for the working point on 2026-09-20) — else the datasheet
        nominal resolved from the fitted datasheet in the URDF tool mount. Never
        motion authority on its own."""
        import tool_spec  # scripts/lib

        offset = self.measured_tip_offset_m()
        spec = tool_spec.load_active_tool(self.repo, self.workspace_section, self.workspace)
        if offset is not None:
            if tool_spec.tip_offset_error_m(spec, offset) > spec.tip_tolerance_m:
                raise ValueError(f"{self.arm}: fitted tool differs from retained tip calibration")
            offset = tool_spec.tcp_from_touchoff_m(spec, offset)
        else:
            offset = tool_spec.resolved_tool_geometry(spec, self.workspace, self.workspace_section,
                                                     self.repo).tcp_offset_m
        pose = self.link_in_link6(self.tool_mount, carriage_m=carriage_m)
        tip = pose[:3, :3] @ np.asarray(offset, float) + pose[:3, 3]
        return tip - carriage_m * self.carriage_axis_in_link6

    def fk_tcp(self, joints, carriage_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """C++ `evaluate_tool`: working point in base, link-6 rotation, jacobian (6,7)."""
        axis = self.carriage_axis_in_link6
        tip = self.tcp_in_link6() + float(carriage_m) * axis
        position, rotation, arm_jacobian = fk(joints, tip)
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = arm_jacobian
        jacobian[:3, 6] = rotation @ axis
        return position, rotation, jacobian

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """The six rotary limits of this arm's controller golden."""
        import yaml

        limits = yaml.safe_load(self.golden_path.read_bytes())["joint_limits"][:6]
        lower = np.array([float(p["position_min"]) for p in limits])
        upper = np.array([float(p["position_max"]) for p in limits])
        if len(limits) != 6 or not np.all(lower < upper):
            raise ValueError(f"{self.golden_path}: malformed rotary joint limits")
        return lower, upper

    def assert_cpp_wxai_compatible(self) -> None:
        """Refuse offline C++ planning unless this chain matches its fixed WXAI FK.

        The C++ samples parser can accept another configured arm prefix, but
        its joint origins, axes and guarded bounds are compiled constants. This
        check admits only a URDF chain and controller whose planned window
        those constants actually represent; it does not qualify a new arm for
        powered use.
        """
        # The samples-to-base conversion also assumes parallel root/base axes.
        _ = self.base_in_root
        joints = self.chain.joints
        document = ET.parse(self.repo / "urdf/tatbot.urdf").getroot()
        declarations = {}
        for element in document.findall("joint"):
            declarations.setdefault(element.get("name"), []).append(element)
        self._assert_cpp_joint_chain(joints, declarations)
        self._assert_cpp_carriage(joints, declarations)
        self._assert_cpp_controller()

    def _cpp_urdf_limit(self, declarations: dict, name: str) -> tuple[float, float]:
        rows = declarations.get(name, ())
        if len(rows) != 1 or rows[0].find("limit") is None:
            raise ValueError(f"{self.arm}: {name} needs one URDF limit declaration")
        bounds = rows[0].find("limit")
        try:
            lower, upper = float(bounds.attrib["lower"]), float(bounds.attrib["upper"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{self.arm}: {name} has invalid URDF limits") from error
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"{self.arm}: {name} has invalid URDF limits")
        return lower, upper

    def _assert_cpp_joint_chain(self, joints: dict, declarations: dict) -> None:
        for index, name in enumerate(self.joint_names):
            joint = joints.get(name)
            parent = self.frame if index == 0 else f"{self.prefix}/link_{index}"
            child = f"{self.prefix}/link_{index + 1}"
            if (joint is None or joint["type"] != "revolute" or joint["parent"] != parent
                    or joint["child"] != child or joint["mimic"] is not None
                    or not np.allclose(joint["origin"][:3, :3], np.eye(3), atol=1e-9, rtol=0)
                    or not np.allclose(joint["origin"][:3, 3], JOINT_ORIGINS[index], atol=1e-9, rtol=0)
                    or not np.allclose(joint["axis"], JOINT_AXES[index], atol=1e-9, rtol=0)
                    or not np.allclose(self._cpp_urdf_limit(declarations, name), (JOINT_LOWER[index], JOINT_UPPER[index]),
                                        atol=1e-9, rtol=0)):
                raise ValueError(f"{self.arm}: {name} differs from the C++ WXAI joint model")

    def _assert_cpp_carriage(self, joints: dict, declarations: dict) -> None:
        carriage = joints.get(self.carriage_joint)
        if (carriage is None or carriage["type"] != "prismatic"
                or carriage["parent"] != self.link6
                or carriage["child"] != f"{self.prefix}/carriage_left"
                or carriage["mimic"] is not None
                or not np.allclose(carriage["origin"][:3, :3], np.eye(3), atol=1e-9, rtol=0)
                or not np.allclose(carriage["axis"], CARRIAGE_AXIS_IN_LINK6, atol=1e-9, rtol=0)
                or not np.allclose(self.carriage_axis_in_link6, CARRIAGE_AXIS_IN_LINK6, atol=1e-9, rtol=0)):
            raise ValueError(f"{self.arm}: carriage differs from the C++ WXAI +Y model")
        carriage_lower, carriage_upper = self._cpp_urdf_limit(declarations, self.carriage_joint)
        if carriage_lower > CARRIAGE_IK_MIN_M or carriage_upper < CARRIAGE_IK_MAX_M:
            raise ValueError(f"{self.arm}: URDF carriage limits exclude the C++ planning window")

    def _assert_cpp_controller(self) -> None:
        import yaml

        document = yaml.safe_load(self.golden_path.read_bytes())
        controller = document.get("joint_limits") if isinstance(document, dict) else None
        if not isinstance(controller, list) or len(controller) != 7:
            raise ValueError(f"{self.arm}: controller golden needs seven joint limits")
        try:
            lower = np.array([float(row["position_min"]) for row in controller])
            upper = np.array([float(row["position_max"]) for row in controller])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{self.arm}: malformed controller joint limits") from error
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower >= upper)):
            raise ValueError(f"{self.arm}: malformed controller joint limits")
        planned_lower = JOINT_LOWER + JOINT_LIMIT_MARGIN_RAD
        planned_upper = JOINT_UPPER - JOINT_LIMIT_MARGIN_RAD
        if (np.any(lower[:6] > planned_lower + 1e-6)
                or np.any(upper[:6] < planned_upper - 1e-6)
                or lower[6] > CARRIAGE_IK_MIN_M + 1e-6
                or upper[6] < CARRIAGE_IK_MAX_M - 1e-6):
            raise ValueError(f"{self.arm}: controller limits exclude the C++ planning window")

    def solve_ik(self, tip_target, rot_target, seed_joints, carriage_m: float = 0.0,
                 label: str = "target") -> np.ndarray:
        """Advisory six-joint IK using this arm's TCP and controller limits."""
        q = np.asarray(seed_joints, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise PlanRefusal("ik", f"{label}: invalid joint seed")
        q = q.copy()
        tip_target = np.asarray(tip_target, float)
        rot_target = np.asarray(rot_target, float)
        if (tip_target.shape != (3,) or rot_target.shape != (3, 3)
                or not np.isfinite(tip_target).all() or not np.isfinite(rot_target).all()):
            raise PlanRefusal("ik", f"{label}: invalid target")
        lower, upper = self.joint_limits()
        lower += JOINT_LIMIT_MARGIN_RAD
        upper -= JOINT_LIMIT_MARGIN_RAD
        if np.any(lower >= upper):
            raise PlanRefusal("joint_limit", f"{self.arm}: no guarded joint range")
        for _ in range(IK_ITERATIONS):
            pos, rot, jac = self.fk_tcp(q, carriage_m)
            e_p = tip_target - pos
            e_r = orientation_error(rot, rot_target)
            if np.linalg.norm(e_p) < IK_POSITION_TOL_M and np.linalg.norm(e_r) < IK_ORIENTATION_TOL_RAD:
                if np.any(q < lower) or np.any(q > upper):
                    raise PlanRefusal("joint_limit", f"{self.arm}: solution reaches a guarded joint limit")
                return q
            dq = damped_least_squares(jac, np.concatenate([e_p, e_r]))
            worst = float(np.max(np.abs(dq)))
            if worst > IK_STEP_CAP_RAD:
                dq *= IK_STEP_CAP_RAD / worst
            q = np.clip(q + dq, lower, upper)
        pos, rot, _ = self.fk_tcp(q, carriage_m)
        raise PlanRefusal("ik", f"{label}: no joint solution — tip residual {np.linalg.norm(tip_target - pos) * 1000:.1f} mm, "
                        f"orientation residual {np.linalg.norm(orientation_error(rot, rot_target)):.3f} rad")


# --- six-joint damped least squares ------------------------------------------


def damped_least_squares(jacobian: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """C++ `damped_least_squares`: the six-joint DLS step, carriage held."""
    jacobian = np.asarray(jacobian, float)[:, :6]
    normal = jacobian @ jacobian.T + (DLS_DAMPING * DLS_DAMPING) * np.eye(6)
    return jacobian.T @ np.linalg.solve(normal, twist)


# --- the wrist camera rig ---------------------------------------------------------------
# A capture is one arm's views, registered through that arm's own measured
# joint readings: `CAMERA_LINKS[arm]` binds the registry's wrist camera on
# `arm` to its depth optical frame, a fixed URDF chain to `<arm>/link_6`
# (`wrist_cameras.optical_frames`). An arm the registry assigns no camera maps
# nothing, so a capture naming its role is refused by name, never registered
# through the other arm's chain. Consumers key on the capture's declared
# `camera_roles` (`capture_geometry.capture_views`).
from wrist_cameras import optical_frames  # noqa: E402
from wrist_cameras import registry as _camera_registry  # noqa: E402


def _camera_links() -> dict[str, dict[str, str]]:
    assigned = {camera.get("arm") for camera in _camera_registry(REPO)}
    return {arm: optical_frames(REPO, arm=arm, stream="depth") if arm in assigned else {}
            for arm in ARM_IDS}


CAMERA_LINKS = _camera_links()
D405_DEPTH_RANGE_M = (0.07, 0.5)


