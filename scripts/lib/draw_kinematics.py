"""Right-arm kinematics for `tatbot draw` — a numpy mirror of square_probe.cpp.

Contract: docs/draw.md. The C++ executor (`cpp/teleop/square_probe.cpp`) is the
authority on what the arm does; this module exists so the Python stages can
(a) compute the same FK the executor uses when they turn a surface into tip
samples, (b) derive the tool axis and the ballpoint tip from the URDF and the
touch-off so the samples file can carry them and the executor can refuse a
mismatch, and (c) run the same 7-DoF carriage-IK loop in advisory mode to say
"this path will pass the executor's caps" before an arm is connected.

Every constant here is copied from square_probe.cpp / square_probe.hpp with
the same name. If the C++ changes, change it here in the same commit — the
tip-constant test and the spiral parity test are what catch the drift.

Frames: the C++ FK works in `right/base_link` ("base"). The URDF root is
`root` = base + BASE_IN_ROOT with no rotation. Only translations differ.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
URDF_PATH = REPO / "urdf" / "tatbot.urdf"

# --- constants mirrored from square_probe.cpp / .hpp -------------------------

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
JOINT_LIMIT_MARGIN_RAD = 0.05

TCP_IN_LINK6 = np.array([0.156062, 0.0, 0.0])
BALLPOINT_TIP_IN_LINK6 = np.array([0.20550927, 0.01083364, -0.00149001])
CARRIAGE_AXIS_IN_LINK6 = np.array([0.0, 1.0, 0.0])

PLAN_MAX_JOINT_VELOCITY_RAD_S = 0.25
PLAN_MAX_MODEL_ERROR_M = 0.0001
PLAN_MAX_MODEL_ERROR_PEN_UP_M = 0.001  # standoff travel: the damped solve trails a 10 mm/s reference
# Pen-down samples of a draw path (plan_joints). The 0.1 mm cap above stays on the
# executor's own spiral; a surface path follows the local normal, and with the
# wrist near its singularity (joint 4 through zero on the first bottle draw) the
# damped solve trails a 3.5 mm/s reference by 0.18 mm -- nothing against a pen
# line, so the path cap is 0.25 mm and the preflight reports the actual value.
PLAN_MAX_MODEL_ERROR_DRAW_M = 0.00025
PLAN_MAX_ORIENTATION_ERROR_RAD = 0.001
DLS_DAMPING = 0.02
POSITION_ERROR_GAIN_S = 4.0
ORIENTATION_ERROR_GAIN_S = 2.0
CARRIAGE_DLS_WEIGHT = 2.0
CARRIAGE_CENTER_GAIN_S = 2.0
PLAN_MAX_CARRIAGE_VELOCITY_M_S = 0.001
PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 = 0.02
CARRIAGE_IK_BIAS_M = 0.002
CARRIAGE_IK_MIN_M = 0.0005
CARRIAGE_IK_MAX_M = 0.0035
PLAN_MAX_TICKS = 250000

# URDF: right/base_link = root + this, no rotation.
BASE_IN_ROOT = np.array([0.0, -0.2675, 0.0])

ARM_JOINT_NAMES = tuple(f"right/joint_{i}" for i in range(6))
CARRIAGE_JOINT_NAME = "right/left_carriage_joint"
LINK6_NAME = "right/link_6"
TOOL_MOUNT_NAME = "right/tool_mount"


class PlanRefusal(RuntimeError):  # noqa: N818 - the contract's name
    """The advisory planner hit one of the executor's caps."""

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


def fk_tcp(joints) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C++ `evaluate`: the ee_gripper TCP."""
    return fk(joints, TCP_IN_LINK6)


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


# --- URDF-derived tool geometry ----------------------------------------------

_CHAIN = None


def urdf_chain(urdf_path: Path | str | None = None):
    """The shared UrdfChain (scripts/vision/urdf_kinematics.py), loaded once."""
    global _CHAIN
    if urdf_path is not None:
        from urdf_kinematics import UrdfChain  # scripts/vision on sys.path
        return UrdfChain(urdf_path)
    if _CHAIN is None:
        import sys
        vision = str(REPO / "scripts" / "vision")
        if vision not in sys.path:
            sys.path.insert(0, vision)
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


def ballpoint_tip_in_link6_from_config(chain=None, workspace: dict | None = None) -> np.ndarray:
    """The touch-off tip (config/workspace.yaml, right/tool_mount) carried to link 6 at carriage 0."""
    import tool_spec  # scripts/lib

    ws = tool_spec.read_workspace(REPO) if workspace is None else workspace
    offset = tool_spec.tip_offset_m(ws, "right")
    if offset is None:
        raise RuntimeError("config/workspace.yaml has no tip offset in right/tool_mount (no touch-off)")
    side = ws.get("right") or {}
    carriage = float(side.get("carriage_m") or 0.0)
    pose = link_in_link6(TOOL_MOUNT_NAME, carriage_m=carriage, chain=chain)
    tip = pose[:3, :3] @ np.asarray(offset, float) + pose[:3, 3]
    # The measured offset was solved at `carriage`; the constant is quoted at carriage 0.
    return tip - carriage * CARRIAGE_AXIS_IN_LINK6


# --- the 7-DoF carriage-IK loop (advisory port of plan_joint_spiral_with_carriage)

_INVERSE_WEIGHTS = np.array([1.0] * 6 + [1.0 / (CARRIAGE_DLS_WEIGHT * CARRIAGE_DLS_WEIGHT)])


def weighted_carriage_dls(jacobian: np.ndarray, twist: np.ndarray, carriage_centering_velocity_m_s: float):
    """C++ `weighted_carriage_dls`: weighted DLS with the carriage centering task projected."""
    jw = jacobian * _INVERSE_WEIGHTS[None, :]
    normal = jw @ jacobian.T + (DLS_DAMPING * DLS_DAMPING) * np.eye(6)

    def pseudoinverse(task):
        return jw.T @ np.linalg.solve(normal, task)

    velocity = pseudoinverse(twist)
    centering = np.zeros(7)
    centering[6] = carriage_centering_velocity_m_s
    projected = pseudoinverse(jacobian[:, 6] * carriage_centering_velocity_m_s)
    return velocity + (centering - projected)


def damped_least_squares(jacobian: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """C++ `damped_least_squares`: the six-joint DLS the flat spiral used (carriage held)."""
    jacobian = np.asarray(jacobian, float)[:, :6]
    normal = jacobian @ jacobian.T + (DLS_DAMPING * DLS_DAMPING) * np.eye(6)
    return jacobian.T @ np.linalg.solve(normal, twist)


def plan_joints(samples, start_joints, start_carriage_m: float, period_s: float,
                start_carriage_velocity_m_s: float = 0.0, lock_carriage_when_up: bool = False,
                carriage_ik: bool = True) -> dict:
    """Integrate the executor's carriage-IK loop over per-sample (p, v, R) references.

    `samples` is anything with `.p` (N,3) tip references in base, `.v` (N,3)
    feedforward tip velocities and `.R` (N,3,3) link-6 target rotations (a
    dict with those keys also works). Sample i is the reference for tick i+1,
    exactly as the C++ loop evaluates the spiral at t = tick * period.

    Raises PlanRefusal with the executor's reason on any cap. Returns
    positions (N,7), velocities (N,7) and the same statistics the C++ plan
    carries, plus `end_carriage_velocity_m_s` so a path planned in slices can
    chain (`start_carriage_velocity_m_s` seeds the acceleration check; the
    executor plans the whole file in one loop from rest). Advisory: the C++
    plan is what runs.

    `lock_carriage_when_up`: on ticks whose sample has `pen == 0`, hold the
    carriage and solve the arm alone (the C++ six-joint DLS). The 7-DoF loop
    hands the carriage ~70 % of any tip motion along its axis, and the tool
    axis has a 0.707 component on it, so a 120 mm approach or lift along the
    tool axis at millimetres per second walks the carriage to its 3.5 mm stop
    — measured here at 1.5 mm/s. Pen-up travel must not use the compliance
    axis; the executor's path planner needs the same rule.
    """
    ref_p, ref_v, ref_r = _references(samples)
    pen = _pen(samples, len(ref_p)) if (lock_carriage_when_up or not carriage_ik) else None
    ticks = len(ref_p)
    if ticks == 0 or ticks > PLAN_MAX_TICKS:
        raise PlanRefusal("sample_count", f"{ticks} ticks is outside the guarded range (1..{PLAN_MAX_TICKS})")
    if not math.isfinite(period_s) or period_s <= 0.0:
        raise PlanRefusal("period", f"period {period_s} must be finite and positive")
    start_carriage_m = float(start_carriage_m)
    if not math.isfinite(start_carriage_m) or not CARRIAGE_IK_MIN_M <= start_carriage_m <= CARRIAGE_IK_MAX_M:
        raise PlanRefusal("carriage_envelope", f"start carriage {start_carriage_m * 1e3:.3f} mm is outside the envelope")
    joints = np.array(start_joints, float)
    check_joint_limits(joints)

    carriage_m = start_carriage_m
    previous_carriage_velocity = float(start_carriage_velocity_m_s)
    positions = np.empty((ticks, 7))
    velocities = np.empty((ticks, 7))
    stats = {
        "max_model_error_mm": 0.0,
        "max_orientation_error_rad": 0.0,
        "max_joint_velocity_rad_s": 0.0,
        "max_cartesian_velocity_m_s": 0.0,
        "max_carriage_velocity_m_s": 0.0,
        "max_carriage_acceleration_m_s2": 0.0,
        "min_carriage_m": carriage_m,
        "max_carriage_m": carriage_m,
    }
    for tick in range(ticks):
        reference = ref_p[tick]
        feedforward = ref_v[tick]
        target_rotation = ref_r[tick]
        stats["max_cartesian_velocity_m_s"] = max(
            stats["max_cartesian_velocity_m_s"], float(np.linalg.norm(feedforward)))

        position, rotation, jacobian = fk_ballpoint(joints, carriage_m)
        twist = np.empty(6)
        twist[:3] = feedforward + POSITION_ERROR_GAIN_S * (reference - position)
        # Angular feedforward from the next sample's rotation (small-angle rotation
        # vector over one tick): without it a moving rotation target lags the
        # proportional loop by omega / K, past the 1 mrad cap on a 15 deg orbit
        # tilt. Exactly zero for a constant rotation, so the spiral is untouched.
        omega_ff = np.zeros(3)
        if tick + 1 < ticks:
            omega_ff = orientation_error(target_rotation, ref_r[tick + 1]) / period_s
        twist[3:] = omega_ff + ORIENTATION_ERROR_GAIN_S * orientation_error(rotation, target_rotation)
        if (pen is not None and pen[tick] == 0) or not carriage_ik:
            # Bring the carriage to rest at half its acceleration cap, then hold it; the
            # arm takes the whole task minus whatever the carriage still contributes.
            step = 0.5 * PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 * period_s
            held = previous_carriage_velocity - float(np.clip(previous_carriage_velocity, -step, step))
            velocity = np.zeros(7)
            velocity[:6] = damped_least_squares(jacobian, twist - jacobian[:, 6] * held)
            velocity[6] = held
        else:
            velocity = weighted_carriage_dls(
                jacobian, twist, CARRIAGE_CENTER_GAIN_S * (CARRIAGE_IK_BIAS_M - carriage_m))
            # Slew-limit the carriage at the pen-down handover (mirrors square_probe.cpp): hold its
            # velocity step to 90 % of the acceleration cap and let the arm take the remainder.
            max_step = 0.9 * PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 * period_s
            max_speed = 0.9 * PLAN_MAX_CARRIAGE_VELOCITY_M_S
            slewed = float(np.clip(np.clip(velocity[6], -max_speed, max_speed),
                                   previous_carriage_velocity - max_step, previous_carriage_velocity + max_step))
            if slewed != velocity[6]:
                velocity = np.concatenate([damped_least_squares(jacobian, twist - jacobian[:, 6] * slewed), [slewed]])

        arm_velocity = velocity[:6]
        arm_abs = np.abs(arm_velocity)
        stats["max_joint_velocity_rad_s"] = max(stats["max_joint_velocity_rad_s"], float(arm_abs.max()))
        bad = ~np.isfinite(arm_velocity) | (arm_abs > PLAN_MAX_JOINT_VELOCITY_RAD_S)
        if bad.any():
            joint = int(np.argmax(bad))
            raise PlanRefusal(
                "joint_velocity",
                f"tick {tick + 1}: joint {joint} needs {arm_velocity[joint]:.4f} rad/s "
                f"(cap {PLAN_MAX_JOINT_VELOCITY_RAD_S})")
        joints = joints + period_s * arm_velocity

        carriage_velocity = float(velocity[6])
        stats["max_carriage_velocity_m_s"] = max(stats["max_carriage_velocity_m_s"], abs(carriage_velocity))
        if not math.isfinite(carriage_velocity) or abs(carriage_velocity) > PLAN_MAX_CARRIAGE_VELOCITY_M_S:
            raise PlanRefusal(
                "carriage_velocity",
                f"tick {tick + 1}: {carriage_velocity * 1e3:.3f} mm/s (cap {PLAN_MAX_CARRIAGE_VELOCITY_M_S * 1e3})")
        carriage_acceleration = (carriage_velocity - previous_carriage_velocity) / period_s
        stats["max_carriage_acceleration_m_s2"] = max(
            stats["max_carriage_acceleration_m_s2"], abs(carriage_acceleration))
        if (not math.isfinite(carriage_acceleration)
                or abs(carriage_acceleration) > PLAN_MAX_CARRIAGE_ACCELERATION_M_S2):
            raise PlanRefusal(
                "carriage_acceleration",
                f"tick {tick + 1}: {carriage_acceleration:.4f} m/s^2 (cap {PLAN_MAX_CARRIAGE_ACCELERATION_M_S2})")
        carriage_m += period_s * carriage_velocity
        previous_carriage_velocity = carriage_velocity
        if not math.isfinite(carriage_m) or not CARRIAGE_IK_MIN_M <= carriage_m <= CARRIAGE_IK_MAX_M:
            raise PlanRefusal(
                "carriage_envelope",
                f"tick {tick + 1}: carriage {carriage_m * 1e3:.3f} mm leaves "
                f"{CARRIAGE_IK_MIN_M * 1e3}..{CARRIAGE_IK_MAX_M * 1e3} mm")
        stats["min_carriage_m"] = min(stats["min_carriage_m"], carriage_m)
        stats["max_carriage_m"] = max(stats["max_carriage_m"], carriage_m)
        check_joint_limits(joints)

        integrated_p, integrated_r, _ = fk_ballpoint(joints, carriage_m)
        model_error_m = float(np.linalg.norm(reference - integrated_p))
        stats["max_model_error_mm"] = max(stats["max_model_error_mm"], model_error_m * 1000.0)
        pen_down = pen is None or pen[tick] != 0
        model_error_cap = PLAN_MAX_MODEL_ERROR_DRAW_M if pen_down else PLAN_MAX_MODEL_ERROR_PEN_UP_M
        if model_error_m > model_error_cap:
            raise PlanRefusal(
                "model_error",
                f"tick {tick + 1}: tip lags the reference by {model_error_m * 1e3:.3f} mm "
                f"(cap {model_error_cap * 1e3})")
        orientation_error_rad = float(np.linalg.norm(orientation_error(integrated_r, target_rotation)))
        stats["max_orientation_error_rad"] = max(stats["max_orientation_error_rad"], orientation_error_rad)
        if orientation_error_rad > PLAN_MAX_ORIENTATION_ERROR_RAD:
            raise PlanRefusal(
                "orientation_error",
                f"tick {tick + 1}: {orientation_error_rad:.5f} rad (cap {PLAN_MAX_ORIENTATION_ERROR_RAD})")
        positions[tick, :6] = joints
        positions[tick, 6] = carriage_m
        velocities[tick] = velocity

    endpoint_p, _, _ = fk_ballpoint(joints, carriage_m)
    stats["endpoint_error_m"] = float(np.linalg.norm(ref_p[-1] - endpoint_p))
    stats["end_carriage_velocity_m_s"] = previous_carriage_velocity
    stats["ticks"] = ticks
    return {"positions": positions, "velocities": velocities, "stats": stats}


def _pen(samples, n: int) -> np.ndarray:
    pen = samples.get("pen") if isinstance(samples, dict) else getattr(samples, "pen", None)
    if pen is None:
        return np.ones(n, dtype=np.int64)
    pen = np.asarray(pen).reshape(-1)
    if len(pen) != n:
        raise ValueError(f"pen has {len(pen)} entries for {n} samples")
    return pen.astype(np.int64)


def _references(samples):
    if isinstance(samples, dict):
        p, v, r = samples["p"], samples["v"], samples["R"]
    else:
        p, v, r = samples.p, samples.v, samples.R
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    r = np.asarray(r, float)
    n = len(p)
    if p.shape != (n, 3) or v.shape != (n, 3) or r.shape != (n, 3, 3):
        raise ValueError(f"reference shapes p{p.shape} v{v.shape} R{r.shape} disagree")
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(r).all()):
        raise PlanRefusal("nan", "a reference sample is not finite")
    return p, v, r


# --- the wrist camera rig ---------------------------------------------------------------
# Both D405s ride link 6 with the tool, so their geometry relative to the tip is fixed:
# 110 mm apart, both looking along +x6 tilted 20 deg inward, converging ~147 mm ahead
# at the tip, wide (87 deg) image axis along y6. The pen points 48 deg off that view.

CAMERA_LINKS = {
    "wrist_upper": "right/realsense_depth_optical_frame",
    "wrist_lower": "right/realsense_lower_depth_optical_frame",
}
# D405 depth stream at 640x480 as the capture server reports it: fx, fy, ppx, ppy, width, height.
D405_INTRINSICS = (387.8, 387.8, 328.9, 238.9, 640, 480)
D405_DEPTH_RANGE_M = (0.07, 0.5)
_RIG: dict | None = None


def wrist_camera_rig() -> dict:
    """Camera geometry in link 6 (joint- and carriage-independent), from the URDF, loaded once.

    ``link6_from_camera`` per role (4x4), ``mean_position`` of the two cameras,
    their mean optical ``view`` direction, and the ``wide_axis`` of the image
    (the lower unit's optical x; the upper's is its negative).
    """
    global _RIG
    if _RIG is None:
        chain = urdf_chain()
        values = joint_map(np.zeros(6), 0.0)
        inv6 = np.linalg.inv(chain.link_pose("right/link_6", values))
        cams = {role: inv6 @ chain.link_pose(link, values) for role, link in CAMERA_LINKS.items()}
        view = np.sum([t[:3, 2] for t in cams.values()], axis=0)
        wide = cams["wrist_lower"][:3, 0].copy()
        _RIG = {
            "link6_from_camera": cams,
            "mean_position": np.mean([t[:3, 3] for t in cams.values()], axis=0),
            "view": view / np.linalg.norm(view),
            "wide_axis": wide / np.linalg.norm(wide),
        }
    return _RIG


def rig_cameras(tip: np.ndarray, rotation: np.ndarray, carriage_m: float) -> dict[str, np.ndarray]:
    """Camera poses (4x4, in the frame ``tip``/``rotation`` are given in) for a tip pose."""
    rig = wrist_camera_rig()
    tip6 = BALLPOINT_TIP_IN_LINK6 + float(carriage_m) * CARRIAGE_AXIS_IN_LINK6
    t6 = np.eye(4)
    t6[:3, :3] = rotation
    t6[:3, 3] = np.asarray(tip, float) - rotation @ tip6
    return {role: t6 @ cam for role, cam in rig["link6_from_camera"].items()}


def camera_view(from_camera: np.ndarray, points: np.ndarray, normals: np.ndarray,
                intrinsics=D405_INTRINSICS, depth_range=D405_DEPTH_RANGE_M) -> tuple[float, float, float]:
    """Frustum score: (fraction of points inside the image and depth range, mean distance m, mean incidence deg)."""
    fx, fy, ppx, ppy, width, height = intrinsics
    q = (np.asarray(points, float) - from_camera[:3, 3]) @ from_camera[:3, :3]
    z = q[:, 2]
    ahead = z > 0.01
    zz = np.where(ahead, z, 1.0)
    u = fx * q[:, 0] / zz + ppx
    v = fy * q[:, 1] / zz + ppy
    inside = (ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
              & (z >= depth_range[0]) & (z <= depth_range[1]))
    rays = np.asarray(points, float) - from_camera[:3, 3]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    incidence = np.degrees(np.arccos(np.clip(-np.einsum("ij,ij->i", rays, np.asarray(normals, float)), -1.0, 1.0)))
    if not inside.any():
        return 0.0, float("nan"), float("nan")
    return float(inside.mean()), float(np.mean(z[inside])), float(np.mean(incidence[inside]))
