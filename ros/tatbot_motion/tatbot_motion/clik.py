"""URDF kinematics and damped-least-squares CLIK; test_clik_parity checks the C++ reference."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


class PlanError(RuntimeError):
    """The planner cannot produce the motion: a joint limit on the way, or a tip it cannot follow."""


def orientation_error(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """0.5 * sum_c cross(R_cur[:, c], R_tgt[:, c]), the C++ term. Written out, the same products and sums as
    np.cross over the columns to the last bit: three calls a tick, np.cross's axis handling was 40% of a plan."""
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = np.asarray(r_cur).tolist()
    (b00, b01, b02), (b10, b11, b12), (b20, b21, b22) = np.asarray(r_tgt).tolist()
    return np.array([0.5 * ((a10 * b20 - a20 * b10) + (a11 * b21 - a21 * b11) + (a12 * b22 - a22 * b12)),
                     0.5 * ((a20 * b00 - a00 * b20) + (a21 * b01 - a01 * b21) + (a22 * b02 - a02 * b22)),
                     0.5 * ((a00 * b10 - a10 * b00) + (a01 * b11 - a11 * b01) + (a02 * b12 - a12 * b02))])


class Kinematics:
    """pinocchio model of one arm from URDF text (tatbot_description.robot_description(), or the
    /robot_description topic). Frames: <arm>/base_link -> <arm>/tcp; joints: the arm's model's, in controller order
    (scripts/lib/arm_model.py, as tatbot_description.arm_model). `revolute` indexes its revolute chain and `carriage` the WXAI's tool carriage, or
    is None. Each thread has its own pinocchio data: sharing one with the stroke pre-planner, a lift read its
    placements (2026-10-21)."""

    def __init__(self, urdf_xml: str, arm: str = "right"):
        import pinocchio as pin
        from tatbot_description import arm_model, names

        self._pin = pin
        self.arm = arm
        self.model = pin.buildModelFromXML(urdf_xml)
        self._local = threading.local()
        self.joint_names = names.joint_names(arm)
        self.arm_model = arm_model.for_arm(arm)
        self.n = self.arm_model.n
        self.revolute = self.arm_model.arm_index
        self.carriage = self.arm_model.carriage
        ids = [self.model.getJointId(name) for name in self.joint_names]
        if any(i >= self.model.njoints for i in ids):
            raise ValueError(f"URDF lacks a joint of {self.joint_names}")
        self._iq = np.array([self.model.idx_qs[i] for i in ids])
        self._iv = np.array([self.model.idx_vs[i] for i in ids])
        self._tcp = self.model.getFrameId(names.tcp_frame(arm))
        self._base = self.model.getFrameId(names.base_frame(arm))
        for frame, fid in ((names.tcp_frame(arm), self._tcp), (names.base_frame(arm), self._base)):
            if fid >= self.model.nframes:
                raise ValueError(f"URDF lacks frame {frame}")
        self.lower = np.array(self.model.lowerPositionLimit[self._iq], float)
        self.upper = np.array(self.model.upperPositionLimit[self._iq], float)
        pin.framesForwardKinematics(self.model, self.data, pin.neutral(self.model))
        world_from_base = self.data.oMf[self._base]
        self._r_bw = np.array(world_from_base.rotation).T
        self._p_bw = -self._r_bw @ np.array(world_from_base.translation)

    @classmethod
    def from_repo(cls, repo=None, arm: str = "right") -> "Kinematics":
        from tatbot_description import robot_description

        return cls(robot_description(repo, arms=(arm,)), arm)

    @property
    def data(self):
        """This thread's pinocchio data (the model is shared and never written)."""
        local = self._local
        if not hasattr(local, "data"):
            local.data, local.q = self.model.createData(), self._pin.neutral(self.model)
        return local.data

    def _full(self, q) -> tuple:
        """(this thread's data, its full configuration set to q)."""
        data, full = self.data, self._local.q
        full[self._iq] = q
        return data, full

    def evaluate(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tcp position (3,), tcp rotation (3, 3), Jacobian (6, n) [linear; angular]), all in the base."""
        pin = self._pin
        data, full = self._full(q)
        jac = pin.computeFrameJacobian(self.model, data, full, self._tcp, pin.LOCAL_WORLD_ALIGNED)
        placement = data.oMf[self._tcp]
        r = self._r_bw
        j = jac[:, self._iv]
        return (r @ placement.translation + self._p_bw, r @ placement.rotation,
                np.vstack([r @ j[:3], r @ j[3:]]))

    def fk(self, q) -> np.ndarray:
        """base_from_tcp (4, 4) at joints q (n,)."""
        return self._frame(q, self._tcp)

    def frame(self, q, name: str) -> np.ndarray:
        """base_from_<name> (4, 4) at joints q (n,), for any URDF frame (e.g. the wrist camera's)."""
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise ValueError(f"URDF lacks frame {name}")
        return self._frame(q, fid)

    def _frame(self, q, fid):
        pin = self._pin
        data, full = self._full(q)
        pin.forwardKinematics(self.model, data, full)
        placement = pin.updateFramePlacement(self.model, data, fid)
        out = np.eye(4)
        out[:3, :3] = self._r_bw @ placement.rotation
        out[:3, 3] = self._r_bw @ placement.translation + self._p_bw
        return out

    def jacobian(self, q) -> np.ndarray:
        """(6, n) [linear; angular] TCP Jacobian, base-aligned (pinocchio LOCAL_WORLD_ALIGNED in the base)."""
        return self.evaluate(q)[2]

    def solve(self, base_from_tcp: np.ndarray, q_seed, iterations: int = 200, tol: float = 1e-9) -> np.ndarray:
        """Joints reaching a TCP pose from q_seed (Newton steps of the revolute chain, a carriage kept): a start
        pose for tests and reach checks, not a path."""
        q = np.array(q_seed, float)
        target = np.asarray(base_from_tcp, float)
        for _ in range(iterations):
            position, rotation, jac = self.evaluate(q)
            error = np.concatenate([target[:3, 3] - position, orientation_error(rotation, target[:3, :3])])
            if np.linalg.norm(error) < tol:
                return q
            chain = jac[:, self.revolute]
            q[self.revolute] += chain.T @ np.linalg.solve(chain @ chain.T + 1e-8 * np.eye(6), error)
        raise PlanError(f"no joints reach the pose from the seed (residual {np.linalg.norm(error):.2e})")


def depth_sensitivity(kin: Kinematics, q, axis, damping: float = 1e-4) -> np.ndarray:
    """(M, n) joints per metre of TCP travel along `axis` (M, 3, unit, base) at each row of q (M, n), the tool's
    rotation and a carriage held: a pen-height change's first-order map to joints, so a trim h composes as
    q + h * dq_dh. On the WXAI follower's drawing poses that is within 18 um of the target at 2 mm and 53 um at 4 mm,
    and on the plan's own IK branch by construction (ros/README.md 4.4, Trim)."""
    out = np.zeros((len(q), kin.n))
    twist, eye6 = np.zeros(6), np.eye(6)
    for k, (qk, ak) in enumerate(zip(np.asarray(q, float), np.asarray(axis, float), strict=True)):
        chain = kin.evaluate(qk)[2][:, kin.revolute]
        twist[:3] = ak
        out[k, kin.revolute] = chain.T @ np.linalg.solve(chain @ chain.T + damping * damping * eye6, twist)
    return out


@dataclass
class ClikResult:
    q: np.ndarray            # (N, n) joints after each sample's tick
    qd: np.ndarray           # (N, n) the velocity integrated over that tick
    tip: np.ndarray          # (N, 3) FK tip after the tick
    model_error_m: np.ndarray        # (N,) |reference - FK tip| after the tick
    orientation_error_rad: np.ndarray  # (N,)


@dataclass(frozen=True)
class _Carriage:
    """The WXAI carriage's terms in the solve: its index, motion.yaml clik.carriage_center_gain_s and carriage.*."""
    index: int
    toward: np.ndarray        # (n,) the unit velocity along the carriage alone
    center_gain: float
    bias: float
    speed: float
    accel: float
    window: list


def _check_chain(kin: Kinematics, carriage_ik: bool) -> None:
    """Refuse carriage IK on an arm without a carriage. (A chain longer than the task's six has its model's posture
    for its self-motion: arm_model refuses one without.)"""
    if kin.carriage is None and carriage_ik:
        raise PlanError("motion.yaml carriage.ik is set, but this arm has no carriage")


def _joint_terms(kin: Kinematics, clik_cfg: dict, carriage_cfg: dict) -> tuple[np.ndarray, np.ndarray, _Carriage | None]:
    """(inverse weights, limit margins) per joint, and the carriage's terms, or None for an arm without one."""
    inverse_weights = np.ones(kin.n)
    margin = np.full(kin.n, float(clik_cfg.get("joint_limit_margin_rad", 0.0)))
    c = kin.carriage
    if c is None:
        return inverse_weights, margin, None
    weight = float(clik_cfg.get("carriage_weight", 2.0))
    inverse_weights[c] = 1.0 / (weight * weight)
    margin[c] = 0.0   # the carriage: its own travel
    toward = np.zeros(kin.n)
    toward[c] = 1.0
    return inverse_weights, margin, _Carriage(
        c, toward, float(clik_cfg.get("carriage_center_gain_s", 2.0)), float(carriage_cfg["bias_m"]),
        float(carriage_cfg["max_m_s"]), float(carriage_cfg.get("max_m_s2", 0.02)), carriage_cfg["window_m"])


class _Solve:
    """CLIK's joint velocities each tick: damped least squares for the task twist, and that with a joint velocity sent
    through the task's null space, which the tip does not feel (the carriage's centring, a long chain's posture)."""

    def __init__(self, kin: Kinematics, damping: float):
        self.damping2 = damping ** 2
        self.eye6 = np.eye(6)
        r = kin.revolute
        posture = None if len(range(kin.n)[r]) <= 6 else kin.arm_model.posture   # a chain longer than the task's six
        self.target = None if posture is None else np.asarray(posture.target)[r]
        self.pull = None if posture is None else posture.gain_s * np.asarray(posture.weights)[r]
        self.self_motion_max = None if posture is None else posture.max_rad_s

    def dls(self, jac6, twist):
        return jac6.T @ np.linalg.solve(jac6 @ jac6.T + self.damping2 * self.eye6, twist)

    def null_space(self, jac, twist, inverse_weights, z):
        """The weighted damped least-squares velocity for `twist`, plus the joint velocity `z` through its null space."""
        jw = jac * inverse_weights[None, :]
        normal = jw @ jac.T + self.damping2 * self.eye6
        velocity = jw.T @ np.linalg.solve(normal, twist)
        velocity += z - jw.T @ np.linalg.solve(normal, jac @ z)
        return velocity

    def chain(self, jac, twist, q, inverse_weights):
        """The revolute chain alone: the task, and a chain longer than six pulled toward its model's posture. The
        pull goes through the task's exact (undamped) null space: through null_space's damped one it leaked into the
        tip, which the position gain held 0.5 mm off its reference, and an xArm dip's dwell rose 0.54 mm in its hold
        and failed (2026-11-09). Its fastest joint moves at the posture's max_rad_s at most: the pull grows with the
        distance from the posture and no slower time law reaches it, so from a calibration hold a radian off the
        posture a plan's first tenth of a second ran a joint past its speed cap however slowly the tip was to move,
        and neither a lift nor a landing planned from there (2026-11-13)."""
        if self.target is None:
            return self.dls(jac, twist)
        z = self.pull * (self.target - q)
        jw = jac * inverse_weights[None, :]
        task = jw.T @ np.linalg.solve(jw @ jac.T + self.damping2 * self.eye6, twist)
        self_motion = z - jw.T @ np.linalg.lstsq(jw @ jac.T, jac @ z, rcond=None)[0]
        fastest = float(np.abs(self_motion).max())
        if fastest > self.self_motion_max:
            self_motion *= self.self_motion_max / fastest
        return task + self_motion


def clik(kin: Kinematics, q0, p, v, rot, pen, period_s: float, clik_cfg: dict, carriage_cfg: dict, *,
         carriage_v=None, carriage_ik: bool = False) -> ClikResult:
    """Integrate per-tick base-frame position/velocity (N,3), rotation (N,3,3) and pen flags (N,).

    The revolute chain (kin.revolute) tracks the task. An arm with a carriage (kin.carriage, the WXAI's):
    pen-down carriage IK uses the weighted, centred solve; other rows hold or follow carriage_v. An arm without
    one solves its chain alone, spending a chain longer than the task's six on its model's posture through the
    task's null space. Joint limits refuse; the caller checks returned velocity and model error. See
    ros/README.md §4.4.
    """
    _check_chain(kin, carriage_ik)
    r, joints = kin.revolute, kin.n
    q = np.array(q0, float).copy()
    n = len(p)
    solve = _Solve(kin, float(clik_cfg["damping"]))
    kp, ko = float(clik_cfg["position_gain_s"]), float(clik_cfg["orientation_gain_s"])
    inverse_weights, margin, cc = _joint_terms(kin, clik_cfg, carriage_cfg)
    # a start already inside the margin (the rest pose lies ~1 deg off its stops; a trip leaves the arm there) may
    # move, never farther toward the stop than it stands: before, its every plan failed at sample 0 (2026-10-21)
    lower = np.minimum(kin.lower + margin, np.where(np.isfinite(q), q, np.inf))
    upper = np.maximum(kin.upper - margin, np.where(np.isfinite(q), q, -np.inf))

    out_q, out_qd = np.empty((n, joints)), np.empty((n, joints))
    tip, model_err, orient_err = np.empty((n, 3)), np.empty(n), np.empty(n)
    previous_cv = 0.0
    position, rotation, jac = kin.evaluate(q)
    for k in range(n):
        twist = np.empty(6)
        twist[:3] = v[k] + kp * (p[k] - position)
        omega_ff = orientation_error(rot[k], rot[k + 1]) / period_s if k + 1 < n else 0.0
        twist[3:] = omega_ff + ko * orientation_error(rotation, rot[k])
        if cc is None:
            velocity = np.empty(joints)
            velocity[r] = solve.chain(jac[:, r], twist, q[r], inverse_weights[r])
        elif pen[k] and carriage_ik:
            c = cc.index
            centering = cc.center_gain * (cc.bias - q[c])
            velocity = solve.null_space(jac, twist, inverse_weights, cc.toward * centering)
            step = 0.9 * cc.accel * period_s
            slewed = float(np.clip(np.clip(velocity[c], -0.9 * cc.speed, 0.9 * cc.speed),
                                   previous_cv - step, previous_cv + step))
            if slewed != velocity[c]:
                velocity[r] = solve.dls(jac[:, r], twist - jac[:, c] * slewed)
                velocity[c] = slewed
            previous_cv = float(velocity[c])
        else:
            c = cc.index
            if carriage_v is not None and np.isfinite(carriage_v[k]):
                held = float(carriage_v[k])
            else:
                step = 0.5 * cc.accel * period_s
                held = previous_cv - float(np.clip(previous_cv, -step, step))
            velocity = np.empty(joints)
            velocity[r] = solve.dls(jac[:, r], twist - jac[:, c] * held)
            velocity[c] = held
            previous_cv = held
        q += period_s * velocity
        bad = ~np.isfinite(q) | (q < lower) | (q > upper)
        if bad.any():
            j = int(np.argmax(bad))
            raise PlanError(f"joint {kin.joint_names[j]} reaches its guarded limit ({q[j]:.4f}) at sample {k}")
        if pen[k] and carriage_ik and not cc.window[0] <= q[cc.index] <= cc.window[1]:
            raise PlanError(f"carriage leaves its drawing window ({q[cc.index] * 1e3:.2f} mm) at sample {k}")
        position, rotation, jac = kin.evaluate(q)
        out_q[k], out_qd[k], tip[k] = q, velocity, position
        model_err[k] = float(np.linalg.norm(p[k] - position))
        orient_err[k] = float(np.linalg.norm(orientation_error(rotation, rot[k])))
    return ClikResult(out_q, out_qd, tip, model_err, orient_err)
