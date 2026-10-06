"""URDF kinematics and damped-least-squares CLIK; test_clik_parity checks the C++ reference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PlanError(RuntimeError):
    """The planner cannot produce the motion: a joint limit on the way, or a tip it cannot follow."""


def orientation_error(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """0.5 * sum_c cross(R_cur[:, c], R_tgt[:, c]), the C++ term."""
    return 0.5 * np.cross(np.asarray(r_cur).T, np.asarray(r_tgt).T).sum(axis=0)


class Kinematics:
    """pinocchio model of one arm from URDF text (tatbot_description.robot_description(), or the
    /robot_description topic). Frames: <arm>/base_link -> <arm>/tcp; joints: names.joint_names(arm)."""

    def __init__(self, urdf_xml: str, arm: str = "right"):
        import pinocchio as pin
        from tatbot_description import names

        self._pin = pin
        self.arm = arm
        self.model = pin.buildModelFromXML(urdf_xml)
        self.data = self.model.createData()
        self.joint_names = names.joint_names(arm)
        ids = [self.model.getJointId(name) for name in self.joint_names]
        if any(i >= self.model.njoints for i in ids):
            raise ValueError(f"URDF lacks a joint of {self.joint_names}")
        self._iq = np.array([self.model.idx_qs[i] for i in ids])
        self._iv = np.array([self.model.idx_vs[i] for i in ids])
        self._q = pin.neutral(self.model)
        self._tcp = self.model.getFrameId(names.tcp_frame(arm))
        self._base = self.model.getFrameId(names.base_frame(arm))
        for frame, fid in ((names.tcp_frame(arm), self._tcp), (names.base_frame(arm), self._base)):
            if fid >= self.model.nframes:
                raise ValueError(f"URDF lacks frame {frame}")
        self.lower = np.array(self.model.lowerPositionLimit[self._iq], float)
        self.upper = np.array(self.model.upperPositionLimit[self._iq], float)
        pin.framesForwardKinematics(self.model, self.data, self._q)
        world_from_base = self.data.oMf[self._base]
        self._r_bw = np.array(world_from_base.rotation).T
        self._p_bw = -self._r_bw @ np.array(world_from_base.translation)

    @classmethod
    def from_repo(cls, repo=None, arm: str = "right") -> "Kinematics":
        from tatbot_description import robot_description

        return cls(robot_description(repo, arms=(arm,)), arm)

    def _full(self, q) -> np.ndarray:
        self._q[self._iq] = q
        return self._q

    def evaluate(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tcp position (3,), tcp rotation (3, 3), Jacobian (6, 7) [linear; angular]), all in the base."""
        pin = self._pin
        jac = pin.computeFrameJacobian(self.model, self.data, self._full(q), self._tcp, pin.LOCAL_WORLD_ALIGNED)
        placement = self.data.oMf[self._tcp]
        r = self._r_bw
        j = jac[:, self._iv]
        return (r @ placement.translation + self._p_bw, r @ placement.rotation,
                np.vstack([r @ j[:3], r @ j[3:]]))

    def fk(self, q) -> np.ndarray:
        """base_from_tcp (4, 4) at joints q (7,)."""
        return self._frame(q, self._tcp)

    def frame(self, q, name: str) -> np.ndarray:
        """base_from_<name> (4, 4) at joints q (7,), for any URDF frame (e.g. the wrist camera's)."""
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise ValueError(f"URDF lacks frame {name}")
        return self._frame(q, fid)

    def _frame(self, q, fid):
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, self._full(q))
        placement = pin.updateFramePlacement(self.model, self.data, fid)
        out = np.eye(4)
        out[:3, :3] = self._r_bw @ placement.rotation
        out[:3, 3] = self._r_bw @ placement.translation + self._p_bw
        return out

    def jacobian(self, q) -> np.ndarray:
        """(6, 7) [linear; angular] TCP Jacobian, base-aligned (pinocchio LOCAL_WORLD_ALIGNED in the base)."""
        return self.evaluate(q)[2]

    def solve(self, base_from_tcp: np.ndarray, q_seed, iterations: int = 200, tol: float = 1e-9) -> np.ndarray:
        """Joints reaching a TCP pose from q_seed (six-joint Newton steps, the carriage kept): a start pose
        for tests and reach checks, not a path."""
        q = np.array(q_seed, float)
        target = np.asarray(base_from_tcp, float)
        for _ in range(iterations):
            position, rotation, jac = self.evaluate(q)
            error = np.concatenate([target[:3, 3] - position, orientation_error(rotation, target[:3, :3])])
            if np.linalg.norm(error) < tol:
                return q
            j6 = jac[:, :6]
            q[:6] += j6.T @ np.linalg.solve(j6 @ j6.T + 1e-8 * np.eye(6), error)
        raise PlanError(f"no joints reach the pose from the seed (residual {np.linalg.norm(error):.2e})")


def depth_sensitivity(kin: Kinematics, q, axis, damping: float = 1e-4) -> np.ndarray:
    """(M, 7) joints per metre of TCP travel along `axis` (M, 3, unit, base) at each row of q (M, 7), the tool's
    rotation and the carriage held: a pen-height change's first-order map to joints, so a trim h composes as
    q + h * dq_dh. On the pink arm's drawing poses that is within 4 um of the target at 1 mm and 18 um at 2 mm,
    and on the plan's own IK branch by construction (ros/README.md 4.4, Trim)."""
    out = np.zeros((len(q), 7))
    twist, eye6 = np.zeros(6), np.eye(6)
    for k, (qk, ak) in enumerate(zip(np.asarray(q, float), np.asarray(axis, float), strict=True)):
        j6 = kin.evaluate(qk)[2][:, :6]
        twist[:3] = ak
        out[k, :6] = j6.T @ np.linalg.solve(j6 @ j6.T + damping * damping * eye6, twist)
    return out


@dataclass
class ClikResult:
    q: np.ndarray            # (N, 7) joints after each sample's tick
    qd: np.ndarray           # (N, 7) the velocity integrated over that tick
    tip: np.ndarray          # (N, 3) FK tip after the tick
    model_error_m: np.ndarray        # (N,) |reference - FK tip| after the tick
    orientation_error_rad: np.ndarray  # (N,)


def clik(kin: Kinematics, q0, p, v, rot, pen, period_s: float, clik_cfg: dict, carriage_cfg: dict, *,
         carriage_v=None, carriage_ik: bool = False) -> ClikResult:
    """Integrate per-tick base-frame position/velocity (N,3), rotation (N,3,3) and pen flags (N,).

    Pen-down carriage IK uses the weighted, centred solve; other rows hold or follow carriage_v.
    Joint limits refuse; the caller checks returned velocity and model error. See ros/README.md §4.4.
    """
    q = np.array(q0, float).copy()
    n = len(p)
    damping2 = float(clik_cfg["damping"]) ** 2
    kp, ko = float(clik_cfg["position_gain_s"]), float(clik_cfg["orientation_gain_s"])
    weight = float(clik_cfg.get("carriage_weight", 2.0))
    center_gain = float(clik_cfg.get("carriage_center_gain_s", 2.0))
    bias = float(carriage_cfg["bias_m"])
    c_speed = float(carriage_cfg["max_m_s"])
    c_accel = float(carriage_cfg.get("max_m_s2", 0.02))
    window = carriage_cfg["window_m"]
    inverse_weights = np.array([1.0] * 6 + [1.0 / (weight * weight)])
    margin = np.array([float(clik_cfg.get("joint_limit_margin_rad", 0.0))] * 6 + [0.0])  # carriage: its own travel
    lower, upper = kin.lower + margin, kin.upper - margin
    eye6 = np.eye(6)

    def dls(jac6, twist):
        return jac6.T @ np.linalg.solve(jac6 @ jac6.T + damping2 * eye6, twist)

    out_q, out_qd = np.empty((n, 7)), np.empty((n, 7))
    tip, model_err, orient_err = np.empty((n, 3)), np.empty(n), np.empty(n)
    previous_cv = 0.0
    position, rotation, jac = kin.evaluate(q)
    for k in range(n):
        twist = np.empty(6)
        twist[:3] = v[k] + kp * (p[k] - position)
        omega_ff = orientation_error(rot[k], rot[k + 1]) / period_s if k + 1 < n else 0.0
        twist[3:] = omega_ff + ko * orientation_error(rotation, rot[k])
        if pen[k] and carriage_ik:
            jw = jac * inverse_weights[None, :]
            normal = jw @ jac.T + damping2 * eye6
            velocity = jw.T @ np.linalg.solve(normal, twist)
            centering = center_gain * (bias - q[6])
            velocity += np.eye(7)[6] * centering - jw.T @ np.linalg.solve(normal, jac[:, 6] * centering)
            step = 0.9 * c_accel * period_s
            slewed = float(np.clip(np.clip(velocity[6], -0.9 * c_speed, 0.9 * c_speed),
                                   previous_cv - step, previous_cv + step))
            if slewed != velocity[6]:
                velocity[:6] = dls(jac[:, :6], twist - jac[:, 6] * slewed)
                velocity[6] = slewed
        else:
            if carriage_v is not None and np.isfinite(carriage_v[k]):
                held = float(carriage_v[k])
            else:
                step = 0.5 * c_accel * period_s
                held = previous_cv - float(np.clip(previous_cv, -step, step))
            velocity = np.empty(7)
            velocity[:6] = dls(jac[:, :6], twist - jac[:, 6] * held)
            velocity[6] = held
        q += period_s * velocity
        previous_cv = float(velocity[6])
        bad = ~np.isfinite(q) | (q < lower) | (q > upper)
        if bad.any():
            j = int(np.argmax(bad))
            raise PlanError(f"joint {kin.joint_names[j]} reaches its guarded limit ({q[j]:.4f}) at sample {k}")
        if pen[k] and carriage_ik and not window[0] <= q[6] <= window[1]:
            raise PlanError(f"carriage leaves its drawing window ({q[6] * 1e3:.2f} mm) at sample {k}")
        position, rotation, jac = kin.evaluate(q)
        out_q[k], out_qd[k], tip[k] = q, velocity, position
        model_err[k] = float(np.linalg.norm(p[k] - position))
        orient_err[k] = float(np.linalg.norm(orientation_error(rotation, rot[k])))
    return ClikResult(out_q, out_qd, tip, model_err, orient_err)
