"""Pen-pointing inverse kinematics on the episode's own MuJoCo model.

The task pins five degrees of freedom: the gap point (``gap_point`` site,
the configured distance beyond the lens along the pen axis) sits on the skin
and the pen axis points into it. Roll about the pen axis is left to a
posture term in the null space, which pulls the wrist toward the rest posture
(the tool rolled 90 degrees) and keeps it away from its stops; the wrist roll
still varies as the pen follows the skin.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tatbot_travel.scene import ARM, GAP_SITE, OPTICAL_FRAME, PEN_BODY

HINGES = tuple(f"{ARM}/joint_{i}" for i in range(6))
LIMIT_MARGIN = 0.05  # rad kept clear of every stop: the real runner clips at the controller's limits less 0.05


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: float  # m
    axis_error: float  # rad

    def ok(self, position_tol: float = 0.003, axis_tol: float = 0.06) -> bool:
        return self.position_error <= position_tol and self.axis_error <= axis_tol


class ArmKinematics:
    """Forward and inverse kinematics of the arm hinges on a private ``MjData``."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(n) for n in HINGES]
        self.qadr = np.array([model.jnt_qposadr[j.id] for j in joints])
        self.dofadr = np.array([model.jnt_dofadr[j.id] for j in joints])
        self.lower = np.array([model.jnt_range[j.id][0] for j in joints]) + LIMIT_MARGIN
        self.upper = np.array([model.jnt_range[j.id][1] for j in joints]) - LIMIT_MARGIN
        self.gap_site = model.site(GAP_SITE).id
        self.pen_body = model.body(PEN_BODY).id
        self.camera_body = model.body(OPTICAL_FRAME).id
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def gap_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World position of the gap point and the pen's unit axis (toward the skin)."""
        self._set(q)
        return self.data.site_xpos[self.gap_site].copy(), self.data.xmat[self.pen_body].reshape(3, 3)[:, 2].copy()

    def camera_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Optical frame (x right, y down, z forward) as (position, rotation) in the world."""
        self._set(q)
        return self.data.xpos[self.camera_body].copy(), self.data.xmat[self.camera_body].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, target: np.ndarray, axis: np.ndarray, q_rest: np.ndarray, *,
              iters: int = 12, damping: float = 0.02, posture_gain: float = 0.08,
              max_step: float = 0.15) -> IKResult:
        """Damped least squares on [position; axis], posture in the null space."""
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lower, self.upper)
        axis = axis / np.linalg.norm(axis)
        for _ in range(iters):
            self._set(q)
            pos = self.data.site_xpos[self.gap_site]
            z = self.data.xmat[self.pen_body].reshape(3, 3)[:, 2]
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.gap_site)
            jp = self._jacp[:, self.dofadr]
            project = np.eye(3) - np.outer(z, z)  # roll about the pen axis is free
            ja = project @ self._jacr[:, self.dofadr]
            jac = np.vstack([jp, ja])
            err = np.concatenate([target - pos, np.cross(z, axis)])
            jjt = jac @ jac.T + (damping ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)
            pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
            null = np.eye(6) - pinv @ jac
            dq += null @ (posture_gain * (q_rest - q))
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.lower, self.upper)
        pos, z = self.gap_pose(q)
        angle = float(np.arccos(np.clip(z @ axis, -1.0, 1.0)))
        return IKResult(q=q, position_error=float(np.linalg.norm(target - pos)), axis_error=angle)
