"""The operator's pen trim (ros/README.md 4.4): a height offset along each knot's depth axis, added to a plan as
q + h * dq_dh when it is sent, so a key press needs no IK and never leaves the plan's IK branch.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from tatbot_motion import timelaw as tl


@dataclass(frozen=True)
class Trim:
    """A trim over a goal's time (s): start_m, then quintic ramps (t0_s, span_s, from_m, to_m), in order, apart."""

    start_m: float = 0.0
    ramps: tuple = ()

    @property
    def end_m(self) -> float:
        return self.ramps[-1][3] if self.ramps else self.start_m

    def settled_s(self) -> float:
        return self.ramps[-1][0] + self.ramps[-1][1] if self.ramps else 0.0

    def value(self, t) -> np.ndarray:
        h = np.full(np.shape(t), self.start_m)
        for t0, span, a, b in self.ramps:
            h = np.where(t >= t0, a + (b - a) * tl._quintic(np.clip((t - t0) / span, 0.0, 1.0)), h)
        return h

    def rate(self, t) -> np.ndarray:
        hd = np.zeros(np.shape(t))
        for t0, span, a, b in self.ramps:
            u = np.clip((t - t0) / span, 0.0, 1.0)
            hd = np.where(t >= t0, (b - a) / span * 30.0 * u * u * (1.0 - u) ** 2, hd)
        return hd

    def to(self, t0: float, target_m: float, cfg: dict) -> Trim:
        """Then, from t0, a ramp to target_m no faster than motion.yaml pen.trim speed_m_s nor shorter than min_s."""
        a = float(self.value(t0))
        span = max(float(cfg["min_s"]), tl.QUINTIC_PEAK * abs(target_m - a) / float(cfg["speed_m_s"]))
        return Trim(self.start_m, (*self.ramps, (float(t0), span, a, float(target_m))))


def compose(traj, trim: Trim):
    """`traj` (a stroke plan, with dq_dh) with `trim` added along its depth axis; info["trim"] keeps the trim."""
    if traj.dq_dh is None:
        raise ValueError("this plan has no depth axis to trim along")
    h, hd = trim.value(traj.t), trim.rate(traj.t)
    qd = traj.qd + hd[:, None] * traj.dq_dh + h[:, None] * np.gradient(traj.dq_dh, traj.t, axis=0)
    qd[-1] = 0.0
    return dataclasses.replace(traj, q=traj.q + h[:, None] * traj.dq_dh, qd=qd, tip=traj.tip + h[:, None] * traj.axis,
                               info={**traj.info, "trim": trim})
