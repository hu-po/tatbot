"""The operator's pen trim (ros/README.md 4.4): a height offset along each knot's depth axis, added to a plan as
q + h * dq_dh when it is sent, so a key press needs no IK and never leaves the plan's IK branch.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from tatbot_motion import timelaw as tl
from tatbot_motion.plan import PHASE_DESCEND


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

    def to(self, t0: float, target_m: float, cfg: dict, by_s: float | None = None) -> Trim:
        """Then, from t0, a ramp to target_m no faster than motion.yaml pen.trim speed_m_s nor shorter than min_s. With
        `by_s` it is over by then, faster than speed_m_s if it must be: the pen is in the air until then (slow_from_s),
        and speed_m_s is a key press's on the skin."""
        a = float(self.value(t0))
        span = tl.QUINTIC_PEAK * abs(target_m - a) / float(cfg["speed_m_s"])
        if by_s is not None:
            span = min(span, by_s - float(t0))
        return Trim(self.start_m, (*self.ramps, (float(t0), max(float(cfg["min_s"]), span), a, float(target_m))))


def slow_from_s(traj, final_m: float) -> float | None:
    """When a stroke plan's descent enters its slow stretch: the time (s) of its first descending knot within
    `final_m` (motion.yaml approach.final_m) of where the pen comes down, along the depth axis there. None for a plan
    that comes down nowhere (joined pen-down to the stroke before, or a hold that is already down).

    A trim eased in from a hover (ArmExecutor.trimmed) is over by then. Eased at a key press's 2 mm/s, a -2.5 mm trim
    took 2.3 s of a 2.4 s descent, and the pen went through its 3 mm/s stretch at 5 mm/s on every stroke of a
    rehearsal; one deeper than -2.9 mm was still easing in when the stroke began (2026-11-12)."""
    down = np.flatnonzero(np.asarray(traj.phase) == PHASE_DESCEND)
    if not len(down):
        return None
    end = down[-1]
    height = (np.asarray(traj.tip)[down] - traj.tip[end]) @ np.asarray(traj.axis)[end]
    return float(traj.t[down[np.argmax(height <= float(final_m))]])


def compose(traj, trim: Trim):
    """`traj` (a stroke plan, with dq_dh) with `trim` added along its depth axis; info["trim"] keeps the trim."""
    if traj.dq_dh is None:
        raise ValueError("this plan has no depth axis to trim along")
    h, hd = trim.value(traj.t), trim.rate(traj.t)
    qd = traj.qd + hd[:, None] * traj.dq_dh + h[:, None] * np.gradient(traj.dq_dh, traj.t, axis=0)
    qd[-1] = 0.0
    return dataclasses.replace(traj, q=traj.q + h[:, None] * traj.dq_dh, qd=qd, tip=traj.tip + h[:, None] * traj.axis,
                               info={**traj.info, "trim": trim})


def dwell(traj, start: int, index: int, span_s: float, dt: float):
    """`traj` from knot `start` to knot `index`, then held at that knot span_s more, a knot each dt: a goal that waits
    there while the keys trim it along its depth axis (the depth hold, ros/README.md 4.4)."""
    rows = np.r_[np.arange(start, index + 1), np.full(max(1, round(span_s / dt)), index)]
    held = len(rows) - (index - start + 1)
    qd = traj.qd[rows].copy()
    qd[-held - 1:] = 0.0
    return dataclasses.replace(
        traj, t=np.r_[traj.t[start:index + 1] - traj.t[start], traj.t[index] - traj.t[start] + dt * np.arange(1, held + 1)],
        q=traj.q[rows], qd=qd, tip=traj.tip[rows], arc_m=traj.arc_m[rows], phase=traj.phase[rows], axis=traj.axis[rows],
        dq_dh=traj.dq_dh[rows])
