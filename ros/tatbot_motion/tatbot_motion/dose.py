"""Needle pokes along a stroke: the planner's own time law times the machine's rate. No IK, no robot I/O.

The tattoo machine runs for the whole of a stroke's goal, so the skin under the needles takes rate x time pokes:
14.6 a millimetre at 51 Hz and 3.5 mm/s, more wherever the pen is slower (an ease, a corner) and a whole dwell's
worth on one spot where it comes down and lifts. An end drawn moving (motion.yaml draw.ends: fly) takes the
steady dose through the marking band and no dwell. This is a count of pokes, not of ink: how dark a poke leaves
the skin is the lab's to measure against it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tatbot_motion import timelaw as tl
from tatbot_motion.plan import FLY, STOP, _cfg, _fly_ends, drawing_time_law, flying_law, stroke_ends

MODEL = "tatbot-dose/1"
JOIN = "join"        # an end joined pen-down to the neighbouring op: eased on the skin, no descent or lift
BIN_M = 0.0001       # the dose is kept in bins this long along the stroke
END_M = 0.001        # a stroke's start and end: its first and last millimetre (half of a stroke under two)
OVER = 2.0           # the line reported as over-dosed takes more than this many times the steady dose
_dwell_cache: dict[str, tuple[float, float]] = {}


def rate_hz(motion: dict) -> float:
    """The machine's rate the dose is counted at: motion.yaml machine.rate_hz, a recorded value."""
    return float(motion["machine"]["rate_hz"])


def band_m(motion: dict) -> float:
    """How far over its pen-down path the pen's needles can still mark (motion.yaml draw.fly.band_m)."""
    return float(motion["draw"]["fly"]["band_m"])


def dwell_s(motion: dict) -> tuple[float, float]:
    """(s, s) a stopping stroke's descent and its lift spend inside the marking band, on one spot: the planner's
    own pen-up legs (the descent's last approach.final_m and the lift's first lift_first_m, at the touchdown budget)."""
    key = json.dumps([motion["approach"], motion["corners"], motion["control_rate_hz"], band_m(motion)], sort_keys=True)
    if key not in _dwell_cache:
        dt, retime = _cfg(motion)
        approach, eye, band = motion["approach"], np.eye(3), band_m(motion)
        standoff, final = float(approach["standoff_m"]), float(approach["final_m"])
        down, _, _ = tl.pen_up_leg(np.array([[0.0, 0.0, standoff], [0.0, 0.0, final], [0.0, 0.0, 0.0]]), eye, eye,
                                   [approach["descend_m_s"], approach["final_m_s"]], retime, dt, gentle_end=True)
        up, _, _ = tl.pen_up_leg(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, standoff]]), eye, eye, [approach["lift_m_s"]],
                                 _cfg(motion, lift=True)[1], dt, gentle_start=True)
        _dwell_cache[key] = (float((down[:, 2] < band).sum() * dt), float((up[:, 2] < band).sum() * dt))
    return _dwell_cache[key]


@dataclass(frozen=True)
class StrokeDose:
    """One stroke's pokes: along its line in BIN_M bins, and on the spot at each end."""

    rate_hz: float
    cruise_m_s: float
    length_m: float
    edges_m: np.ndarray     # (n + 1,) bin edges, m from the stroke's first point along it; a flying end's marking band
                            # reaches a little outside [0, length]
    pokes: np.ndarray       # (n,) pokes in each bin
    touchdown: float        # pokes on the first point while the pen comes down and settles there
    liftoff: float          # pokes on the last point while it lifts
    start: str              # STOP, FLY or JOIN
    end: str
    corners: int            # vertices the corner law slowed
    drawing_s: float        # the time its needles are over its line (a dot: its dwell), the dwells at its ends apart

    @property
    def steady_per_mm(self) -> float:
        return self.rate_hz / (self.cruise_m_s * 1000.0)

    @property
    def per_mm(self) -> np.ndarray:
        return self.pokes / (np.diff(self.edges_m) * 1000.0)

    @property
    def total(self) -> float:
        return float(self.pokes.sum()) + self.touchdown + self.liftoff

    def laid(self, at_m) -> np.ndarray:
        """Pokes laid from the first bin up to each arc position along the stroke (the dwells at its ends apart)."""
        return np.interp(at_m, self.edges_m, np.r_[0.0, np.cumsum(self.pokes)])

    def between(self, a_m: float, b_m: float) -> float:
        """Pokes laid between two arc positions along the stroke (the spot dwells at its ends not counted)."""
        lo, hi = self.laid([a_m, b_m])
        return float(hi - lo)

    def summary(self) -> dict:
        """Pokes per millimetre over the stroke's first and last END_M and its middle (the median between them),
        its densest BIN_M away from its ends when a corner slowed it, the share of its line over OVER times the
        steady dose, and the pokes on the spot at each end."""
        if self.length_m <= 1e-6:   # a dot: every poke on one spot
            return {"start_per_mm": None, "middle_per_mm": None, "end_per_mm": None, "corner_peak_per_mm": None,
                    "over_share": 1.0, "over_mm": 0.0, "touchdown_pokes": round(self.touchdown, 1),
                    "liftoff_pokes": round(self.liftoff, 1), "pokes": round(self.total, 1)}
        end = min(END_M, 0.5 * self.length_m)
        centre, per_mm = 0.5 * (self.edges_m[:-1] + self.edges_m[1:]), self.per_mm
        inner = (centre > end) & (centre < self.length_m - end)
        middle = float(np.median(per_mm[inner if inner.any() else np.ones(len(centre), bool)]))
        peak = float(per_mm[inner].max()) if self.corners and inner.any() else None
        on = (centre > 0.0) & (centre < self.length_m)
        over = on & (per_mm > OVER * self.steady_per_mm)
        return {"start_per_mm": round(self.between(0.0, end) / (end * 1000.0), 1), "middle_per_mm": round(middle, 1),
                "end_per_mm": round(self.between(self.length_m - end, self.length_m) / (end * 1000.0), 1),
                "corner_peak_per_mm": None if peak is None else round(peak, 1),
                "over_share": round(float(over.sum() / max(on.sum(), 1)), 3),
                "over_mm": round(float(over.sum() * BIN_M * 1000.0), 2),
                "touchdown_pokes": round(self.touchdown, 1), "liftoff_pokes": round(self.liftoff, 1),
                "pokes": round(self.total, 1)}


def stroke(points_m, cruise_m_s: float, motion: dict, *, joined: bool = False, lift_at_end: bool = True,
           clear_m=None, settle_s: float = 0.0, hz: float | None = None) -> StrokeDose:
    """The dose of a stroke along `points_m` (its page polyline, closed back to its start when it is closed) drawn
    at `cruise_m_s`, as plan_op plans it under `motion`: its ends stop or fly by draw.ends, an end `joined` pen-down
    to the op before it (or, without `lift_at_end`, to the one after) is eased on the skin with no dwell, and a
    flying end whose run would leave the page's clear area (`clear_m`) stops (plan._fly_ends). `settle_s` is the
    pen's dwell at touchdown (motion.yaml pen.<mode>.settle_s)."""
    hz = rate_hz(motion) if hz is None else float(hz)
    dt, retime = _cfg(motion)
    points = np.asarray(points_m, float)
    length = tl.polyline_length(points)
    down_s, up_s = dwell_s(motion)
    start, end = JOIN if joined else STOP, STOP if lift_at_end else JOIN
    if length <= 1e-6:   # a dot: a pen-down dwell of twice the shortest ease
        hold = 2.0 * float(motion["draw"]["min_ease_s"])
        return StrokeDose(hz, cruise_m_s, 0.0, np.zeros(1), np.zeros(0),
                          hz * ((0.0 if joined else down_s + settle_s) + hold), hz * (up_s if lift_at_end else 0.0),
                          start, end, 0, hold)
    fly = stroke_ends(motion)
    fly_start, fly_end, _ = _fly_ends(fly, points, points, clear_m, cruise_m_s, joined=joined, from_arc_m=0.0,
                                      lift_at_end=lift_at_end)
    if fly_start or fly_end:
        _, s, lift, s0 = flying_law(points, cruise_m_s, motion, fly, fly_start, fly_end)
        corners = int((tl.corner_speed_limits(*tl.polyline_turn_angles(points), cruise_m_s, retime.corner_accel_m_s2,
                                              retime.window_s) < cruise_m_s).sum())
        marking = np.flatnonzero(lift < fly.band_m)   # one stretch: in through the band, along, out through it
        at = np.r_[0.0, s][marking[0]:marking[-1] + 2] - s0
        start, end = FLY if fly_start else start, FLY if fly_end else end
    else:
        _, s, _, info = drawing_time_law(points, cruise_m_s, motion)
        at, corners = np.r_[0.0, s], int(info["limited_vertices"])
    # the time the pen spends in each bin: its arc at every control tick, read back at the bin edges
    lo = min(0.0, float(np.floor(at[0] / BIN_M + 1e-9)) * BIN_M)
    edges = lo + BIN_M * np.arange(int(np.ceil((max(length, at[-1]) - lo) / BIN_M - 1e-9)) + 1)
    rising = at + 1e-15 * np.arange(len(at))   # a pen at rest for a tick still gets an abscissa of its own
    reached = np.interp(edges, rising, dt * np.arange(len(at)))
    reached[-1] = dt * (len(at) - 1)           # the last bin holds the ticks spent on its far edge
    pokes = np.diff(reached) * hz
    return StrokeDose(hz, cruise_m_s, length, edges, pokes,
                      hz * (down_s + settle_s) if start == STOP else 0.0, hz * up_s if end == STOP else 0.0,
                      start, end, corners, float(reached[-1]))
