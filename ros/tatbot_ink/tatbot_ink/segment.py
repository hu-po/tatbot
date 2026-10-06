"""Split strokes at a contact budget, cutting at sharp corners and keeping arc addresses (README 4.2 step 5).

Ported from scripts/lib/stroke_material.py (`cut_points`, `material_candidates`,
`resample_polyline_by_arclength`) and human_rep/ink_program.py (`_split_polyline`).
The uniform partition fixes the piece count; each interior cut may move up to a fifth of the
budget to the sharpest corner in reach, and every piece still fits the budget. A piece keeps its
source arc interval, so a stroke can resume from an arc position.
"""
from __future__ import annotations

import math

import numpy as np

from tatbot_ink.errors import CompileError

CUT_WINDOW_FRACTION = 0.2
CORNER_MIN_RAD = math.radians(5.0)
MAX_PIECES = 10_000


def arc_lengths(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample(points: np.ndarray, s) -> np.ndarray:
    """Points of a polyline at arc lengths `s`."""
    points = np.asarray(points, float)
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    s = np.clip(np.asarray(s, float), 0.0, cum[-1])
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(seg_len[index] > 0.0, (s - cum[index]) / seg_len[index], 0.0)
    return points[index] + frac[:, None] * seg[index]


def cut_points(stroke: np.ndarray, arc: np.ndarray, count: int, max_length: float) -> np.ndarray:
    """count + 1 arc positions from 0 to the length, interior cuts moved to nearby sharp corners."""
    stroke = np.asarray(stroke, float)
    arc = np.asarray(arc, float)
    length = float(arc[-1])
    uniform = np.linspace(0.0, length, count + 1)
    if count < 2 or len(stroke) < 3:
        return uniform
    tangents = np.diff(stroke, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-300)
    turning = np.arccos(np.clip(np.einsum("ij,ij->i", tangents[:-1], tangents[1:]), -1.0, 1.0))
    corners = np.flatnonzero(turning >= CORNER_MIN_RAD) + 1
    window = CUT_WINDOW_FRACTION * max_length
    cuts = uniform.copy()
    for index in range(1, count):
        previous = float(cuts[index - 1])
        lo = max(uniform[index] - window, length - (count - index) * max_length, previous)
        hi = min(uniform[index] + window, previous + max_length)
        cuts[index] = min(max(float(uniform[index]), lo), hi)
        candidates = corners[(arc[corners] > lo) & (arc[corners] < hi)]
        if len(candidates):
            cuts[index] = float(arc[candidates[np.argmax(turning[candidates - 1])]])
    return cuts


def split(points: np.ndarray, max_length_m: float, arc_start_m: float = 0.0) -> list[tuple[np.ndarray, tuple[float, float]]]:
    """Pieces of one polyline no longer than `max_length_m`, each with its (lo, hi) source arc.

    `arc_start_m` offsets the addresses when `points` is itself a piece of a longer stroke.
    Consecutive pieces share their cut point. A stroke within the budget comes back whole.
    """
    points = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0]
    points = points[keep]
    if len(points) < 2:
        raise CompileError("a stroke has no length")
    arc = arc_lengths(points)
    length = float(arc[-1])
    if not (max_length_m > 0.0 and math.isfinite(max_length_m)):
        raise CompileError("the segment budget must be positive")
    count = max(1, math.ceil(length / max_length_m - 1e-12))
    if count > MAX_PIECES:
        raise CompileError(f"a stroke needs {count} pieces at a {max_length_m * 1000:g} mm budget")
    if count == 1:
        return [(points, (arc_start_m, arc_start_m + length))]
    cuts = cut_points(points, arc, count, max_length_m)
    ends = resample(points, cuts)
    ends[0], ends[-1] = points[0], points[-1]
    pieces = []
    for index in range(count):
        lo, hi = float(cuts[index]), float(cuts[index + 1])
        interior = points[(arc > lo) & (arc < hi)]
        piece = np.vstack([ends[index], interior, ends[index + 1]])
        pieces.append((piece, (arc_start_m + lo, arc_start_m + hi)))
    return pieces


def split_timed(points, max_seconds, speed, motion):
    """Computational chunks bounded by the actual Cartesian drawing law, with source arcs.

    Chunk boundaries do not imply a lift or replenish the tool. IK may further
    retime execution; this is the same offline model used by preparation estimates.
    """
    from tatbot_motion.estimate import drawing_seconds

    speed = min(speed, motion["tip_speed"]["pen_down_max_m_s"])
    minimum = 2 * motion["draw"]["min_ease_s"] + 1 / motion["control_rate_hz"]
    if max_seconds < minimum:
        raise CompileError(f"chunk budget must allow at least {minimum:g} s for easing and a control tick")
    pending = list(reversed(split(points, max_seconds * speed)))
    result = []
    while pending:
        piece, (lo, hi) = pending.pop()
        seconds = drawing_seconds({"points_m": piece, "closed": False}, speed, motion)
        if seconds <= max_seconds + 1e-10:
            result.append((piece, (lo, hi), seconds))
            continue
        if len(result) + len(pending) + 2 > MAX_PIECES or hi - lo < 1e-9:
            raise CompileError("drawing cannot fit the requested chunk budget")
        children = split(piece, (hi - lo) / 2, lo)
        # Preserve the original endpoint address against floating point accumulation.
        children[-1] = (children[-1][0], (children[-1][1][0], hi))
        pending.extend(reversed(children))
    return result
