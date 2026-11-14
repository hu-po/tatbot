"""Each technique's strokes in a cell: page millimetres, in drawing order, with a deposition per point for fades.

A stroke is (points (n, 2), deposition (n,) or None). Every technique lays its strokes about the origin along +x,
then turns them by `angle_deg` and moves them to the cell's centre. A pattern that would leave its cell, less
MARGIN_MM at every edge, is refused rather than clipped: the cell's reading assumes the whole pattern is in it.
"""
from __future__ import annotations

import numpy as np

from tatbot_lab.space import resolve

MARGIN_MM = 1.0
Stroke = tuple[np.ndarray, "np.ndarray | None"]


def _parallel(lines: int, length: float, pitch: float, *, serpentine: bool) -> list[np.ndarray]:
    """`lines` strokes along x, `pitch` apart in y, top first; alternate ones reversed when `serpentine`."""
    ys = (np.arange(lines)[::-1] - (lines - 1) / 2) * pitch
    out = []
    for k, y in enumerate(ys):
        a, b = np.array([-length / 2, y]), np.array([length / 2, y])
        out.append(np.array([b, a] if serpentine and k % 2 else [a, b]))
    return out


def _corners(length: float) -> list[np.ndarray]:
    """An L in the top third, a three-turn zigzag in the middle, three short strokes at the bottom."""
    third, half = length / 3, length / 2
    ell = np.array([[-half, third + third / 2], [-half, third / 2 + 0.5], [half, third / 2 + 0.5]])
    zig_x = np.linspace(-half, half, 5)
    zig = np.stack([zig_x, np.array([1, -1, 1, -1, 1]) * third / 3], axis=1)
    short = min(3.0, length / 4)
    ends = [np.array([[x - short / 2, -third - 0.5], [x + short / 2, -third - 0.5]]) for x in (-half + short / 2, 0.0,
                                                                                            half - short / 2)]
    return [ell, zig, *ends]


def strokes(technique: str, settings: dict) -> list[Stroke]:
    """The technique's strokes about the origin, turned, before placing; passes repeat each stroke in place."""
    s = resolve(technique, settings)
    if technique in ("line", "carry", "whip"):
        lines = _parallel(int(s["lines"]), s["length_mm"], s["pitch_mm"], serpentine=technique == "carry")
    elif technique == "solid":
        lines = _parallel(max(1, int(round(s["length_mm"] / s["pitch_mm"])) + 1), s["length_mm"], s["pitch_mm"],
                          serpentine=True)
    else:
        lines = _corners(s["length_mm"])
    turn = np.radians(s["angle_deg"])
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    out: list[Stroke] = []
    for pts in lines:
        pts = pts @ rot.T
        dep = None
        if technique == "whip":   # full at the start, falling linearly to the far end's deposition
            pts = np.linspace(pts[0], pts[-1], 9)
            dep = np.linspace(1.0, s["deposition"], len(pts))
        out += [(pts, dep)] * int(s["passes"])
    return out


def place(technique: str, settings: dict, box_mm) -> list[Stroke]:
    """The technique's strokes at the centre of a cell box (x0, y0, x1, y1), refused if they leave it."""
    x0, y0, x1, y1 = box_mm
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    placed = [(pts + centre, dep) for pts, dep in strokes(technique, settings)]
    allpts = np.vstack([pts for pts, _ in placed])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    if lo[0] < x0 + MARGIN_MM - 1e-9 or lo[1] < y0 + MARGIN_MM - 1e-9 or hi[0] > x1 - MARGIN_MM + 1e-9 \
            or hi[1] > y1 - MARGIN_MM + 1e-9:
        span = hi - lo
        raise ValueError(f"{technique} spans {span[0]:.1f} x {span[1]:.1f} mm, more than the "
                         f"{x1 - x0 - 2 * MARGIN_MM:.1f} x {y1 - y0 - 2 * MARGIN_MM:.1f} mm a cell holds")
    return placed
