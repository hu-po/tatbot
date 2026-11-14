"""Ink coverage rasters and their composite over a substrate, in linear light."""
from __future__ import annotations

import numpy as np

from .color import srgb_to_linear

RETURN_STEP_PX = 0.5   # a stroke is walked this finely to count the pen's returns over a pixel


def _segment_distance(xs, ys, a, b):
    d = b - a
    t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.hypot(xs - (a[0] + t * d[0]), ys - (a[1] + t * d[1]))


def _swept_row_distance(xs, ys, a, b, row):
    """Distance to the parallelogram a row (centred on the path, `row` end to end) sweeps from a to b; 0 inside."""
    h = row / 2
    dist = np.minimum.reduce([_segment_distance(xs, ys, p, q)
                              for p, q in ((a - h, b - h), (a + h, b + h), (a - h, a + h), (b - h, b + h))])
    d = b - a
    det = d[0] * row[1] - d[1] * row[0]
    if abs(det) > 1e-9:   # dragged along the row the parallelogram is its edges, which the distances already hold
        rx, ry = xs - a[0], ys - a[1]
        s, u = (rx * row[1] - ry * row[0]) / det, (d[0] * ry - d[1] * rx) / det
        dist = np.where((s >= 0) & (s <= 1) & (np.abs(u) <= 0.5), 0.0, dist)
    return dist


def stroke_coverage(shape: tuple[int, int], points_px: np.ndarray, width_px: float,
                    opacity: np.ndarray | float = 1.0,
                    row_px=None, dose=None, dwell=(0.0, 0.0)) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """One pen-down polyline as an antialiased coverage patch: (row0, row1, col0, col1) and the patch.

    Segments of one stroke combine by max, so a polyline's own joints never double its ink; where a round mark
    comes back over a pixel after going more than its width along the line (a retrace, a pendulum's next swing)
    the pixel takes another pass, as tatbot_ink.passes counts them (_returns). `opacity` is per segment (ink fading
    since the last dip, a deposition) or one value. A round mark is a disc `width_px` across swept along the line.
    A magnum's is its needle row, the vector `row_px` (held at one angle on the page), swept along it and rounded
    by `width_px`: wide across the row, thin along it, square where it starts and stops; it stays one pass.

    `dose` is the needle pokes each segment lays over the steady dose of one pass (one value, or one per segment):
    a segment at twice the steady dose darkens as two passes do (1 - (1 - c)^2), an ease or a corner more. `dwell`
    is the same for the pokes on the spot at the stroke's first and last point (a pen that comes down, or lifts,
    with the machine running), in passes over a disc `width_px` across there.
    """
    pts = np.asarray(points_px, float)
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        pts = np.vstack([pts, pts])
    row = None if row_px is None else np.asarray(row_px, float)
    half = width_px / 2 + 1.0 + (0.0 if row is None else float(np.linalg.norm(row)) / 2)
    c0, r0 = np.floor(pts.min(axis=0) - half).astype(int)
    c1, r1 = np.ceil(pts.max(axis=0) + half).astype(int) + 1
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    if r0 >= r1 or c0 >= c1:
        return None
    patch = np.zeros((r1 - r0, c1 - c0))
    alpha = np.broadcast_to(np.asarray(opacity, float), (len(pts) - 1,))
    times = np.broadcast_to(np.asarray(1.0 if dose is None else dose, float), (len(pts) - 1,))
    # the spot at each end: a segment without length there, at the opacity of the stroke's nearest
    spots = [(pts[k], pts[k], alpha[k], passes) for k, passes in zip((0, -1), dwell, strict=True) if passes > 0.0]
    for a, b, op, n in [*zip(pts[:-1], pts[1:], alpha, times, strict=True), *spots]:
        lo = np.floor(np.minimum(a, b) - half).astype(int)
        hi = np.ceil(np.maximum(a, b) + half).astype(int) + 1
        sc0, sr0 = max(lo[0], c0), max(lo[1], r0)
        sc1, sr1 = min(hi[0], c1), min(hi[1], r1)
        if sr0 >= sr1 or sc0 >= sc1:
            continue
        ys, xs = np.mgrid[sr0:sr1, sc0:sc1] + 0.5
        dist = _segment_distance(xs, ys, a, b) if row is None else _swept_row_distance(xs, ys, a, b, row)
        cov = np.clip(width_px / 2 - dist + 0.5, 0.0, 1.0) * op
        if n != 1.0:
            cov = 1.0 - (1.0 - cov) ** n
        view = patch[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0]
        np.maximum(view, cov, out=view)
    if row is None and len(pts) > 2:   # two points cannot come back over themselves
        patch = 1.0 - (1.0 - patch) ** _returns(pts, width_px, (r0, r1, c0, c1))
    return (r0, r1, c0, c1), patch


def _returns(pts, width_px, box):
    """(rows, cols) of the patch `box`: the passes a disc `width_px` across makes over each pixel walking the line,
    another each time it comes back after going more than its width along it (tatbot_ink.passes' rule); at least 1."""
    r0, r1, c0, c1 = box
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0]]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / RETURN_STEP_PX)) + 2)
    xy = np.c_[np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])]
    off = np.arange(-int(np.ceil(width_px / 2)) - 1, int(np.ceil(width_px / 2)) + 2)
    col = np.floor(xy[:, 0]).astype(int)[:, None, None] + off[None, None, :]
    row = np.floor(xy[:, 1]).astype(int)[:, None, None] + off[None, :, None]
    near = (col + 0.5 - xy[:, 0, None, None]) ** 2 + (row + 0.5 - xy[:, 1, None, None]) ** 2 <= (width_px / 2) ** 2
    sample, i, j = np.nonzero(near)
    r, c = row[sample, i, 0] - r0, col[sample, 0, j] - c0
    inside = (r >= 0) & (r < r1 - r0) & (c >= 0) & (c < c1 - c0)
    flat, sample = r[inside] * (c1 - c0) + c[inside], sample[inside]
    order = np.lexsort((sample, flat))
    flat, sample = flat[order], sample[order]
    first = np.r_[True, (np.diff(flat) != 0) | (np.diff(sample) > width_px / RETURN_STEP_PX)]
    passes = np.bincount(flat[first], minlength=(r1 - r0) * (c1 - c0)).reshape(r1 - r0, c1 - c0)
    return np.maximum(passes, 1)


def add_stroke(coverage: np.ndarray, points_px, width_px: float, opacity=1.0, row_px=None, dose=None,
               dwell=(0.0, 0.0)) -> None:
    """Accumulate a stroke into an ink's coverage: repeated passes darken (1 - (1-c)(1-s)), and so do the pokes of
    a slow or dwelling pen (`dose`, `dwell`: stroke_coverage)."""
    hit = stroke_coverage(coverage.shape, points_px, width_px, opacity, row_px, dose, dwell)
    if hit:
        (r0, r1, c0, c1), patch = hit
        view = coverage[r0:r1, c0:c1]
        view[...] = 1.0 - (1.0 - view) * (1.0 - patch)


def composite(substrate_srgb, layers) -> np.ndarray:
    """Linear-light image of inks laid in order over a substrate.

    `layers` is [(InkLook, coverage)]. A translucent ink filters what is under it by its own colour
    against white (tattoo and ballpoint inks); an opaque one (white, silver) covers it. Returns sRGB
    linear values; callers convert for display.
    """
    shape = next((c.shape for _, c in layers), (1, 1))
    out = np.broadcast_to(srgb_to_linear(np.asarray(substrate_srgb, float)), (*shape, 3)).copy()
    for lk, cov in layers:
        ink = srgb_to_linear(np.asarray(lk.srgb, float))
        c = cov[..., None]
        out = out * (1 - c) + c * (ink if lk.opaque else out * ink)
    return out
