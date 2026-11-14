"""A page that may bend: its base plane, a smooth height field over it, and the print's chart on it.

The flat page every plan used (`base_from_page`, z out of the paper) stays the base. Over it a height h(x, y) along its
normal gives the surface; h = 0 is the plane. The field is a cubic spline on a regular grid of knots, fitted to
measured points (the wrist gauge's holds and the depth around them) with a penalty on its bending. How stiff it is
comes from the points themselves: the stiffest surface that still predicts measurements it was not shown. So it
follows whatever the skin does where it was measured (a crest, a trough, a saddle), stays smooth through their noise,
and runs on as a plane where nothing was measured. Nothing in it names a shape.

A print transferred onto gently curved skin wraps it without stretching, so the print's own millimetres (u, v), the
coordinates every design is placed in, are arc lengths on the surface, not its shadow on the base plane: the chart
maps one to the other by unfolding the surface from the page's centre, triangle by triangle from its edge lengths
alone (Inkmap's intrinsic unfold, web/inkmap/src/core/unfold.ts, on a regular grid). On the plane the chart is the
identity, so a flat page plans exactly as before.

Stdlib and numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

CHART_STEP_M = 0.001      # the chart's grid spacing
CHART_MARGIN_M = 0.005    # the chart covers the fitted region and this much around it
FOOT_ITERATIONS = 30      # PageSurface.foot's fixed point, until it moves less than FOOT_TOL_M: three left a point 10 mm
FOOT_TOL_M = 1e-7         # over a 35 mm forearm's flank, 10 mm off the crest, 0.2 mm from its foot
KNOT_M = 0.004            # the field's knots are at most this far apart: a bend of 10 mm radius is met within 0.01 mm
STIFFNESS = tuple(10.0 ** e for e in range(1, -7, -1))   # the bending penalties a fit chooses among, stiffest first
STIFF_ENOUGH = 1.1        # ... the stiffest whose held-out error is within this of the best one's
FOLDS = 5                 # the parts the points are held out in, one at a time, to choose it
EXTENT = (-0.05, -0.08, 0.05, 0.08)


def _blank(extent) -> HeightField:
    """The flat field over an extent (each side at least a knot wide), its knots at most KNOT_M apart."""
    lo, hi = np.asarray(extent[:2], float), np.asarray(extent[2:], float)
    lo, hi = np.minimum(lo, (lo + hi - KNOT_M) / 2), np.maximum(hi, (lo + hi + KNOT_M) / 2)
    nx, ny = (max(1, int(np.ceil(span / KNOT_M - 1e-9))) for span in hi - lo)
    return HeightField((*map(float, lo), *map(float, hi)), np.zeros((ny + 3, nx + 3)))


def _spline(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The four uniform cubic B-splines over a cell at t in [0, 1]: (N, 4) values and (N, 4) d/dt."""
    s = 1.0 - t
    value = np.stack([s ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3], axis=1) / 6.0
    slope = np.stack([-s ** 2, 3 * t ** 2 - 4 * t, -3 * t ** 2 + 2 * t + 1, t ** 2], axis=1) / 2.0
    return value, slope


@dataclass
class HeightField:
    """h(x, y) in page metres: a cubic spline whose (ny + 3, nx + 3) coefficients stand on a regular grid of knots
    over `extent`. Past the extent it runs on as the plane it left along."""

    extent: tuple[float, float, float, float] = EXTENT   # xmin, ymin, xmax, ymax it was fitted over
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    stiffness: float = 0.0   # the bending penalty its fit chose (STIFFNESS); 0: not fitted

    @property
    def flat(self) -> bool:
        return not np.any(self.coeffs)

    def _cells(self, xy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per point: the first coefficient (iy, ix) of the sixteen that reach it, and their (N, 4, 4) weights for
        h, dh/dx and dh/dy. A point past the extent takes the plane the field leaves its edge along."""
        xy = np.atleast_2d(np.asarray(xy, float))
        lo, hi = np.asarray(self.extent[:2]), np.asarray(self.extent[2:])
        n = np.array(self.coeffs.shape[::-1]) - 3
        step = (hi - lo) / n
        inside = np.clip(xy, lo, hi)
        at = (inside - lo) / step
        cell = np.minimum(at.astype(int), n - 1)
        (bx, dx), (by, dy) = _spline(at[:, 0] - cell[:, 0]), _spline(at[:, 1] - cell[:, 1])
        along_x = by[:, :, None] * dx[:, None, :] / step[0]
        along_y = dy[:, :, None] * bx[:, None, :] / step[1]
        past = (xy - inside)[:, :, None, None]
        return cell[:, 1], cell[:, 0], by[:, :, None] * bx[:, None, :] + past[:, 0] * along_x + past[:, 1] * along_y, \
            along_x, along_y

    def _patch(self, iy, ix) -> np.ndarray:
        four = np.arange(4)
        return self.coeffs[iy[:, None, None] + four[None, :, None], ix[:, None, None] + four[None, None, :]]

    def __call__(self, xy) -> np.ndarray:
        iy, ix, weights, _, _ = self._cells(xy)
        return np.einsum("nab,nab->n", self._patch(iy, ix), weights)

    def gradient(self, xy) -> np.ndarray:
        """(N, 2) dh/dx, dh/dy."""
        iy, ix, _, along_x, along_y = self._cells(xy)
        patch = self._patch(iy, ix)
        return np.stack([np.einsum("nab,nab->n", patch, along_x), np.einsum("nab,nab->n", patch, along_y)], axis=1)

    def rows(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """What gives h at points from the raveled coefficients: per point the sixteen coefficients that reach it,
        (N, 16) their indices and (N, 16) their weights."""
        iy, ix, weights, _, _ = self._cells(xy)
        four, width = np.arange(4), self.coeffs.shape[1]
        columns = (iy[:, None, None] + four[None, :, None]) * width + ix[:, None, None] + four[None, None, :]
        return columns.reshape(len(iy), 16), weights.reshape(len(iy), 16)

    def bending(self) -> np.ndarray:
        """The quadratic form of the field's bending over its extent in its raveled coefficients (the thin plate's
        h_xx^2 + 2 h_xy^2 + h_yy^2, by the coefficients' differences): zero on every plane."""
        rows, columns = self.coeffs.shape
        step = (np.asarray(self.extent[2:]) - np.asarray(self.extent[:2])) / (np.array([columns, rows]) - 3)

        def difference(n: int, order: int) -> np.ndarray:
            return np.diff(np.eye(n), order, axis=0)
        xx = np.kron(np.eye(rows), difference(columns, 2)) / step[0] ** 2
        yy = np.kron(difference(rows, 2), np.eye(columns)) / step[1] ** 2
        xy = np.kron(difference(rows, 1), difference(columns, 1)) / (step[0] * step[1])
        return xx.T @ xx + 2.0 * xy.T @ xy + yy.T @ yy

    def as_dict(self) -> dict:
        return {"extent": [float(v) for v in self.extent], "coeffs": self.coeffs.tolist(), "stiffness": self.stiffness}

    @classmethod
    def from_dict(cls, data: dict) -> HeightField:
        """A field as_dict wrote. A page recorded before the spline holds a polynomial ("degree"): it is read as
        the spline through that polynomial over its extent."""
        extent = tuple(float(v) for v in data["extent"])
        if "degree" not in data:
            return cls(extent, np.asarray(data["coeffs"], float), float(data.get("stiffness", 0.0)))
        xy = np.stack(np.meshgrid(np.arange(extent[0], extent[2] + CHART_STEP_M, CHART_STEP_M),
                                  np.arange(extent[1], extent[3] + CHART_STEP_M, CHART_STEP_M)), -1).reshape(-1, 2)
        x, y = (xy / float(data["scale"])).T
        terms = [(i, d - i) for d in range(int(data["degree"]) + 1) for i in range(d, -1, -1)]
        h = sum(c * x ** i * y ** j for c, (i, j) in zip(data["coeffs"], terms, strict=True))
        return fit_height(np.c_[xy, h], extent=extent, stiffness=STIFFNESS[-1], min_points=1)


def fit_fields(z, weights, fields, extra=None, *, groups=None, stiffness: float | None = None,
               reject_m: float = 0.0008, rounds: int = 4) -> tuple[list[HeightField], np.ndarray]:
    """Height fields through measured heights `z` (N,), robustly, with any unknowns that are linear in them.

    `fields`: [(rows, xy, extent)], one field per entry over the points `rows` (indices into z) at page (x, y).
    `extra` (N, E): columns of unknowns every point may share (a camera's turn: z = h(x, y) + extra @ e). Returns
    the fields and e. Each round refits to the points within max(reject_m, 3 MADs) of the last. `weights` (N,)
    favour points measured better.

    The bending penalty is `stiffness`, or chosen: the points are held out a part at a time (`groups` (N,), the
    hold each came from, so that what one hold saw is predicted from the others; else at random) and the stiffest
    penalty that predicts them within STIFF_ENOUGH of the best is kept. One penalty serves every field."""
    z, w = np.asarray(z, float), np.asarray(weights, float)
    extra = np.zeros((len(z), 0)) if extra is None else np.asarray(extra, float)
    blank = [_blank(extent) for _, _, extent in fields]
    start = np.cumsum([0] + [f.coeffs.size for f in blank])
    size = start[-1] + extra.shape[1]
    # a point's height is sixteen of its field's coefficients (column, weight), and the shared unknowns
    column, weight = np.zeros((len(z), 16), int), np.zeros((len(z), 16))
    for k, (rows, xy, _) in enumerate(fields):
        column[rows], weight[rows] = blank[k].rows(xy)
        column[rows] += start[k]
    bend = np.zeros((size, size))
    for k, f in enumerate(blank):
        bend[start[k]:start[k + 1], start[k]:start[k + 1]] = f.bending()
    fold = np.random.default_rng(0).integers(0, FOLDS, len(z)) if groups is None else np.asarray(groups) % FOLDS

    def normal(rows):
        """The weighted normal equations of the points `rows`: (gram, right side, z's own square)."""
        c, v, e, zw = column[rows], weight[rows], extra[rows], z[rows] * w[rows]
        vw = v * w[rows, None]
        gram = np.zeros((size, size))
        pairs = (c[:, :, None] * start[-1] + c[:, None, :]).ravel()
        gram[:start[-1], :start[-1]] = np.bincount(
            pairs, (vw[:, :, None] * v[:, None, :]).ravel(), start[-1] ** 2).reshape(start[-1], start[-1])
        rhs = np.r_[np.bincount(c.ravel(), (v * zw[:, None]).ravel(), start[-1]), e.T @ zw]
        for j in range(extra.shape[1]):
            gram[:start[-1], start[-1] + j] = gram[start[-1] + j, :start[-1]] = np.bincount(
                c.ravel(), (vw * e[:, j, None]).ravel(), start[-1])
        gram[start[-1]:, start[-1]:] = e.T @ (e * w[rows, None])
        return gram, rhs, float(z[rows] @ zw)

    def solve(gram, rhs, rho):   # the ridge only names an answer where neither a point nor the bending reaches
        size = np.trace(gram)
        penalty = rho * size / max(np.trace(bend), 1e-300) * bend
        return np.linalg.solve(gram + penalty + 1e-12 * size / len(rhs) * np.eye(len(rhs)), rhs)

    def choose(keep) -> float:
        parts = [normal(keep & (fold == k)) for k in range(FOLDS)]
        gram, rhs = sum(p[0] for p in parts), sum(p[1] for p in parts)
        missed = []
        for rho in STIFFNESS:
            fits = [(solve(gram - g, rhs - r, rho), g, r, zz) for g, r, zz in parts]
            missed.append(sum(zz - 2.0 * c @ r + c @ g @ c for c, g, r, zz in fits))
        return next(rho for rho, miss in zip(STIFFNESS, missed, strict=True) if miss <= STIFF_ENOUGH ** 2 * min(missed))

    keep = np.ones(len(z), bool)
    coeffs = np.zeros(size)
    for _ in range(rounds):
        if stiffness is None:
            stiffness = choose(keep)
        gram, rhs, _ = normal(keep)
        coeffs = solve(gram, rhs, stiffness)
        r = z - np.einsum("nk,nk->n", weight, coeffs[column]) - extra @ coeffs[start[-1]:]
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(reject_m, 3.0 * 1.4826 * mad)
    out = [HeightField(f.extent, coeffs[start[k]:start[k + 1]].reshape(f.coeffs.shape), float(stiffness))
           for k, f in enumerate(blank)]
    return out, coeffs[start[-1]:]


def fit_height(points, weights=None, *, extent=None, groups=None, stiffness: float | None = None,
               min_points: int = 30, reject_m: float = 0.0008, rounds: int = 4) -> HeightField:
    """The height field through page-frame points (N, 3: x, y, height over the base plane): fit_fields with one
    field and nothing else unknown. Raises ValueError with too few points."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < min_points:
        raise ValueError(f"{len(pts)} surface points: too few to fit a height field (at least {min_points})")
    if extent is None:
        extent = (*pts[:, :2].min(axis=0), *pts[:, :2].max(axis=0))
    w = np.ones(len(pts)) if weights is None else weights
    return fit_fields(pts[:, 2], w, [(np.arange(len(pts)), pts[:, :2], extent)], groups=groups, stiffness=stiffness,
                      reject_m=reject_m, rounds=rounds)[0][0]


@dataclass
class Chart:
    """The print's (u, v) on the base plane's (x, y): U, V sampled on a regular grid over the height field's extent
    (the unfolded position of each grid vertex). Identity when the field is flat."""

    x0: float
    y0: float
    step: float
    u: np.ndarray   # (ny, nx)
    v: np.ndarray

    def to_chart(self, xy) -> np.ndarray:
        """Base-plane (x, y) -> print (u, v), bilinear on the grid; outside it, the identity plus the edge's offset."""
        xy = np.atleast_2d(np.asarray(xy, float))
        fx = np.clip((xy[:, 0] - self.x0) / self.step, 0.0, self.u.shape[1] - 1.000001)
        fy = np.clip((xy[:, 1] - self.y0) / self.step, 0.0, self.u.shape[0] - 1.000001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy

        def bilinear(grid):
            return ((1 - ax) * (1 - ay) * grid[iy, ix] + ax * (1 - ay) * grid[iy, ix + 1]
                    + (1 - ax) * ay * grid[iy + 1, ix] + ax * ay * grid[iy + 1, ix + 1])
        cx = self.x0 + fx * self.step
        cy = self.y0 + fy * self.step
        return np.stack([bilinear(self.u) + xy[:, 0] - cx, bilinear(self.v) + xy[:, 1] - cy], axis=1)

    def to_plane(self, uv, iterations: int = 8) -> np.ndarray:
        """Print (u, v) -> base-plane (x, y): to_chart inverted by fixed-point iteration (the chart is near the
        identity: a slope of 0.3 shortens it ~5%)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        xy = uv.copy()
        for _ in range(iterations):
            xy = xy + (uv - self.to_chart(xy))
        return xy


def unfold(height: HeightField, step: float = CHART_STEP_M, margin: float = CHART_MARGIN_M) -> Chart:
    """The chart by unfolding the surface over a grid from the vertex nearest the page's origin: the seed's two
    triangles laid flat with their first edge along +x, then breadth first every triangle across a placed edge, its
    third vertex placed from the edge's two by its 3D edge lengths, on the far side from the triangle already there.
    A vertex keeps the first place it got (nearest the seed). Exact on a developable surface; on a doubly curved one
    the walk's paths disagree by the curvature they enclose (millimetres over a palm), which is the print's own
    stretch there."""
    xmin, ymin, xmax, ymax = height.extent
    xs = np.arange(min(xmin, 0.0) - margin, max(xmax, 0.0) + margin + step / 2, step)
    ys = np.arange(min(ymin, 0.0) - margin, max(ymax, 0.0) + margin + step / 2, step)
    gx, gy = np.meshgrid(xs, ys)
    if height.flat:
        return Chart(float(xs[0]), float(ys[0]), step, gx.copy(), gy.copy())
    ny, nx = gx.shape
    z = height(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(ny, nx)
    p3 = np.stack([gx, gy, z], axis=-1)
    placed = np.full((ny, nx, 2), np.nan)
    si, sj = int(np.argmin(np.abs(ys))), int(np.argmin(np.abs(xs)))
    si, sj = min(si, ny - 2), min(sj, nx - 2)

    def dist(a, b):
        return float(np.linalg.norm(p3[a] - p3[b]))

    a, b, c = _cell(si, sj)[0]
    placed[a] = (0.0, 0.0)
    ab = dist(a, b)
    placed[b] = (ab, 0.0)
    placed[c] = _apex(placed[a], placed[b], dist(a, c), dist(b, c), side=+1.0)
    queue = deque([(si, sj, 0), (si, sj, 1)])
    seen = {(si, sj, 0)}
    while queue:
        i, j, k = queue.popleft()
        tri = _cell(i, j)[k]
        for e0, e1, other in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
            for ni, nj, nk in _across(e0, e1, ny, nx):
                if (ni, nj, nk) in seen:
                    continue
                seen.add((ni, nj, nk))
                new = [v for v in _cell(ni, nj)[nk] if v not in (e0, e1)][0]
                if np.isnan(placed[new][0]):
                    side = -np.sign(_cross(placed[e1] - placed[e0], placed[other] - placed[e0])) or 1.0
                    placed[new] = _apex(placed[e0], placed[e1], dist(e0, new), dist(e1, new), side=side)
                queue.append((ni, nj, nk))
    # the seed vertex sits at the grid point nearest the origin: shift so that (0, 0) maps to (0, 0)
    offset = np.array([xs[sj], ys[si]])
    u = placed[..., 0] + offset[0]
    v = placed[..., 1] + offset[1]
    chart = Chart(float(xs[0]), float(ys[0]), step, u, v)
    zero = chart.to_chart([[0.0, 0.0]])[0]
    chart.u, chart.v = u - zero[0], v - zero[1]
    return chart


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _apex(p0, p1, d0: float, d1: float, side: float) -> np.ndarray:
    """The point d0 from p0 and d1 from p1, on `side` (+1 left of p0 -> p1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    base = p1 - p0
    length = float(np.linalg.norm(base))
    along = (d0 * d0 - d1 * d1 + length * length) / (2.0 * length)
    across = np.sqrt(max(d0 * d0 - along * along, 0.0))
    unit = base / length
    return p0 + along * unit + side * across * np.array([-unit[1], unit[0]])


def _cell(i, j):
    """The grid cell (i, j)'s two triangles, as vertex index triples."""
    return [((i, j), (i, j + 1), (i + 1, j)), ((i + 1, j + 1), (i + 1, j), (i, j + 1))]


def _across(e0, e1, ny, nx):
    """The triangles (cell i, j, which) that share the edge e0-e1."""
    out = []
    for i in range(min(e0[0], e1[0]) - 1, max(e0[0], e1[0]) + 1):
        for j in range(min(e0[1], e1[1]) - 1, max(e0[1], e1[1]) + 1):
            if 0 <= i < ny - 1 and 0 <= j < nx - 1:
                out += [(i, j, k) for k, tri in enumerate(_cell(i, j)) if e0 in tri and e1 in tri]
    return out


class PageSurface:
    """The page a plan draws on: `base_from_page` (the page pose, z out of the paper) and a height field over it,
    with the print's chart. Every point is given in print coordinates (u, v), as stroke points are."""

    def __init__(self, base_from_page, height: HeightField | None = None, chart: Chart | None = None):
        self.base = np.asarray(base_from_page, float)
        self.height = height or HeightField()
        self.chart = chart if chart is not None else unfold(self.height)

    @property
    def flat(self) -> bool:
        return self.height.flat

    def moved(self, base_from_page) -> PageSurface:
        """The same surface carried rigidly with its page (a followed slide, a resumed page)."""
        return PageSurface(base_from_page, self.height, self.chart)

    def normals_page(self, xy) -> np.ndarray:
        """(N, 3) unit normals in the page frame at base-plane points."""
        g = self.height.gradient(xy)
        n = np.column_stack([-g[:, 0], -g[:, 1], np.ones(len(g))])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def lift(self, uv, offset_m=0.0) -> tuple[np.ndarray, np.ndarray]:
        """((N, 3) base points `offset_m` (one, or one per point) along the surface's normal over print points uv,
        (N, 3) base unit normals)."""
        xy = self.chart.to_plane(uv)
        h = self.height(xy)
        n = self.normals_page(xy)
        local = np.column_stack([xy, h]) + np.asarray(offset_m, float).reshape(-1, 1) * n
        return (local @ self.base[:3, :3].T + self.base[:3, 3]), n @ self.base[:3, :3].T

    def height_over(self, point_base) -> tuple[float, np.ndarray]:
        """(a base point's height over the surface along the normal under it, that unit normal in the base). A point
        over the page's plane past the field's extent is measured from that plane: nothing was measured
        there, and at the arm's rest it put the pen under the standoff of a skin on a holder (2026-10-08)."""
        local = np.linalg.inv(self.base) @ np.r_[np.asarray(point_base, float), 1.0]
        x0, y0, x1, y1 = self.height.extent
        if not self.flat and not (x0 <= local[0] <= x1 and y0 <= local[1] <= y1):
            return float(local[2]), self.base[:3, 2].copy()
        _, gap, normal = self.foot(point_base)
        return gap, normal

    def heights_over(self, points_base) -> np.ndarray:
        """height_over's heights for base points (N, 3), their feet found together."""
        local = self._local(points_base)
        x0, y0, x1, y1 = self.height.extent
        on = self.flat | ((x0 <= local[:, 0]) & (local[:, 0] <= x1) & (y0 <= local[:, 1]) & (local[:, 1] <= y1))
        gaps = local[:, 2].copy()
        gaps[on] = self._feet(local[on])[1]
        return gaps

    def foot(self, point_base) -> tuple[np.ndarray, float, np.ndarray]:
        """(the print point (u, v) of the surface under a base point, along the surface's normal there; the point's
        height over it; that unit normal in the base). Far over a curved skin the fixed point below diverges: a tip
        42 mm over a forearm's field ran off its extent, and the height it brought back held a landing as one at the
        skin (2026-10-23). An iterate that runs away (non-finite, or ending off the fitted extent from a point
        over it) takes the field's nearest point instead (_nearest); every other foot is the fixed point's."""
        xy, gap, n = self._feet(self._local(point_base))
        return self.chart.to_chart(xy)[0], float(gap[0]), n[0] @ self.base[:3, :3].T

    def _local(self, points_base) -> np.ndarray:   # base points, (N, 3) or one, in the page frame
        page_from_base = np.linalg.inv(self.base)
        return np.atleast_2d(np.asarray(points_base, float)) @ page_from_base[:3, :3].T + page_from_base[:3, 3]

    def _feet(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """foot's fixed point for page-frame points (N, 3), each iterated until its own end: ((N, 2) the base-plane
        feet, (N,) the points' heights over them, (N, 3) the page-frame unit normals there)."""
        x0, y0, x1, y1 = self.height.extent
        lo, hi = np.array([x0, y0]) - CHART_MARGIN_M, np.array([x1, y1]) + CHART_MARGIN_M

        def within(xy) -> np.ndarray:
            return np.all((lo <= xy) & (xy <= hi), axis=1)

        over, xy, going = within(local[:, :2]), local[:, :2].copy(), np.ones(len(local), bool)
        with np.errstate(over="ignore", invalid="ignore"):   # a diverging iterate is caught below, not warned of
            for _ in range(FOOT_ITERATIONS):   # the foot of the normal through the point, from its shadow on the plane
                if not going.any():
                    break
                at, last = local[going], xy[going]
                n = self.normals_page(last)
                step = at[:, :2] - (at[:, 2] - self.height(last))[:, None] * n[:, :2] / np.maximum(n[:, 2], 1e-6)[:, None]
                moved = np.linalg.norm(step - last, axis=1)
                xy[going] = step
                going[going] = ~((moved < FOOT_TOL_M) | ~np.isfinite(moved) | (over[going] & ~within(step)))
        # past its extent the field is the plane it left along, where a runaway iterate can come to rest: a foot
        # there, from a point over the field, is not one
        fitted = np.all((lo + CHART_MARGIN_M <= xy) & (xy <= hi - CHART_MARGIN_M), axis=1)
        for i in np.flatnonzero(~np.isfinite(xy).all(axis=1) | (over & ~fitted)):
            xy[i] = self._nearest(local[i])
        n = self.normals_page(xy)
        return xy, np.einsum("ij,ij->i", local - np.c_[xy, self.height(xy)], n), n

    def _nearest(self, local) -> np.ndarray:
        """The base-plane point (x, y) of the field nearest a page-frame point, on a CHART_STEP_M grid over its
        fitted extent (a foot 0.7 mm off moves a 40 mm height by under 0.01 mm)."""
        x0, y0, x1, y1 = self.height.extent
        xs = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / CHART_STEP_M)) + 1))
        ys = np.linspace(y0, y1, max(2, int(np.ceil((y1 - y0) / CHART_STEP_M)) + 1))
        grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        off = np.c_[grid, self.height(grid)] - np.asarray(local, float)[:3]
        return grid[int(np.argmin(np.einsum("ij,ij->i", off, off)))]

    def as_dict(self) -> dict:
        return {"height": self.height.as_dict()}

    @classmethod
    def from_dict(cls, base_from_page, data: dict) -> PageSurface:
        return cls(base_from_page, HeightField.from_dict(data["height"]))
