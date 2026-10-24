"""A page that may bend: its base plane, a smooth height field over it, and the print's chart on it.

The flat page every plan used (`base_from_page`, z out of the paper) stays the base. Over it a height h(x, y) along its
normal, a low-order polynomial fitted to measured points (the wrist gauge's holds and the depth around them), gives
the surface; h = 0 is the plane. A print transferred onto gently curved skin wraps it without stretching, so the print's
own millimetres (u, v), the coordinates every design is placed in, are arc lengths on the surface, not its shadow on
the base plane: the chart maps one to the other by unfolding the surface from the page's centre, triangle by triangle
from its edge lengths alone (Inkmap's intrinsic unfold, web/inkmap/src/core/unfold.ts, on a regular grid). On the plane
the chart is the identity, so a flat page plans exactly as before.

Stdlib and numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

CHART_STEP_M = 0.001      # the chart's grid spacing
CHART_MARGIN_M = 0.005    # the chart covers the fitted region and this much around it
FOOT_ITERATIONS = 30      # PageSurface.foot's fixed point, until it moves less than FOOT_TOL_M: three left a point 10 mm
FOOT_TOL_M = 1e-7         # over a 35 mm forearm's flank, 10 mm off the crest, 0.2 mm from its foot


def _terms(degree: int) -> list[tuple[int, int]]:
    """The polynomial's monomials x^i y^j, i + j <= degree, constant and linear first."""
    return [(i, d - i) for d in range(degree + 1) for i in range(d, -1, -1)]


def _design(xy: np.ndarray, terms, scale: float) -> np.ndarray:
    x, y = xy[:, 0] / scale, xy[:, 1] / scale
    return np.stack([x ** i * y ** j for i, j in terms], axis=1)


@dataclass
class HeightField:
    """h(x, y) = sum c_k (x/s)^i (y/s)^j in page metres (s = `scale` keeps the fit conditioned)."""

    degree: int = 0
    coeffs: tuple[float, ...] = (0.0,)
    scale: float = 0.05
    extent: tuple[float, float, float, float] = (-0.05, -0.08, 0.05, 0.08)   # xmin, ymin, xmax, ymax it was fitted over

    @property
    def flat(self) -> bool:
        return not any(abs(c) > 0.0 for c in self.coeffs)

    def __call__(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        return _design(xy, _terms(self.degree), self.scale) @ np.asarray(self.coeffs, float)

    def gradient(self, xy) -> np.ndarray:
        """(N, 2) dh/dx, dh/dy."""
        xy = np.atleast_2d(np.asarray(xy, float))
        x, y, s = xy[:, 0] / self.scale, xy[:, 1] / self.scale, self.scale
        gx = np.zeros(len(xy))
        gy = np.zeros(len(xy))
        for c, (i, j) in zip(self.coeffs, _terms(self.degree), strict=True):
            if i:
                gx += c * i * x ** (i - 1) * y ** j / s
            if j:
                gy += c * j * x ** i * y ** (j - 1) / s
        return np.stack([gx, gy], axis=1)

    def as_dict(self) -> dict:
        return {"degree": self.degree, "coeffs": [float(c) for c in self.coeffs], "scale": self.scale,
                "extent": [float(v) for v in self.extent]}

    @classmethod
    def from_dict(cls, data: dict) -> HeightField:
        return cls(int(data["degree"]), tuple(float(c) for c in data["coeffs"]), float(data["scale"]),
                   tuple(float(v) for v in data["extent"]))


def fit_height(points, weights=None, *, degree: int = 3, extent=None, min_points: int = 30,
               reject_m: float = 0.0008, rounds: int = 4) -> HeightField:
    """The height field through page-frame points (N, 3: x, y, height over the base plane), robustly: refitted to the
    points within max(reject_m, 3 MADs) of it. `weights` (N,) favour points measured better (the gauge's own point
    under the tip over the depth around it). The degree drops to what the points' spread supports. Raises ValueError
    with too few points."""
    pts = np.asarray(points, float).reshape(-1, 3)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float)
    if len(pts) < min_points:
        raise ValueError(f"{len(pts)} surface points: too few to fit a height field (at least {min_points})")
    if extent is None:
        extent = (*pts[:, :2].min(axis=0), *pts[:, :2].max(axis=0))
    span = max(extent[2] - extent[0], extent[3] - extent[1], 1e-3)
    scale = span / 2.0
    degree = int(min(degree, 1 if span < 0.008 else degree))
    terms = _terms(degree)
    a = _design(pts[:, :2], terms, scale)
    keep = np.ones(len(pts), bool)
    coeffs = np.zeros(len(terms))
    for _ in range(rounds):
        sw = np.sqrt(w[keep])
        coeffs = np.linalg.lstsq(a[keep] * sw[:, None], pts[keep, 2] * sw, rcond=None)[0]
        r = pts[:, 2] - a @ coeffs
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(reject_m, 3.0 * 1.4826 * mad)
    return HeightField(degree, tuple(float(c) for c in coeffs), float(scale), tuple(float(v) for v in extent))


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

    def lift(self, uv, offset_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """((N, 3) base points `offset_m` along the surface's normal over print points uv, (N, 3) base unit normals)."""
        xy = self.chart.to_plane(uv)
        h = self.height(xy)
        n = self.normals_page(xy)
        local = np.column_stack([xy, h]) + float(offset_m) * n
        return (local @ self.base[:3, :3].T + self.base[:3, 3]), n @ self.base[:3, :3].T

    def height_over(self, point_base) -> tuple[float, np.ndarray]:
        """(a base point's height over the surface along the normal under it, that unit normal in the base). A point
        over the page's plane past the field's extent is measured from that plane: the polynomial is an extrapolation
        there, and at the arm's rest it put the pen under the standoff of a skin on a holder (2026-10-08)."""
        local = np.linalg.inv(self.base) @ np.r_[np.asarray(point_base, float), 1.0]
        x0, y0, x1, y1 = self.height.extent
        if not self.flat and not (x0 <= local[0] <= x1 and y0 <= local[1] <= y1):
            return float(local[2]), self.base[:3, 2].copy()
        _, gap, normal = self.foot(point_base)
        return gap, normal

    def foot(self, point_base) -> tuple[np.ndarray, float, np.ndarray]:
        """(the print point (u, v) of the surface under a base point, along the surface's normal there; the point's
        height over it; that unit normal in the base). Far over a curved skin the fixed point below diverges: a tip
        42 mm over a forearm's degree-4 field ran off its extent into the polynomial's overflow, and the NaN height
        held a landing as one at the skin (2026-10-23). An iterate that runs away (non-finite, or off the fitted
        field, its extent and the chart's margin, from a point over it) takes the field's nearest point instead
        (_nearest); every other foot is the fixed point's, as before."""
        local = np.linalg.inv(self.base) @ np.r_[np.asarray(point_base, float), 1.0]
        x0, y0, x1, y1 = self.height.extent
        lo, hi = np.array([x0, y0]) - CHART_MARGIN_M, np.array([x1, y1]) + CHART_MARGIN_M

        def within(xy) -> bool:
            return bool(np.all((lo <= xy) & (xy <= hi)))

        over, xy, moved = within(local[:2]), local[:2].copy(), np.inf
        with np.errstate(over="ignore", invalid="ignore"):   # a diverging iterate is caught below, not warned of
            for _ in range(FOOT_ITERATIONS):   # the foot of the normal through the point, from its shadow on the plane
                n = self.normals_page(xy)[0]
                h = float(self.height(xy)[0])
                xy, last = local[:2] - (local[2] - h) * n[:2] / max(n[2], 1e-6), xy
                moved = float(np.linalg.norm(xy - last))
                if moved < FOOT_TOL_M or not np.isfinite(moved) or (over and not within(xy)):
                    break
        if not np.all(np.isfinite(xy)) or (over and not within(xy)):
            xy = self._nearest(local[:3])
        n = self.normals_page(xy)[0]
        h = float(self.height(xy)[0])
        gap = float((local[:3] - np.r_[xy, h]) @ n)
        return self.chart.to_chart(xy)[0], gap, n @ self.base[:3, :3].T

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
