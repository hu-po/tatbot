"""Layer B of `tatbot draw`: the mapped surface, in numpy.

A numpy port of ``tatbot_sim.surface`` (``Chart``, ``PlaneChart``,
``CylinderChart``, ``_catmull_rom`` and ``DisplacedSurface``) so the drawing
stages can run in a NumPy-only environment without torch or ManiSkill, plus what the sim never
needed: fitting a chart to wrist-camera points, fusing those points into the
height grid, and reading/writing ``surface.npz`` (``docs/draw.md``).

The math is kept line-for-line with the torch version on purpose: the sim
loads the same ``.npz`` into ``DisplacedSurface`` and a parity test pins the
two within a micrometre. If you change a formula here, change it there.

Conventions (``docs/draw.md``, "surface.npz"): a chart is parameterised in
canvas metres ``(u, v)`` with ``u`` in ``[-w/2, w/2]`` and ``v`` in
``[-h/2, h/2]``; ``rot`` columns are ``(e_u, e_v, n)``; ``height`` is
``(rows, cols)`` metres along the chart normal, rows index ``v``, cols index
``u``, nodes on the border. Everything is in the ``root`` frame.

Numpy only: no scipy, no torch.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

SCHEMA = "tatbot.surface/1"

sim_parity_note = """Parity with ``tatbot_sim.surface.DisplacedSurface``.

``HeightFieldSurface.to_npz`` writes the ``tatbot.surface/1`` layout and
``tatbot_sim.surface_io.displaced_surface_from_npz`` reads it back into a
batch-1 ``DisplacedSurface`` on the same chart with the same ``height`` grid.
Both implement the same Catmull-Rom sampling (border-replicated 4x4 patch,
u first then v), the same frame assembly (chart point + h * n, tangents with
the ``h * dn/duv`` curvature term, cross-product normal re-oriented to the
chart's outward sense) and the same Gauss-Newton projection on the tangential
residual. ``python/tatbot_sim/tests/test_surface_io.py`` checks frame points
and normals agree within 1e-6 m / 1e-6 on a grid of uv. The sim's texel
resolution (``cols``/``rows``) is not stored in the file: it is a rendering
concern, chosen by the loader so texels stay isotropic.
"""


# --------------------------------------------------------------------------- charts


def _rows(x, n: int) -> np.ndarray:
    """(3,) -> (n, 3) writable copy."""
    return np.repeat(np.asarray(x, dtype=np.float64)[None, :], n, axis=0)


class Chart:
    """The smooth base a displacement is measured from (see the torch docstring).

    Unbatched: one chart, ``(N, 2)`` queries. Arc-length parameterised in canvas
    metres, so a millimetre of ``(u, v)`` is a millimetre along the base.
    """

    kind = "chart"

    def __init__(self, center, rot):
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)

    @property
    def normal(self) -> np.ndarray:
        return self.rot[:, 2]

    def frame(self, uv):
        """(N, 2) canvas metres -> point, d/du, d/dv, outward normal, each (N, 3)."""
        raise NotImplementedError

    def normal_derivatives(self, uv):
        """(N, 2) -> (dn/du, dn/dv), each (N, 3). Zero for a flat chart."""
        raise NotImplementedError

    def invert(self, points) -> np.ndarray:
        """World points (N, 3) -> (N, 2) canvas coordinates of the nearest base point."""
        raise NotImplementedError

    def offset(self, shift_m: float) -> "Chart":
        """The chart moved ``shift_m`` along its normal so it stays the surface's own shape.

        Anchoring a curved chart by shifting the HEIGHT breaks its isometry: a
        cylinder chart of radius r carrying a constant +3 mm displacement is
        really a cylinder of radius r+3, and canvas v (arc length on the chart)
        then overstates the surface arc by 3/r -- the live bottle capture drew
        a 57 mm spiral 3 % short. Offsetting the chart instead keeps the
        displacement near zero and the metric honest.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "chart_kind": self.kind,
            "center": self.center.tolist(),
            "rot": self.rot.tolist(),
            "radius_m": float(getattr(self, "radius", math.nan)),
        }

    @staticmethod
    def from_dict(d: dict) -> Chart:
        return chart_from_dict(d)


class PlaneChart(Chart):
    """A flat base: the pad, and the case a curved surface must reduce to."""

    kind = "plane"

    def offset(self, shift_m: float) -> "PlaneChart":
        return PlaneChart(self.center + float(shift_m) * self.rot[:, 2], self.rot.copy())

    def frame(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        ex, ey, n = self.rot[:, 0], self.rot[:, 1], self.rot[:, 2]
        point = self.center + uv[:, 0:1] * ex + uv[:, 1:2] * ey
        m = uv.shape[0]
        return point, _rows(ex, m), _rows(ey, m), _rows(n, m)

    def normal_derivatives(self, uv):
        m = np.asarray(uv).reshape(-1, 2).shape[0]
        return np.zeros((m, 3)), np.zeros((m, 3))

    def invert(self, points) -> np.ndarray:
        d = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.center
        return np.stack([d @ self.rot[:, 0], d @ self.rot[:, 1]], axis=-1)


class CylinderChart(Chart):
    """A limb: a cylinder whose axis runs along canvas u, wrapped by canvas v.

    ``center`` is the surface point at ``(0, 0)`` (the crest), ``rot`` columns
    are the axis, the crest tangent and the outward normal there, and ``v`` is
    ARC LENGTH around the circumference (``theta = v / radius``). The axis
    passes through ``center - radius * n``.
    """

    kind = "cylinder"

    def __init__(self, center, rot, radius_m: float):
        super().__init__(center, rot)
        self.radius = float(radius_m)
        if not (self.radius > 0.0):
            raise ValueError(f"cylinder radius must be positive (convex); got {self.radius}")

    def offset(self, shift_m: float) -> "CylinderChart":
        # The offset surface of a cylinder is a cylinder: same axis, radius r + s,
        # and the crest point moves s along the crest normal.
        return CylinderChart(self.center + float(shift_m) * self.rot[:, 2], self.rot.copy(),
                             float(self.radius) + float(shift_m))

    @property
    def axis_point(self) -> np.ndarray:
        return self.center - self.radius * self.rot[:, 2]

    def _sc(self, uv):
        th = uv[:, 1:2] / self.radius
        return np.sin(th), np.cos(th)

    def frame(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        ex, ey, ez = self.rot[:, 0], self.rot[:, 1], self.rot[:, 2]
        r = self.radius
        s, c = self._sc(uv)
        point = self.center + uv[:, 0:1] * ex + (r * s) * ey + (r * c - r) * ez
        d_du = _rows(ex, uv.shape[0])
        d_dv = c * ey - s * ez
        normal = s * ey + c * ez
        return point, d_du, d_dv, normal

    def normal_derivatives(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        ey, ez = self.rot[:, 1], self.rot[:, 2]
        s, c = self._sc(uv)
        dn_dv = (c * ey - s * ez) / self.radius
        return np.zeros((uv.shape[0], 3)), dn_dv

    def invert(self, points) -> np.ndarray:
        d = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.center
        lx, ly, lz = d @ self.rot[:, 0], d @ self.rot[:, 1], d @ self.rot[:, 2]
        # the axis sits one radius below the crest, so measure the angle from there
        theta = np.arctan2(ly, lz + self.radius)
        return np.stack([lx, self.radius * theta], axis=-1)


def chart_from_dict(d: dict) -> Chart:
    kind = str(d["chart_kind"])
    if kind == "plane":
        return PlaneChart(d["center"], d["rot"])
    if kind == "cylinder":
        return CylinderChart(d["center"], d["rot"], float(d["radius_m"]))
    raise ValueError(f"unknown chart_kind {kind!r}")


# --------------------------------------------------------------------------- height field


def catmull_rom(p, t):
    """Catmull-Rom through 4 samples ``(..., 4)`` at ``t`` in [0, 1] between p1 and p2.

    Returns ``(value, d value / d t)``; ``t`` broadcasts against ``p[..., 0]``.
    Same coefficients as ``tatbot_sim.surface._catmull_rom``.
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a = 2.0 * p1
    b = p2 - p0
    c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
    d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    t2 = t * t
    val = 0.5 * (a + b * t + c * t2 + d * t2 * t)
    der = 0.5 * (b + 2.0 * c * t + 3.0 * d * t2)
    return val, der


class HeightFieldSurface:
    """A chart plus a height grid: the mapped surface the tool draws on.

    ``height`` is ``(rows, cols)`` metres along the chart normal on a regular
    grid spanning the canvas with nodes on the border; outside the extent the
    border replicates, exactly as in ``DisplacedSurface``. ``count`` and
    ``residual_m`` (same shape) are the fusion evidence per cell — ``count == 0``
    marks a hole the preflight must refuse — and ``anchor_*`` record the
    layer-C touch-off shift already folded into ``height``.
    """

    def __init__(self, chart: Chart, height, width_m: float, height_m: float):
        height = np.asarray(height, dtype=np.float64)
        if height.ndim != 2 or height.shape[0] < 2 or height.shape[1] < 2:
            raise ValueError(f"height must be (rows>=2, cols>=2); got {height.shape}")
        if not np.isfinite(height).all():
            raise ValueError("height contains non-finite values")
        self.chart = chart
        self.height = height
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        rows, cols = height.shape
        self.count = np.zeros((rows, cols), dtype=np.int32)
        self.residual_m = np.zeros((rows, cols), dtype=np.float64)
        self.anchor_uv = np.full(2, np.nan)
        self.anchor_point = np.full(3, np.nan)
        self.anchor_shift_m = 0.0
        self.unconverged = 0  # points the last projection could not place

    @property
    def rows(self) -> int:
        return self.height.shape[0]

    @property
    def cols(self) -> int:
        return self.height.shape[1]

    def grid_uv(self):
        """Node coordinates: ``(us (cols,), vs (rows,))``."""
        us = np.linspace(-self.width_m / 2, self.width_m / 2, self.cols)
        vs = np.linspace(-self.height_m / 2, self.height_m / 2, self.rows)
        return us, vs

    # -- sampling ---------------------------------------------------------

    def sample_height(self, uv):
        """(N, 2) canvas metres -> (h, dh/du, dh/dv), each (N,), metres."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        hr, hc = self.height.shape
        gu = (uv[:, 0] + self.width_m / 2) / self.width_m * (hc - 1)
        gv = (uv[:, 1] + self.height_m / 2) / self.height_m * (hr - 1)
        gu = np.clip(gu, 0.0, hc - 1.0)
        gv = np.clip(gv, 0.0, hr - 1.0)
        iu = np.clip(np.floor(gu).astype(np.int64), 0, hc - 1)
        iv = np.clip(np.floor(gv).astype(np.int64), 0, hr - 1)
        tu = (gu - iu)[:, None]
        tv = gv - iv

        off = np.array([-1, 0, 1, 2])
        ou = np.clip(iu[:, None] + off[None, :], 0, hc - 1)  # (N, 4)
        ov = np.clip(iv[:, None] + off[None, :], 0, hr - 1)  # (N, 4)
        patch = self.height[ov[:, :, None], ou[:, None, :]]  # (N, 4, 4): [v, u]

        # interpolate along u within each of the four rows, then across v
        rows_val, rows_du = catmull_rom(patch, tu)  # (N, 4) each
        val, dval_dtv = catmull_rom(rows_val, tv)
        dval_dtu, _ = catmull_rom(rows_du, tv)
        du_per_m = (hc - 1) / self.width_m
        dv_per_m = (hr - 1) / self.height_m
        return val, dval_dtu * du_per_m, dval_dtv * dv_per_m

    def frame(self, uv):
        """(N, 2) -> surface point, d/du, d/dv, unit outward normal, each (N, 3)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        c_pt, c_du, c_dv, c_n = self.chart.frame(uv)
        dn_du, dn_dv = self.chart.normal_derivatives(uv)
        h, h_du, h_dv = self.sample_height(uv)
        h = h[:, None]
        point = c_pt + h * c_n
        s_du = c_du + h_du[:, None] * c_n + h * dn_du
        s_dv = c_dv + h_dv[:, None] * c_n + h * dn_dv
        n = np.cross(s_du, s_dv)
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        # keep the chart's outward sense; the cross product's sign follows the
        # parameterisation and would flip the tool through the surface
        flip = (n * c_n).sum(-1, keepdims=True) < 0
        n = np.where(flip, -n, n)
        return point, s_du, s_dv, n

    def first_fundamental_form(self, uv) -> np.ndarray:
        _, s_du, s_dv, _ = self.frame(uv)
        e = (s_du * s_du).sum(-1)
        f = (s_du * s_dv).sum(-1)
        g = (s_dv * s_dv).sum(-1)
        return np.stack([np.stack([e, f], -1), np.stack([f, g], -1)], -2)

    # -- projection -------------------------------------------------------

    def project(self, points, iters: int = 8, tol_m: float = 1e-6):
        """World points (N, 3) -> (uv (N, 2), signed distance (N,) along the LOCAL normal).

        Batched Gauss-Newton on the tangential residual from the chart's exact
        inverse. A point the iteration could not place (tangential residual
        above ``tol_m`` after ``iters`` steps) gets ``inf`` distance rather than
        a plausible-looking wrong place; ``self.unconverged`` counts them.
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        uv = self.chart.invert(p)
        for _ in range(int(iters)):
            point, s_du, s_dv, _ = self.frame(uv)
            res = p - point
            e = (s_du * s_du).sum(-1)
            f = (s_du * s_dv).sum(-1)
            g = (s_dv * s_dv).sum(-1)
            ru = (res * s_du).sum(-1)
            rv = (res * s_dv).sum(-1)
            det = np.maximum(e * g - f * f, 1e-12)
            step = np.stack([(g * ru - f * rv) / det, (e * rv - f * ru) / det], axis=-1)
            uv = uv + step
            if np.abs(step).max() < 1e-13:
                break
        point, _, _, n = self.frame(uv)
        res = p - point
        dist = (res * n).sum(-1)
        tangential = np.linalg.norm(res - dist[:, None] * n, axis=-1)
        ok = tangential <= tol_m
        self.unconverged = int((~ok).sum())
        dist = np.where(ok, dist, np.inf)
        return uv, dist

    # -- shifting / anchoring --------------------------------------------

    def _clone(self, height) -> HeightFieldSurface:
        s = HeightFieldSurface(self.chart, height, self.width_m, self.height_m)
        s.count = self.count.copy()
        s.residual_m = self.residual_m.copy()
        s.anchor_uv = self.anchor_uv.copy()
        s.anchor_point = self.anchor_point.copy()
        s.anchor_shift_m = float(self.anchor_shift_m)
        return s

    def shifted(self, delta_m: float) -> HeightFieldSurface:
        """The same surface translated ``delta_m`` along the chart normal."""
        s = self._clone(self.height + float(delta_m))
        s.anchor_shift_m = float(self.anchor_shift_m) + float(delta_m)
        return s

    def anchor_to(self, point, iters: int = 6):
        """Shift the height so the surface passes through ``point`` along its normal.

        Returns ``(surface, shift_m, uv)``. The shift is along the CHART normal
        while the projected distance is along the LOCAL normal, so on a slope
        one shift leaves a second-order residual; a few re-projections drive it
        to zero. Raises if the point cannot be projected.
        """
        p = np.asarray(point, dtype=np.float64).reshape(3)
        s = self
        total = 0.0
        uv = None
        for _ in range(int(iters)):
            uv, dist = s.project(p[None])
            d = float(dist[0])
            if not np.isfinite(d):
                raise ValueError("anchor point does not project onto the surface")
            total += d
            s = s.shifted(d)
            if abs(d) < 1e-12:
                break
        s.anchor_uv = np.asarray(uv[0], dtype=np.float64).copy()
        s.anchor_point = p.copy()
        return s, total, uv[0]

    # -- io ----------------------------------------------------------------

    def to_npz(self, path, extra_json: dict | None = None) -> Path:
        """Write ``surface.npz`` (``docs/draw.md``); with ``extra_json`` also a ``.json`` sidecar."""
        path = Path(path)
        radius = float(getattr(self.chart, "radius", math.nan))
        np.savez(
            path,
            schema=np.str_(SCHEMA),
            chart_kind=np.str_(self.chart.kind),
            center=self.chart.center.astype(np.float64),
            rot=self.chart.rot.astype(np.float64),
            radius_m=np.float64(radius),
            width_m=np.float64(self.width_m),
            height_m=np.float64(self.height_m),
            height=self.height.astype(np.float64),
            count=self.count.astype(np.int32),
            residual_m=self.residual_m.astype(np.float64),
            anchor_uv=np.asarray(self.anchor_uv, dtype=np.float64).reshape(2),
            anchor_point=np.asarray(self.anchor_point, dtype=np.float64).reshape(3),
            anchor_shift_m=np.float64(self.anchor_shift_m),
        )
        if extra_json is not None:
            side = path.with_suffix(".json")
            payload = {
                "schema": SCHEMA,
                "chart": self.chart.to_dict(),
                "width_m": self.width_m,
                "height_m": self.height_m,
                "rows": self.rows,
                "cols": self.cols,
                "anchor_uv": [float(x) for x in self.anchor_uv],
                "anchor_point": [float(x) for x in self.anchor_point],
                "anchor_shift_m": float(self.anchor_shift_m),
                "holes": int((self.count == 0).sum()),
                **extra_json,
            }
            side.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_npz(cls, path) -> HeightFieldSurface:
        with np.load(Path(path)) as d:
            schema = str(d["schema"])
            if schema != SCHEMA:
                raise ValueError(f"{path}: schema {schema!r} != {SCHEMA!r}")
            chart = chart_from_dict(
                {
                    "chart_kind": str(d["chart_kind"]),
                    "center": d["center"],
                    "rot": d["rot"],
                    "radius_m": float(d["radius_m"]),
                }
            )
            s = cls(chart, d["height"], float(d["width_m"]), float(d["height_m"]))
            s.count = d["count"].astype(np.int32)
            s.residual_m = d["residual_m"].astype(np.float64)
            s.anchor_uv = d["anchor_uv"].astype(np.float64)
            s.anchor_point = d["anchor_point"].astype(np.float64)
            s.anchor_shift_m = float(d["anchor_shift_m"])
        return s


# --------------------------------------------------------------------------- fitting


def _orient(v: np.ndarray, hint) -> np.ndarray:
    hint = np.asarray(hint, dtype=np.float64).reshape(3)
    return -v if float(v @ hint) < 0.0 else v


def _basis_in_plane(n: np.ndarray, u_hint) -> tuple[np.ndarray, np.ndarray]:
    """(e_u, e_v) with e_u the in-plane projection of ``u_hint`` (world x, then y, if degenerate)."""
    candidates = [] if u_hint is None else [np.asarray(u_hint, dtype=np.float64).reshape(3)]
    candidates += [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for cand in candidates:
        eu = cand - (cand @ n) * n
        if np.linalg.norm(eu) > 1e-6:
            eu = eu / np.linalg.norm(eu)
            return eu, np.cross(n, eu)
    raise ValueError("could not choose an in-plane u axis")


def fit_plane(points, normal_hint, u_hint=None) -> PlaneChart:
    """PCA plane through ``points`` (N>=3, 3); normal oriented along ``normal_hint``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 3:
        raise ValueError("fit_plane needs at least 3 points")
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = _orient(vt[2], normal_hint)
    eu, ev = _basis_in_plane(n, u_hint)
    return PlaneChart(c, np.stack([eu, ev, n], axis=1))


def plane_rms(points, chart: PlaneChart) -> float:
    d = np.asarray(points, dtype=np.float64).reshape(-1, 3) - chart.center
    return float(np.sqrt(np.mean((d @ chart.normal) ** 2)))


def _cylinder_residual(p, c, a, r):
    d = p - c
    t = d @ a
    q = d - t[:, None] * a[None, :]
    rho = np.linalg.norm(q, axis=-1)
    return rho - r, t, q, rho


def fit_cylinder(points, axis_hint, normal_hint, iters: int = 50):
    """Gauss-Newton cylinder fit -> ``(CylinderChart, rms_m)``.

    Parameters are an axis point, the axis direction and the radius, started
    from a plane fit plus a quadratic curvature estimate across ``axis_hint``.
    The Jacobian is analytic; the two gauge directions (axis point along the
    axis, axis vector norm) are handled by the minimum-norm least-squares step
    and re-normalisation. The crest (``uv = 0``) is the surface point nearest
    the centroid, and the axis is oriented along ``axis_hint``.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 6:
        raise ValueError("fit_cylinder needs at least 6 points")
    plane = fit_plane(p, normal_hint, u_hint=axis_hint)
    n0, a, ev = plane.rot[:, 2], plane.rot[:, 0].copy(), plane.rot[:, 1]
    d = p - plane.center
    w, z = d @ ev, d @ n0
    coef = np.linalg.lstsq(np.stack([np.ones_like(w), w, w * w], axis=1), z, rcond=None)[0]
    kappa = 2.0 * coef[2] / (1.0 + coef[1] ** 2) ** 1.5
    r = -1.0 / kappa if kappa < -1e-3 else 1.0  # convex crest, else a weak-curvature start
    r = float(np.clip(r, 1e-3, 10.0))
    c = plane.center - r * n0
    centroid = p.mean(axis=0)
    ones = np.ones(p.shape[0])

    f, t, q, rho = _cylinder_residual(p, c, a, r)
    for _ in range(int(iters)):
        qn = q / np.maximum(rho, 1e-12)[:, None]
        jac = np.concatenate([-qn, -t[:, None] * qn, -ones[:, None]], axis=1)  # (N, 7)
        dx = np.linalg.lstsq(jac, -f, rcond=None)[0]
        c_new, a_new, r_new = c + dx[:3], a + dx[3:6], r + dx[6]
        a_new = a_new / np.linalg.norm(a_new)
        r_new = float(max(abs(r_new), 1e-4))
        c_new = c_new + ((centroid - c_new) @ a_new) * a_new  # gauge: nearest axis point to the cloud
        f_new, t_new, q_new, rho_new = _cylinder_residual(p, c_new, a_new, r_new)
        if not (np.isfinite(f_new).all() and np.isfinite(c_new).all()):
            break
        c, a, r, f, t, q, rho = c_new, a_new, r_new, f_new, t_new, q_new, rho_new
        if np.linalg.norm(dx) < 1e-12:
            break
    rms = float(np.sqrt(np.mean(f * f)))

    a = _orient(a, axis_hint)
    nc = centroid - c
    nc = nc - (nc @ a) * a
    if np.linalg.norm(nc) < 1e-9:
        nc = n0 - (n0 @ a) * a
    nc = nc / np.linalg.norm(nc)
    center = c + ((centroid - c) @ a) * a + r * nc
    rot = np.stack([a, np.cross(nc, a), nc], axis=1)  # e_u x e_v = n
    return CylinderChart(center, rot, r), rms


def choose_chart(points, normal_hint, u_hint=None, prefer: str = "auto"):
    """Fit both charts and pick one -> ``(chart, report)``.

    ``auto`` takes the cylinder when its rms is at least 25 % below the plane's,
    the cloud's own sagitta on that radius is at least 3x the cylinder rms
    (the curvature is resolved, not fitted to noise), its radius is under 0.5 m
    and its crest normal agrees with ``normal_hint`` (convex toward the tool);
    otherwise the plane.
    """
    if prefer not in ("auto", "plane", "cylinder"):
        raise ValueError(f"prefer must be auto|plane|cylinder, got {prefer!r}")
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    plane = fit_plane(p, normal_hint, u_hint)
    rms_p = plane_rms(p, plane)
    report = {"plane_rms_m": rms_p, "points": int(p.shape[0]), "prefer": prefer}
    cyl, rms_c = None, math.inf
    # The Gauss-Newton fit is seeded across ``axis_hint``; a hint 90 deg off the
    # true axis (a pipe standing across the canvas) collapses it to a plane. Start
    # it from six in-plane directions and keep the best finite fit (live capture
    # 2026-09-01: a single u-hint start returned radius 9.5e6 m on a 45 mm bottle).
    e_u, e_v = plane.rot[:, 0], plane.rot[:, 1]
    errors = []
    for angle in np.deg2rad(np.arange(0.0, 180.0, 30.0)):
        try:
            candidate, rms = fit_cylinder(p, math.cos(angle) * e_u + math.sin(angle) * e_v, normal_hint)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if np.isfinite(rms) and 0.005 <= candidate.radius <= 0.5 and rms < rms_c:
            cyl, rms_c = candidate, rms
    if cyl is None and errors:
        report["cylinder_error"] = errors[0]
    report["cylinder_rms_m"] = rms_c
    report["cylinder_radius_m"] = math.nan if cyl is None else cyl.radius
    convex = cyl is not None and float(cyl.normal @ np.asarray(normal_hint, dtype=np.float64)) > 0.0
    # How much the cloud actually bends across its own extent: the sagitta of the fitted
    # radius over the cloud's span around the axis. A 3x rms margin over the plane was too
    # strict once five viewpoints' hand-eye errors sat in the same cloud (first live scan:
    # plane 2.87 mm, cylinder 1.99 mm at 41 mm on a 45 mm bottle -> plane chosen).
    sagitta = 0.0
    if cyl is not None and np.isfinite(rms_c):
        span = float(np.ptp((p - cyl.center) @ cyl.rot[:, 1]))
        sagitta = min(span, 2.0 * cyl.radius) ** 2 / (8.0 * cyl.radius)
    report["cylinder_sagitta_m"] = sagitta
    if prefer == "plane" or cyl is None or not np.isfinite(rms_c):
        chosen, reason = plane, "plane requested" if prefer == "plane" else "cylinder fit unavailable"
    elif prefer == "cylinder":
        chosen, reason = cyl, "cylinder requested"
    elif rms_c <= 0.75 * rms_p and sagitta >= 3.0 * rms_c and cyl.radius < 0.5 and convex:
        chosen, reason = cyl, (f"cylinder rms {rms_c:.2e} <= 0.75 x plane rms {rms_p:.2e}, "
                               f"sagitta {sagitta:.2e} >= 3 x rms")
    else:
        chosen, reason = plane, (
            "cylinder not 25% better than the plane" if rms_c > 0.75 * rms_p
            else "curvature not resolved (sagitta < 3 x cylinder rms)" if sagitta < 3.0 * rms_c
            else "cylinder radius >= 0.5 m" if cyl.radius >= 0.5
            else "cylinder concave toward the tool"
        )
    report["chart_kind"] = chosen.kind
    report["reason"] = reason
    report["rms_m"] = rms_c if chosen is cyl else rms_p
    return chosen, report


# --------------------------------------------------------------------------- fusion


def _group_median(cell: np.ndarray, values: np.ndarray, ncells: int):
    """Per-cell median of ``values`` grouped by integer ``cell``; NaN where empty."""
    order = np.lexsort((values, cell))
    cs, vs = cell[order], values[order]
    counts = np.bincount(cs, minlength=ncells)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    mid = starts + counts // 2
    med = np.full(ncells, np.nan)
    odd = (counts > 0) & (counts % 2 == 1)
    even = (counts > 0) & ~odd
    med[odd] = vs[mid[odd]]
    med[even] = 0.5 * (vs[mid[even] - 1] + vs[mid[even]])
    return med, counts


def _shifted(arr: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """``arr`` moved so out[y, x] = arr[y + dy, x + dx], ``fill`` outside."""
    out = np.full_like(arr, fill)
    rows, cols = arr.shape
    ys, ye = max(0, -dy), min(rows, rows - dy)
    xs, xe = max(0, -dx), min(cols, cols - dx)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = arr[ys + dy:ye + dy, xs + dx:xe + dx]
    return out


_NEIGH8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
_NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _fill_holes(height: np.ndarray, filled: np.ndarray, relax_iters: int = 200) -> np.ndarray:
    """Define ``height`` on every cell: nearest-neighbour dilation, then Laplacian relaxation of holes."""
    h = np.where(filled, height, 0.0)
    cur = filled.copy()
    while not cur.all():
        acc = np.zeros_like(h)
        cnt = np.zeros_like(h)
        for dy, dx in _NEIGH8:
            acc += _shifted(np.where(cur, h, 0.0), dy, dx, 0.0)
            cnt += _shifted(cur.astype(np.float64), dy, dx, 0.0)
        new = ~cur & (cnt > 0)
        if not new.any():  # unreachable for a connected grid, but never spin
            break
        h[new] = acc[new] / cnt[new]
        cur |= new
    holes = ~filled
    for _ in range(int(relax_iters)):
        if not holes.any():
            break
        acc = np.zeros_like(h)
        for dy, dx in _NEIGH4:
            acc += _shifted(h, dy, dx, np.nan)
        # replicate the border: a missing neighbour repeats the cell itself
        cnt = np.zeros_like(h)
        for dy, dx in _NEIGH4:
            cnt += _shifted(np.ones_like(h), dy, dx, 0.0)
        acc = np.nan_to_num(acc) + (4.0 - cnt) * h
        h[holes] = (acc / 4.0)[holes]
    return h


def _gaussian_weighted(height: np.ndarray, weight: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Normalised-convolution Gaussian: smooth(h * w) / smooth(w), separable, edge-replicated.

    Holes carry zero weight and infilled cells neither pull nor are pulled by
    fewer than their neighbours' samples; where every weight is zero the input
    is returned unchanged.
    """
    radius = max(1, int(np.ceil(3.0 * sigma_cells)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_cells) ** 2)
    kernel /= kernel.sum()

    def blur(a: np.ndarray) -> np.ndarray:
        padded = np.pad(a, radius, mode="edge")
        rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
        return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, rows)

    num = blur(height * weight)
    den = blur(weight)
    out = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), height)
    return out


def _median3(height: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """3x3 median over filled cells only, applied to filled cells only."""
    stack = [_shifted(np.where(filled, height, np.nan), dy, dx, np.nan) for dy, dx in _NEIGH8]
    stack.append(np.where(filled, height, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN only where ~filled, discarded below
        med = np.nanmedian(np.stack(stack), axis=0)
    return np.where(filled, med, height)


def fuse(points, chart: Chart, width_m: float, height_m: float, cell_m: float, min_count: int = 3,
         smooth_m: float = 0.004):
    """Bin surface points into a height grid over ``chart`` -> ``HeightFieldSurface``.

    Each point's height is its signed distance along the chart normal at its
    own foot (``chart.frame(chart.invert(p))``), which on a plane is exactly
    the cell-normal distance and on a cylinder avoids the ``d^2 / 2R`` sag an
    off-node point shows against the node's normal. Points are assigned to
    the nearest grid node (nodes on the border, spacing about ``cell_m``);
    points more than half a cell outside the canvas are dropped. Per cell:
    ``height`` = median, ``residual_m`` = median absolute deviation, ``count``.
    Cells with fewer than ``min_count`` points are holes: their height is
    infilled so the surface is defined everywhere, their ``count`` is 0 so the
    preflight can refuse a design that lands on one. Filled cells then get one
    3x3 median pass (over filled neighbours only), then a count-weighted
    Gaussian of sigma ``smooth_m`` (normalised convolution, so holes do not
    pull). The chart carries the curvature; the displacement it leaves is
    smooth at the scale of a limb, while per-cell medians of D405 depth
    (3 mm per pixel) are not -- unsmoothed, the Catmull-Rom normals follow
    the noise and the pen leans by degrees on a flat sheet. ``smooth_m=0``
    disables it.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = max(2, int(round(width_m / cell_m)) + 1)
    rows = max(2, int(round(height_m / cell_m)) + 1)
    uv = chart.invert(p)
    cp, _, _, cn = chart.frame(uv)
    h = ((p - cp) * cn).sum(-1)

    ku = np.rint((uv[:, 0] + width_m / 2) / width_m * (cols - 1)).astype(np.int64)
    kv = np.rint((uv[:, 1] + height_m / 2) / height_m * (rows - 1)).astype(np.int64)
    keep = (ku >= 0) & (ku < cols) & (kv >= 0) & (kv < rows) & np.isfinite(h)
    cell = (kv[keep] * cols + ku[keep]).astype(np.int64)
    h = h[keep]
    ncells = rows * cols
    med, counts = _group_median(cell, h, ncells)
    mad, _ = _group_median(cell, np.abs(h - med[cell]), ncells)

    med = med.reshape(rows, cols)
    mad = mad.reshape(rows, cols)
    counts = counts.reshape(rows, cols)
    filled = counts >= int(min_count)
    if not filled.any():
        raise ValueError("fuse: no cell reached min_count; nothing to build a surface from")
    height = _fill_holes(med, filled)
    height = _median3(height, filled)
    if smooth_m > 0.0:
        height = _gaussian_weighted(height, np.where(filled, counts, 0).astype(float), smooth_m / cell_m)

    surf = HeightFieldSurface(chart, height, width_m, height_m)
    surf.count = np.where(filled, counts, 0).astype(np.int32)
    surf.residual_m = np.where(filled, np.nan_to_num(mad), 0.0)
    return surf


# --------------------------------------------------------------------------- mesh


def mesh(surface: HeightFieldSurface, step: float):
    """Triangulate the surface at about ``step`` metres -> (vertices (M, 3), faces (F, 3), normals (M, 3)).

    Faces wind counter-clockwise seen from the outward normal, for Rerun/PLY.
    """
    nu = max(2, int(round(surface.width_m / step)) + 1)
    nv = max(2, int(round(surface.height_m / step)) + 1)
    us = np.linspace(-surface.width_m / 2, surface.width_m / 2, nu)
    vs = np.linspace(-surface.height_m / 2, surface.height_m / 2, nv)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    verts, _, _, normals = surface.frame(uv)
    idx = np.arange(nv * nu).reshape(nv, nu)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate(
        [np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)], axis=0
    ).astype(np.int64)
    return verts, faces, normals
