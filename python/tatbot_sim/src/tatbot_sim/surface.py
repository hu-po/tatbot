"""The geometry a tool draws on: a base chart, a displacement, and the metric.

A drawable surface is a smooth chart plus a height field over it. ``Chart`` is
the base -- a plane for a pad, a cylinder for a limb -- and the displacement is
DATA sampled on a grid, so no particular shape is ever written into the code. A
procedural generator, a depth frame and a scan all reduce to filling that array.

Charts are parameterised by CANVAS METRES, the same (u, v) the sheet textures
are baked in. That is what buys a curved surface without an unwrapping step:
texel space is the chart, and a single chart has no seams for ``splat`` to be
aware of. Charts are also arc-length parameterised, so the chart alone
contributes no distortion; what a curved surface does change is the metric,
because displacement stretches the surface along the height gradient and a
curved chart stretches it where the height lifts off the base. That is why
``first_fundamental_form`` exists and why kernels are built from it rather than
from a single texels-per-metre scalar.

Deliberately limited to ONE chart and no overhangs: a limb is a cylinder, a pad
is a plane, and geometry outside that is mostly geometry the tool cannot reach.
A multi-chart atlas slots in beside this as another ``Surface`` if it is ever
needed.
"""

from __future__ import annotations

import numpy as np
import torch

from tatbot_sim.textures import SHEET_H_M, SHEET_W_M, SIZE_X, SIZE_Y


class Surface:
    """A drawable surface: the map between world points and texel space.

    Implementations own the metric -- how many texels a millimetre spans at a
    given point -- so kernels stay in world units. The interface is deliberately
    the minimum a splat needs: where on the sheet, how far off it, and at what
    angle the tool meets it.

    The sheet mapping itself is shared: every surface carries the same baked
    texture, so extent, resolution and ``canvas_to_px`` live here and only the
    geometry differs below.
    """

    def __init__(
        self,
        width_m: float = SHEET_W_M,
        height_m: float = SHEET_H_M,
        cols: int = SIZE_X,
        rows: int = SIZE_Y,
    ):
        self.width_m, self.height_m = float(width_m), float(height_m)
        self.cols, self.rows = int(cols), int(rows)
        self.m_per_px_x = self.width_m / self.cols
        self.m_per_px_y = self.height_m / self.rows
        # The sheet is ~2.4 px/mm on BOTH axes by construction. Kernels handle
        # anisotropy of the SURFACE (slope, curvature) through the metric, but
        # they assume the sheet's own pixels are square; a rectangular texel
        # would skew every kernel by a constant nobody measured.
        skew = abs(self.m_per_px_x - self.m_per_px_y) / self.m_per_px_x
        if skew > 0.01:
            raise ValueError(f"anisotropic sheet pixels ({skew:.1%}); kernels assume isotropy")
        self.texel_per_m = 0.5 * (self.cols / self.width_m + self.rows / self.height_m)

    def canvas_to_px(self, xy_m: torch.Tensor) -> torch.Tensor:
        """Canvas-frame metres (..., 2) -> texel coordinates (..., 2) as (col, row).

        Reproduces textures.py: column 0 at canvas x = -W/2 with +u along +x,
        row 0 at canvas y = -H/2 with rows along +y, pixel k centred at
        ``(k + 0.5) * m_per_px - half``. Field arrays therefore use the same
        [row, col] convention as the images ``_make_sheet`` writes, and
        compositing needs no flip.
        """
        col = (xy_m[..., 0] + self.width_m / 2) / self.m_per_px_x - 0.5
        row = (xy_m[..., 1] + self.height_m / 2) / self.m_per_px_y - 0.5
        return torch.stack([col, row], dim=-1)

    def project(self, points_w: torch.Tensor, axis_w: torch.Tensor | None = None):
        raise NotImplementedError

    def first_fundamental_form(self, uv_m: torch.Tensor) -> torch.Tensor:
        """Metric tensor (B, 2, 2) in canvas coordinates: [[E, F], [F, G]].

        The world length of a canvas-frame step ``d`` is ``sqrt(d @ M @ d)``.
        Identity means a millimetre of canvas is a millimetre of surface, which
        is the flat case and the only one the old scalar metric could express.
        """
        raise NotImplementedError

    def frame(self, uv_m: torch.Tensor):
        """(B, 2) canvas metres -> point, d/du, d/dv, unit outward normal, (B, 3) each."""
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        raise NotImplementedError

    def env_view(self, i: int, n: int) -> Surface:
        """This surface as seen by env ``i``, with its geometry repeated ``n`` times.

        The planner works one episode at a time over a whole trajectory, which
        is the opposite batching to the env's one-point-per-env. Rather than a
        second implementation of the geometry for that shape, an env's surface
        is restated as a batch of identical copies and the same ``frame`` runs.
        """
        raise NotImplementedError

    def frame_np(self, i: int, uv_m: np.ndarray):
        """Env ``i`` at (T, 2) canvas metres -> world points and normals, numpy.

        The planner is numpy end to end; this is the whole of its dependency on
        how a surface is actually shaped.
        """
        uv = torch.as_tensor(np.asarray(uv_m, dtype=np.float32), device=self._device)
        point, _, _, normal = self.env_view(i, uv.shape[0]).frame(uv)
        return point.cpu().numpy(), normal.cpu().numpy()

    def base_normal_np(self, i: int) -> np.ndarray:
        """(3,) the UNDISPLACED normal for env ``i`` — the pad's own, not the
        skin's. Leaning is measured from this, because it is the direction the
        arm's workspace is built around; the displacement is what asks the
        wrist to leave it.
        """
        raise NotImplementedError

    def origin_world_np(self) -> np.ndarray:
        """(B, 3) world position of each env's canvas origin."""
        point, _, _, _ = self.frame(
            torch.zeros(self.batch_size, 2, dtype=torch.float32, device=self._device)
        )
        return point.cpu().numpy()


class PlanarSurface(Surface):
    """The flat pad: a plane plus the sheet's pixel mapping.

    Built per episode from the env's canvas frame (``pad_top_center``,
    ``pad_rot``; rotation columns are the canvas x axis, y axis and outward
    normal).
    """

    def __init__(
        self,
        center: torch.Tensor,
        rot: torch.Tensor,
        width_m: float = SHEET_W_M,
        height_m: float = SHEET_H_M,
        cols: int = SIZE_X,
        rows: int = SIZE_Y,
    ):
        super().__init__(width_m, height_m, cols, rows)
        self.center = center  # (B, 3) top-face centre, world
        self.rot = rot  # (B, 3, 3), column 2 = outward normal

    @property
    def normal(self) -> torch.Tensor:
        return self.rot[:, :, 2]

    @property
    def batch_size(self) -> int:
        return self.center.shape[0]

    @property
    def _device(self):
        return self.center.device

    def frame(self, uv_m: torch.Tensor):
        ex, ey, n = self.rot[:, :, 0], self.rot[:, :, 1], self.rot[:, :, 2]
        point = self.center + uv_m[:, 0:1] * ex + uv_m[:, 1:2] * ey
        return point, ex, ey, n

    def env_view(self, i: int, n: int) -> PlanarSurface:
        return PlanarSurface(
            self.center[i].expand(n, 3),
            self.rot[i].expand(n, 3, 3),
            self.width_m, self.height_m, self.cols, self.rows,
        )

    def base_normal_np(self, i: int) -> np.ndarray:
        return self.rot[i, :, 2].cpu().numpy()

    def first_fundamental_form(self, uv_m: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(2, dtype=uv_m.dtype, device=uv_m.device)
        return eye.expand(uv_m.shape[0], 2, 2)

    def project(self, points_w: torch.Tensor, axis_w: torch.Tensor | None = None):
        """World points (B, 3) -> (uv_m (B, 2), signed_dist (B,), incidence_cos (B,)).

        ``signed_dist`` is distance along the outward normal, matching the
        env's deposit gate. ``incidence_cos`` is |axis . normal| for a supplied
        tool axis and 1 otherwise.
        """
        d = points_w - self.center
        local = torch.stack(
            [(d * self.rot[:, :, 0]).sum(-1), (d * self.rot[:, :, 1]).sum(-1)], dim=-1
        )
        dist = (d * self.rot[:, :, 2]).sum(-1)
        if axis_w is None:
            inc = torch.ones_like(dist)
        else:
            inc = (axis_w * self.rot[:, :, 2]).sum(-1).abs().clamp(0.0, 1.0)
        return local, dist, inc


class Chart:
    """The smooth base a displacement is measured from.

    Arc-length parameterised in canvas metres, so a step of one millimetre in
    (u, v) is one millimetre along the base regardless of curvature. Charts
    supply their frame and the derivatives of their normal; the surface built
    on top needs both to get its metric right, because on a curved chart a
    point lifted by ``h`` sweeps more arc than the base does.
    """

    def frame(self, uv_m: torch.Tensor):
        """(B, 2) canvas metres -> point, d/du, d/dv, outward normal, each (B, 3)."""
        raise NotImplementedError

    def normal_derivatives(self, uv_m: torch.Tensor):
        """(B, 2) -> (dn/du, dn/dv), each (B, 3). Zero for a flat chart."""
        raise NotImplementedError

    def env_view(self, i: int, n: int) -> "Chart":
        """This chart as seen by env ``i``, repeated ``n`` times."""
        raise NotImplementedError

    def invert(self, points_w: torch.Tensor) -> torch.Tensor:
        """World points (B, 3) -> the (B, 2) canvas coordinates of the nearest base point.

        Only a starting guess for the surface's own projection, but an exact
        one for the undisplaced chart, which is why a low displacement
        converges in a couple of iterations.
        """
        raise NotImplementedError


class PlaneChart(Chart):
    """A flat base: the pad, and the case a curved surface must reduce to."""

    def __init__(self, center: torch.Tensor, rot: torch.Tensor):
        self.center = center  # (B, 3)
        self.rot = rot  # (B, 3, 3): columns are canvas x, canvas y, outward normal

    def frame(self, uv_m: torch.Tensor):
        ex, ey, n = self.rot[:, :, 0], self.rot[:, :, 1], self.rot[:, :, 2]
        point = self.center + uv_m[:, 0:1] * ex + uv_m[:, 1:2] * ey
        return point, ex, ey, n

    def normal_derivatives(self, uv_m: torch.Tensor):
        zero = torch.zeros_like(self.center)
        return zero, zero

    def invert(self, points_w: torch.Tensor) -> torch.Tensor:
        d = points_w - self.center
        return torch.stack(
            [(d * self.rot[:, :, 0]).sum(-1), (d * self.rot[:, :, 1]).sum(-1)], dim=-1
        )

    def env_view(self, i: int, n: int) -> "PlaneChart":
        return PlaneChart(self.center[i].expand(n, 3), self.rot[i].expand(n, 3, 3))


class CylinderChart(Chart):
    """A limb: a cylinder whose axis runs along canvas x, wrapped by canvas y.

    ``center`` is the point on the surface at (0, 0) -- the crest of the
    cylinder -- so it plays the same role the pad's top-face centre does, and
    ``rot``'s columns are the axis, the crest tangent and the outward normal
    there. Canvas y is ARC LENGTH around the circumference, so the chart is
    isometric and a stroke drawn 30 mm across the canvas is 30 mm of skin.
    """

    def __init__(self, center: torch.Tensor, rot: torch.Tensor, radius: torch.Tensor):
        if bool((radius <= 0).any()):
            raise ValueError("cylinder radius must be positive (convex); got a non-positive value")
        self.center = center  # (B, 3)
        self.rot = rot  # (B, 3, 3): axis, crest tangent, outward normal at v=0
        self.radius = radius  # (B,)

    def _theta(self, v_m: torch.Tensor) -> torch.Tensor:
        """Arc length (B, 1) -> angle (B, 1). Radius is (B,), so it must be
        unsqueezed or the division broadcasts into a (B, B) grid."""
        return v_m / self.radius[:, None]

    def frame(self, uv_m: torch.Tensor):
        ex, ey, ez = self.rot[:, :, 0], self.rot[:, :, 1], self.rot[:, :, 2]
        r = self.radius[:, None]
        th = self._theta(uv_m[:, 1:2])
        s, c = torch.sin(th), torch.cos(th)
        point = self.center + uv_m[:, 0:1] * ex + (r * s) * ey + (r * c - r) * ez
        d_du = ex
        d_dv = c * ey - s * ez
        normal = s * ey + c * ez
        return point, d_du, d_dv, normal

    def normal_derivatives(self, uv_m: torch.Tensor):
        ey, ez = self.rot[:, :, 1], self.rot[:, :, 2]
        r = self.radius[:, None]
        th = self._theta(uv_m[:, 1:2])
        s, c = torch.sin(th), torch.cos(th)
        dn_dv = (c * ey - s * ez) / r  # = d_dv / r
        return torch.zeros_like(ey), dn_dv

    def invert(self, points_w: torch.Tensor) -> torch.Tensor:
        d = points_w - self.center
        lx = (d * self.rot[:, :, 0]).sum(-1)
        ly = (d * self.rot[:, :, 1]).sum(-1)
        lz = (d * self.rot[:, :, 2]).sum(-1)
        # the axis sits one radius below the crest, so measure the angle from there
        theta = torch.atan2(ly, lz + self.radius)
        return torch.stack([lx, self.radius * theta], dim=-1)

    def env_view(self, i: int, n: int) -> "CylinderChart":
        return CylinderChart(
            self.center[i].expand(n, 3), self.rot[i].expand(n, 3, 3), self.radius[i].expand(n)
        )


def _catmull_rom(p: torch.Tensor, t: torch.Tensor):
    """Catmull-Rom through 4 samples (..., 4) at t in [0, 1] between p1 and p2.

    Returns (value, d value / d t). Interpolating and C1, which is what the
    tool orientation needs: bilinear height would give a normal field that
    steps at every cell boundary, and the wrist would chase those steps.
    """
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a = 2.0 * p1
    b = p2 - p0
    c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
    d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    t2 = t * t
    val = 0.5 * (a + b * t + c * t2 + d * t2 * t)
    der = 0.5 * (b + 2.0 * c * t + 3.0 * d * t2)
    return val, der


class DisplacedSurface(Surface):
    """A chart plus a height field: the general drawable surface.

    ``height`` is (B, hr, hc) metres of displacement along the chart normal,
    sampled on a regular grid spanning the canvas extent with nodes on the
    border. Its resolution is independent of the texture's: the surface is
    smooth, so a coarse grid interpolated bicubically carries it, and the
    height grid is the one thing a generator, a depth frame or a scan all
    produce.

    Outside the canvas extent the grid replicates its border, so the surface
    stays defined (and flat) where no data exists rather than extrapolating a
    shape nobody measured.
    """

    def __init__(
        self,
        chart: Chart,
        height: torch.Tensor,
        width_m: float = SHEET_W_M,
        height_m: float = SHEET_H_M,
        cols: int = SIZE_X,
        rows: int = SIZE_Y,
        iters: int = 3,
        tol_m: float = 1e-5,
    ):
        super().__init__(width_m, height_m, cols, rows)
        if height.ndim != 3 or height.shape[-1] < 2 or height.shape[-2] < 2:
            raise ValueError(f"height must be (B, hr>=2, hc>=2); got {tuple(height.shape)}")
        self.chart = chart
        self.height = height
        self.iters = int(iters)
        self.tol_m = float(tol_m)
        self.unconverged = 0  # points the last projection could not place

    @property
    def batch_size(self) -> int:
        return self.height.shape[0]

    @property
    def _device(self):
        return self.height.device

    def env_view(self, i: int, n: int) -> "DisplacedSurface":
        return DisplacedSurface(
            self.chart.env_view(i, n),
            self.height[i].expand(n, *self.height.shape[1:]),
            self.width_m, self.height_m, self.cols, self.rows, self.iters, self.tol_m,
        )

    def base_normal_np(self, i: int) -> np.ndarray:
        _, _, _, n = self.chart.env_view(i, 1).frame(torch.zeros(1, 2, device=self._device))
        return n[0].cpu().numpy()

    def sample_height(self, uv_m: torch.Tensor):
        """(B, 2) canvas metres -> (h, dh/du, dh/dv), each (B,), metres."""
        b, hr, hc = self.height.shape
        gu = (uv_m[:, 0] + self.width_m / 2) / self.width_m * (hc - 1)
        gv = (uv_m[:, 1] + self.height_m / 2) / self.height_m * (hr - 1)
        gu = gu.clamp(0.0, hc - 1.0)
        gv = gv.clamp(0.0, hr - 1.0)
        iu = torch.floor(gu).long().clamp(0, hc - 1)
        iv = torch.floor(gv).long().clamp(0, hr - 1)
        tu = (gu - iu).unsqueeze(-1)
        tv = gv - iv

        off = torch.tensor([-1, 0, 1, 2], device=self.height.device)
        ou = (iu[:, None] + off[None, :]).clamp(0, hc - 1)  # (B, 4)
        ov = (iv[:, None] + off[None, :]).clamp(0, hr - 1)  # (B, 4)
        idx = (ov[:, :, None] * hc + ou[:, None, :]).reshape(b, 16)
        patch = self.height.reshape(b, hr * hc).gather(1, idx).reshape(b, 4, 4)

        # interpolate along u within each of the four rows, then across v
        rows_val, rows_du = _catmull_rom(patch, tu)  # (B, 4) each
        val, dval_dtv = _catmull_rom(rows_val, tv)
        dval_dtu, _ = _catmull_rom(rows_du, tv)
        du_per_m = (hc - 1) / self.width_m
        dv_per_m = (hr - 1) / self.height_m
        return val, dval_dtu * du_per_m, dval_dtv * dv_per_m

    def frame(self, uv_m: torch.Tensor):
        """(B, 2) -> surface point, d/du, d/dv, unit outward normal, each (B, 3)."""
        c_pt, c_du, c_dv, c_n = self.chart.frame(uv_m)
        dn_du, dn_dv = self.chart.normal_derivatives(uv_m)
        h, h_du, h_dv = self.sample_height(uv_m)
        h = h.unsqueeze(-1)
        point = c_pt + h * c_n
        s_du = c_du + h_du.unsqueeze(-1) * c_n + h * dn_du
        s_dv = c_dv + h_dv.unsqueeze(-1) * c_n + h * dn_dv
        n = torch.cross(s_du, s_dv, dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # keep the chart's outward sense; the cross product's sign follows the
        # parameterisation and would flip the tool through the surface
        n = torch.where(((n * c_n).sum(-1, keepdim=True) < 0), -n, n)
        return point, s_du, s_dv, n

    def first_fundamental_form(self, uv_m: torch.Tensor) -> torch.Tensor:
        _, s_du, s_dv, _ = self.frame(uv_m)
        e = (s_du * s_du).sum(-1)
        f = (s_du * s_dv).sum(-1)
        g = (s_dv * s_dv).sum(-1)
        return torch.stack([torch.stack([e, f], -1), torch.stack([f, g], -1)], -2)

    def invert(self, points_w: torch.Tensor) -> torch.Tensor:
        """World points (B, 3) -> canvas metres (B, 2) of the nearest surface point.

        Gauss-Newton on the tangential residual, started from the chart's exact
        inverse. Two or three steps is plenty for the low slopes a tool can
        actually work on; the residual is checked afterwards rather than
        trusted.
        """
        uv = self.chart.invert(points_w)
        for _ in range(self.iters):
            point, s_du, s_dv, _ = self.frame(uv)
            res = points_w - point
            e = (s_du * s_du).sum(-1)
            f = (s_du * s_dv).sum(-1)
            g = (s_dv * s_dv).sum(-1)
            ru = (res * s_du).sum(-1)
            rv = (res * s_dv).sum(-1)
            det = (e * g - f * f).clamp(min=1e-12)
            uv = uv + torch.stack([(g * ru - f * rv) / det, (e * rv - f * ru) / det], dim=-1)
        return uv

    def project(self, points_w: torch.Tensor, axis_w: torch.Tensor | None = None):
        """World points (B, 3) -> (uv_m (B, 2), signed_dist (B,), incidence_cos (B,)).

        ``signed_dist`` is measured along the LOCAL normal, so the deposit gate
        keeps meaning the same thing it does on a plane. A point the iteration
        could not place -- off the end of the chart, or too far away for the
        linearisation -- is returned infinitely far from the surface rather
        than at a plausible-looking wrong place, because a bad uv would deposit
        ink somewhere nobody asked for.
        """
        uv = self.invert(points_w)
        point, _, _, n = self.frame(uv)
        res = points_w - point
        dist = (res * n).sum(-1)
        tangential = (res - dist.unsqueeze(-1) * n).norm(dim=-1)
        ok = tangential <= self.tol_m
        self.unconverged = int((~ok).sum())
        dist = torch.where(ok, dist, torch.full_like(dist, float("inf")))
        inc = (
            torch.ones_like(dist)
            if axis_w is None
            else (axis_w * n).sum(-1).abs().clamp(0.0, 1.0)
        )
        return uv, dist, inc


def _border_taper(rows: int, cols: int, frac: float) -> np.ndarray:
    """Smoothstep window that is 1 inside and 0 on the border, (rows, cols)."""
    if frac <= 0.0:
        return np.ones((rows, cols))

    def ramp(n):
        t = np.linspace(0.0, 1.0, n)
        edge = np.clip(np.minimum(t, 1.0 - t) / max(frac, 1e-9), 0.0, 1.0)
        return edge * edge * (3.0 - 2.0 * edge)

    return ramp(rows)[:, None] * ramp(cols)[None, :]


def random_height_field(
    rng,
    num_envs: int,
    rows: int,
    cols: int,
    feature_m,
    max_slope_rad,
    amplitude_m,
    width_m: float = SHEET_W_M,
    height_m: float = SHEET_H_M,
    components: int = 6,
    taper_frac: float = 0.15,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A smooth random displacement, (B, rows, cols) metres.

    Band-limited by construction -- a sum of plane waves whose wavelengths sit
    around ``feature_m`` -- which is what makes it read as skin over tissue
    rather than as noise. The point of generating surfaces instead of measuring
    one is that a policy has to contour whatever it meets, so what matters is
    the DISTRIBUTION being wide enough to contain the real ones, not any
    particular sample matching a particular pad.

    Each env is scaled to hit BOTH a sampled peak amplitude and a sampled
    steepest slope, whichever binds first. Slope is the quantity worth
    controlling: it is what costs reach, what tilts the tool off vertical, and
    what stretches texel space. Amplitude alone would let a short-wavelength
    draw hide a cliff inside a modest height.

    ``feature_m``, ``max_slope_rad`` and ``amplitude_m`` are per-env arrays, so
    the caller owns the ranges and the whole tree lands in run_meta.

    The field tapers to zero at the canvas border. That is what a silicone skin
    draped over a pad actually does -- the edges lie flat on it -- and it is
    also what keeps the rendered sheet flush with the pad body underneath
    instead of floating off its rim.
    """
    us = np.linspace(-width_m / 2, width_m / 2, cols)
    vs = np.linspace(-height_m / 2, height_m / 2, rows)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    window = _border_taper(rows, cols, taper_frac)
    feature_m = np.asarray(feature_m, dtype=np.float64).reshape(num_envs)
    max_slope_rad = np.asarray(max_slope_rad, dtype=np.float64).reshape(num_envs)
    amplitude_m = np.asarray(amplitude_m, dtype=np.float64).reshape(num_envs)

    out = np.zeros((num_envs, rows, cols), dtype=np.float64)
    for i in range(num_envs):
        acc = np.zeros((rows, cols), dtype=np.float64)
        for _ in range(components):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            wavelength = feature_m[i] * rng.uniform(0.7, 1.4)
            k = 2.0 * np.pi / wavelength
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.5, 1.0)
            acc += amp * np.cos(
                k * (np.cos(theta) * uu + np.sin(theta) * vv) + phase
            )
        # A skin lying on a support can bulge UP off it and cannot sink into
        # it, so the field is lifted to rest on zero rather than centred on it.
        # Centring put roughly half the surface below the substrate's top face,
        # where the body underneath poked through the sheet and hid both the
        # ruling and the ink -- and where the flat workspace floor spent its
        # time clamping motion that was perfectly legal.
        acc -= acc.min()
        acc *= window
        peak = float(np.abs(acc).max())
        if peak <= 0.0:
            continue
        g_v, g_u = np.gradient(acc, vs, us)
        slope = float(np.hypot(g_u, g_v).max())
        by_amplitude = amplitude_m[i] / peak
        by_slope = np.tan(max_slope_rad[i]) / slope if slope > 0 else by_amplitude
        out[i] = acc * min(by_amplitude, by_slope)
    return torch.as_tensor(out, dtype=torch.float32, device=device)


def drape_height_field(
    rng,
    num_envs: int,
    rows: int,
    cols: int,
    peak_m,
    radius_u_m,
    radius_v_m,
    center_u_m=None,
    center_v_m=None,
    width_m: float = SHEET_W_M,
    height_m: float = SHEET_H_M,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A single broad rise: skin draped over a pad, (B, rows, cols) metres.

    A raised cosine, which is what draping looks like -- flat where the skin
    overhangs onto the table, rising smoothly to one summit, with no crease at
    the foot because the profile leaves the flat with zero slope.

    This is a SAMPLER, not a shape in the code: it fills the same displacement
    array a depth capture or a scan would, and everything downstream still only
    sees a grid of numbers. What it buys is a distribution centred on the
    surface the operator actually records on, instead of a general field that
    spends most of its range on skins nobody owns.
    """
    us = np.linspace(-width_m / 2, width_m / 2, cols)
    vs = np.linspace(-height_m / 2, height_m / 2, rows)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    peak_m = np.asarray(peak_m, dtype=np.float64).reshape(num_envs)
    ru = np.asarray(radius_u_m, dtype=np.float64).reshape(num_envs)
    rv = np.asarray(radius_v_m, dtype=np.float64).reshape(num_envs)
    cu = np.zeros(num_envs) if center_u_m is None else np.asarray(center_u_m).reshape(num_envs)
    cv = np.zeros(num_envs) if center_v_m is None else np.asarray(center_v_m).reshape(num_envs)

    out = np.zeros((num_envs, rows, cols), dtype=np.float64)
    for i in range(num_envs):
        r = np.hypot((uu - cu[i]) / max(ru[i], 1e-6), (vv - cv[i]) / max(rv[i], 1e-6))
        out[i] = np.where(r < 1.0, 0.5 * peak_m[i] * (1.0 + np.cos(np.pi * np.clip(r, 0, 1))), 0.0)
    del rng  # the caller samples the parameters; this only draws them
    return torch.as_tensor(out, dtype=torch.float32, device=device)


def cylinder_amplitude_ceiling(radius_m, fraction: float = 0.15):
    """Largest displacement a cylinder chart may carry, per env.

    A point displaced INWARD on a cylinder sweeps less arc than the base does,
    so texel space compresses by (1 + h/R) and a kernel has to grow to keep its
    world size. Capping the displacement at a fraction of the radius keeps that
    growth inside the field's margin, so the bound is enforced where a surface
    is built rather than discovered by a raise in the middle of a run.
    """
    return np.asarray(radius_m, dtype=np.float64) * float(fraction)
