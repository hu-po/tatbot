"""Pigment as a field on the surface, and tools as operators on it.

Ink used to be a pool of kinematic cylinder actors teleported under the pen —
monotone-additive and stateless, with no per-pixel quantity a removal tool
could subtract from. The laser pen needs an inverse, so pigment becomes a
per-env scalar field in the sheet's own texture space: one float in [0, 1]
per texel, 0 = bare paper, 1 = fully opaque ink.

Tools are operators on that field, dispatched on the registry's ``kind``:

    pen    field <- clamp(field + opacity * K)
    laser  field <- field * (1 - eta * K)

The laser's multiplicative form is the physically right shape — a pass clears
a FRACTION of what is there, so repeated passes fade a stroke asymptotically
instead of deleting it, and partial removal falls out of the arithmetic rather
than being special-cased.

Kernels are specified in MILLIMETRES and converted to texels by the surface,
never the other way round. That is the hinge the curved surface turns on: a
curved surface stretches texel space, so a circular spot in world space is not
a circle in texels, and a kernel that thinks in pixels silently changes size
across a limb. The geometry itself — charts, displacement and the metric that
comes out of them — lives in ``surface``; this module only ever asks it where
a point lands and how the texels are stretched there.

Deliberately free of sapien: everything here is torch + numpy, so the tests
run without a render device. The sheet constants come from ``textures`` —
imported precisely so the pixel<->canvas mapping cannot drift from the module
that bakes the paper.
"""

from __future__ import annotations

import numpy as np
import torch

from tatbot_sim.surface import Surface


def kernel_half(radii_m: torch.Tensor, texel_per_m: float, profile: str, stretch: float = 1.0) -> int:
    """Texels from the centre a kernel of these radii can reach.

    ``stretch`` is how far the surface metric can spread a kernel in texel
    space (1.0 on a plane). It sizes the window and, through it, the field's
    margin, so a curved surface never needs the array to be reallocated.
    """
    reach = float(radii_m.max()) * float(texel_per_m) * float(stretch)
    return int(np.ceil(2.0 * reach)) + 1 if profile == "gaussian" else int(np.ceil(reach)) + 1


def stamp_kernels(
    radii_m: torch.Tensor,
    texel_per_m: float,
    profile: str = "disc",
    metric: torch.Tensor | None = None,
    half: int | None = None,
) -> torch.Tensor:
    """Per-env kernels, (B, k, k), peak 1, sized from world radii.

    One window for the batch, sized to the largest radius; smaller envs simply
    carry more zeros, which keeps the splat a single batched op while radius
    stays a per-env randomization.

    ``disc`` is the pen: flat top, one-texel anti-aliased edge, the shape a
    nib leaves. ``gaussian`` is the laser spot, whose dose falls off from the
    beam centre.

    ``metric`` is the surface's first fundamental form at the stamp, (B, 2, 2)
    in canvas coordinates. It is what makes a kernel ELLIPTICAL: on a slope a
    texel spans more world distance along the gradient than across it, so a
    round spot in world space is an ellipse in texel space, and a kernel that
    ignored this would quietly change size as the tool crossed a limb. Identity
    -- the flat case -- reduces the expression to the plain radial one.
    """
    r_px = radii_m.to(torch.float32) * float(texel_per_m)
    if metric is None:
        metric_mat = torch.eye(2, device=radii_m.device, dtype=torch.float32).expand(len(r_px), 2, 2)
    else:
        metric_mat = metric
    e, f, g = metric_mat[:, 0, 0], metric_mat[:, 0, 1], metric_mat[:, 1, 1]
    # a world radius reaches furthest along the metric's softest direction
    lam_min = 0.5 * ((e + g) - torch.sqrt((e - g) ** 2 + 4.0 * f**2))
    stretch = float(torch.rsqrt(lam_min.clamp(min=1e-9)).max())
    need = kernel_half(radii_m, texel_per_m, profile, stretch)
    if half is None:
        half = need
    elif need > half:
        raise ValueError(
            f"{profile} kernel needs {need} texels of window but the field was built for {half}; "
            f"the surface stretches texel space by {stretch:.2f}x — raise InkField(max_stretch=)"
        )
    ar = torch.arange(-half, half + 1, device=radii_m.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    # distance in texels that a world-space circle traces out under this metric
    ee, ff, gg = e[:, None, None], f[:, None, None], g[:, None, None]
    d = torch.sqrt((ee * xx**2 + 2.0 * ff * xx * yy + gg * yy**2).clamp(min=0.0))
    r = r_px[:, None, None].clamp(min=1e-6)
    if profile == "disc":
        return (r + 0.5 - d).clamp(0.0, 1.0)
    if profile == "gaussian":
        return torch.exp(-0.5 * (d / (r / 2.0)) ** 2)
    raise ValueError(f"unknown kernel profile: {profile}")


def splat(
    field: torch.Tensor,
    centers_px: torch.Tensor,
    kernels: torch.Tensor,
    weights: torch.Tensor,
    op: str = "add",
    active: torch.Tensor | None = None,
) -> None:
    """Write one stamp per env into ``field`` (B, H, W), in place.

    THE only place pixels are written — deposition, removal and pre-inking all
    come through here, so the seam-aware version a UV-unwrapped mesh needs is
    an upgrade to this one function rather than a rewrite of the tools.

    ``field`` is expected to carry a margin wide enough for the kernel (see
    ``InkField``): a stamp whose window would leave the array is dropped
    outright rather than clipped, which keeps every scattered index unique and
    the write order irrelevant. Coordinates are (col, row), matching
    ``Surface.canvas_to_px``.
    """
    b, h, w = field.shape
    half = kernels.shape[-1] // 2
    col = torch.round(centers_px[:, 0]).long()
    row = torch.round(centers_px[:, 1]).long()
    ok = (row >= half) & (row < h - half) & (col >= half) & (col < w - half)
    if active is not None:
        ok = ok & active
    if not bool(ok.any()):
        return
    # dropped envs keep weight 0, so their gather->scatter writes back exactly
    # what was there; clamping their index is then harmless
    weights = torch.where(ok, weights, torch.zeros_like(weights))
    row = row.clamp(half, h - half - 1)
    col = col.clamp(half, w - half - 1)

    ar = torch.arange(-half, half + 1, device=field.device)
    rows = row[:, None] + ar[None, :]  # (B, k)
    cols = col[:, None] + ar[None, :]  # (B, k)
    idx = (rows[:, :, None] * w + cols[:, None, :]).reshape(b, -1)  # (B, k*k)
    kern = (kernels * weights[:, None, None]).reshape(b, -1)

    flat = field.view(b, h * w)
    cur = flat.gather(1, idx)
    if op == "add":
        new = (cur + kern).clamp(0.0, 1.0)
    elif op == "erode":
        new = (cur * (1.0 - kern)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown splat op: {op}")
    flat.scatter_(1, idx, new)


def laser_eta(
    clearance: torch.Tensor,
    incidence_cos: torch.Tensor,
    dwell_s: torch.Tensor | None = None,
    standoff_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of the REMAINING pigment one laser pass clears.

    v0 is ``clearance * incidence_cos``: a per-pass fraction drawn from the DR
    tree, reduced when the beam meets the surface obliquely. Dwell and standoff
    are accepted and unused — the handheld pen this registry describes is a
    CONTACT tool (plastic tip on the skin, emitter focused at it), so there is
    no working distance to model and tip speed is already folded into how many
    passes a texel receives. They are in the signature because a bench Nd:YAG
    is held 3-5 cm off the skin where standoff SETS the dose, and that
    hardware should not require changing every caller.

    Deliberately NOT derived from the datasheet's optical figures: that file
    says plainly that every wavelength, pulse width and power on the consumer
    unit is marketing copy. Clearance is a tunable of the sim, logged per run.
    """
    del dwell_s, standoff_m  # see docstring
    return (clearance * incidence_cos).clamp(0.0, 1.0)


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Polyline (N, 2) resampled to a point every ``spacing_m`` along its length."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts.astype(np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return pts[:1].astype(np.float32)
    n = max(int(np.ceil(total / max(spacing_m, 1e-6))) + 1, 2)
    s = np.linspace(0.0, total, n)
    return np.stack(
        [np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1
    ).astype(np.float32)


class InkField:
    """Per-env pigment, the single source of truth for what is on the surface.

    Stored with a margin on every side so a stamp near the sheet edge falls off
    the array instead of needing clip logic in the hot path; only the interior
    is ever composited or measured.

    Appearance is per env. The dot pool forced one ink colour and width across
    the batch (a shared actor pool), which the field does not, so line weight
    and darkness now vary env-to-env like every other visual axis.
    """

    def __init__(
        self,
        num_envs: int,
        surface: Surface,
        pen_radius_m: torch.Tensor,
        laser_radius_m: torch.Tensor,
        ink_rgb: torch.Tensor,
        device: torch.device | str = "cpu",
        max_stretch: float = 1.5,
    ):
        self.num_envs = num_envs
        self.rows, self.cols = surface.rows, surface.cols
        self.device = device
        self.ink_rgb = ink_rgb.to(device)  # (B, 3) in [0, 1]
        self.pen_radius_m = pen_radius_m.to(device)
        self.laser_radius_m = laser_radius_m.to(device)
        self.texel_per_m = surface.texel_per_m
        # Kernels are built per stamp, because their shape depends on where the
        # tool is: the surface metric turns a round spot into an ellipse. The
        # WINDOW they are built in is fixed here, sized for the most a surface
        # is allowed to stretch texel space, so the field's margin never has to
        # change and the splat stays one batched op.
        self.max_stretch = float(max_stretch)
        self.pen_half = kernel_half(self.pen_radius_m, self.texel_per_m, "disc", self.max_stretch)
        self.laser_half = kernel_half(
            self.laser_radius_m, self.texel_per_m, "gaussian", self.max_stretch
        )
        # margin: the widest stamp, so no in-bounds tool position is ever
        # dropped for want of room
        self.pad = max(self.pen_half, self.laser_half) + 1
        self._f = torch.zeros(
            (num_envs, self.rows + 2 * self.pad, self.cols + 2 * self.pad),
            dtype=torch.float32,
            device=device,
        )
        self.dirty = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def field(self) -> torch.Tensor:
        """The visible sheet, (B, rows, cols) — the margin is never shown."""
        return self._f[:, self.pad : self.pad + self.rows, self.pad : self.pad + self.cols]

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        if env_idx is None:
            self._f.zero_()
            self.dirty.zero_()
        else:
            self._f[env_idx] = 0.0
            self.dirty[env_idx] = False

    def _stamp(self, surface, uv_m, radii, profile, half, weights, op, active) -> None:
        """One stamp per env at canvas-frame ``uv_m``, shaped by the surface there.

        Texel space is this class's business, so the surface is asked for the
        two things only it knows -- where the point lands and how the metric
        stretches there -- and everything downstream stays in pixels.
        """
        kernels = stamp_kernels(
            radii, self.texel_per_m, profile, surface.first_fundamental_form(uv_m), half
        )
        # Round to a texel BEFORE the margin is added. torch.round breaks ties
        # to even, so rounding px + pad would move a stamp sitting exactly on a
        # texel boundary by one texel whenever the margin is odd -- and the
        # margin is sized from the sampled tool radii, so where the ink landed
        # would depend on a number nobody thinks of as geometry.
        px = torch.round(surface.canvas_to_px(uv_m))
        splat(self._f, px + self.pad, kernels, weights, op, active)
        if active is not None:
            self.dirty |= active
        else:
            self.dirty[:] = True

    def deposit(self, surface: Surface, uv_m, opacity, active=None) -> None:
        """Pen: add pigment. Overlapping passes saturate at fully opaque."""
        self._stamp(
            surface, uv_m, self.pen_radius_m, "disc", self.pen_half, opacity, "add", active
        )

    def remove(self, surface: Surface, uv_m, eta, active=None) -> None:
        """Laser: clear a fraction of the pigment under the spot."""
        self._stamp(
            surface, uv_m, self.laser_radius_m, "gaussian", self.laser_half, eta, "erode", active
        )

    def rasterize(self, surface: Surface, strokes_per_env, opacity) -> None:
        """Pre-ink each env's strokes — canvas-frame polylines, metres.

        Used to open an episode on a sheet that already carries a motif, which
        is what a removal task needs: the laser's target has to exist before
        the episode starts. Points are laid down at half a line width so the
        stroke reads continuous, through the same splat the pen uses, so
        pre-inked and drawn ink are the same substance.
        """
        spacing = float(self.pen_radius_m.min()) * 0.5
        sampled = [
            [resample_polyline(s.points, spacing) for s in strokes]
            for strokes in strokes_per_env
        ]
        flat = [np.concatenate(s, axis=0) if s else np.zeros((0, 2), np.float32) for s in sampled]
        n_max = max((len(p) for p in flat), default=0)
        if n_max == 0:
            return
        for i in range(n_max):
            pts = torch.as_tensor(
                np.stack([p[min(i, len(p) - 1)] if len(p) else np.zeros(2, np.float32)
                          for p in flat]),
                dtype=torch.float32, device=self.device,
            )
            active = torch.as_tensor(
                [i < len(p) for p in flat], dtype=torch.bool, device=self.device
            )
            self.deposit(surface, pts, opacity, active)

    def coverage(self) -> torch.Tensor:
        """Mean pigment per env, (B,) — the ground truth for how much is on the sheet."""
        return self.field.mean(dim=(1, 2))

    def composite_rgba(self, base_rgb01: torch.Tensor) -> torch.Tensor:
        """Ink over paper, (B, rows, cols, 4) uint8, ready to upload.

        Multiplicative over the sheet rather than a hard overwrite, so the
        printed ruling stays visible under thin ink the way real pen over
        paper does — and so a laser thinning ink reveals the ruling again
        instead of leaving a bleached patch.
        """
        f = self.field.unsqueeze(-1)
        rgb = base_rgb01 * (1.0 - f) + self.ink_rgb[:, None, None, :] * f
        rgba = torch.cat([rgb, torch.ones_like(f)], dim=-1)
        # round, not truncate: a cast would bias every channel down by up to
        # 1/255, which on bare paper is a visible tint shift against the
        # file-loaded sheet it replaces
        return (rgba.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
