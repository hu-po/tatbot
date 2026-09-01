"""Sensor-realism corruption for the wrist cameras — a first-class DR axis.

Depth: the D405 model below. RGB: :class:`RGBJitter` at the bottom — real
wrist cameras auto-expose and white-balance against the scene while sim
renders one fixed response, so each env's RGB gets a per-episode exposure
gain, per-channel white-balance gains, and a gamma, applied on the GPU.

Sim depth is a perfect z-buffer. The D405 is a short-range stereo camera: it
returns per-pixel noise that grows with range, holes where matching fails
(specular spots, low texture, grazing angles), and zeros for "no measurement".
A policy trained mostly on clean sim depth would lean on precision the real
sensor never delivers, so corruption is randomized per episode exactly like
lighting: every environment sees a differently-bad sensor.

The effects, all vectorized on the GPU alongside rendering:

- **Range-dependent noise, split by spatial scale.** Stereo depth error grows
  ~quadratically with distance; ``sigma_at_ref_mm`` is the TOTAL per-pixel
  temporal noise at ``ref_mm`` (~150 mm working distance), scaled by
  ``(z/ref)^2``. Its variance is split between an i.i.d. component and a
  spatially CORRELATED speckle field (coarse per-frame noise, bilinearly
  upsampled): real stereo error is correlated across neighbouring pixels, and
  purely white noise is something a network can average away in a way the
  real sensor never allows.
- **Static calibration warp.** A per-episode low-frequency bias field, also
  scaled by ``(z/ref)^2`` — the slowly-varying surface warp a miscalibrated
  stereo pair bakes into every frame (the probe's plane-fit residual exceeds
  its temporal noise; that gap is this term).
- **Edge dropout with width.** Stereo fails in BANDS at depth
  discontinuities, not on one-pixel lines: the gradient mask (computed on
  clean depth) is dilated by a per-episode 1-5 px kernel before dropping.
- **Blob dropout, mostly persistent, and shaped by the scene.** Texture-poor
  patches lose whole regions: a coarse random field is upsampled to the frame
  and thresholded, so dropouts arrive as blobs rather than salt noise. Most of
  the budget is a STATIC per-episode pattern with a smaller per-frame flicker
  component — real holes sit still (the probe had always-valid pixels across
  150 frames), and holes that teleport every frame are a different, easier
  nuisance. Two properties matter as much as the fraction:

  *Smooth boundaries.* The field is upsampled BILINEARLY and thresholded, so
  a hole's edge is a level set with curvature. Nearest upsampling of a coarse
  grid — what this did until the 2026-08-26 comparison against the bench —
  draws axis-aligned rectangles on a lattice, which reads as synthetic at a
  glance however well the dropped fraction matches.

  *Content correlation.* The score is pushed down where the surface turns away
  from the camera (depth gradient) and where it is far, both of which cost a
  stereo match, so holes gather on the pen barrel and the flanks the way the
  real ones do instead of landing on near, flat, well-lit skin. The threshold
  is each env's own order statistic, so the dropped fraction stays exactly
  what was sampled and the probe calibration survives the weighting.

A fourth effect the phase 0 probe demanded: **minimum-range cutoff.** The
D405 cannot measure closer than ~70 mm, so on the real rig the pen shaft and
fingers in the bottom of the frame come back as 0 — while sim's z-buffer
happily returns valid depth there. Everything nearer than a per-episode
min-z is dropped, matching the real sensor's blind zone.

Depth stays integer millimetres with 0 = invalid, the same contract the real
camera and the writer use. The default ranges are centred on the 2026-08-21
phase 0 probe (pad at the pen tip, ~156-159 mm): per-pixel temporal noise
3.0-3.8 mm median (p95 5.3-6.4, includes servo micro-motion, so the low end
of the range dips below), ~21-27% of the ROI invalid — much of it the pen's
blind zone, which the min-z cutoff reproduces geometrically — and 65-76% of
pixels landing in the paper band.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812  (torch's own universal spelling)


@dataclass
class DepthNoiseConfig:
    """Per-episode randomization ranges; each env samples its own values."""

    sigma_at_ref_mm: tuple[float, float] = (1.0, 4.5)
    ref_mm: float = 155.0
    # fraction of the noise VARIANCE that is spatially correlated speckle
    corr_frac: tuple[float, float] = (0.4, 0.8)
    speckle_cells: int = 48  # correlated-noise resolution (~13 px patches)
    warp_mm: tuple[float, float] = (0.0, 2.5)  # static calibration-warp amplitude
    warp_cells: int = 6
    min_z_mm: tuple[float, float] = (60.0, 90.0)  # D405 blind zone starts ~70 mm
    edge_grad_mm: float = 6.0
    edge_drop_prob: tuple[float, float] = (0.2, 0.8)
    edge_dilate_px: tuple[int, int] = (1, 5)  # occlusion-band width (rounded to odd)
    blob_drop_frac: tuple[float, float] = (0.08, 0.30)
    blob_cells: int = 24  # dropout-grid resolution; coarser = larger holes
    blob_static_frac: float = 0.7  # share of the blob budget that stays put all episode
    # How strongly a hole prefers ground the sensor finds hard. Stereo fails on
    # surfaces turned away from it and on anything far, so real dropouts sit in
    # patches that follow the scene; weight 0 puts them anywhere and reproduces
    # the content-blind behaviour these were built with.
    blob_grazing_weight: tuple[float, float] = (0.3, 1.2)
    blob_range_weight: tuple[float, float] = (0.1, 0.7)
    # slope, as a depth gradient in mm per pixel, at which a surface counts as
    # fully turned away — past this the grazing term saturates
    grazing_ref_mm_px: float = 4.0


class DepthCorruptor:
    """Samples one sensor-quality profile per env at reset, applies it per frame."""

    def __init__(self, num_envs: int, device: torch.device, cfg: DepthNoiseConfig | None = None,
                 seed: int | None = None):
        self.cfg = cfg or DepthNoiseConfig()
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.reset()

    def _uniform(self, lo: float, hi: float) -> torch.Tensor:
        u = torch.rand(self.num_envs, device=self.gen.device,
                       generator=self.gen).to(self.device)
        return lo + u * (hi - lo)

    def reset(self):
        """Draw a fresh sensor profile for every environment."""
        c = self.cfg
        self.sigma = self._uniform(*c.sigma_at_ref_mm).view(-1, 1, 1, 1)
        corr = self._uniform(*c.corr_frac).view(-1, 1, 1, 1)
        # bilinear upsampling averages the coarse field, shrinking its
        # per-pixel variance to ~0.44x (E[sum w_i^2] for random offsets);
        # 1.5x restores the sampled total so the probe calibration holds
        self.sigma_corr = self.sigma * corr.sqrt() * 1.5
        self.sigma_iid = self.sigma * (1 - corr).sqrt()
        self.warp_amp = self._uniform(*c.warp_mm).view(-1, 1, 1, 1)
        self.min_z = self._uniform(*c.min_z_mm).view(-1, 1, 1, 1)
        self.edge_p = self._uniform(*c.edge_drop_prob).view(-1, 1, 1, 1)
        # one kernel per batch (shared across envs — per-env widths would
        # need grouped pooling for little gain). Must be ODD: max_pool2d
        # with even k and padding k//2 emits W+1 columns and the next
        # tensor op crashes (found the hard way — even draws made runs die
        # ~1 batch in, odd draws sailed through)
        lo, hi = c.edge_dilate_px
        self.edge_k = 1 + 2 * int(torch.randint(lo // 2, hi // 2 + 1, (1,),
                                                generator=self.gen,
                                                device=self.gen.device).item())
        self.blob_f = self._uniform(*c.blob_drop_frac).view(-1, 1, 1, 1)
        self.graze_w = self._uniform(*c.blob_grazing_weight).view(-1, 1, 1, 1)
        self.range_w = self._uniform(*c.blob_range_weight).view(-1, 1, 1, 1)
        self._blob_static = torch.randn(
            (self.num_envs, 1, c.blob_cells, c.blob_cells),
            device=self.gen.device, generator=self.gen).to(self.device)
        self._warp_field = None  # lazily built at first frame (needs H, W)

    def _blobs(self, d0: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor,
               shape: tuple[int, int, int]) -> torch.Tensor:
        """Where texture-poor patches lose their measurement.

        Two things make these look like a stereo camera's holes rather than a
        mask. First they are SMOOTH: a coarse field is upsampled bilinearly and
        then thresholded, so the boundary is a level set with curvature.
        Thresholding a coarse field with nearest upsampling instead -- which is
        what this did -- draws axis-aligned rectangles on a lattice, and that
        reads as synthetic at a glance even when the dropped fraction is right.

        Second they follow the SCENE: the score is pushed down where the
        surface turns away from the camera and where it is far, both of which
        cost a stereo match. Locations were content-random before, which put
        holes in the middle of the near, flat, well-lit skin as readily as on
        the pen barrel.

        The threshold is a per-env quantile of the score, so the dropped
        fraction stays exactly what was sampled however the weighting shifts
        the distribution -- the calibration against the probe survives.
        """
        c = self.cfg
        b, h, w = shape
        # coarse field, half static for the episode and half fresh per frame:
        # real holes mostly sit still, and ones that teleport every frame are
        # an easier nuisance than the sensor actually presents
        flicker = self._sample("randn", (b, 1, c.blob_cells, c.blob_cells))
        s = c.blob_static_frac
        coarse = self._blob_static * s + flicker * (1.0 - s)
        field = F.interpolate(coarse, size=(h, w), mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)
        # normalise: bilinear upsampling shrinks the spread, and mixing two
        # fields shrinks it again, so the raw scale is not comparable per env
        field = (field - field.mean((1, 2, 3), keepdim=True)) / \
            field.std((1, 2, 3), keepdim=True).clamp_min(1e-6)

        # the scene's contribution: slope as seen by the camera, and range
        graze = (torch.maximum(gx, gy) / c.grazing_ref_mm_px).clamp(0, 1)
        # quadratic, like the noise term above it: a match gets harder with
        # range the same way its error grows, so a linear cost understates how
        # much worse the far half of a frame is
        far = ((d0.to(torch.float32) / (2 * c.ref_mm)) ** 2).clamp(0, 1)
        score = field - self.graze_w * graze - self.range_w * far

        # exact fraction: cut at each env's own order statistic. Sub-sampled,
        # because a full-frame sort per env buys a cut point that moves by far
        # less than one pixel's worth of area. Each env has its own target
        # fraction, which is why this is a sort-and-gather rather than
        # torch.quantile with a scalar q.
        flat = score.reshape(b, -1)[:, ::7].sort(dim=1).values
        k = (self.blob_f.reshape(b, 1) * (flat.shape[1] - 1)).round().long()
        thr = flat.gather(1, k).reshape(b, 1, 1, 1)
        return score < thr


    def _follow(self, device: torch.device) -> None:
        """Move per-batch state to the frames' device (a cpu sim can still
        render on the GPU). The generator stays put so seeds stay honest;
        on the normal path devices already match and this is a no-op."""
        if device == self.device:
            return
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.device = device

    def _sample(self, kind, shape):
        fn = torch.randn if kind == "randn" else torch.rand
        return fn(shape, device=self.gen.device, generator=self.gen).to(self.device)

    def __call__(self, depth_mm: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) integer millimetres -> corrupted uint16-range int32."""
        self._follow(depth_mm.device)
        c = self.cfg
        b, h, w, _ = depth_mm.shape
        d = depth_mm.to(torch.float32)
        # blind zone: the stereo baseline cannot resolve anything this close,
        # so the pen and fingers in frame come back invalid on the real D405
        invalid = (d <= 0) | (d < self.min_z)

        # Edge mask BEFORE noise: on the noised map every neighbour pair
        # differs by ~sqrt(2)*sigma and the whole frame reads as "edges" —
        # measured as ~25% spurious dropout before this ordering was fixed.
        # Real discontinuities live in the clean geometry. Dilation turns the
        # one-pixel line into the multi-pixel band real occlusions produce.
        gx = torch.zeros_like(d)
        gy = torch.zeros_like(d)
        gx[:, :, 1:, :] = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs()
        gy[:, 1:, :, :] = (d[:, 1:, :, :] - d[:, :-1, :, :]).abs()
        edge = torch.maximum(gx, gy) > c.edge_grad_mm
        if self.edge_k > 1:
            edge = F.max_pool2d(edge.permute(0, 3, 1, 2).float(), self.edge_k,
                                stride=1, padding=self.edge_k // 2)
            edge = edge.permute(0, 2, 3, 1) > 0.5

        range_gain = (d / c.ref_mm) ** 2
        # static calibration warp: one low-frequency field per episode
        if self._warp_field is None or self._warp_field.shape[1:3] != (h, w):
            f = self._sample("randn", (b, 1, c.warp_cells, c.warp_cells))
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            self._warp_field = f.permute(0, 2, 3, 1)
        d = d + self._warp_field * self.warp_amp * range_gain
        # correlated speckle (fresh each frame) + iid residue
        sp = self._sample("randn", (b, 1, c.speckle_cells, c.speckle_cells))
        sp = F.interpolate(sp, size=(h, w), mode="bilinear", align_corners=False)
        d = d + sp.permute(0, 2, 3, 1) * self.sigma_corr * range_gain
        d = d + self._sample("randn", d.shape) \
                * self.sigma_iid * range_gain
        u = self._sample("rand", d.shape)
        drop = edge & (u < self.edge_p)

        drop = drop | self._blobs(d0=depth_mm, gx=gx, gy=gy, shape=(b, h, w)) | invalid

        d = d.round().clamp(0, 65535)
        d[drop] = 0
        return d.to(torch.int32)


@dataclass
class RGBJitterConfig:
    """Per-episode camera-response jitter for the wrist RGB streams."""

    enabled: bool = True
    exposure: tuple[float, float] = (0.82, 1.22)
    white_balance: tuple[float, float] = (0.93, 1.07)  # per-channel gains
    gamma: tuple[float, float] = (0.88, 1.15)
    # sensor grain, fresh each frame: sim renders are noiseless while a real
    # sensor at indoor light is not (std in [0,1] units; ~2/255 at the top)
    noise_std: tuple[float, float] = (0.0, 0.008)


class RGBJitter:
    """Samples one camera response per env at reset, applies it per frame.

    rgb' = clip(rgb/255 * exposure * wb_c, 0, 1) ** gamma — cheap tensor
    math on the GPU before the device->host transfer.
    """

    def __init__(self, num_envs: int, device: torch.device,
                 cfg: RGBJitterConfig | None = None, seed: int | None = None):
        self.cfg = cfg or RGBJitterConfig()
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.reset()

    def _u(self, lo, hi, shape):
        u = torch.rand(shape, device=self.gen.device, generator=self.gen)
        return lo + u.to(self.device) * (hi - lo)

    def reset(self):
        c = self.cfg
        b = self.num_envs
        self.exposure = self._u(*c.exposure, (b, 1, 1, 1))
        self.wb = self._u(*c.white_balance, (b, 1, 1, 3))
        self.gamma = self._u(*c.gamma, (b, 1, 1, 1))
        self.noise = self._u(*c.noise_std, (b, 1, 1, 1))

    def __call__(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 -> jittered uint8."""
        if not self.cfg.enabled:
            return rgb_uint8
        # The renderer's device can differ from the sim backend's (a cpu sim
        # still renders on the GPU); follow the frames, keep the generator on
        # its own device so sampling stays deterministic per seed.
        if rgb_uint8.device != self.device:
            for name, value in list(vars(self).items()):
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to(rgb_uint8.device))
            self.device = rgb_uint8.device
        f = rgb_uint8.to(torch.float32) / 255.0
        f = (f * self.exposure * self.wb).clamp(0, 1) ** self.gamma
        # grain after the response curve — readout noise sits on the output
        grain = torch.randn(f.shape, device=self.gen.device, generator=self.gen)
        f = f + grain.to(f.device) * self.noise
        return (f.clamp(0, 1) * 255.0).round().to(torch.uint8)
