"""What drawing a cell changed: its after raster aligned to its before, matched in brightness, and differenced.

The difference cancels the print: a cell sits over markers (2026-11-08: the dense tag tiles stay), and ink is
read as what changed on them, not as what is dark. The rasters come aligned (frames.pair); a per-channel gain and
offset fitted on the skin outside the strokes matches the light; then, per pixel:
- `dl`: L* the cell lost (darkening, positive where ink went in), in CIELAB from the 8-bit colour;
- `da`, `db`: a* and b* it gained;
- `c`: the share of light the ink takes, 1 - after / before in linear luminance, which survives blur.
Hue is only meaningful where the before raster is light: on the violet print red reads as darkening, not as red
(2026-11-05), so `light` marks where the print was not.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KEEP_OUT_MM = 1.5    # the strokes' box grown by this is left out of alignment and brightness matching
LIGHT_L = 60.0       # before-raster L* above this is unprinted skin


@dataclass
class Diff:
    dl: np.ndarray
    da: np.ndarray
    db: np.ndarray
    c: np.ndarray
    valid: np.ndarray
    outside: np.ndarray
    light: np.ndarray
    noise: float     # sd of dl outside the strokes: the read's own floor


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _linear_y(bgr: np.ndarray) -> np.ndarray:
    v = np.clip(bgr, 0, 255) / 255.0
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def difference(before: np.ndarray, after: np.ndarray, valid: np.ndarray, keep_out: np.ndarray) -> Diff:
    """`before` and `after`: aligned float32 BGR rasters on one Frame (frames.pair, fuse); `valid`: where both are;
    `keep_out`: the strokes' neighbourhood, left out of the brightness match."""
    aligned = after
    outside = valid & ~keep_out
    if outside.sum() < 100:
        raise ValueError("too little skin outside the strokes to align and match brightness on")
    matched = np.empty_like(aligned)
    for k in range(3):
        gain, offset = np.polyfit(aligned[..., k][outside], before[..., k][outside], 1)
        matched[..., k] = aligned[..., k] * gain + offset
    la, lb = _lab(before), _lab(matched)
    dl = la[..., 0] - lb[..., 0]
    ya, yb = _linear_y(before), _linear_y(matched)
    c = np.where(ya > 1e-3, 1.0 - yb / np.maximum(ya, 1e-3), 0.0)
    return Diff(dl=dl, da=lb[..., 1] - la[..., 1], db=lb[..., 2] - la[..., 2], c=c, valid=valid, outside=outside,
                light=la[..., 0] > LIGHT_L, noise=float(dl[outside].std()))
