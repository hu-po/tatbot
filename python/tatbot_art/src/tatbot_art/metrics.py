"""How far a drawing's predicted look is from the colours it was meant to have, as seen from a distance.

Per-pixel ΔE misjudges hatching: up close a tint is lines and paper. Following S-CIELAB's idea (Zhang and
Wandell), both images are blurred in CIELAB before CIEDE2000, lightness less than the two opponent
channels, with widths that scale with viewing distance. The widths here are an approximation stated in
millimetres at 300 mm (lightness 0.5 mm, a and b 1.5 mm), not S-CIELAB's fitted filters.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .color import cielab, delta_e00

SIGMA_MM_AT_300 = (0.5, 1.5, 1.5)


def blurred_delta_e(a_srgb: np.ndarray, b_srgb: np.ndarray, px_per_mm: float, view_mm: float = 300.0,
                    where: np.ndarray | None = None) -> dict:
    if a_srgb.shape != b_srgb.shape:
        raise ValueError(f"images differ in shape: {a_srgb.shape} and {b_srgb.shape}")
    scale = view_mm / 300.0 * px_per_mm
    labs = []
    for img in (a_srgb, b_srgb):
        lab = cielab(img)
        labs.append(np.stack([gaussian_filter(lab[..., c], s * scale) for c, s in enumerate(SIGMA_MM_AT_300)], axis=-1))
    de = delta_e00(*labs)
    sel = de[where] if where is not None and where.any() else de.ravel()
    return {"view_mm": view_mm, "mean": round(float(sel.mean()), 2), "p90": round(float(np.percentile(sel, 90)), 2)}
