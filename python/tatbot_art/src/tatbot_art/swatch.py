"""The achievable-palette sheet: every swatch hatched as it would be drawn, over its predicted far colour.

It is the picture of what a palette load can make: the reference image a generator is given, and the plan
for the swatch card that later measures it.
"""
from __future__ import annotations

import numpy as np

from .color import linear_to_srgb, oklch
from .gamut import Swatch
from .raster import add_stroke, composite

CELL_MM, GAP_MM = 12.0, 2.0


def hatch_lines(size_px: int, angle_deg: float, pitch_px: float) -> list[np.ndarray]:
    """Parallel lines across a square cell at an angle (degrees, counter-clockwise from +x, y up)."""
    theta = np.radians(angle_deg)
    direction = np.array([np.cos(theta), -np.sin(theta)])  # image rows grow downward
    normal = np.array([-direction[1], direction[0]])
    centre, reach = np.array([size_px / 2, size_px / 2]), size_px
    offsets = np.arange(-reach, reach + pitch_px, pitch_px)
    return [np.array([centre + o * normal - reach * direction, centre + o * normal + reach * direction])
            for o in offsets]


def order(swatches: list[Swatch]) -> list[Swatch]:
    """Chromatic swatches by hue then lightness, near-greys last by lightness."""
    def key(sw):
        lightness, chroma, hue = oklch(sw.srgb)
        return (1, -lightness, 0.0) if chroma < 0.03 else (0, round(hue / 20), -lightness)
    return sorted(swatches, key=key)


def sheet(swatches: list[Swatch], looks: dict, substrate_srgb, px_per_mm: float = 10.0,
          columns: int = 12) -> np.ndarray:
    cell, gap = int(CELL_MM * px_per_mm), int(GAP_MM * px_per_mm)
    rows = max(1, -(-len(swatches) // columns))
    img = np.ones((rows * (cell + gap) + gap, columns * (cell + gap) + gap, 3)) * np.asarray(substrate_srgb)
    hatched = int(cell * 2 / 3)
    for k, sw in enumerate(order(swatches)):
        layers = []
        for lay in sw.layers:
            cov = np.zeros((hatched, cell))
            width_px = lay.pitch_mm * lay.coverage * px_per_mm
            for line in hatch_lines(max(hatched, cell), lay.angle_deg, lay.pitch_mm * px_per_mm):
                add_stroke(cov, line, width_px)
            layers.append((looks[lay.ink_id], cov))
        r, c = gap + (k // columns) * (cell + gap), gap + (k % columns) * (cell + gap)
        img[r:r + hatched, c:c + cell] = linear_to_srgb(composite(substrate_srgb, layers)) if layers else substrate_srgb
        img[r + hatched:r + cell, c:c + cell] = sw.srgb
    return img
