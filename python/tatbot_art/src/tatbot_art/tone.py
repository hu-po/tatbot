"""DBV3's tone response: which source grey draws which coverage, measured with a step wedge.

The hatch PFM is Streamlines Flow Field with its field flattened (amplitude and frequencies at DBV3's
floor of 0.001), so its streamlines are parallel at one angle, spaced between Min and Max Spacing by the
source's luminance, and end where the source turns white (Threshold 250). Sketch Lines was measured first
and rejected: with a fixed angle its squiggles run straight on out of their regions for 10-20 mm.

A wedge of grey stripes is acquired, its paths drawn at the pen width, and the covered fraction of each
stripe read back. Near black DBV3 only piles on line (429 mm/cm² at grey 0 against 172 at grey 80 for
the same full cover), so a solid fill uses the curve's `solid_grey` rather than black.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .raster import add_stroke

WEDGE_GREYS = (0, 40, 80, 120, 160, 200, 230)
# The flattened Streamlines Flow Field of `acquire.settings("hatch")` at a 0.6 mm pen, measured on the
# Premium 1.6.22 worker on 2026-10-30 (grey 230 read 0.300, taken as the 200 stripe's 0.275: monotone).
STREAMLINES_06 = {"pfm": "Streamlines Flow Field", "pen_width_mm": 0.6, "solid_grey": 70,
                  "points": [[0, 1.0], [40, 0.999], [80, 0.917], [120, 0.568], [160, 0.403], [200, 0.275],
                             [230, 0.275], [255, 0.0]],
                  "source": "step wedge, 7 stripes of 10 x 30 mm at 10 px/mm, paths drawn at the pen width"}


@dataclass(frozen=True)
class ToneCurve:
    pfm: str
    pen_width_mm: float
    points: tuple[tuple[float, float], ...]
    solid_grey: int = 70
    source: str = ""
    extra: dict = field(default_factory=dict)

    def coverage(self, grey) -> np.ndarray:
        g, c = np.array(self.points).T
        return np.interp(grey, g, c)

    def grey(self, coverage) -> np.ndarray:
        """The source grey for a target coverage; full coverage takes `solid_grey`, none stays white."""
        g, c = np.array([p for p in self.points if p[0] >= self.solid_grey]).T
        c = np.minimum.accumulate(c)  # coverage must not rise with grey for the inverse to exist
        cov = np.asarray(coverage, float)
        out = np.interp(-cov, -c, g)
        out = np.where(cov >= float(self.coverage(self.solid_grey)), self.solid_grey, out)
        return np.where(cov <= 0, 255, np.clip(np.rint(out), 0, 255)).astype(np.uint8)

    def as_dict(self) -> dict:
        return {"pfm": self.pfm, "pen_width_mm": self.pen_width_mm, "solid_grey": self.solid_grey,
                "points": [list(p) for p in self.points], "source": self.source, **self.extra}


def load(path: Path | str | None = None) -> ToneCurve:
    data = json.loads(Path(path).read_text()) if path else STREAMLINES_06
    return ToneCurve(data["pfm"], float(data["pen_width_mm"]), tuple(tuple(p) for p in data["points"]),
                     int(data.get("solid_grey", 70)), data.get("source", ""))


def wedge(px_per_mm: float = 10.0) -> tuple[np.ndarray, dict]:
    """The wedge source (RGB uint8) and the layout `measure` reads back."""
    stripe_w, stripe_h, gap = int(10 * px_per_mm), int(30 * px_per_mm), int(px_per_mm)
    img = np.full((stripe_h + 2 * gap, len(WEDGE_GREYS) * (stripe_w + gap) + gap, 3), 255, np.uint8)
    for k, g in enumerate(WEDGE_GREYS):
        x0 = gap + k * (stripe_w + gap)
        img[gap:gap + stripe_h, x0:x0 + stripe_w] = g
    return img, {"greys": list(WEDGE_GREYS), "px_per_mm": px_per_mm, "stripe_px": [stripe_w, stripe_h], "gap_px": gap}


def measure(artwork: dict, layout: dict, pen_width_mm: float) -> list[list[float]]:
    """[grey, covered fraction] per stripe of an acquired wedge, 1 mm inside each stripe's edge."""
    program = artwork["program"]
    px = layout["px_per_mm"]
    height_mm = program["canvas_m"]["height"] * 1000
    shape = (int(round(height_mm * px)), int(round(program["canvas_m"]["width"] * 1000 * px)))
    cov = np.zeros(shape)
    for layer in program["layers"]:
        for el in layer["elements"]:
            p = np.asarray(el["points_m"], float) * 1000
            add_stroke(cov, np.column_stack([p[:, 0] * px, (height_mm - p[:, 1]) * px]), pen_width_mm * px)
    (sw, sh), gap, inset = layout["stripe_px"], layout["gap_px"], int(px)
    out = []
    for k, g in enumerate(layout["greys"]):
        x0 = gap + k * (sw + gap)
        out.append([g, round(float(cov[gap + inset:gap + sh - inset, x0 + inset:x0 + sw - inset].mean()), 3)])
    return out + [[255, 0.0]]
