"""Strokes on the phantom, and the arm's chart they are drawn in.

A stroke (``DrawnLine``) is a path on the skin in the phantom frame, with its
normals and arclength: the thing the expert traces. Strokes are not drawn
here -- they are thinned out of the ink painted on the forearm (``ink.py``)
-- but Sharpie lines are laid out here, as curves in the arm's chart (x along
the arm, angle around it; see ``shell.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tatbot_travel.phantom import Phantom

if TYPE_CHECKING:
    from tatbot_travel.shell import SkinShell


@dataclass(frozen=True)
class DrawnLine:
    chart: np.ndarray  # (N, 2) x (m), angle (rad; 0 = back of the forearm)
    points: np.ndarray  # (N, 3) on the skin, phantom frame
    normals: np.ndarray  # (N, 3) outward unit normals
    arclength: np.ndarray  # (N,) metres from the first point
    width_m: float
    colour: str
    face_indices: np.ndarray | None = None
    barycentric: np.ndarray | None = None
    surface: SkinShell | None = None

    @property
    def length(self) -> float:
        return float(self.arclength[-1])

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point, normal and unit tangent at arclength ``s`` (clamped to the ends)."""
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.arclength, s) - 1, 0, len(self.arclength) - 2))
        t = (s - self.arclength[i]) / max(self.arclength[i + 1] - self.arclength[i], 1e-9)
        point = (1 - t) * self.points[i] + t * self.points[i + 1]
        normal = (1 - t) * self.normals[i] + t * self.normals[i + 1]
        if self.surface is not None:
            chart = (1 - t) * self.chart[i] + t * self.chart[i + 1]
            point, normal = self.surface.lift(*chart)
        tangent = self.points[i + 1] - self.points[i]
        return point, normal / np.linalg.norm(normal), tangent / max(np.linalg.norm(tangent), 1e-9)


def sample_chart_curve(rng: np.random.Generator, phantom: Phantom, *, length_range=(0.10, 0.24),
                       off_top_prob=0.10, samples: int = 400) -> np.ndarray:
    """A smooth curve on the forearm: straight, wavy or diagonal, mostly along the back."""
    lo, hi = phantom.forearm
    length = min(rng.uniform(*length_range), (hi - lo) - 0.02)
    # A line as long as a small forearm leaves no room to slide; rounding can make the room negative.
    x0 = rng.uniform(lo + 0.01, max(lo + 0.01, hi - 0.01 - length))
    x = np.linspace(x0, x0 + length, samples)
    theta0 = rng.uniform(-np.pi * 0.8, np.pi * 0.8) if rng.random() < off_top_prob else rng.normal(0, 0.3)
    shape = rng.choice(["straight", "wavy", "diagonal"], p=[0.3, 0.45, 0.25])
    theta = np.full_like(x, theta0)
    t = (x - x0) / length
    if shape in ("wavy", "diagonal") and rng.random() < 0.8:
        wavelength = rng.uniform(0.05, 0.2)
        theta += rng.uniform(0.08, 0.5) * np.sin(2 * np.pi * (x - x0) / wavelength + rng.uniform(0, 2 * np.pi))
    if shape == "diagonal":
        theta += rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * (t - 0.5)
    return np.stack([x, np.clip(theta, -2.6, 2.6)], axis=1)
