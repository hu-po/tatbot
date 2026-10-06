"""Procedural flash: random tattoo designs, so the policy meets ink it cannot memorise.

The practice arm's tattoos are not settled, and whatever it carries should
look like ink to the policy rather than like one of a dozen designs it has
seen. Each design here is composed from one to four primitives -- regular
polygons and stars, smooth blobs, hearts, rose curves, spirals, waves, vines,
concentric rings, radial motifs, and lettering in the Hershey fonts -- each
outlined, filled or hatched at a random pen width, then rasterised at the
ink module's millimetre scale and cropped to the ink.

Geometry is built in a unit square ([-1, 1] on both axes) and mapped to the
raster at the end, so a primitive never needs to know the design's size.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Path = tuple[np.ndarray, bool]  # (N, 2) points in the unit square, closed?

WORDS = ("LOVE", "MOM", "HOPE", "FAITH", "WILD", "FREE", "TATBOT", "ROBOT", "ATX", "2026", "1999", "XO",
         "amor", "dream", "wave", "sun", "moon", "home", "vida", "fate")
FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX)


def _polar(radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


# ---- primitives: each returns paths in the unit square -------------------------------------------
def polygon(rng: np.random.Generator) -> list[Path]:
    """A regular polygon, or a star when the inner radius drops."""
    n = int(rng.integers(3, 9))
    star = rng.random() < 0.45
    k = 2 * n if star else n
    radius = np.where(np.arange(k) % 2 == 1, rng.uniform(0.35, 0.6), 1.0) if star else np.ones(k)
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 2 * np.pi)
    return [(_polar(radius, theta), True)]


def blob(rng: np.random.Generator) -> list[Path]:
    """A smooth closed curve: a circle with a few random harmonics."""
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    radius = np.ones_like(theta)
    for k in range(2, int(rng.integers(3, 7))):
        radius += rng.uniform(0, 0.35 / k) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    squash = np.array([1.0, rng.uniform(0.5, 1.0)])
    return [(_polar(radius / radius.max(), theta) * squash, True)]


def heart(rng: np.random.Generator) -> list[Path]:
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return [(np.stack([x, y], axis=1) / 17.0, True)]


def rose(rng: np.random.Generator) -> list[Path]:
    """Petals: the polar rose r = cos(k theta), with a centre disc."""
    k = int(rng.integers(2, 8))
    theta = np.linspace(0, 2 * np.pi * (1 if k % 2 else 2), 720, endpoint=False)
    petals = _polar(np.abs(np.cos(k * theta)) ** rng.uniform(0.5, 1.0), theta)
    centre = _polar(np.full(60, rng.uniform(0.1, 0.25)), np.linspace(0, 2 * np.pi, 60, endpoint=False))
    return [(petals, True), (centre, True)]


def spiral(rng: np.random.Generator) -> list[Path]:
    turns = rng.uniform(1.5, 4.0)
    theta = np.linspace(0, 2 * np.pi * turns, int(200 * turns))
    return [(_polar(theta / theta[-1], theta * rng.choice([-1, 1])), False)]


def wave(rng: np.random.Generator) -> list[Path]:
    """A sine, a zigzag or a scalloped line across the square; sometimes several."""
    x = np.linspace(-1, 1, 300)
    phase = x * np.pi * rng.uniform(1.5, 5.0)
    shape = rng.integers(3)
    if shape == 0:
        y = np.sin(phase)
    elif shape == 1:
        y = (2 / np.pi) * np.arcsin(np.sin(phase))  # zigzag
    else:
        y = -np.abs(np.sin(phase))  # scallops
    rows = int(rng.integers(1, 4))
    offsets = np.linspace(-0.5, 0.5, rows) if rows > 1 else [0.0]
    return [(np.stack([x, 0.25 * y + dy], axis=1), False) for dy in offsets]


def vine(rng: np.random.Generator) -> list[Path]:
    """A meandering stem with leaves along it."""
    n = 200
    heading = np.cumsum(rng.normal(0, 0.06, n)) + rng.uniform(0, 2 * np.pi)
    stem = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], axis=1), axis=0)
    stem = (stem - stem.mean(axis=0)) / np.abs(stem - stem.mean(axis=0)).max()
    paths: list[Path] = [(stem, False)]
    leaf = np.linspace(0, np.pi, 30)
    for i in rng.choice(np.arange(10, n - 10), size=int(rng.integers(2, 8)), replace=False):
        d = stem[i + 1] - stem[i - 1]
        angle = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
        outline = np.concatenate([np.stack([leaf, 0.35 * np.sin(leaf)], 1),
                                  np.stack([leaf[::-1], -0.35 * np.sin(leaf[::-1])], 1)])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        paths.append((stem[i] + (outline / np.pi * rng.uniform(0.12, 0.25)) @ rot.T, True))
    return paths


def rings(rng: np.random.Generator) -> list[Path]:
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    radii = np.sort(rng.uniform(0.25, 1.0, int(rng.integers(2, 5))))
    return [(_polar(np.full_like(theta, r), theta), True) for r in radii]


def radial(rng: np.random.Generator) -> list[Path]:
    """A motif repeated around a centre: a small mandala."""
    motif = rng.choice([polygon, blob, heart, wave])(rng)
    count = int(rng.integers(4, 13))
    scale, reach = rng.uniform(0.15, 0.3), rng.uniform(0.45, 0.75)
    paths: list[Path] = []
    for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        paths += [((pts * scale + [reach, 0.0]) @ rot.T, closed) for pts, closed in motif]
    return paths


PRIMITIVES: tuple[Callable[[np.random.Generator], list[Path]], ...] = (
    polygon, blob, heart, rose, spiral, wave, vine, rings, radial)


# ---- drawing --------------------------------------------------------------------------------------
class _Canvas:
    """A square raster of ink coverage, drawn in unit-square coordinates."""

    def __init__(self, size_mm: float, mm_per_px: float):
        self.mm_per_px = mm_per_px
        self.n = max(16, int(round(size_mm / mm_per_px)))
        self.alpha = np.zeros((self.n, self.n), np.float32)

    def _px(self, pts: np.ndarray) -> np.ndarray:
        return np.round((np.asarray(pts) * 0.47 + 0.5) * self.n * 16).astype(np.int32)  # 4 fractional bits

    def _thickness(self, pen_mm: float) -> int:
        return max(1, int(round(pen_mm / self.mm_per_px)))

    def outline(self, path: Path, pen_mm: float) -> None:
        cv2.polylines(self.alpha, [self._px(path[0])], path[1], 1.0, self._thickness(pen_mm), cv2.LINE_AA, 4)

    def fill(self, path: Path) -> None:
        cv2.fillPoly(self.alpha, [self._px(path[0])], 1.0, cv2.LINE_AA, 4)

    def hatch(self, path: Path, pen_mm: float, rng: np.random.Generator) -> None:
        """Parallel lines clipped to a closed path, then its outline."""
        mask = np.zeros_like(self.alpha)
        cv2.fillPoly(mask, [self._px(path[0])], 1.0, cv2.LINE_AA, 4)
        lines = np.zeros_like(self.alpha)
        spacing = self._thickness(pen_mm) * rng.uniform(2.5, 5.0)
        angle = rng.uniform(0, np.pi)
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]]) * self.n * 2
        for offset in np.arange(-self.n, self.n, spacing):
            a = np.array([self.n / 2, self.n / 2]) + u * offset
            p, q = (tuple(int(c) for c in np.round(end)) for end in (a - v, a + v))
            cv2.line(lines, p, q, 1.0, self._thickness(pen_mm * 0.7), cv2.LINE_AA)
        np.maximum(self.alpha, lines * mask, out=self.alpha)
        self.outline(path, pen_mm)

    def text(self, rng: np.random.Generator, pen_mm: float) -> None:
        """A word in a Hershey font, fitted to the square."""
        word = str(rng.choice(WORDS))
        font = int(rng.choice(FONTS)) | (cv2.FONT_ITALIC if rng.random() < 0.3 else 0)
        thickness = self._thickness(pen_mm)
        (w, h), base = cv2.getTextSize(word, font, 1.0, thickness)
        scale = 0.9 * self.n / max(w, 1)
        (w, h), base = cv2.getTextSize(word, font, scale, thickness)
        origin = (int((self.n - w) / 2), int((self.n + h) / 2))
        layer = np.zeros((self.n, self.n), np.uint8)  # putText draws only into 8-bit images
        cv2.putText(layer, word, origin, font, scale, 255, thickness, cv2.LINE_AA)
        np.maximum(self.alpha, layer.astype(np.float32) / 255.0, out=self.alpha)


def _placed(paths: list[Path], rng: np.random.Generator, spread: float) -> list[Path]:
    """Scale, rotate and shift a primitive's paths within the square."""
    scale = rng.uniform(0.35, 1.0) if spread else 1.0
    angle = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * scale
    shift = rng.uniform(-spread, spread, 2)
    if rng.random() < 0.2:  # mirrored copy: bilateral symmetry
        paths = paths + [(pts * [-1.0, 1.0], closed) for pts, closed in paths]
    return [(np.clip(pts @ rot.T + shift, -1.0, 1.0), closed) for pts, closed in paths]


def _draw(canvas: _Canvas, paths: list[Path], pen_mm: float, rng: np.random.Generator) -> None:
    style = rng.choice(["outline", "fill", "hatch"], p=[0.6, 0.25, 0.15])
    for path in paths:
        if style == "fill" and path[1]:
            canvas.fill(path)
        elif style == "hatch" and path[1]:
            canvas.hatch(path, pen_mm, rng)
        else:
            canvas.outline(path, pen_mm)


def procedural(rng: np.random.Generator, size_mm: float, mm_per_px: float) -> np.ndarray:
    """A random design's ink coverage (float32, ``mm_per_px``), cropped to the ink."""
    canvas = _Canvas(size_mm, mm_per_px)
    pen_mm = float(rng.uniform(0.4, 2.2))
    if rng.random() < 0.2:
        canvas.text(rng, pen_mm)
    count = int(rng.integers(1, 5)) if canvas.alpha.max() == 0 else int(rng.integers(0, 2))
    for i in range(count):
        primitive = PRIMITIVES[int(rng.integers(len(PRIMITIVES)))]
        _draw(canvas, _placed(primitive(rng), rng, spread=0.0 if i == 0 else 0.5), pen_mm, rng)
    if rng.random() < 0.3:  # healed ink: edges soften a little
        canvas.alpha = cv2.GaussianBlur(canvas.alpha, (0, 0), rng.uniform(0.5, 2.0))
    rows, cols = np.nonzero(canvas.alpha > 0.05)
    if len(rows) == 0:
        return canvas.alpha
    return np.ascontiguousarray(canvas.alpha[rows.min():rows.max() + 1, cols.min():cols.max() + 1])
