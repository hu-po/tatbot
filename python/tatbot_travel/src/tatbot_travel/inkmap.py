"""Ink on the practice forearm, found in a wrist view and lifted onto the measured surface.

Ink is what is darker than the skin's local colour or off its hue, against a median over a window
several line widths wide (after the ROS close-up gauge's local-median rule). Only pixels on the
forearm count -- the depth says which.
The mask thins to centrelines and splits into strokes between ends and junctions
(``ink._skeleton_graph``); each stroke's pixels go to the base frame through their own depth and
then onto the surface, resampled every millimetre.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tatbot_travel.ink import _skeleton_graph
from tatbot_travel.surface import Frame, Surface

STEP_M = 0.001  # stroke resampling
RIM_PX = 17  # silhouette band ignored for ink
JUMP_M = 0.003  # neighbouring centreline pixels further apart than this in 3D are a depth edge, not ink
MIN_FACING = 0.45  # cosine between the skin's normal and the view ray; below it the skin is seen edge-on


def ink_mask(rgb: np.ndarray, valid: np.ndarray, *, keep: np.ndarray | None = None, window_px: int = 31,
             darker: float = 20.0, chroma: float = 18.0, min_area_px: int = 30) -> np.ndarray:
    """Pixels darker than the local skin (Lab L) or off its hue (a*b* distance), inside ``valid``.

    The local skin is a median over ``window_px``, several line widths. On the pale silicone, black and
    blue ink drop L; red ink also moves a*b*. Shading across the curved arm moves neither much, which a
    plain colour distance does not separate (it reads the lit side of the forearm as ink). The local
    colour comes from all of ``valid``; ``keep`` only trims the answer -- filling holes in the background
    with one colour would draw edges around them that read as ink."""
    lab = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    if valid.any():
        lab[~valid] = np.median(lab[valid], axis=0)
    background = np.stack([cv2.medianBlur(np.clip(lab[..., i], 0, 255).astype(np.uint8), window_px | 1)
                           for i in range(3)], axis=-1).astype(np.float32)
    off_hue = np.linalg.norm(lab[..., 1:] - background[..., 1:], axis=2)
    mask = ((background[..., 0] - lab[..., 0] > darker) | (off_hue > chroma)) & valid
    if keep is not None:
        mask &= keep
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[labels]


@dataclass
class Stroke:
    """One ink stroke on the skin: points every millimetre, their surface normals, its pixels."""

    points: np.ndarray  # (N, 3) base frame
    normals: np.ndarray  # (N, 3)
    pixels: np.ndarray  # (M, 2) the source view's (row, col) centreline

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


def resample(path: np.ndarray, step_m: float = STEP_M) -> np.ndarray:
    """Points every ``step_m`` along a polyline (its ends kept)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step_m:
        return path[[0, -1]]
    t = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step_m)) + 1)
    return np.stack([np.interp(t, s, path[:, i]) for i in range(path.shape[1])], axis=1)


def smooth(path: np.ndarray, window: int = 5) -> np.ndarray:
    """A moving average that keeps both ends where they are (pixel staircases out)."""
    if len(path) <= window:
        return path
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.repeat(path[:1], pad, 0), path, np.repeat(path[-1:], pad, 0)])
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(path.shape[1])], axis=1)


def strokes(frame: Frame, surface: Surface, forearm: np.ndarray, *, min_length_m: float = 0.008,
            **mask_options) -> tuple[list[Stroke], np.ndarray]:
    """The view's ink strokes on the surface, longest first, and the ink mask they came from.

    ``forearm`` marks the pixels on the arm (the caller's depth segmentation); its rim is dropped: depth
    edges spill a few pixels past the silhouette, and the background there reads as dark ink."""
    interior = cv2.erode(forearm.astype(np.uint8), np.ones((RIM_PX, RIM_PX), np.uint8)).astype(bool)
    mask = ink_mask(frame.rgb, interior, keep=facing(frame), **mask_options)
    paths, _ = _skeleton_graph(mask)
    out = []
    for path in paths:
        for px, base in _continuous(frame, np.asarray(path)):
            on_skin, normals, supported = surface.project_path(resample(smooth(base), STEP_M))
            if supported.sum() < 3:
                continue
            stroke = Stroke(points=on_skin[supported], normals=normals[supported], pixels=px)
            if stroke.length_m >= min_length_m:
                out.append(stroke)
    return sorted(out, key=lambda s: -s.length_m), mask


def facing(frame: Frame, min_cos: float = MIN_FACING) -> np.ndarray:
    """Pixels whose skin faces the camera: where it curves away toward the silhouette it shades darker
    and reads as ink. Normals from the depth image's own neighbours, view rays from the intrinsics."""
    h, w = frame.depth_m.shape
    v, u = np.mgrid[0:h, 0:w].astype(float)
    x, y = frame.intr.undistort_normalized(u.ravel(), v.ravel())
    depth = cv2.medianBlur(frame.depth_m.astype(np.float32), 5)
    pts = np.stack([x.reshape(h, w) * depth, y.reshape(h, w) * depth, depth], axis=-1)
    step = 6  # a wide baseline: D405 depth on pale silicone is speckled pixel to pixel
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, step:-step] = pts[:, 2 * step:] - pts[:, :-2 * step]
    dv[step:-step] = pts[2 * step:] - pts[:-2 * step]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    ray = pts / np.maximum(np.linalg.norm(pts, axis=-1, keepdims=True), 1e-9)
    cos = np.abs((n * ray).sum(axis=-1)) / np.maximum(norm, 1e-12)
    ok = ((cos >= min_cos) & (depth > 0) & (norm > 0)).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(ok, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel).astype(bool)


def _continuous(frame: Frame, px: np.ndarray):
    """A centreline's pixels with depth, in base-frame points, split wherever the depth jumps."""
    depth = frame.depth_m[px[:, 0], px[:, 1]]
    px = px[depth > 0]
    depth = depth[depth > 0]
    if len(px) < 3:
        return
    x, y = frame.intr.undistort_normalized(px[:, 1].astype(float), px[:, 0].astype(float))
    base = np.stack([x * depth, y * depth, depth], axis=1) @ frame.cam_r.T + frame.cam_p
    cuts = np.flatnonzero(np.linalg.norm(np.diff(base, axis=0), axis=1) > JUMP_M) + 1
    for part_px, part in zip(np.split(px, cuts), np.split(base, cuts), strict=True):
        if len(part) >= 3:
            yield part_px, part


def forearm_pixels(frame: Frame, plane: tuple[np.ndarray, float], *, min_m: float = 0.005,
                   min_fraction: float = 0.01, exclude: np.ndarray | None = None) -> np.ndarray:
    """The forearm's pixels: of what stands above the table plane, the large region with the most colour
    (skin tones carry chroma; the black foam, grey palette station, white tag cubes and the arm's own
    parts do not), holes filled (ink and glare drop depth)."""
    from tatbot_travel.surface import deproject

    cam, px = deproject(frame.depth_m, frame.intr, None, *frame.range_m)
    base = cam @ frame.cam_r.T + frame.cam_p
    n, d = plane
    standing = np.zeros(frame.depth_m.shape, np.uint8)
    up = base @ n - d > min_m
    standing[px[up, 0], px[up, 1]] = 1
    if exclude is not None:
        standing[exclude] = 0  # the arm's own parts (selfview.robot_mask) stand above the table too
    standing = cv2.morphologyEx(standing, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(standing, connectivity=8)
    lab = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    large = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_fraction * standing.size]
    if not large:
        return np.zeros(standing.shape, bool)
    best = max(large, key=lambda i: np.median(chroma[labels == i]))
    region = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    if exclude is not None:
        filled[exclude] = 0
    return filled.astype(bool)
