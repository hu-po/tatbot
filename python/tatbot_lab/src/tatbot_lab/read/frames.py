"""Camera frames onto a cell's page raster: markers, local homographies, and the fusion of a scan's many views.

A cell is read on a raster turned to its strokes (`Frame`): rows advance along the strokes from where they start,
columns across them, PX_PER_MM samples a millimetre, the cell's strokes padded by PAD_MM so the ring around them and
the skin that brightness is matched on are in it. A view is a before and an after frame from one pose: the rest
view, or one hold of a scan, which the arm repeats. Its before frame maps onto the raster by a homography from the
markers within RADIUS_MM of the cell (2026-11-05's readers: the skin curves, so a homography is only good locally),
and its after frame through the same one once aligned to the before (`pair`). `fuse` combines a scan's views.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

PX_PER_MM = 10.0     # raster samples a millimetre: above the wrist's ~3.7 native, so resampling loses nothing
PAD_MM = 14.0        # raster margin about the strokes: the 8 mm speckle ring and skin to match brightness on
RADIUS_MM = 52.0     # markers this near the cell's centre carry its homography
MIN_MARKERS = 2
WEIGHT_MM = 25.0     # a marker's weight in the cell's homography falls as exp(-(d / WEIGHT_MM)^2) with its distance
GROSS_PX = 25.0      # a marker the fit misses by more than this, on average over its corners, is a bad read
ECC_PAD_MM = 40.0    # skin about the cell the after frame is aligned on: 2026-11-05's ~150 px at the rest view
SHEET_PAD_MM = 15.0  # a chosen sheet's region reaches this far past its print: the tracker reads a page mm off


@dataclass(frozen=True)
class Frame:
    """A raster grid on the page: pixel (col, row) is page mm origin + col/px * across + row/px * along."""
    origin: np.ndarray
    along: np.ndarray
    across: np.ndarray
    shape: tuple[int, int]   # rows, cols
    px_per_mm: float = PX_PER_MM

    def affine(self) -> np.ndarray:
        """3x3: raster (col, row, 1) to page mm (x, y, 1)."""
        out = np.eye(3)
        out[:2, 0] = self.across / self.px_per_mm
        out[:2, 1] = self.along / self.px_per_mm
        out[:2, 2] = self.origin
        return out

    def page(self) -> tuple[np.ndarray, np.ndarray]:
        """Page x, y (mm) of every raster pixel."""
        rows, cols = np.mgrid[0:self.shape[0], 0:self.shape[1]].astype(float)
        p = self.origin[:, None, None] + (cols[None] * self.across[:, None, None]
                                          + rows[None] * self.along[:, None, None]) / self.px_per_mm
        return p[0], p[1]

    def uv(self, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Along (u) and across (v) mm of every pixel from page point `start`."""
        x, y = self.page()
        dx, dy = x - start[0], y - start[1]
        return dx * self.along[0] + dy * self.along[1], dx * self.across[0] + dy * self.across[1]


def frame_for(strokes_mm: list[np.ndarray], pad_mm: float = PAD_MM, px_per_mm: float = PX_PER_MM) -> Frame:
    """The raster for a cell's strokes, turned to the first stroke's direction."""
    first = strokes_mm[0]
    along = first[-1] - first[0]
    along = along / np.linalg.norm(along) if np.linalg.norm(along) > 1e-9 else np.array([1.0, 0.0])
    across = np.array([-along[1], along[0]])
    pts = np.vstack(strokes_mm)
    u, v = pts @ along, pts @ across
    u0, v0 = u.min() - pad_mm, v.min() - pad_mm
    rows = int(np.ceil((u.max() - u.min() + 2 * pad_mm) * px_per_mm))
    cols = int(np.ceil((v.max() - v.min() + 2 * pad_mm) * px_per_mm))
    return Frame(u0 * along + v0 * across, along, across, (rows, cols), px_per_mm)


def tile(print_dir):
    """The print's markers as fiducials.stencil reads them (scripts/vision on the path)."""
    from fiducials import stencil

    t = stencil.load(print_dir)
    if t is None:
        raise ValueError(f"{print_dir}: no tile.json, not a generated tile print")
    return t


def sheet_roi(world_from_page, tile_mm, intrinsics: dict, pad_mm: float = SHEET_PAD_MM) -> list[int] | None:
    """The image box (x0, y0, x1, y1 px) one sheet's print covers in the overhead's frame, padded by pad_mm: the
    tracker's world is the D555 colour optical frame, so its page pose (`world_from_page`, the ROS page frame) and the
    capture's intrinsics (tatbot.camera-intrinsics/1, Brown-Conrady) place it. None when it lies wholly outside."""
    half = np.array([tile_mm[0], tile_mm[1]], float) / 2000.0 + pad_mm / 1000.0
    page = np.array([[sx * half[0], sy * half[1], 0.0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
    camera = (np.asarray(world_from_page, float) @ page.T).T[:, :3]
    k = np.array([[intrinsics["fx"], 0.0, intrinsics["ppx"]], [0.0, intrinsics["fy"], intrinsics["ppy"]], [0, 0, 1]])
    px, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k,
                              np.asarray(intrinsics.get("distortion_coefficients") or [0.0] * 5, float))
    px = px.reshape(-1, 2)
    x0, y0 = np.clip(np.floor(px.min(axis=0)), 0, [intrinsics["width"], intrinsics["height"]])
    x1, y1 = np.clip(np.ceil(px.max(axis=0)), 0, [intrinsics["width"], intrinsics["height"]])
    return None if x1 <= x0 or y1 <= y0 else [int(x0), int(y0), int(x1), int(y1)]


def markers(image: np.ndarray, t, instrument: str = "wrist", roi=None) -> dict[int, np.ndarray]:
    """id -> (4, 2) image corners of every marker of the print read in a BGR frame, the largest read of each id, in
    the skin-contrast view; the overhead's frames as the stencil observer reads them (stencil.overhead_reading: small
    markers upsampled). Not stencil.find's violet view first: on 2026-11-05's red block its corners bent the cell's
    homography enough to leave the read at 4x its floor (4.9 against 1.1 in L*). `roi` (x0, y0, x1, y1, image px)
    reads only inside it: two sheets of one print in the overhead's view give every marker twice (2026-11-08, both
    mounts carrying forearm18 981a6476)."""
    from fiducials import stencil

    reading = stencil.overhead_reading(t) if instrument == "overhead" else {}
    found: dict[int, tuple[np.ndarray, float]] = {}
    for d in stencil.find_all(image, t, roi_xyxy=None if roi is None else tuple(int(v) for v in roi), **reading):
        corners = np.asarray(d.corners_px, float).reshape(-1, 2)
        area = 0.5 * abs(np.dot(corners[:, 0], np.roll(corners[:, 1], 1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], 1)))
        if int(d.tag_id) not in found or area > found[int(d.tag_id)][1]:
            found[int(d.tag_id)] = (corners, area)
    return {i: c for i, (c, _) in found.items()}


def _weighted_dlt(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The homography taking src to dst (n, 2) by a normalized DLT, each correspondence weighted by w."""
    def normalize(p):
        c = p.mean(axis=0)
        k = np.sqrt(2) / max(float(np.mean(np.linalg.norm(p - c, axis=1))), 1e-9)
        return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1]])
    ts, td = normalize(src), normalize(dst)
    a = np.c_[src, np.ones(len(src))] @ ts.T
    b = np.c_[dst, np.ones(len(dst))] @ td.T
    zero = np.zeros((len(a), 3))
    rows = np.vstack([np.hstack([-a, zero, b[:, :1] * a]), np.hstack([zero, -a, b[:, 1:2] * a])])
    rows *= np.concatenate([w, w])[:, None]
    h = np.linalg.svd(rows)[2][-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


def homography(found: dict[int, np.ndarray], t, centre_mm, radius_mm: float = RADIUS_MM):
    """Page mm -> image px about `centre_mm` from the read markers within radius_mm, and the ids it kept; None when
    fewer than MIN_MARKERS. Deterministic, and local: each marker weighs exp(-(d / WEIGHT_MM)^2) by its distance d
    from the cell, so the skin's curve bends the fit least where the cell is, and a marker the fit misses by more
    than GROSS_PX (a false or misplaced read) is dropped, worst first, before a refit. RANSAC was not: on the curve
    no one homography fits every marker within its threshold, so the inliers it drew, and the cell's place with
    them, moved by a millimetre between calls (2026-11-05's fresh red block, read a line off)."""
    centre = np.asarray(centre_mm, float)
    dist = {i: float(np.hypot(*(t.corners[i][:, :2].mean(axis=0) * 1000 - centre))) for i in found if i in t.corners}
    near = sorted(i for i, d in dist.items() if d < radius_mm)
    while len(near) >= MIN_MARKERS:
        src = np.vstack([t.corners[i][:, :2] * 1000 for i in near])
        dst = np.vstack([found[i] for i in near])
        h = _weighted_dlt(src, dst, np.repeat([np.exp(-(dist[i] / WEIGHT_MM) ** 2) for i in near], 4))
        miss = np.linalg.norm(cv2.perspectiveTransform(src[None], h)[0] - dst, axis=1).reshape(-1, 4).mean(axis=1)
        if miss.max() <= GROSS_PX:
            return h, near
        near.pop(int(np.argmax(miss)))
    return None, near


def rectify(image: np.ndarray, h: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """The frame's raster sampled from `image` through page->image homography `h`: float32 BGR, and where valid."""
    size = (frame.shape[1], frame.shape[0])
    m = h @ frame.affine()
    raster = cv2.warpPerspective(image, m, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP).astype(np.float32)
    inside = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), m, size,
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    valid = cv2.erode(inside, np.ones((5, 5), np.uint8)) > 0   # cubic taps reach two pixels past the edge
    return raster, valid


class UnplacedError(ValueError):
    """A view whose before frame does not see enough of the print about a cell to place it."""


def pair(before: np.ndarray, after: np.ndarray, t, frame: Frame, centre_mm, changed_mm, instrument: str = "wrist",
         roi=None) -> dict:
    """One view's before and after frames, taken from the same pose, on the cell's raster. The before frame's markers
    place it on the page (none is inked yet); the after frame is aligned to the before in the image by ECC with the
    cell left out, then sampled through the same homography, so the skin's curve bends both alike and cancels in
    their difference. Ink over a marker cannot move the after frame's registration (2026-11-05's readers did the
    same; registering the after frame by its own markers cost a red block 4x its floor). `changed_mm` lists every
    box (x0, y0, x1, y1, page mm) drawn between the two frames, this cell's first and then its run's others: ECC
    aligns on the skin about the cell that did not change."""
    h, near = homography(markers(before, t, instrument, roi), t, centre_mm)
    if h is None:
        raise UnplacedError(f"the before frame reads fewer than {MIN_MARKERS} markers within {RADIUS_MM:g} mm of the cell")
    ga = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bx0, by0, bx1, by1 = changed_mm[0]   # the cell itself: the window is it and ECC_PAD_MM of skin about it
    window = np.array([[[bx0 - ECC_PAD_MM, by0 - ECC_PAD_MM]], [[bx1 + ECC_PAD_MM, by0 - ECC_PAD_MM]],
                       [[bx1 + ECC_PAD_MM, by1 + ECC_PAD_MM]], [[bx0 - ECC_PAD_MM, by1 + ECC_PAD_MM]]], float)
    outline = cv2.perspectiveTransform(window, h).reshape(-1, 2)
    x0, y0 = np.maximum(np.floor(outline.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(outline.max(axis=0)).astype(int), [before.shape[1], before.shape[0]])
    mask = np.full(ga.shape, 255, np.uint8)
    for bx0, by0, bx1, by1 in changed_mm:
        quad = cv2.perspectiveTransform(np.array([[[bx0, by0]], [[bx1, by0]], [[bx1, by1]], [[bx0, by1]]], float), h)
        cv2.fillPoly(mask, [quad.reshape(-1, 2).astype(np.int32)], 0)
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    ecc, warp = cv2.findTransformECC(ga[y0:y1, x0:x1], gb[y0:y1, x0:x1], warp, cv2.MOTION_HOMOGRAPHY, criteria,
                                     mask[y0:y1, x0:x1], 5)
    shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
    to_after = shift @ warp.astype(float) @ np.linalg.inv(shift)   # before image px -> after image px
    raster_a, valid_a = rectify(before, h, frame)
    raster_b, valid_b = rectify(after, to_after @ h, frame)
    return {"before": raster_a, "after": raster_b, "valid": valid_a & valid_b, "ecc": float(ecc), "markers": near}


def fuse(pairs: list[dict]) -> dict:
    """A rest view's one pair, or a scan's many, as one before and one after raster: the per-pixel median of the
    views that cover it, which drops a wet-ink glint only some views see and samples finer than any one view."""
    if len(pairs) == 1:
        return {"before": pairs[0]["before"], "after": pairs[0]["after"], "valid": pairs[0]["valid"]}
    valid = np.stack([p["valid"] for p in pairs])
    out = {"valid": valid.any(axis=0)}
    for side in ("before", "after"):
        stack = np.stack([p[side] for p in pairs]).astype(np.float32)
        stack[~valid] = np.nan
        with warnings.catch_warnings():   # a pixel no view covers is all-NaN: invalid, and zero below
            warnings.simplefilter("ignore", RuntimeWarning)
            out[side] = np.nan_to_num(np.nanmedian(stack, axis=0)).astype(np.float32)
    return out
