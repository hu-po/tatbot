"""The fiducial-tile skin stencil (scripts/lib/stencil_tile.py): its markers in the page frame, read in a page image.

A tile print is a grid of AprilTag 16h5 markers, every code on the sheet distinct, so one decoded marker names its
place on the print: there is no repeat to fit a step off, as the coded border's lattice and the artwork's bands
had. The wrist readings rectify a frame onto the skin about the believed page (tatbot_session.inspect.rectify, a
curved page unrolled in the print's millimetres), and `fit` takes that page image to the print: every marker
found there against its printed place, as one rigid move in the page plane.

Page frame: metres, origin at the tile's centre, x right, y toward the print's top. The layout record's corners
are millimetres from the tile's top-left, y down, TL/TR/BR/BL of the upright marker: the order both OpenCV and the
AprilTag C library report a tag's corners in, so a marker's turn on the page is read from its corners.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

LAYOUT_FILE = "tile.json"   # the layout record beside an installed tile print's tracking.json


@dataclasses.dataclass(frozen=True)
class Tile:
    family: str
    size_m: tuple[float, float]         # the tile, the page a tile print's designs are placed on
    corners: dict[int, np.ndarray]      # id -> (4, 2) page metres, TL/TR/BR/BL
    sides_m: dict[int, float]           # id -> printed side
    anchors: frozenset[int]             # the corner anchors the overhead reads (silicone)
    record: dict

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(self.corners)

    @property
    def curve_radius_m(self) -> float:
        """The tightest skin this layout is read on (CURVE_RADIUS_M); a layout not named there, the forearm's."""
        return CURVE_RADIUS_M.get(self.record.get("layout"), CURVE_RADIUS_M["forearm"])


def tile(record: dict) -> Tile:
    """A layout record (stencil_tile.generate's JSON) in the page frame."""
    width, height = (float(v) for v in record["tile_mm"])
    corners, sides, anchors = {}, {}, set()
    for marker in record["markers"]:
        mm = np.asarray(marker["corners_mm"], float).reshape(4, 2)
        corners[int(marker["id"])] = np.c_[mm[:, 0] - width / 2, height / 2 - mm[:, 1]] / 1000.0
        sides[int(marker["id"])] = float(marker["side_mm"]) / 1000.0
        if marker.get("role") == "anchor":
            anchors.add(int(marker["id"]))
    if len(corners) != len(record["markers"]):
        raise ValueError("a tile's marker codes must be distinct")
    return Tile(record["family"], (width / 1000.0, height / 1000.0), corners, sides, frozenset(anchors), record)


def load(directory) -> Tile | None:
    """The tile beside an installed reference (LAYOUT_FILE), None for any other print."""
    path = Path(directory) / LAYOUT_FILE
    return tile(json.loads(path.read_text())) if path.exists() else None


SKIN_CONTRAST = (2.0, 8)   # skin_gray's CLAHE: clip limit, tiles a side


def skin_gray(image: np.ndarray, contrast=SKIN_CONTRAST) -> np.ndarray:
    """The view as the markers read best on skin: the green channel (a violet transfer absorbs it, skin reflects
    it), local contrast evened (CLAHE). On the 2026-10-17 forearm frames this read markers in 70 of 92 frames."""
    import cv2

    green = image if image.ndim == 2 else image[:, :, 1]
    clip, tiles = contrast
    return cv2.createCLAHE(clip, (tiles, tiles)).apply(np.ascontiguousarray(green, np.uint8))


def detector(t: Tile, min_side_px: float = 10.0, *, upsample: bool = False):
    """A detector of this tile's codes only, edge-fitted corners; with `upsample`, SMALL's."""
    from .detector import DetectorConfig, FiducialDetector

    config = DetectorConfig(adaptive_window_max=53, min_side_px=min_side_px, refinement="edges", cell_margin=0.25,
                            skin=True, **(SMALL if upsample else {}))
    return FiducialDetector(t.ids, config, ids_by_family={t.family: t.ids})


# The overhead D555's colour (1280x720) reads the forearm tile's 12 mm markers at ~10 px a side, 1.5-1.9 px a code
# cell, their perimeter (40.8 px median, 37.4 p5) at OpenCV's floor of 0.03 x 1280 = 38.4 px. Over 32 snapshots each
# (2026-10-19) the plain reading placed the paper print in none and its skin transfer in 15; read at twice the
# resolution (bicubic), half the perimeter floor, a threshold constant of 4 and a stronger CLAHE it placed both in
# all 32, 7 and 9 markers median (without the CLAHE and bicubic: 29 and 32, 3 and 4 markers, the turn stepping up
# to 2.3 deg frame to frame against 1.4). A tile whose smallest marker is under SMALL_MARKER_M is read this way from
# the start; larger ones (the silicone tile's 18 mm) keep the plain reading.
SMALL_MARKER_M = 0.015
SMALL = {"scale": 2.0, "min_perimeter_rate": 0.015, "threshold_constant": 4.0}
SMALL_CONTRAST = (4.0, 16)


def small_markers(t: Tile) -> bool:
    """Whether the overhead reads this tile upsampled (SMALL, SMALL_CONTRAST) and in a window about its placement."""
    return min(t.sides_m.values()) < SMALL_MARKER_M


def overhead_reading(t: Tile) -> dict:
    """rgbd_pose's reading of this tile in the stencil observer's overhead colour: {"det", "contrast"}."""
    small = small_markers(t)
    return {"det": detector(t, min_side_px=6.0, upsample=small), "contrast": SMALL_CONTRAST if small else SKIN_CONTRAST}


def find_all(image: np.ndarray, t: Tile, det=None, roi_xyxy=None, *, contrast=SKIN_CONTRAST, only=None) -> list:
    """Every read of this tile's codes in an image (fiducials.detector.Detection; corners in image pixels, pixel
    centres at integers), within roi_xyxy when given; with `only`, of those codes alone. A code can read more than
    once: a false decode elsewhere in the scene beside the print's own."""
    return (det or detector(t)).detect("tile", skin_gray(image, contrast), 0, roi_xyxy, only=only)


def violet_gray(image: np.ndarray) -> np.ndarray:
    """The view with only a violet transfer dark: blue and red over green, inverted. Skin and black tattoo ink both
    read white, so a stroke inked across the markers no longer joins them into one dark shape, which the quad
    search cannot split (2026-10-21: a 2x2 piece whose first stroke crossed it read 0-1 markers a wrist frame in
    green and CLAHE, 2-4 here in 22 of 24 frames, none of them a false code)."""
    import cv2

    if image.ndim == 2:
        return image
    b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
    v = cv2.GaussianBlur(np.clip((b + r) // 2 - g, 0, 255).astype(np.float32), (3, 3), 0)
    return (255 - np.clip(v * 255.0 / max(float(np.percentile(v, 99.5)), 1.0), 0, 255)).astype(np.uint8)


def find(image: np.ndarray, t: Tile, det=None) -> dict[int, np.ndarray]:
    """id -> (4, 2) corners in image pixels (pixel centres at integers) of this tile's markers in an image: the
    largest instance of each id read in violet_gray, then of each id it missed in skin_gray (a print in black), those
    alone fitted to their edges there: fitting every code read again cost 15 ms of a frame's 67 on the silicone tile's
    19 markers in view (OpenCV 4.6, x86, 2026-10-24), each frame read in a gauge hold's 25 by two cameras."""
    det = det or detector(t)
    found = _largest(det.detect("tile", violet_gray(image), 0))
    if missing := frozenset(t.ids) - found.keys():
        found.update(_largest(find_all(image, t, det, only=missing)))
    return {tag_id: corners for tag_id, (corners, _) in found.items()}


def _largest(reads) -> dict[int, tuple[np.ndarray, float]]:   # id -> (corners, side) of each id's largest read
    seen = {}
    for d in reads:
        if d.tag_id not in seen or d.side_px > seen[d.tag_id][1]:
            seen[d.tag_id] = (d.corners_px, d.side_px)
    return seen


def rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of the 2-D rigid move taking src onto dst in least squares."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rot = vt.T @ d @ u.T
    return rot, md - rot @ ms


# fit_points' acceptance, from the 2026-10-17 forearm frames (handheld D405, 12-25 cm, the arm's curve unrolled by a
# height field fitted to the frame's depth): 1.5 mm / 0.6 mm accepted 3 of 60 views, 2.5 mm / 1.2 mm 42, their tip
# reads 0.52 mm median from the frames' own RGB-D pose of the print (itself good to ~2 mm). A misread code stands
# where another marker is printed, a pitch (16 mm and more) away, far outside either. On the arm (2026-10-18, the
# silicone tile on the curved holder, three rehearsals, 66 wrist readings) the rms ran 0.96 mm median, 1.24 p95: the
# five between 1.2 and 1.45 mm sat on the plan as tightly as the rest (0.14 mm MAD against ~0.7-1.1), so the cap is
# 1.4 mm (65 of 66 taken).
AGREE_M = 0.0025
MAX_RMS_M = 0.0014


def fit_points(seen: dict[int, np.ndarray], t: Tile, *, tol_m: float = AGREE_M, min_markers: int = 2,
               max_rms_m: float = MAX_RMS_M) -> dict | None:
    """The in-plane correction p_print = R(theta) q + t taking markers seen at page points q (id -> (4, 2) believed
    page metres) onto their printed places: one marker's four corners propose a move, the move most markers agree
    with within tol_m is refitted on all of their corners. An extra marker of the right id where another of the
    print is, a misread, agrees with none and is left out. None with no marker of the print seen.
    Keys: the move (tx_m, ty_m, theta_rad, scale), rms_m, sigma, points, sides, score, margin and tone of an
    artwork fit, plus "markers" (agreeing), "seen", "ids" and "accepted": at least min_markers
    agree and their corners' rms is within max_rms_m. A marker read has no rival peak a band away to beat, so its
    "margin" is 1 when accepted and 0 when not, on the scale the artwork's acceptance reads. A lone marker is also
    accepted when the move it reads is under half the way to its nearest neighbour on the print (lone_ok): a
    misread code stands where another marker is printed, so it reads a move of that distance or more. At the curved
    holder's gauge holds (2026-10-18) two of six views saw one marker, read to 0.27 and 0.61 mm rms."""
    ids = [i for i in seen if i in t.corners]
    if not ids:
        return None
    best = None
    for i in ids:
        rot, shift = rigid(seen[i], t.corners[i])
        agree = [j for j in ids if np.abs(seen[j] @ rot.T + shift - t.corners[j]).max() < tol_m]
        if best is None or len(agree) > len(best):
            best = agree
    if not best:   # no marker's corners keep its printed shape within tol_m: a quad read across ink or a fold
        return None
    q = np.concatenate([seen[i] for i in best])
    p = np.concatenate([t.corners[i] for i in best])
    rot, shift = rigid(q, p)
    res = np.linalg.norm(q @ rot.T + shift - p, axis=1)
    rms = float(np.sqrt(np.mean(res ** 2)))
    accepted = rms <= max_rms_m and (len(best) >= min_markers or lone_ok(t, best[0], rot, shift))
    # the turn's sigma from the corners' spread about their centre, the shift's from the residual
    spread = float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))
    sigma_xy = max(rms, 1e-4) / np.sqrt(len(p))
    return {"tx_m": float(shift[0]), "ty_m": float(shift[1]), "theta_rad": float(np.arctan2(rot[1, 0], rot[0, 0])),
            "scale": [1.0, 1.0], "rms_m": rms,
            "sigma": [sigma_xy, sigma_xy, float(max(rms, 1e-4) / max(spread, 1e-3) / np.sqrt(len(p)))],
            "points": int(len(p)), "sides": ["markers"], "score": len(best) / len(ids), "margin": float(accepted),
            "tone": "markers", "markers": len(best), "seen": len(ids), "ids": sorted(best), "accepted": accepted}


def lone_ok(t: Tile, tag_id: int, rot, shift) -> bool:
    """Whether one marker's read move (p = rot q + shift) carries it under half the way to its nearest neighbour."""
    centre = t.corners[tag_id].mean(axis=0)
    gap = min(float(np.linalg.norm(c.mean(axis=0) - centre)) for i, c in t.corners.items() if i != tag_id)
    believed = (centre - shift) @ rot   # rot.T (p - shift)
    return float(np.linalg.norm(centre - believed)) < gap / 2


def fit(page_image: np.ndarray, t: Tile, extent_m, px_per_mm: float, det=None, **accept) -> dict | None:
    """fit_points on a page image covering extent_m = (xmin, ymin, xmax, ymax) at px_per_mm, row 0 at ymax
    (tatbot_session.inspect.page_grid): its pixel (c, r) centre is page (xmin + (c + 0.5) s, ymax - (r + 0.5) s)."""
    xmin, _, _, ymax = extent_m
    step = 1e-3 / px_per_mm
    seen = {i: np.c_[xmin + (c[:, 0] + 0.5) * step, ymax - (c[:, 1] + 0.5) * step]
            for i, c in find(page_image, t, det).items()}
    return fit_points(seen, t, **accept)


def target_corners(t: Tile, tag_id: int) -> np.ndarray:
    """A marker's corners in the stencil observer's target frame (origin the page centre, x along u, y along v,
    down the print, z = x cross y, into it): the page frame with y and z turned over."""
    c = t.corners[tag_id]
    return np.c_[c[:, 0], -c[:, 1], np.zeros(4)]


def _ray(rays: np.ndarray, px: np.ndarray) -> np.ndarray:
    """The camera ray through each subpixel point, interpolated in a per-pixel ray image (rows, cols, 3)."""
    h, w = rays.shape[:2]
    x = np.clip(px[:, 0], 0, w - 1.001)
    y = np.clip(px[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * rays[y0, x0] + fx * (1 - fy) * rays[y0, x0 + 1] + (1 - fx) * fy * rays[y0 + 1, x0]
            + fx * fy * rays[y0 + 1, x0 + 1])


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 rigid transform taking src (N, 3) onto dst in least squares."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    out = np.eye(4)
    out[:3, :3] = vt.T @ d @ u.T
    out[:3, 3] = md - out[:3, :3] @ ms
    return out


# rgbd_pose's pose on skin. The flat layout's rigid fit is the chord plane of the markers in view, its z turned toward
# their side of the curve and its origin under the skin. Over the overhead's readings of the forearm transfer on a live
# forearm (2026-10-20, 02:47-03:07Z) it tilted 14-33 deg where the layout laid round a cylinder through the same
# corners put the normal at the print's centre at 11-18 deg and that centre 0.3-2 mm higher; centre and turn moved
# under 0.3 mm and 0.3 deg. The wrist locate plans its views about this pose: at the chord's 19-20 deg none was
# reachable, at 12.5-13 deg 16-25 were. A cylinder needs markers both ways across the print (BEND_SPAN_M, the smaller
# spread of their printed centres): along one row or column its radius, or its axis, is unread.
BEND_SPAN_M = 0.005
BEND_STEPS = np.array([1e-6] * 6 + [1e-3, 1e-5])   # the fit's numerical derivative: rad, m, 1/m, rad


def _rotations(v: np.ndarray) -> np.ndarray:
    """(P, 3, 3) the rotation matrices of rotation vectors v (P, 3) (Rodrigues)."""
    theta = np.linalg.norm(v, axis=1)[:, None, None]
    k = np.cross(np.eye(3)[None], (v / np.maximum(theta[:, :, 0], 1e-12))[:, None, :])   # each axis's cross matrix
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def bent(t: Tile, ids, points: np.ndarray, pose: np.ndarray,
         iterations: int = 40) -> tuple[np.ndarray, float, float, np.ndarray]:
    """(camera <- target, the skin's radius m (inf: flat), the corners' rms m, each corner's place on that layout less
    its point (N, 3) m) of the layout laid unstretched round a
    cylinder through `points` (ids[k]'s corners at rows 4k:4k+4, camera metres), from `pose`, their flat rigid fit
    (kabsch): the axis any way in the print's plane, the skin convex to the camera and no tighter than 0.75 of the
    tile's curve_radius_m; Levenberg-Marquardt, numpy only (the observer's interpreter has no scipy). Its origin is
    the print's centre on that skin, its z the skin's normal there (into it), its x along u there. `pose` itself,
    radius inf, when the markers do not span BEND_SPAN_M both ways or no bend fits them better."""
    q, points = np.concatenate([target_corners(t, i)[:, :2] for i in ids]), np.asarray(points, float)
    most, centres = 1.0 / (0.75 * t.curve_radius_m), np.array([t.corners[i].mean(axis=0) for i in ids])

    def residual(p):   # (P, 8): a turn (rotation vector) and a move of `pose`, the curvature, the axis's angle from u
        c, s, k = np.cos(p[:, 7:]), np.sin(p[:, 7:]), p[:, 6:7]
        along, across = q[:, 0] * c + q[:, 1] * s, q[:, 1] * c - q[:, 0] * s
        arc = across * np.sinc(k * across / np.pi)                              # sin(k a) / k
        sag = 0.5 * k * across ** 2 * np.sinc(k * across / (2 * np.pi)) ** 2   # (1 - cos(k a)) / k, into the skin
        local = np.stack([along * c - arc * s, along * s + arc * c, sag], axis=-1)
        turned = pose[:3, :3] @ _rotations(p[:, :3])
        return (local @ turned.transpose(0, 2, 1) + pose[:3, 3] + p[:, None, 3:6] - points).reshape(len(p), -1)

    flat = residual(np.zeros((1, 8)))[0]
    best = (float(flat @ flat), np.zeros(8))
    if len(ids) < 3 or np.linalg.eigvalsh(np.cov(centres.T))[0] < BEND_SPAN_M ** 2:
        return pose, np.inf, float(np.sqrt(best[0] / len(q))), flat.reshape(-1, 3)
    for axis in (0.0, np.pi / 2):   # round v (a forearm under the print's long side) or round u
        p, damp = np.r_[np.zeros(6), most / 2, axis], 1e-3
        r = residual(p[None])[0]
        for _ in range(iterations):
            jac = ((residual(p + np.diag(BEND_STEPS)) - r) / BEND_STEPS[:, None]).T
            a = jac.T @ jac
            trial = p - np.linalg.solve(a + damp * (np.diag(np.diag(a)) + 1e-9 * np.trace(a) * np.eye(8)), jac.T @ r)
            trial[6] = np.clip(trial[6], 0.0, most)
            moved = residual(trial[None])[0]
            if moved @ moved < r @ r:
                done = r @ r - moved @ moved < 1e-6 * (r @ r)
                p, r, damp = trial, moved, damp / 3
                if done:
                    break
            elif (damp := damp * 10) > 1e6:
                break
        if r @ r < best[0]:
            best = (float(r @ r), p)
    p = best[1]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = pose[:3, :3] @ _rotations(p[None, :3])[0], pose[:3, 3] + p[3:6]
    return out, (1.0 / p[6] if p[6] > 0 else np.inf), float(np.sqrt(best[0] / len(q))), residual(p[None])[0].reshape(-1, 3)


# rgbd_pose's lift: each corner sits at the depth about its own pixel (corner_depth), so no surface is fitted under
# a marker or across the sheet. 15 x 15 colour px is about 7 x 7 of the D555's own 640 x 360 depth. Over the
# transfer's 32 still views (2026-10-19) a 7 x 7 colour square read the depth's own ripple across one 10 px marker
# and stretched it past the size check in 9 views; 15 x 15 kept 294 markers to the sheet plane's 290, its corners
# 0.35 mm median from their own median along their rays (the plane's 0.23 mm, smoothed over the sheet).
CORNER_PX = 7
DEPTH_RANGE_M = (0.05, 3.0)
# The tightest skin each layout is laid on, by its record's "layout". A print laid on skin without stretching keeps
# its distances along the skin, so two of its points' chord falls short of their printed distance d by at most
# d - 2 r sin(d / 2r) (under 1 mm over 33 mm at r 40 mm), and nothing lies farther apart than printed. The forearm
# tile goes on a forearm (r ~40 mm) or the silicone mount; the silicone tile on a practice skin, flat or over the
# mount (r 146-191 mm along v under the forearm transfer's markers, 32 views, 2026-10-19), never round a forearm: at
# 35 mm its anchors printed 118 mm apart agreed down to 66 mm, and a code read 45 mm off its place beside two
# anchors was kept for one of them (2026-10-19 review, synthetic).
CURVE_RADIUS_M = {"forearm": 0.035, "silicone": 0.10}


def corner_depth(depth_m: np.ndarray, px: np.ndarray, half: int = CORNER_PX, band_m: float = 0.01) -> np.ndarray | None:
    """(N,) the depth (metres along z) at each subpixel point px (image pixels, centres at integers): about each of
    the four pixels around it, the mean of the middle half of the depths in the (2 half + 1)^2 px square there that
    are valid (in DEPTH_RANGE_M, within band_m of the square's median: not the arm over the skin) together with their
    mirror through that pixel, interpolated bilinearly. Mirrored pairs keep a sloped skin's mean at the pixel itself
    when the arm, a hole or the image's edge takes one side of the square; taken at the nearest pixel alone, a sheet
    turned 37.5 deg from the camera read 0.3 mm rms off along the rays. None when any square keeps under one row
    of its pixels (a corner on the arm's edge)."""
    h, w = depth_m.shape
    step = np.arange(-half, half + 1)
    out = []
    for x, y in np.clip(px, 0, [w - 1.001, h - 1.001]):
        x0, y0 = int(x), int(y)
        near = []
        for yy, xx in ((y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)):
            rows, cols = yy + step, xx + step
            patch = depth_m[np.clip(rows, 0, h - 1)[:, None], np.clip(cols, 0, w - 1)]
            valid = (((rows >= 0) & (rows < h))[:, None] & ((cols >= 0) & (cols < w)) & np.isfinite(patch)
                     & (patch > DEPTH_RANGE_M[0]) & (patch < DEPTH_RANGE_M[1]))
            if not valid.any():
                return None
            valid &= np.abs(patch - np.median(patch[valid])) < band_m
            valid &= valid[::-1, ::-1]
            if valid.sum() < len(step):
                return None
            kept = np.sort(patch[valid])
            quarter = len(kept) // 4
            near.append(kept[quarter:len(kept) - quarter].mean())
        fx, fy = x - x0, y - y0
        out.append((1 - fy) * ((1 - fx) * near[0] + fx * near[1]) + fy * ((1 - fx) * near[2] + fx * near[3]))
    return np.asarray(out)


def rgbd_pose(image: np.ndarray, depth_m: np.ndarray, rays: np.ndarray, t: Tile, det=None, *,
              tol_m: float = 0.004, min_markers: int = 2, radius_m: float | None = None, roi_xyxy=None,
              contrast=SKIN_CONTRAST) -> dict:
    """camera <- target of a tile print in one aligned RGB-D view. Every read of its codes (find_all) has each corner
    lifted along its own ray to the depth about its own pixel (corner_depth) and is kept when it spans its printed
    side (a false decode elsewhere in the scene does not hide the print's own read of that code). The reads kept are
    the most, one per code, whose corners all lie as far apart as printed (_consensus): no surface is fitted, so a
    skin curved under the print, or the arm hiding part of it, moves no marker's points. The pose is the layout laid
    round a cylinder through them (bent): its origin the print's centre on the skin, its z the skin's normal there;
    the flat layout's rigid fit (the chord plane of the markers in view) when they cannot show a bend. `rays`: the
    camera ray of every pixel (z = 1), aligned with `depth_m` (metres along z). Fewer than min_markers agreeing is no
    pose: under the arm one stray read decoded off the wrist-tag cube is as likely as one of the print's.
    `roi_xyxy`, `contrast`: find_all's. {"found", "camera_from_target", "markers", "ids", "seen", "rms_m" (the
    corners about that layout), "radius_m" (the skin's, inf flat), "points_camera" (the kept markers' corners,
    ids[k]'s at rows 4k:4k+4), "pixels" (the same corners in the image)} or {"found": False, "reason"}."""
    # keyed (id, n): a code can read in more than one place, a false decode elsewhere beside the print's own
    seen = {(d.tag_id, n): d.corners_px for n, d in enumerate(find_all(image, t, det, roi_xyxy, contrast=contrast))}
    if not seen:
        return {"found": False, "reason": "no marker of the print read"}
    lifted = {}
    for key, px in seen.items():
        depth = corner_depth(depth_m, px)
        if depth is not None:
            lifted[key] = _ray(rays, px) * depth[:, None]
    sized = {i: c for i, c in lifted.items() if _printed_size(c, t.sides_m[i[0]])}
    if not sized:
        why = [f"{len(lifted)} reads of its codes off the print (not its printed size on the depth)"] if lifted else []
        why += [f"{len(seen) - len(lifted)} without depth under them"] if len(seen) > len(lifted) else []
        return {"found": False, "reason": "no marker of the print read: " + ", ".join(why)}
    keep = _consensus(t, sized, tol_m, radius_m)
    if len(keep) < min_markers:
        return {"found": False, "reason": f"{len(sized)} reads of its codes at its printed size, {len(keep)} agreeing"}
    ids, points = [k[0] for k in keep], np.concatenate([sized[k] for k in keep])
    pose, radius, rms, _ = bent(t, ids, points, kabsch(np.concatenate([target_corners(t, i) for i in ids]), points))
    return {"found": True, "camera_from_target": pose, "markers": len(keep), "ids": ids,
            "seen": len({i[0] for i in seen}), "rms_m": rms, "radius_m": radius, "points_camera": points,
            "pixels": np.concatenate([seen[k] for k in keep])}


def _printed_size(corners: np.ndarray, side_m: float, tol: float = 0.15) -> bool:
    """Whether a marker's lifted corners span its printed side within tol. Under full correction other codes
    decode as the tile's (2026-10-18): the 12 mm ArUco grid beside the 18 mm silicone tile placed it 260 mm away, and
    two 47 mm AprilTag 36h11 faces of the wrist-tag cube, taken for 40 mm anchors (+17.5%), put it on the cube."""
    sides = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
    return bool(np.all(np.abs(sides / side_m - 1.0) < tol))


def _consensus(t: Tile, lifted: dict, tol_m: float, radius_m: float | None = None, stretch: float = 0.0) -> list:
    """The most reads, one per code, every two of which agree: each corner of one as far from each corner of the
    other as printed, within tol_m, and on skin curved to radius_m (the tile's curve_radius_m) no nearer than its
    chord. A false read, or depth off the skin, sits a pitch (16 mm and more) or a lift from where the others put it
    and agrees with few. Keys (id, n), in that order. Each read in turn seeds a group of the reads agreeing with it,
    the one disagreeing with the most others (then the farthest from its printed distances to the rest) left out
    until every two agree; of the largest groups, the one nearest its printed distances. Never the keys' order: two
    true reads and a stray agreeing with one of them tie on every count but that one. `stretch`: the skin's own give,
    a share of each printed distance allowed either way on top of tol_m (a transfer on a forearm turned since)."""
    radius_m = t.curve_radius_m if radius_m is None else radius_m
    keys = sorted(lifted)
    points = np.stack([lifted[k] for k in keys])                     # (n, 4, 3)
    printed = np.stack([t.corners[k[0]] for k in keys])              # (n, 4, 2)
    d3 = np.linalg.norm(points[:, :, None, None] - points[None, None], axis=-1)       # (n, 4, n, 4)
    dp = np.linalg.norm(printed[:, :, None, None] - printed[None, None], axis=-1)
    chord = 2 * radius_m * np.sin(np.minimum(dp / (2 * radius_m), np.pi / 2))   # held at 2r past half a turn
    over = np.maximum(np.maximum(d3 - dp * (1 + stretch), chord * (1 - stretch) - d3) - tol_m, 0.0).max(axis=(1, 3))
    off = np.abs(d3 - dp).max(axis=(1, 3))                          # (n, n): how far from the printed distances
    codes = np.array([k[0] for k in keys])
    agree = (over == 0.0) & (codes[:, None] != codes[None]) | np.eye(len(keys), dtype=bool)
    best, best_off = [], np.inf
    for seed in range(len(keys)):
        group = list(np.flatnonzero(agree[seed]))
        while True:
            bad = (~agree[np.ix_(group, group)]).sum(axis=1)
            if not bad.any():
                break
            spread = off[np.ix_(group, group)].sum(axis=1)
            group.pop(max(range(len(group)), key=lambda k: (bad[k], spread[k])))
        spread = off[np.ix_(group, group)].sum()
        if (len(group), -spread) > (len(best), -best_off):
            best, best_off = group, spread
    return [keys[i] for i in sorted(best)]
