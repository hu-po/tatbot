"""The coded stencil's pose in one wrist-camera frame, and the pen tip on the print.

Seed-agnostic: the coded border's junction knots (2.2 mm discs) sit on a lattice that depends only on the page
layout (stencil_coded.Layout), so any print of the layout serves. A prior pose of the print in the camera frame
(FK of the camera and the run's page) seeds the search, and the frame's own depth gives the paper's plane, so
only the in-plane offset is searched. Poses a lattice step apart fit the knots equally well; the band's edges
tell them apart. The right pose matches the knots in clear view, leaves none predicted there but missing, and
puts no detected knot one step into the clear centre or past the band (`support`). Every lattice translation of
a seed fit is scored, and the margin over the best one more than 3 mm away says how sure the choice is.

The tip on the print is the wrist gauge's tip point (tatbot_session.gauge, in the colour optical frame) carried
into the print's frame: where the pen will touch, with no arm kinematics. Lengths are metres in the page frame
(centre origin, x right, y toward the print's top) unless noted. Stencil tracking: ros/README.md 4.4.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

MARGIN_MIN = 40       # a fit whose support beats a lattice step's by less is refused (recorded holds: 6 of 7
#                       frames a step off scored under it, p10 of the rest 92)


@dataclass
class Layout:
    knots: np.ndarray       # (N, 2) junction centres
    knot_m: float           # disc diameter
    inner: tuple            # clear centre: left, bottom, right, top
    outer: tuple            # the band's outer edge: left, bottom, right, top
    step_m: float           # lattice spacing


def layout(settings: dict) -> Layout:
    """The knots and band of a coded print from its settings.json (the seed does not enter)."""
    from stencil_coded import Layout as Coded

    lay = Coded(**settings)
    pts = np.array([lay.position(q, r) for q, r in lay.nodes], float)
    w, h, m = settings["width_mm"], settings["height_mm"], settings["margin_mm"]
    x0, y0, x1, y1 = settings["border_inner_mm"]
    return Layout(np.c_[pts[:, 0] - w / 2, h / 2 - pts[:, 1]] / 1000, settings["knot_mm"] / 1000,
                  ((x0 - w / 2) / 1000, (h / 2 - y1) / 1000, (x1 - w / 2) / 1000, (h / 2 - y0) / 1000),
                  ((m - w / 2) / 1000, (m - h / 2) / 1000, (w / 2 - m) / 1000, (h / 2 - m) / 1000),
                  settings["spacing_mm"] / 1000)


def load_layout(pattern_id: str, root=None) -> Layout:
    """The installed reference's layout (<log root>/stencils/references/<pattern_id>/settings.json)."""
    from stencil_reference import observer_references

    return layout(json.loads((observer_references(root) / pattern_id / "settings.json").read_text()))


def paper_plane(depth_raw: np.ndarray, meta: dict, stride: int = 8, iters: int = 120, tol: float = 0.0015,
                seed: int = 1) -> tuple[np.ndarray, float]:
    """(unit normal, distance) of the paper in the colour optical frame, n.x = d, n_z > 0: RANSAC on the native
    depth (the wrist gauge's npz metadata), its points carried into the colour frame."""
    units, di = float(meta["depth_units_m"]), meta["intrinsics"]
    z = depth_raw[::stride, ::stride].astype(np.float64) * units
    v, u = np.mgrid[0:depth_raw.shape[0]:stride, 0:depth_raw.shape[1]:stride]
    ok = (z > 0.07) & (z < 0.40)
    pts = np.c_[(u[ok] - di["ppx"]) / di["fx"] * z[ok], (v[ok] - di["ppy"]) / di["fy"] * z[ok], z[ok]]
    r = np.asarray(meta["color_from_depth"]["rotation"], float).reshape(3, 3, order="F")
    pts = pts @ r.T + np.asarray(meta["color_from_depth"]["translation_m"], float)
    if len(pts) < 50:
        raise ValueError("no paper in the depth frame")
    best, best_n = None, 0
    for a, b, c in pts[np.random.default_rng(seed).integers(0, len(pts), (iters, 3))]:
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(pts @ n - n @ a) < tol
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    p = pts[best]
    n = np.linalg.svd(p - p.mean(0), full_matrices=False)[2][2]
    n = -n if n[2] < 0 else n
    return n, float(n @ p.mean(0))


def snap_to_plane(cam_from_print: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """The pose moved onto a measured plane: its origin along its camera ray, its x projected into the plane."""
    o, x = cam_from_print[:3, 3], cam_from_print[:3, 0]
    zc = n if cam_from_print[:3, 2] @ n >= 0 else -n
    x2 = x - (x @ zc) * zc
    x2 /= np.linalg.norm(x2)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x2, np.cross(zc, x2), zc, o * (d / (n @ o))
    return out


def detect_discs(bgr: np.ndarray, min_area: int = 12, max_area: int = 600, big: int = 4000):
    """Dark filled blobs (knot candidates): centroids (N, 2) px, areas (N,), and a mask of the large dark bodies
    (the pen and its machine) behind which no knot can be seen."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = (gray / np.maximum(cv2.GaussianBlur(gray, (0, 0), 15), 1) < 0.62).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    ys, xs = np.nonzero(lab)
    ll, a = lab[ys, xs], np.maximum(area, 1)
    mx, my = np.bincount(ll, xs, n) / a, np.bincount(ll, ys, n) / a
    cxx, cyy = np.bincount(ll, xs * xs, n) / a - mx ** 2, np.bincount(ll, ys * ys, n) / a - my ** 2
    cxy = np.bincount(ll, xs * ys, n) / a - mx * my
    half, root = (cxx + cyy) / 2, np.sqrt(np.maximum(((cxx - cyy) / 2) ** 2 + cxy ** 2, 0))
    l1, l2 = np.maximum(half + root, 1e-6), np.maximum(half - root, 1e-6)
    keep = (area >= min_area) & (area <= max_area) & (area / (4 * math.pi * np.sqrt(l1 * l2)) > 0.78) & (l1 / l2 < 12.25)
    keep[0] = False
    bodies = np.nonzero(area > big)[0]
    occluded = cv2.dilate(np.isin(lab, bodies[bodies > 0]).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    return cents[keep], area[keep], occluded


def project(pts: np.ndarray, cam_from_print: np.ndarray, k, dist) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of page points and whether each is in front of the camera."""
    obj = np.c_[pts, np.zeros(len(pts))]
    front = (obj @ cam_from_print[:3, :3].T + cam_from_print[:3, 3])[:, 2] > 0.02
    px, _ = cv2.projectPoints(obj, cv2.Rodrigues(cam_from_print[:3, :3])[0], cam_from_print[:3, 3], k, dist)
    return px.reshape(-1, 2), front


def _knot_size(lay, pose, k, dist):
    pred, front = project(lay.knots, pose, k, dist)
    ex, _ = project(lay.knots + [lay.knot_m / 2, 0], pose, k, dist)
    ey, _ = project(lay.knots + [0, lay.knot_m / 2], pose, k, dist)
    return pred, front, math.pi * np.linalg.norm(ex - pred, axis=1) * np.linalg.norm(ey - pred, axis=1)


def _shifts(pred_u, det_u, search, step=3, tol=3.5, keep=8, min_sep=10):
    """Image shifts of the predicted knots that land the most of them on a detection, best first."""
    grid = np.array([(dx, dy) for dx in range(-search, search + 1, step) for dy in range(-search, search + 1, step)], float)
    d, _ = cKDTree(det_u).query((pred_u[None] + grid[:, None]).reshape(-1, 2), distance_upper_bound=tol)
    scores = np.isfinite(d).reshape(len(grid), -1).sum(1)
    out = []
    for i in np.argsort(-scores):
        if all(np.hypot(*(grid[i] - o)) >= min_sep for o in out):
            out.append(grid[i])
        if len(out) == keep:
            break
    return out


def _seed(shift, lay, idx, pred_u, det, det_u, k, dist):
    """From one image shift: a RANSAC homography on the matched knots, re-matched through it, then PnP."""
    tree, page = cKDTree(det_u), lay.knots * 1000
    d, j = tree.query(pred_u + shift, distance_upper_bound=6.0)
    ok = np.isfinite(d)
    if ok.sum() < 12:
        return None
    h, _ = cv2.findHomography(page[idx[ok]].astype(np.float32), det_u[j[ok]].astype(np.float32), cv2.RANSAC, 2.5,
                              maxIters=2000, confidence=0.999)
    for gate in (3.0, 2.0):
        if h is None:
            return None
        d, j = tree.query(cv2.perspectiveTransform(page[idx].reshape(-1, 1, 2).astype(np.float32), h).reshape(-1, 2),
                          distance_upper_bound=gate)
        ok = np.isfinite(d)
        if ok.sum() < 12:
            return None
        h, _ = cv2.findHomography(page[idx[ok]].astype(np.float32), det_u[j[ok]].astype(np.float32), 0)
    obj = np.c_[lay.knots[idx[ok]], np.zeros(ok.sum())]
    good, rvec, tvec = cv2.solvePnP(obj, det[j[ok]].astype(np.float64), k, dist, flags=cv2.SOLVEPNP_IPPE)
    if not good:
        return None
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return _refine(pose, lay, det, k, dist)


def _refine(pose, lay, det, k, dist):
    """Re-match the knots from a pose and refine it by PnP: (pose, inliers, rms px) or None."""
    pred, front = project(lay.knots, pose, k, dist)
    idx = np.nonzero(front & (pred[:, 0] > 0) & (pred[:, 0] < 1280) & (pred[:, 1] > 0) & (pred[:, 1] < 720))[0]
    if len(idx) < 12:
        return None
    tree = cKDTree(det)
    for gate in (4.0, 2.5):
        d, j = tree.query(pred[idx], distance_upper_bound=gate)
        ok = np.isfinite(d)
        if ok.sum() < 12:
            return None
        obj, img = np.c_[lay.knots[idx[ok]], np.zeros(ok.sum())], det[j[ok]].astype(np.float64)
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, dist, cv2.Rodrigues(pose[:3, :3])[0],
                                          pose[:3, 3].reshape(3, 1).copy())
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        pred, _ = project(lay.knots, pose, k, dist)
    res = np.linalg.norm(cv2.projectPoints(obj, rvec, tvec, k, dist)[0].reshape(-1, 2) - img, axis=1)
    keep = res < 2.0
    return pose, int(keep.sum()), float(np.sqrt(np.mean(res[keep] ** 2))) if keep.any() else math.inf


def lattice_steps(step_m: float, reach: int = 2) -> list[np.ndarray]:
    """Translations of the knot lattice within `reach` steps along each basis vector."""
    a1, a2 = np.array([step_m, 0.0]), np.array([step_m / 2, step_m * math.sqrt(3) / 2])
    return [i * a1 + j * a2 for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)]


def support(pose, lay: Layout, det, det_u, occluded, k, dist, tol=3.0, min_px2=8.0) -> tuple[int, int, int]:
    """(hits, misses, misplaced). Hits and misses: knots the pose puts in clear view (inside the frame, not behind
    the pen, big enough to detect) with and without a detection within tol px. Misplaced: detections the pose
    puts one lattice step into the clear centre or out past the band, where a pose a step off finds real knots."""
    pred, front, size = _knot_size(lay, pose, k, dist)
    seen = front & (pred[:, 0] > 8) & (pred[:, 0] < 1272) & (pred[:, 1] > 8) & (pred[:, 1] < 712) & (size > min_px2)
    p = pred[seen]
    occ = occluded[np.clip(p[:, 1].astype(int), 0, occluded.shape[0] - 1), np.clip(p[:, 0].astype(int), 0, occluded.shape[1] - 1)]
    d, _ = cKDTree(det).query(p[~occ])
    rays = (np.linalg.inv(k) @ np.c_[det_u, np.ones(len(det_u))].T).T @ pose[:3, :3]   # in the page frame
    o = -pose[:3, :3].T @ pose[:3, 3]
    xy = o[:2] + (-o[2] / rays[:, 2])[:, None] * rays[:, :2]

    def ring(rect, a, b):
        x0, y0, x1, y1 = rect
        dm = np.minimum(np.minimum(xy[:, 0] - x0, x1 - xy[:, 0]), np.minimum(xy[:, 1] - y0, y1 - xy[:, 1]))
        return int(((dm > a) & (dm < b)).sum())
    st = lay.step_m
    grown = (lay.outer[0] - st, lay.outer[1] - st, lay.outer[2] + st, lay.outer[3] + st)
    return int((d < tol).sum()), int((d >= tol).sum()), ring(lay.inner, 0.4 * st, st) + ring(grown, 0.0, 0.6 * st)


@dataclass
class Fit:
    cam_from_print: np.ndarray
    inliers: int
    rms_px: float
    hits: int
    misses: int
    misplaced: int
    score: int
    margin: int                       # the score over the best candidate more than 3 mm away
    candidates: list = field(default_factory=list)


def fit_print(bgr: np.ndarray, lay: Layout, prior: np.ndarray, k, dist, plane=None, search_px: int = 110) -> Fit | None:
    """The print's pose in the camera frame. prior: a cam_from_print guess; plane: (n, d) from paper_plane,
    which replaces the prior's tilt and distance. None when too few knots are seen."""
    if plane is not None:
        prior = snap_to_plane(prior, *plane)
    det, area, occluded = detect_discs(bgr)
    pred, front, size = _knot_size(lay, prior, k, dist)
    if len(det) and front.any():                           # blobs of a knot's projected size
        _, j = cKDTree(pred[front]).query(det)
        ratio = area / size[front][j]
        det = det[(ratio > 0.45) & (ratio < 2.2)]
    if len(det) < 12:
        return None
    det_u = cv2.undistortPoints(det.reshape(-1, 1, 2), k, dist, P=k).reshape(-1, 2)
    idx = np.nonzero(front & (pred[:, 0] > -30) & (pred[:, 0] < 1310) & (pred[:, 1] > -30) & (pred[:, 1] < 750))[0]
    if len(idx) < 12:
        return None
    pred_u = cv2.undistortPoints(pred[idx].reshape(-1, 1, 2), k, dist, P=k).reshape(-1, 2)
    seeds = [s for s in (_seed(sh, lay, idx, pred_u, det, det_u, k, dist) for sh in _shifts(pred_u, det_u, search_px))
             if s is not None and s[1] >= 20]
    if not seeds:
        return None
    seed = max(seeds, key=lambda s: s[1])[0]
    fits = []
    for step in lattice_steps(lay.step_m):
        moved = seed.copy()
        moved[:3, 3] = seed[:3, 3] + seed[:3, :3] @ np.array([step[0], step[1], 0.0])
        r = _refine(moved, lay, det, k, dist)
        if r is None or r[1] < 12 or any(np.linalg.norm(r[0][:3, 3] - f[0][:3, 3]) < 0.001 for f in fits):
            continue
        hits, misses, misplaced = support(r[0], lay, det, det_u, occluded, k, dist)
        fits.append((*r, hits, misses, misplaced, hits - misses - 5 * misplaced))
    if not fits:
        return None
    fits.sort(key=lambda f: -f[6])
    best = fits[0]
    rival = max((f[6] for f in fits[1:] if np.linalg.norm(f[0][:3, 3] - best[0][:3, 3]) > 0.003), default=-10 ** 6)
    return Fit(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[6] - rival,
               [(f[0][:3, 3].tolist(), f[6]) for f in fits])


def tip_on_print(cam_from_print: np.ndarray, tip_cam) -> np.ndarray:
    """A camera-frame point in the print's frame: (x, y, height over the paper)."""
    pfc = np.linalg.inv(cam_from_print)
    return pfc[:3, :3] @ np.asarray(tip_cam, float) + pfc[:3, 3]


def measure(bgr, depth_raw, meta: dict, base_from_camera, base_from_page, lay: Layout, tip_cam) -> dict:
    """One wrist frame (colour, native depth and the gauge's npz metadata, the camera's FK pose and the run's
    page): the fit's numbers, the tip on the print, and the print's pose in the base. Plain lists, for a log."""
    ci = meta["color_intrinsics"]
    k, dist = np.asarray(ci["k"], float), np.asarray(ci["coeffs"], float)
    prior = np.linalg.inv(base_from_camera) @ base_from_page
    fit = fit_print(bgr, lay, prior, k, dist, plane=paper_plane(depth_raw, meta))
    if fit is None:
        return {"fit": None}
    return {"fit": True, "inliers": fit.inliers, "rms_px": round(fit.rms_px, 3), "hits": fit.hits,
            "misses": fit.misses, "misplaced": fit.misplaced, "margin": fit.margin,
            "accepted": fit.margin >= MARGIN_MIN, "tip_m": tip_on_print(fit.cam_from_print, tip_cam).tolist(),
            "cam_from_print": fit.cam_from_print.tolist(),
            "base_from_print": (np.asarray(base_from_camera) @ fit.cam_from_print).tolist()}
