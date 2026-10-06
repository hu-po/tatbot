"""The practice forearm's surface, measured: wrist D405 frames fused in the arm's base frame.

Every frame carries the camera pose its joints put it at (FK through the URDF), so frames from a
few views above the forearm land in one cloud without any external registration. The surface is
queried locally: a point's neighbours within a few millimetres give a plane (moving least squares
of degree one), which is the point on the skin and the normal the pen points against. The table
is a plane fitted to the lowest large flat region and removed; what stands above it is the arm.

The overhead D555 sees the whole forearm at once but has no trusted transform to this arm after a
re-layout; ``icp`` aligns its cloud of the forearm onto the wrist scan instead, so the forearm
itself is the calibration target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from tatbot_travel.camera import Intrinsics

UP = np.array([0.0, 0.0, 1.0])  # the arm base frame's z: away from the table


def deproject(depth_m: np.ndarray, intr: Intrinsics, mask: np.ndarray | None = None,
              min_m: float = 0.07, max_m: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points of the valid depth pixels, and those pixels' (row, col) indices."""
    valid = (depth_m > min_m) & (depth_m < max_m)
    if mask is not None:
        valid &= mask
    rows, cols = np.nonzero(valid)
    x, y = intr.undistort_normalized(cols.astype(float), rows.astype(float))
    z = depth_m[rows, cols]
    return np.stack([x * z, y * z, z], axis=1), np.stack([rows, cols], axis=1)


@dataclass
class Frame:
    """One aligned RGB-D view and where its camera stood (optical frame in the base frame)."""

    rgb: np.ndarray  # HxWx3 uint8
    depth_m: np.ndarray  # HxW float, aligned to rgb
    intr: Intrinsics
    cam_p: np.ndarray  # (3,)
    cam_r: np.ndarray  # (3, 3), columns are the optical axes x right, y down, z forward
    range_m: tuple[float, float] = (0.07, 0.6)  # depth worth trusting (the wrist D405's)

    def points(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base-frame points, their colours and pixel indices."""
        cam, px = deproject(self.depth_m, self.intr, mask, *self.range_m)
        return cam @ self.cam_r.T + self.cam_p, self.rgb[px[:, 0], px[:, 1]], px

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame points to pixels (u, v) and their depth along the optical axis."""
        cam = (np.asarray(points, float) - self.cam_p) @ self.cam_r
        return self.intr.project(cam), cam[:, 2]


def voxel(points: np.ndarray, size_m: float, *extra: np.ndarray) -> tuple[np.ndarray, ...]:
    """One averaged point per occupied voxel (and the matching rows of ``extra`` arrays)."""
    keys = np.floor(points / size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out = []
    for arr in (points, *extra):
        acc = np.zeros((len(counts), *arr.shape[1:]))
        np.add.at(acc, inverse, arr)
        out.append(acc / counts.reshape(-1, *([1] * (arr.ndim - 1))))
    return tuple(out)


def fuse(frames: list[Frame], size_m: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Every frame's points in the base frame, voxel-averaged: (points, rgb)."""
    pts, cols = zip(*(f.points()[:2] for f in frames), strict=True)
    points, rgb = voxel(np.concatenate(pts), size_m, np.concatenate(cols).astype(float))
    return points, rgb


def table_plane(points: np.ndarray, *, iters: int = 300, tol_m: float = 0.003, seed: int = 0,
                up: np.ndarray = UP) -> tuple[np.ndarray, float]:
    """RANSAC: the plane with the most support among candidates within 20 degrees of ``up`` (the base
    frame's z, or a camera frame's estimate of it), as (unit normal toward ``up``, offset)."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, float) / np.linalg.norm(up)
    best, best_count = (up, 0.0), -1
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.radians(20)):
            continue
        count = int(np.count_nonzero(np.abs(points @ n - n @ a) < tol_m))
        if count > best_count:
            best, best_count = (n, float(n @ a)), count
    n, d = best
    inliers = np.abs(points @ n - d) < tol_m  # refit on the inliers
    centred = points[inliers] - points[inliers].mean(axis=0)
    n = np.linalg.svd(centred, full_matrices=False)[2][-1]
    n = n if n @ up > 0 else -n
    return n, float(n @ points[inliers].mean(axis=0))


def above(points: np.ndarray, plane: tuple[np.ndarray, float], min_m: float = 0.005) -> np.ndarray:
    """Which points stand at least ``min_m`` above the plane."""
    n, d = plane
    return points @ n - d > min_m


def largest_cluster(points: np.ndarray, link_m: float = 0.004) -> np.ndarray:
    """Mask of the largest connected component (points within ``link_m`` of each other)."""
    from scipy.sparse.csgraph import connected_components

    tree = cKDTree(points)
    pairs = tree.query_pairs(link_m, output_type="ndarray")
    from scipy.sparse import coo_matrix

    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


class Surface:
    """Local planes over a measured cloud: the nearest point on the skin and its outward normal."""

    def __init__(self, points: np.ndarray, radius_m: float = 0.006, min_neighbours: int = 12):
        self.points = np.asarray(points, float)
        self.tree = cKDTree(self.points)
        self.radius, self.min_neighbours = radius_m, min_neighbours
        self.centroid = self.points.mean(axis=0)

    def query(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """The local plane at the measured point nearest ``p``: ``p`` projected onto it, its unit normal
        (pointing out of the arm) and the neighbour count (0: too few points to call it surface)."""
        _, nearest = self.tree.query(p)
        idx = self.tree.query_ball_point(self.points[nearest], self.radius)
        if len(idx) < self.min_neighbours:
            return np.asarray(p, float), UP.copy(), 0
        nb = self.points[idx]
        c = nb.mean(axis=0)
        n = np.linalg.svd(nb - c, full_matrices=False)[2][-1]
        if n @ (c - self.centroid) < 0 and n @ UP < 0.5:
            n = -n  # a side of the forearm faces outward from its axis
        elif n @ UP < 0:
            n = -n  # the top faces up
        return p - ((p - c) @ n) * n, n, len(idx)

    def project_path(self, path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every path point moved onto the surface: (points, normals, supported mask)."""
        out = [self.query(p) for p in np.asarray(path, float)]
        pts, normals, counts = zip(*out, strict=True)
        return np.array(pts), np.array(normals), np.array(counts) > 0


def icp(source: np.ndarray, target: np.ndarray, init: np.ndarray | None = None, *,
        iters: int = 40, max_pair_m: float = 0.02, radius_m: float = 0.006,
        surface: Surface | None = None, normals: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP moving ``source`` onto ``target``: (4x4 transform, rms residual of the pairs).
    ``surface``/``normals`` of the target may be passed in when several starts share it."""
    surface = surface or Surface(target, radius_m=radius_m)
    normals = normals if normals is not None else np.array([surface.query(p)[1] for p in target])
    tf = np.eye(4) if init is None else np.array(init, float)
    rms = np.inf
    for _ in range(iters):
        moved = source @ tf[:3, :3].T + tf[:3, 3]
        dist, idx = surface.tree.query(moved)
        keep = dist < max_pair_m
        if keep.sum() < 6:
            raise RuntimeError("icp: too few pairs within reach; the initial transform is too far off")
        p, q, n = moved[keep], target[idx[keep]], normals[idx[keep]]
        r = ((p - q) * n).sum(axis=1)
        a = np.hstack([np.cross(p, n), n])
        x = np.linalg.lstsq(a, -r, rcond=None)[0]
        step = np.eye(4)
        step[:3, :3] = _rotation(x[:3])
        step[:3, 3] = x[3:]
        tf = step @ tf
        rms = float(np.sqrt(np.mean(r ** 2)))
        if np.linalg.norm(x) < 1e-7:
            break
    return tf, rms


def _rotation(w: np.ndarray) -> np.ndarray:
    """Rodrigues: the rotation by the vector ``w`` (radians)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx
