"""The overhead D555's view of the forearm, put in the blue arm's base frame by the forearm itself.

A re-layout moves the D555 relative to the arm, and a stored registration goes stale with it. The wrist
scan already measures the forearm in the base frame, so the overhead view is aligned to that: the two
table planes fix tilt and height, and point-to-plane ICP of the wrist scan's forearm onto the
overhead's, started from a ring of yaws, finds the turn that pairs the most of the scan. The overhead then adds the whole forearm, and a view the audience can
read, in the same frame as the pen.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tatbot_travel import inkmap, surface
from tatbot_travel.camera import Intrinsics

CAMERA_UP = np.array([0.0, 0.0, -1.0])  # looking down: the table's normal points back at the camera
OVERHEAD_RANGE_M = (0.2, 1.5)


def capture(repo: Path) -> surface.Frame:
    """One aligned colour/depth set from the D555 owner over the tatbot bus, in its own optical frame."""
    for sub in ("ros/tatbot_bridge", "ros/tatbot_description"):
        if str(repo / sub) not in sys.path:
            sys.path.insert(0, str(repo / sub))
    from tatbot_bridge.capture import Camera, intrinsics_of

    camera = Camera(repo)
    try:
        cap = camera.capture(time.time_ns() - 1_000_000_000)
    finally:
        camera.close()
    if cap["depth_m"] is None:
        raise RuntimeError("the overhead capture carries no depth")
    return frame_of(cap["image"][..., ::-1], cap["depth_m"], intrinsics_of(cap["metadata"]))


def frame_of(rgb: np.ndarray, depth_m: np.ndarray, intrinsics: dict) -> surface.Frame:
    model = "inverse_brown_conrady" if "Inverse" in intrinsics["distortion_model"] else "brown_conrady"
    intr = Intrinsics(int(intrinsics["width"]), int(intrinsics["height"]), intrinsics["fx"], intrinsics["fy"],
                      intrinsics["ppx"], intrinsics["ppy"],
                      tuple((list(intrinsics.get("distortion_coefficients") or []) + [0.0] * 5)[:5]), model)
    return surface.Frame(rgb=np.ascontiguousarray(rgb), depth_m=depth_m, intr=intr, cam_p=np.zeros(3), cam_r=np.eye(3),
                         range_m=OVERHEAD_RANGE_M)


def forearm_cloud(frame: surface.Frame, up: np.ndarray) -> tuple[np.ndarray, tuple]:
    """The forearm's points (in the frame's coordinates) and the table plane under it."""
    points, _, _ = frame.points()
    plane = surface.table_plane(points, up=up)
    mask = inkmap.forearm_pixels(frame, plane)
    arm, _, _ = frame.points(mask)
    (arm,) = surface.voxel(arm, 0.002)
    return arm, plane


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _facing(points: np.ndarray, up: np.ndarray, min_cos: float = 0.6) -> np.ndarray:
    """The points whose surface faces up off the table: what an overhead and a wrist view both see. The
    wrist also sees the forearm's flank, which the overhead cannot; left in, ICP slides that flank onto
    the overhead's top and calls the bad turn a better fit."""
    skin = surface.Surface(points)
    normals = np.array([skin.query(p)[1] for p in points])
    return points[np.abs(normals @ up) >= min_cos]


def align(scan_arm: np.ndarray, scan_plane: tuple, over_arm: np.ndarray, over_plane: tuple, *,
          seeds: int = 12, inlier_m: float = 0.005) -> tuple[np.ndarray, float, float]:
    """``overhead_from_base`` (4x4) moving the wrist scan's forearm onto the overhead's, its ICP rms and
    the fraction of scan points it pairs within ``inlier_m``.

    The table planes fix tilt and height. Yaw is searched from ``seeds`` starts about the table normal,
    each refined by ICP; the winner pairs the most scan points. Rms alone would not do: a partial scan
    can fit one round feature (the fist) at any turn, with a small residual and little of the arm."""
    from scipy.spatial import cKDTree

    n_s, n_o = scan_plane[0], over_plane[0]
    scan_arm, over_arm = _facing(scan_arm, n_s), _facing(over_arm, n_o)
    (src,) = surface.voxel(scan_arm, 0.003)
    target = surface.Surface(over_arm)
    normals = np.array([target.query(p)[1] for p in over_arm])
    tree = cKDTree(over_arm)
    r1 = _rotation_between(n_s, n_o)
    best = (None, np.inf, 0.0)
    for yaw in np.linspace(0.0, 2 * np.pi, seeds, endpoint=False):
        k = np.array([[0, -n_o[2], n_o[1]], [n_o[2], 0, -n_o[0]], [-n_o[1], n_o[0], 0]])
        r = (np.eye(3) + np.sin(yaw) * k + (1 - np.cos(yaw)) * k @ k) @ r1
        tf = np.eye(4)
        tf[:3, :3] = r
        tf[:3, 3] = over_arm.mean(axis=0) - r @ src.mean(axis=0)
        # Height from the planes, not the centroids (each camera sees a different part of the arm).
        tf[:3, 3] += n_o * ((over_plane[1] - scan_plane[1]) - n_o @ tf[:3, 3])
        try:
            for reach in (0.06, 0.03, 0.015, 0.008):
                tf, rms = surface.icp(src, over_arm, tf, iters=25, max_pair_m=reach, surface=target, normals=normals)
        except RuntimeError:
            continue
        paired = float(np.mean(tree.query(src @ tf[:3, :3].T + tf[:3, 3])[0] < inlier_m))
        if paired > best[2] or (paired == best[2] and rms < best[1]):
            best = (tf, rms, paired)
    if best[0] is None:
        raise RuntimeError("the overhead forearm did not align with the wrist scan")
    return best


def load(path: Path) -> surface.Frame:
    data = np.load(path, allow_pickle=True)
    return frame_of(data["bgr"][..., ::-1], data["depth"], json.loads(str(data["intrinsics"])))
