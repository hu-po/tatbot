"""Wrist camera observation, page-depth fitting and print/ink inspection; see ros/README.md."""
from __future__ import annotations

import sys
import time

import numpy as np
from tatbot_description import repo_root
from tatbot_description.transforms import rpy_matrix

from tatbot_session.geometry import SIDES, inner_edges

sys.path.insert(0, str(repo_root(None) / "scripts/lib"))
from rgbd_geometry import depth_points, page_height_from_depth  # noqa: E402,F401

CAMERA_FRAME = "{arm}/realsense_color_optical_frame"   # optical: z forward, x right, y down
PAGE_SIZE = (0.100, 0.150)                              # the nominal stencil page


def page_extent(size_m, more_m: float = 0.010) -> tuple[float, float, float, float]:
    """A page image's extent in the page frame (xmin, ymin, xmax, ymax), metres: the page and `more_m` round it."""
    hx, hy = size_m[0] / 2.0 + more_m, size_m[1] / 2.0 + more_m
    return (-hx, -hy, hx, hy)


PAGE_EXTENT = page_extent(PAGE_SIZE)                    # page images of the nominal page


def camera_frame(arm: str) -> str:
    return CAMERA_FRAME.format(arm=arm)


def look_at(base_from_page: np.ndarray, xy, *, tilt_rad: float, azimuth_rad: float, distance_m: float,
            roll_rad: float = 0.0) -> np.ndarray:
    """base_from_camera `distance_m` from page point xy, tilted `tilt_rad` off the page normal toward
    azimuth `azimuth_rad` (from page x toward page y), looking at the point. At roll 0 the image x axis
    is as close to page y as the view allows; roll_rad turns the image about the view axis."""
    page = np.asarray(base_from_page, dtype=float)
    target = (page @ [xy[0], xy[1], 0.0, 1.0])[:3]
    out_dir = (np.cos(tilt_rad) * page[:3, 2] + np.sin(tilt_rad) *
               (np.cos(azimuth_rad) * page[:3, 0] + np.sin(azimuth_rad) * page[:3, 1]))
    z = -out_dir / np.linalg.norm(out_dir)
    x = page[:3, 1] - (page[:3, 1] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = page[:3, 0] - (page[:3, 0] @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    x, y = np.cos(roll_rad) * x + np.sin(roll_rad) * y, -np.sin(roll_rad) * x + np.cos(roll_rad) * y
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, target - distance_m * z
    return out


def look_down(base_from_page: np.ndarray, xy, height_m: float) -> np.ndarray:
    """look_at straight down the page normal from height_m."""
    return look_at(base_from_page, xy, tilt_rad=0.0, azimuth_rad=0.0, distance_m=height_m)


def tcp_target(base_from_camera: np.ndarray, tcp_from_camera: np.ndarray) -> np.ndarray:
    """The tcp pose that puts the camera at base_from_camera (the camera rides the wrist with the tool)."""
    return np.asarray(base_from_camera) @ np.linalg.inv(tcp_from_camera)


REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0])   # the landing sleep pose
JOINT_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0])            # wrist moves cost more


SIDE_LINKS = ("{arm}/link_3", "{arm}/link_4", "{arm}/link_5", "{arm}/link_6")
# The wrist's own bodies, which the pen's clearance says nothing about: the fiducial cube's tag faces, the
# camera and the tool mount. A view keeps each on the arm's side and BODY_CLEARANCE_M over the page: the last
# post-draw view (roll 90) put the cube's lowest tag face 33 mm over the table and 140-180 mm past the camera
# toward the palette (2026-09-30). The cube's tags are 47 mm, so 70 mm to a face centre keeps its corners clear.
WRIST_BODY = ("{arm}/wrist_tag2", "{arm}/wrist_tag3", "{arm}/wrist_tag4", "{arm}/realsense_link", "{arm}/tool_mount")
BODY_CLEARANCE_M = 0.070
# A view stays near the pen-down pose over its target, not the folded sleep pose: scored from the sleep pose, the
# chooser picked contortions, and one folded the wrist's fiducial cube into the arm's own upper link (2026-09-30).
# The pen within MAX_PEN_TILT_DEG of straight down into the page, each wrist joint within MAX_WRIST_RAD of the
# reference, and the cube's tag faces CUBE_UP_M or more over the tool mount along the page normal.
CUBE = ("{arm}/wrist_tag2", "{arm}/wrist_tag3", "{arm}/wrist_tag4")
MAX_PEN_TILT_DEG = 30.0
MAX_WRIST_RAD = 0.7         # 40 deg
CUBE_UP_M = 0.030
HOVER_M = 0.120          # the reference: the pen straight down this high over the target


def pen_down_reference(kin, solve_ik, page, xy, rest=REST, height_m: float = HOVER_M):
    """Joints with the pen straight down over page point xy at `height_m`, at the pen heading (its turn about its
    own axis, free for a round ballpoint) that keeps every joint farthest from its limits; None if none solves."""
    from tatbot_session import geometry

    lower, upper = np.asarray(kin.lower)[:6], np.asarray(kin.upper)[:6]
    best = None
    for yaw in np.radians(np.arange(0.0, 360.0, 30.0)):
        heading = np.eye(4)
        heading[:3, 0] = np.cos(yaw) * page[:3, 0] + np.sin(yaw) * page[:3, 1]
        try:
            q = solve_ik(kin, geometry.tool_down_pose(page, xy, height_m, heading), np.asarray(rest, float))
        except ValueError:
            continue
        margin = float(np.min(np.minimum(q[:6] - lower, upper - q[:6])))
        if best is None or margin > best[0]:
            best = (margin, np.asarray(q, float))
    return None if best is None else best[1]


def upright(kin, q, ref, page, cube, mount, *, max_pen_tilt_deg: float = MAX_PEN_TILT_DEG,
            max_wrist_rad: float = MAX_WRIST_RAD, cube_up_m: float = CUBE_UP_M) -> bool:
    """The pen mostly down, the wrist joints near the pen-down reference `ref`, the cube over the tool mount."""
    down = kin.fk(ref)[:3, 2]
    if float(np.degrees(np.arccos(np.clip(kin.fk(q)[:3, 2] @ down, -1.0, 1.0)))) > max_pen_tilt_deg:
        return False
    if np.any(np.abs(np.asarray(q)[3:6] - np.asarray(ref)[3:6]) > max_wrist_rad):
        return False
    if not cube or not mount:
        return True
    under = kin.frame(q, mount)[:3, 3] @ page[:3, 2]
    return all(float(kin.frame(q, f)[:3, 3] @ page[:3, 2] - under) >= cube_up_m for f in cube)


def frames_of(kin, q, names) -> list[str]:
    """The named frames (`{arm}` filled) that this arm's description has; the others are skipped."""
    out = []
    for name in (n.format(arm=kin.arm) for n in names):
        try:
            kin.frame(q, name)
        except ValueError:
            continue
        out.append(name)
    return out


def on_side(kin, q, frames, max_y: float | None) -> bool:
    """Every named frame at base y <= max_y (None: no limit)."""
    return max_y is None or all(float(kin.frame(q, f)[1, 3]) <= max_y for f in frames)


def view_clear(kin, q, page, *, margin_rad: float, clearance_m: float, body, body_clearance_m: float, side,
               max_y: float | None, posture=None) -> bool:
    """A view's joints clear of their limits, the pen `clearance_m` and the wrist's bodies `body_clearance_m`
    over the page, the side frames at base y <= max_y, and `posture(q)` (upright) when given."""
    if np.any(q[:6] < np.asarray(kin.lower)[:6] + margin_rad) or np.any(q[:6] > np.asarray(kin.upper)[:6] - margin_rad):
        return False
    if posture is not None and not posture(q):
        return False
    if float((kin.fk(q)[:3, 3] - page[:3, 3]) @ page[:3, 2]) < clearance_m:
        return False
    if any(float((kin.frame(q, f)[:3, 3] - page[:3, 3]) @ page[:3, 2]) < body_clearance_m for f in body):
        return False
    return on_side(kin, q, side, max_y)


VIEW_SELF_MARGIN_M = 0.015    # a view itself stands this far past the guard's floor; its way at least at the floor


def path_clear(self_gap, q_from, q_to, samples: int = 12, end_margin_m: float = VIEW_SELF_MARGIN_M) -> bool:
    """The straight joint-space way from q_from to q_to (ready.joint_move is one) never comes onto the arm itself:
    `self_gap(q)` (tatbot_motion.collision.Guard.self_gap, m) stays >= end_margin_m at q_to and nowhere under
    0 or where it starts. The client's inspect moves bypass the session's guard, so the choice checks its way."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    floor = min(0.0, float(self_gap(q_from)))
    if float(self_gap(q_to)) < end_margin_m:
        return False
    return all(float(self_gap(q_from + s * (q_to - q_from))) >= floor for s in np.linspace(0.0, 1.0, samples)[1:-1])


def view_targets(page, xy, tcp_from_cam, down, *, tilts_deg, distances_m, azimuths, rolls_deg,
                 max_pen_tilt_deg: float):
    """Each candidate view ({"tilt_deg", "azimuth_deg", "distance_m", "roll_deg"}, base_from_tcp target) whose
    pen stays within max_pen_tilt_deg of `down`, before any IK: most of the grid never needs solving."""
    cos_max = float(np.cos(np.radians(max_pen_tilt_deg)))
    for tilt in np.asarray(list(tilts_deg), dtype=float):
        for d in distances_m:
            for az in np.asarray(azimuths, dtype=float):
                for roll in np.asarray(list(rolls_deg), dtype=float):
                    cam = look_at(page, xy, tilt_rad=np.radians(tilt), azimuth_rad=az, distance_m=d,
                                  roll_rad=np.radians(roll))
                    target = tcp_target(cam, tcp_from_cam)
                    if float(target[:3, 2] @ down) >= cos_max:
                        yield ({"tilt_deg": float(tilt), "azimuth_deg": float(np.degrees(az)) % 360,
                                "distance_m": float(d), "roll_deg": float(roll)}, target)


def aim(kin, solve_ik, base_from_page, xy, q_seed, *, frame: str, clearance_m: float, azimuths=None,
        tilts_deg=(20, 30, 35, 40, 45, 50), distances_m=(0.18, 0.20, 0.22, 0.24), rolls_deg=range(0, 360, 45),
        rest=REST, max_y: float | None = None, margin_rad: float = 0.0667, body_clearance_m: float = BODY_CLEARANCE_M,
        reference=None, self_gap=None, max_pen_tilt_deg: float = MAX_PEN_TILT_DEG, cube_up_m: float = CUBE_UP_M):
    """Joints that point the camera at page point xy with the pen at least `clearance_m` over the page: the
    least contorted of every reachable view, tilt x azimuth (default every 15 deg) x distance x roll about the
    view axis (rectification does not care which way the image is turned).

    A view stays upright (upright): the pen within `max_pen_tilt_deg` of straight down, each wrist joint within
    MAX_WRIST_RAD of the pen-down `reference` over xy (pen_down_reference, solved from `rest`), and the wrist's
    fiducial cube `cube_up_m` over the tool mount. It is seeded at the reference and scored by the weighted
    joint distance from it: scored from the folded sleep pose, the chooser picked contortions, and one folded
    the cube into the arm's own upper link (2026-09-30). With `self_gap` (Guard.self_gap for this arm) the view
    and the straight joint way to it from q_seed keep the arm off itself (path_clear).
    The right wrist D405 sits ~48 deg off the pen axis; the pen reaches ~160 mm past it, so the nearest pose
    with the pen 15 mm clear is ~180 mm out (~3.7 px/mm). With `max_y`, the camera and the wrist links stay at
    base y <= max_y, so the right arm keeps to its own side of the table, clear of the palette mid-table; the
    wrist's bodies (WRIST_BODY) keep to it too, and stay `body_clearance_m` over the page wherever the pen is.
    A view whose revolute joints come within `margin_rad` of their limits is skipped (ready.MARGIN_RAD, twice
    the planner's guard): the next plan starts from it, and one left joint_1 at -0.004 rad, which the planner
    refused to leave (bench 2026-09-27).
    Returns (q, {"tilt_deg", "azimuth_deg", "distance_m", "roll_deg", "cost"}) or None."""
    page = np.asarray(base_from_page, dtype=float)
    tcp_from_cam = np.linalg.inv(kin.fk(q_seed)) @ kin.frame(q_seed, frame)
    body = frames_of(kin, q_seed, WRIST_BODY)
    side = [frame] + [f.format(arm=kin.arm) for f in SIDE_LINKS] + body
    azimuths = np.radians(np.arange(0, 360, 15)) if azimuths is None else np.asarray(azimuths, dtype=float)
    ref = pen_down_reference(kin, solve_ik, page, xy, rest) if reference is None else np.asarray(reference, float)
    if ref is None:
        return None
    seed = ref.copy()
    seed[6] = q_seed[6]
    cube, mount = frames_of(kin, q_seed, CUBE), (frames_of(kin, q_seed, ("{arm}/tool_mount",)) or [None])[0]

    def accept(q):
        return (upright(kin, q, seed, page, cube, mount, max_pen_tilt_deg=max_pen_tilt_deg, cube_up_m=cube_up_m)
                and (self_gap is None or path_clear(self_gap, q_seed, q)))

    best = None
    for view, target in view_targets(page, xy, tcp_from_cam, kin.fk(seed)[:3, 2], tilts_deg=tilts_deg,
                                      distances_m=distances_m, azimuths=azimuths, rolls_deg=rolls_deg,
                                      max_pen_tilt_deg=max_pen_tilt_deg):
        try:
            q = solve_ik(kin, target, seed)
        except ValueError:
            continue
        if not view_clear(kin, q, page, margin_rad=margin_rad, clearance_m=clearance_m, body=body,
                          body_clearance_m=body_clearance_m, side=side, max_y=max_y, posture=accept):
            continue
        cost = float(np.sum(JOINT_WEIGHTS * (q - seed) ** 2))
        if best is None or cost < best[1]["cost"]:
            best = (q, {**view, "cost": cost})
    return best


def newest_run(root, name: str = ""):
    """The ros-draw run to inspect: `name`, else the newest holding page.json and program.json."""
    runs = sorted(p for p in root.glob("2*") if (p / "page.json").is_file() and (p / "program.json").is_file())
    run = (root / name) if name else (runs[-1] if runs else None)
    return run if run is not None and (run / "page.json").is_file() else None


def page_grid(extent_m, px_per_mm: float):
    """Page-frame (x, y) of every pixel centre of a page image covering extent_m = (xmin, ymin, xmax,
    ymax), row 0 at ymax (the top of the print), column 0 at xmin."""
    xmin, ymin, xmax, ymax = extent_m
    step = 1e-3 / px_per_mm
    xs = xmin + (np.arange(int(round((xmax - xmin) / step))) + 0.5) * step
    ys = ymax - (np.arange(int(round((ymax - ymin) / step))) + 0.5) * step
    return np.meshgrid(xs, ys)


def page_to_pixel(points_page: np.ndarray, base_from_page: np.ndarray, base_from_camera: np.ndarray,
                  k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Pixel (u, v) of page-plane points (N, 2), with the Brown-Conrady distortion (k1 k2 p1 p2 k3)."""
    pts = np.asarray(points_page, dtype=float).reshape(-1, 2)
    homog = np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))]
    cam = (np.linalg.inv(base_from_camera) @ base_from_page @ homog.T)[:3].T
    x, y = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    k1, k2, p1, p2, k3 = (list(np.asarray(dist, float)) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]]


def rectify(image: np.ndarray, base_from_page, base_from_camera, k, dist, extent_m, px_per_mm: float) -> np.ndarray:
    """The camera image resampled onto the page plane (page_grid layout); outside the frame is black."""
    import cv2

    gx, gy = page_grid(extent_m, px_per_mm)
    uv = page_to_pixel(np.c_[gx.ravel(), gy.ravel()], base_from_page, base_from_camera, k, dist)
    mx = uv[:, 0].reshape(gx.shape).astype(np.float32)
    my = uv[:, 1].reshape(gx.shape).astype(np.float32)
    return cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_image_px(points_m, extent_m, px_per_mm: float) -> np.ndarray:
    """Page-frame metres -> page-image pixel coordinates (float)."""
    xmin, _, _, ymax = extent_m
    p = np.asarray(points_m, dtype=float).reshape(-1, 2)
    return np.c_[(p[:, 0] - xmin) * 1000 * px_per_mm, (ymax - p[:, 1]) * 1000 * px_per_mm]


def overlay(page_image: np.ndarray, program: dict, clear_m, extent_m, px_per_mm: float) -> np.ndarray:
    """Planned strokes (thin red) and the clear-centre outline (green) over a page image."""
    import cv2

    out = page_image.copy() if page_image.ndim == 3 else cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)
    hx, hy = clear_m[0] / 2, clear_m[1] / 2
    box = to_image_px([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], extent_m, px_per_mm)
    cv2.polylines(out, [np.round(box).astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
    for op in program.get("ops", []):
        if op.get("op") != "stroke":
            continue
        pts = op["points_m"] + ([op["points_m"][0]] if op.get("closed") else [])
        px = np.round(to_image_px(pts, extent_m, px_per_mm)).astype(np.int32)
        cv2.polylines(out, [px], False, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def median_stack(images) -> np.ndarray:
    """Pixelwise median of same-sized page images, ignoring black (outside a frame) pixels."""
    arr = np.stack([np.asarray(i, dtype=np.float32) for i in images])
    mask = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], -1).max(axis=-1) > 0
    arr[~mask] = np.nan
    med = np.nanmedian(arr, axis=0)
    return np.nan_to_num(med, nan=0.0).astype(np.uint8)


class WristCamera:
    """Exclusive ROS wrist D405: colour and optional native or colour-aligned depth in metres."""

    def __init__(self, serial: str = "", width: int = 1280, height: int = 720, fps: int = 30, depth: bool = False,
                 aligned: bool = False):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5], dtype=float)
        self.model = str(intr.model)
        self.serial = profile.get_device().get_info(rs.camera_info.serial_number)
        self.depth = depth
        self.align = rs.align(rs.stream.color) if depth and aligned else None
        if depth:
            dp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            di = dp.get_intrinsics()
            ex = dp.get_extrinsics_to(profile.get_stream(rs.stream.color))
            self.k_depth = self.k if self.align else np.array([[di.fx, 0, di.ppx], [0, di.fy, di.ppy], [0, 0, 1]])
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self.depth_metadata = {"serial": self.serial, "aligned_to_color": bool(self.align),
                                   "depth_units_m": self.depth_scale, "width": di.width, "height": di.height,
                                   "color_intrinsics": {"k": self.k.tolist(), "coeffs": self.dist.tolist(), "model": self.model},
                                   "color_from_depth": {"rotation": list(ex.rotation), "translation_m": list(ex.translation),
                                                        "rotation_layout": "column_major"},
                                   "intrinsics": {"fx": di.fx, "fy": di.fy, "ppx": di.ppx, "ppy": di.ppy,
                                                  "model": str(di.model), "coeffs": list(di.coeffs)}}
        for _ in range(30):   # auto exposure settles
            self.pipe.wait_for_frames(5000)

    def grab(self, n: int) -> list[np.ndarray]:
        return self.grab_both(n)[0]

    def grab_both(self, n: int, observe=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """n colour frames, and n depth frames in metres (empty without depth)."""
        colors, depths = [], []
        self.capture_timestamps = []
        self.raw_depth = []
        for _ in range(n):
            fs = self.pipe.wait_for_frames(5000)
            if self.align is not None:
                fs = self.align.process(fs)
            colors.append(np.asanyarray(fs.get_color_frame().get_data()).copy())
            if self.depth:
                frame = fs.get_depth_frame()
                self.raw_depth.append(np.asanyarray(frame.get_data()).copy())
                depths.append(self.raw_depth[-1].astype(np.float32) * self.depth_scale)
                self.capture_timestamps.append({"received_at_ns": time.time_ns(), "depth_ms": frame.get_timestamp(),
                                                "received_monotonic_s": time.monotonic(),
                                                "joints": None if observe is None else observe(),
                                                "depth_domain": str(frame.get_frame_timestamp_domain()),
                                                "color_ms": fs.get_color_frame().get_timestamp(),
                                                "color_domain": str(fs.get_color_frame().get_frame_timestamp_domain())})
        return colors, depths

    def close(self) -> None:
        self.pipe.stop()


DEPTH_FRAME = "{arm}/realsense_depth_optical_frame"


def fuse_heights(sources: list[tuple[str, float, float]], *, warn_m: float) -> dict:
    """Inverse-variance fusion of centre-normal heights; warn on disagreement without dropping sources."""
    vals = np.array([s[1] for s in sources], dtype=float)
    w = 1.0 / np.maximum(np.array([s[2] for s in sources], dtype=float), 1e-6) ** 2
    fused = float((w * vals).sum() / w.sum())
    warn = [f"{name} {(v - fused) * 1e3:+.1f} mm" for (name, v, _), v in zip(sources, vals, strict=True)
            if abs(v - fused) > warn_m]
    return {"offset_m": fused, "sigma_m": float(1.0 / np.sqrt(w.sum())),
            "sources": [{"name": n, "offset_m": v, "sigma_m": s} for n, v, s in sources], "warn": warn}


# --- the printed border: where the print really is on the page plane ------------------------------


def print_mask(page_image: np.ndarray, px_per_mm: float, *, darkness: float = 0.3, feature_mm: float = 6.0):
    """(dark, valid) masks of a page image: dark where the pixel is `darkness` below the local paper
    white (a max filter wider than the border's ~4 mm filled shapes, so shadows fall out), valid where
    the camera saw the plane (rectify leaves the rest black)."""
    import cv2

    gray = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    k = max(3, int(round(feature_mm * px_per_mm)) | 1)
    white = cv2.GaussianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), (k, k), 0)
    valid = gray > 8
    dark = valid & (gray < (1.0 - darkness) * np.maximum(white, 1.0))
    return dark, valid




def _strip_density(dark, valid, extent_m, px_per_mm: float, axis: int, scan, along, step_m: float,
                   min_valid: float) -> np.ndarray:
    """The dark share of each strip `step_m` wide at the scan positions, across the segment `along`
    (a0, a1) of a side along `axis`; nan where the camera saw less than `min_valid` of it."""
    xmin, _, _, ymax = extent_m
    px = 1e-3 / px_per_mm
    a0, a1 = along

    def to_px(x, y):
        return int(round((x - xmin) / px)), int(round((ymax - y) / px))

    dens = []
    for s in scan:
        if axis == 0:
            (c0, r0), (c1, r1) = to_px(s - step_m / 2, a1), to_px(s + step_m / 2, a0)
        else:
            (c0, r0), (c1, r1) = to_px(a0, s + step_m / 2), to_px(a1, s - step_m / 2)
        r0, r1, c0, c1 = max(r0, 0), max(r1, r0 + 1), max(c0, 0), max(c1, c0 + 1)
        v = valid[r0:r1, c0:c1]
        dens.append(np.nan if v.size == 0 or v.mean() < min_valid else float(dark[r0:r1, c0:c1][v].mean()))
    return np.asarray(dens)


def _first_run(dens: np.ndarray, inside: int, need: int, need_clear: int, density: float) -> int | None:
    """The first scan index that starts `need` strips over `density` after `need_clear` clean ones, or
    None; None too when the first `inside` strips (inside the nominal edge) were not all seen."""
    if np.isnan(dens[:inside]).any():
        return None   # the scan inside the nominal edge must be seen whole, or a cut edge could pass for the border
    over = np.nan_to_num(dens, nan=0.0) > density
    hit = next((i for i in range(len(over) - need) if over[i:i + need].all()), None)
    if hit is not None and (hit < need_clear or over[hit - need_clear:hit].any()):
        return None
    return hit


def drawing_extent(program: dict | None):
    """(xmin, ymin, xmax, ymax), page metres, of every stroke the program plans; None without one."""
    points = [p for op in (program or {}).get("ops", []) if op.get("op") == "stroke" for p in op["points_m"]]
    if not points:
        return None
    pts = np.asarray(points, dtype=float)[:, :2]
    return (*(float(v) for v in pts.min(axis=0)), *(float(v) for v in pts.max(axis=0)))


def border_edges(page_image: np.ndarray, extent_m, px_per_mm: float, clear_m, *, inner_m=None, size_m=PAGE_SIZE,
                 drawn_m=None, search_m: float = 0.020, segments: int = 8, step_m: float = 0.00025,
                 density: float = 0.12, run_m: float = 0.002, clear_run_m: float = 0.003,
                 min_valid: float = 0.6) -> list[dict]:
    """The inner edge of the printed border, segment by segment along each side of the clear centre:
    scanning outward from `search_m` inside the side's nominal edge (inner_edges: `inner_m`, else
    +-clear_m / 2), or from the drawing's own edge where that is nearer (`drawn_m`, drawing_extent),
    the first place where the dark density across a strip stays over `density` for
    `run_m` (a 0.5 mm ink line does not), after at least `clear_run_m` of clean paper (the pen cone and
    arm shadows start dark and fail). Outward the scan stops at `search_m`, or sooner, where a run
    found would end `run_m` short of the page's own edge (`size_m`, the page centred; None: no bound):
    the side's margin + frame is all paper and border, and a dark mat beyond the page, which a border
    too faint to find lets the scan reach, could otherwise pass for it. The mat's run then needs the
    page's edge 2 run_m inside where it is believed to be. Returns [{"side", "axis", "offset_m" (along
    the edge normal from the nominal edge, page frame), "edge_m", "along_m"}] in the page image's
    (believed) frame; occluded or shadow-cut segments are left out. Started inside a drawing, the first
    dark run after clean paper is ink: a resume's wrist locate took a drawn wave 14-17 mm inside the
    border for it, and the fit's free x scale (1.21-1.26) carried that into a page 10-13 mm off
    (2026-10-05). Started at the drawing's edge, a scan either begins on clean paper or on ink, which the
    clean run refuses."""
    dark, valid = print_mask(page_image, px_per_mm)
    edges = inner_edges(clear_m, inner_m)
    mid = ((edges["left"] + edges["right"]) / 2.0, (edges["bottom"] + edges["top"]) / 2.0)
    span = ((edges["right"] - edges["left"]) / 2.0, (edges["top"] - edges["bottom"]) / 2.0)
    deepest = dict.fromkeys(SIDES, search_m)
    if drawn_m is not None:
        x0, y0, x1, y1 = drawn_m
        room = {"left": x0 - edges["left"], "right": edges["right"] - x1,
                "bottom": y0 - edges["bottom"], "top": edges["top"] - y1}
        deepest = {side: float(np.clip(room[side], 0.0, search_m)) for side in SIDES}
    need, need_clear = (int(round(v / step_m)) for v in (run_m, clear_run_m))
    out = []
    for side, (axis, sign) in SIDES.items():
        nominal = edges[side]
        inside = int(round(deepest[side] / step_m))
        half = span[1 - axis] * 0.85   # segments over the middle of the side, between its neighbours' edges
        bounds = mid[1 - axis] + np.linspace(-half, half, segments + 1)
        outward = search_m if size_m is None else min(search_m, size_m[axis] / 2.0 - sign * nominal - run_m)
        scan = nominal + sign * np.arange(-deepest[side], outward, step_m)   # inside -> outside
        for a0, a1 in zip(bounds[:-1], bounds[1:], strict=True):
            dens = _strip_density(dark, valid, extent_m, px_per_mm, axis, scan, (a0, a1), step_m, min_valid)
            hit = _first_run(dens, inside, need, need_clear, density)
            if hit is not None:
                out.append({"side": side, "axis": axis, "offset_m": float(scan[hit] - nominal),
                            "edge_m": float(scan[hit]), "along_m": float((a0 + a1) / 2)})
    return out


def solve_in_plane(edges: list[dict], clear_m, *, inner_m=None, centre_m=(0.0, 0.0),
                   outlier_m: float = 0.0012, free_scale=(True, True)) -> dict | None:
    """The in-plane correction (tx, ty, theta) that moves the detected border edges onto the print's
    nominal inner edges (inner_edges: `inner_m`, else the clear-centre outline): p_true = R(theta)
    (c + S (p - c)) + t in page coordinates, linearised in theta. S = diag(sx, sy) absorbs the page
    image's scale error (the camera's mount pose is CAD, so an oblique view comes out a few % too large
    or small along an axis); an axis is scaled only when both of its opposite sides were seen, else its
    scale is 1. S is about c = `centre_m`, the page point the views aim at: the view axis meets the
    page there wherever the camera sits along it, so the error scales the image about c and t is the
    print's own offset; about another point c', t would carry (I - S)(c - c'). The per-side edges
    change only the targets, not c: for an edge point on a left/right side (x = e, y = a), sx (e - cx)
    + cx - theta a + tx = that side's nominal x; on a top/bottom side (y = e, x = a), sy (e - cy) + cy
    + theta a + ty = its nominal y. The worst point is dropped and the fit redone until every residual
    is within `outlier_m`. `free_scale` (x, y) False holds that axis's scale at 1 even with both sides
    seen: a side found by one or two segments is fitted exactly by the scale, so they can never be
    rejected. Returns {"tx_m", "ty_m", "theta_rad", "scale", "rms_m", "sigma" (tx, ty, theta),
    "points", "sides"} or None without three sides covering both axes."""
    nominal = inner_edges(clear_m, inner_m)
    cx, cy = centre_m
    keep = np.ones(len(edges), dtype=bool)
    while True:
        kept = [e for e, k in zip(edges, keep, strict=True) if k]
        sides = {e["side"] for e in kept}
        if keep.sum() < 5 or len(sides) < 3 or len({e["axis"] for e in kept}) < 2:
            return None
        free_x = bool(free_scale[0]) and {"left", "right"} <= sides
        free_y = bool(free_scale[1]) and {"bottom", "top"} <= sides
        rows, rhs = [], []
        for e in edges:
            if e["axis"] == 0:
                rows.append([1.0, 0.0, -e["along_m"], e["edge_m"] - cx if free_x else 0.0, 0.0])
                rhs.append(nominal[e["side"]] - (cx if free_x else e["edge_m"]))
            else:
                rows.append([0.0, 1.0, e["along_m"], 0.0, e["edge_m"] - cy if free_y else 0.0])
                rhs.append(nominal[e["side"]] - (cy if free_y else e["edge_m"]))
        cols = [0, 1, 2] + ([3] if free_x else []) + ([4] if free_y else [])
        a, b = np.asarray(rows)[:, cols], np.asarray(rhs)
        x, *_ = np.linalg.lstsq(a[keep], b[keep], rcond=None)
        res = np.where(keep, np.abs(b - a @ x), -1.0)
        worst = int(np.argmax(res))
        if res[worst] <= outlier_m:
            break
        keep[worst] = False
    full = dict(zip(cols, x, strict=True))
    res = (b - a @ x)[keep]
    var = float(res @ res) / max(1, int(keep.sum()) - len(cols))
    cov = var * np.linalg.pinv(a[keep].T @ a[keep])
    return {"tx_m": float(x[0]), "ty_m": float(x[1]), "theta_rad": float(x[2]),
            "scale": [float(full.get(3, 1.0)), float(full.get(4, 1.0))], "rms_m": float(np.sqrt(var)),
            "sigma": [float(np.sqrt(max(cov[i, i], 0.0))) for i in range(3)], "points": int(keep.sum()),
            "sides": sorted(sides)}


def fit_view(page_image: np.ndarray, extent_m, px_per_mm: float, page: dict, *, inner_m, drawn_m, centre_m,
             free_scale, artwork=None) -> dict | None:
    """One wrist view's in-plane correction: its border's inner edges (border_edges, solve_in_plane), else,
    with the print's `artwork`, its frame as a whole (artwork_fit); None when neither fits."""
    clear = page["clear_m"]
    found = border_edges(page_image, extent_m, px_per_mm, clear, inner_m=inner_m, size_m=page["size_m"],
                         drawn_m=drawn_m)
    sol = solve_in_plane(found, clear, inner_m=inner_m, centre_m=centre_m, free_scale=free_scale)
    if sol is None and artwork is not None:
        sol = artwork_fit(page_image, artwork, extent_m, px_per_mm, page["size_m"], clear)
    return sol


def print_artwork(pattern_id: str | None):
    """The installed print's artwork (stencil.png beside its tracking.json; grey, u right, v down), None
    without one."""
    import cv2
    from stencil_reference import observer_references

    return cv2.imread(str(observer_references() / pattern_id / "stencil.png"), cv2.IMREAD_GRAYSCALE) \
        if pattern_id else None


def artwork_fit(page_image: np.ndarray, artwork: np.ndarray, extent_m, px_per_mm: float, size_m, clear_m, *,
                step_m: float = 0.002, search_m: float = 0.015, max_turn_deg: float = 6.0,
                min_score: float = 0.3) -> dict | None:
    """The in-plane correction from the printed frame as a whole, for a view whose border_edges fit fails: the
    page image averaged to `step_m` per pixel, where the frame's 14 mm band shows but not its 2 mm knots or code
    bits (so any print of the layout matches, and a lattice step is never taken for the page), correlated with
    the print's `artwork` over the frame only, never the clear centre (stencil_plane_match's band-pass and
    frame mask), within `search_m` and `max_turn_deg` of the believed page. A thermal transfer on silicone
    showed a half-skin print's top and right bands solid and the other two as scattered knots, and the hex
    lattice's inner edge steps ~3 mm row to row: the edge scan found three agreeing sides in no view, while
    this put all three views within 3 mm of each other (2026-10-06). Returns solve_in_plane's correction with
    the peak's "score", or None when the peak scores under `min_score` or sits on the search window's edge."""
    import cv2

    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    import stencil_plane_match as pm

    block = max(1, int(round(step_m * 1000 * px_per_mm)))
    gray = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape[0] // block, gray.shape[1] // block
    small = cv2.resize(gray[:rows * block, :cols * block].astype(np.float32), (cols, rows),
                       interpolation=cv2.INTER_AREA)
    step = block / px_per_mm / 1000.0   # small pixel j's centre is at xmin + (j + 0.5) step
    template = pm.reference_template(artwork, size_m, step)
    u0, v0 = 0.5 - clear_m[0] / size_m[0] / 2.0, 0.5 - clear_m[1] / size_m[1] / 2.0
    keep = pm.frame_mask(template, (u0, v0, 1.0 - u0, 1.0 - v0))
    template = pm.band_pass(template)
    pad = int(np.ceil(search_m / step))
    image = cv2.copyMakeBorder(pm.band_pass(small), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
    th, tw = template.shape
    centre = ((tw - 1) / 2.0, (th - 1) / 2.0)   # the page centre in the template

    def correlate(degrees):   # the template turned counter-clockwise on the page (y up)
        turn = cv2.getRotationMatrix2D(centre, degrees, 1.0)
        turned = cv2.warpAffine(template, turn, (tw, th), flags=cv2.INTER_LINEAR, borderValue=0.0)
        mask = cv2.warpAffine(keep, turn, (tw, th), flags=cv2.INTER_NEAREST, borderValue=0.0)
        result = cv2.matchTemplate(image, turned, cv2.TM_CCOEFF_NORMED, mask=mask)
        return np.where(np.abs(result) <= 1.0 + 1e-6, result, -1.0)   # OpenCV 4.6 gives a blank window FLT_MAX

    def vertex(a, b, c):   # the parabola's peak through three equally spaced samples, in steps from the middle
        curve = a - 2.0 * b + c
        return 0.5 * (a - c) / curve if curve < 0 else 0.0

    coarse = np.arange(-max_turn_deg, max_turn_deg + 1e-9, 1.0)
    turn = coarse[int(np.argmax([correlate(d).max() for d in coarse]))]
    fine = np.arange(turn - 0.75, turn + 0.76, 0.25)
    peaks = [float(correlate(d).max()) for d in fine]
    i = int(np.clip(np.argmax(peaks), 1, len(fine) - 2))
    degrees = fine[i] + 0.25 * float(np.clip(vertex(*peaks[i - 1:i + 2]), -1.0, 1.0))
    result = correlate(degrees)
    score = float(result.max())
    y, x = np.unravel_index(int(np.argmax(result)), result.shape)
    if score < min_score or not (0 < x < result.shape[1] - 1 and 0 < y < result.shape[0] - 1):
        return None
    col = x + vertex(*result[y, x - 1:x + 2]) + centre[0] - pad
    row = y + vertex(*result[y - 1:y + 2, x]) + centre[1] - pad
    xmin, _, _, ymax = extent_m
    shift = np.array([xmin + (col + 0.5) * step, ymax - (row + 0.5) * step])   # the print's centre, believed frame
    phi = np.radians(degrees)
    c, s = np.cos(phi), np.sin(phi)
    t = -np.array([c * shift[0] + s * shift[1], -s * shift[0] + c * shift[1]])   # p_print = R(-phi) (p - shift)
    return {"tx_m": float(t[0]), "ty_m": float(t[1]), "theta_rad": float(-phi), "scale": [1.0, 1.0],
            "rms_m": None, "sigma": [step / 2.0, step / 2.0, float(np.radians(1.0))], "points": 0,
            "sides": ["artwork"], "score": score}


def hold_tip(colors, frames, meta: dict, tip_cam, cam_from_page: np.ndarray, page: dict, artwork,
             px_per_mm: float = 5.0) -> dict | None:
    """Where the pen's tip is on the print in one gauge hold's wrist frames, from the camera's own geometry: the
    surface the gauge fitted (gauge.Frame, the paper's plane by the pen: a whole-frame plane fit takes the mat under
    a 5 mm silicone skin, 3.8 mm off), the frames rectified onto it about the page believed there (`cam_from_page`,
    FK: it only seeds the fit), the print's frame matched to its `artwork` (artwork_fit), and the gauge's tip point
    `tip_cam` carried into the print's frame. Neither FK nor the camera's CAD mount enters: the wrist locate's far
    views put a half-skin print 6 mm apart in y, these frames within ~1-2 mm of each other (2026-10-06). Not the
    border's edges: on that print the edge scan's answer jumped 5 mm when its seed moved 2.6 mm (the hex lattice's
    inner edge steps row to row), the artwork's moved 0.4 mm at most. {"tip_m": (x, y, height over the print),
    "score"}, or None when the gauge saw no surface or the artwork does not match."""
    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    import stencil_tip

    seen = [(c, f) for c, f in zip(colors, frames, strict=True) if f is not None]
    if not seen:
        return None
    surface = seen[0][1]
    n = -np.asarray(surface.n, float)   # away from the camera, as stencil_tip takes a plane
    snapped = stencil_tip.snap_to_plane(np.asarray(cam_from_page, float), n, float(n @ np.asarray(surface.c, float)))
    k, dist = (np.asarray(meta["color_intrinsics"][key], float) for key in ("k", "coeffs"))
    extent = page_extent(page["size_m"])
    view = median_stack([rectify(c, snapped, np.eye(4), k, dist, extent, px_per_mm) for c, _ in seen])
    fit = artwork_fit(view, artwork, extent, px_per_mm, page["size_m"], page["clear_m"])
    if fit is None:
        return None
    on_print = snapped @ np.linalg.inv(correction_matrix(fit["tx_m"], fit["ty_m"], fit["theta_rad"]))
    return {"tip_m": [float(v) for v in stencil_tip.tip_on_print(on_print, tip_cam)], "score": fit["score"]}


def holds_shift(holds, *, outlier_m: float = 0.003, min_holds: int = 3) -> dict | None:
    """The page's in-plane shift from the gauge holds: each hold's commanded page point "xy" and the tip's measured
    point on the print "tip_m" (hold_tip), believed + shift = print, the median of tip - xy over the holds within
    `outlier_m` of the first median (one hold's fit jumped a band, 20 mm, 2026-10-06). The turn stays the
    locate's: across a 35 x 40 mm square the holds' 1-2 mm scatter is a 3 deg turn. {"shift_m", "spread_m" (1.4826
    MAD per axis), "holds", "dropped"}, or None with fewer than `min_holds` agreeing."""
    r = np.array([np.subtract(h["tip_m"][:2], h["xy"]) for h in holds], float).reshape(-1, 2)
    if len(r) < min_holds:
        return None
    keep = np.linalg.norm(r - np.median(r, axis=0), axis=1) <= outlier_m
    if keep.sum() < min_holds:
        return None
    shift = np.median(r[keep], axis=0)
    spread = 1.4826 * np.median(np.abs(r[keep] - shift), axis=0)
    return {"shift_m": shift.tolist(), "spread_m": spread.tolist(), "holds": int(keep.sum()),
            "dropped": int((~keep).sum())}


def combine_views(solutions: list[dict], floor=(0.0, 0.0, 0.0)) -> dict | None:
    """One wrist estimate from per-pose border fits: the median correction, with each sigma the
    largest of the fits' own, the spread between poses (1.4826 MAD, or half the range for two), and
    `floor` (the camera mount's CAD error). A pose seeing all four sides counts double."""
    sols = [s for s in solutions if s is not None]
    if not sols:
        return None
    weights = [2 if len(s["sides"]) == 4 else 1 for s in sols]
    vals = np.repeat(np.array([[s["tx_m"], s["ty_m"], s["theta_rad"]] for s in sols]), weights, axis=0)
    med = np.median(vals, axis=0)
    if len(sols) >= 3:
        spread = 1.4826 * np.median(np.abs(vals - med), axis=0)
    elif len(sols) == 2:
        spread = np.abs(vals.max(axis=0) - vals.min(axis=0)) / 2
    else:
        spread = np.zeros(3)
    fit = np.max([s["sigma"] for s in sols], axis=0)
    sigma = np.maximum.reduce([fit, spread, np.asarray(floor, dtype=float)])
    return {"tx_m": float(med[0]), "ty_m": float(med[1]), "theta_rad": float(med[2]),
            "sigma": [float(v) for v in sigma], "poses": len(sols)}


def correction_matrix(tx: float, ty: float, theta: float) -> np.ndarray:
    """4x4 page-frame transform p_true = R(theta) p + t (rotation about page z)."""
    return rpy_matrix([tx, ty, 0.], [0., 0., theta])


def fuse(estimates) -> dict:
    """Inverse-variance fusion of in-plane page corrections [(tx, ty, theta), (sx, sy, stheta)], each
    relative to the same prior page; the prior itself is (0, 0, 0) with its own sigmas. Returns the
    fused correction and sigmas."""
    vals = np.array([e[0] for e in estimates], dtype=float)
    sig = np.array([e[1] for e in estimates], dtype=float)
    w = 1.0 / np.maximum(sig, 1e-9) ** 2
    fused = (w * vals).sum(axis=0) / w.sum(axis=0)
    return {"tx_m": float(fused[0]), "ty_m": float(fused[1]), "theta_rad": float(fused[2]),
            "sigma": [float(v) for v in 1.0 / np.sqrt(w.sum(axis=0))]}


def plan_samples(program: dict, step_m: float = 0.0002) -> np.ndarray:
    """Points every `step_m` along every planned stroke (page metres)."""
    out = []
    for op in program.get("ops", []):
        if op.get("op") != "stroke":
            continue
        pts = np.asarray(op["points_m"] + ([op["points_m"][0]] if op.get("closed") else []), dtype=float)
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
            out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        out.append(pts[-1:])
    return np.vstack(out) if out else np.zeros((0, 2))


def ink_alignment(page_image: np.ndarray, program: dict, extent_m, px_per_mm: float, clear_m, *,
                  search_m: float = 0.010, step_m: float = 0.00025, near_m: float = 0.001,
                  min_coverage: float = 0.3) -> dict | None:
    """Where the ink sits against the plan in a page image: the plan shift (dx, dy) that brings the
    planned strokes nearest the dark marks inside the clear centre (mean distance to the nearest ink
    over every plan sample), and the coverage there: the share of the planned path with ink within
    `near_m`. The shift is the camera mount's error plus the arm's; coverage is how much of the design
    was drawn. Only ink within reach of this plan counts (its box plus the search), so other drawings
    on the page do not pull it. Returns {"dx_m", "dy_m", "coverage", "mean_gap_m", "p50_gap_m",
    "p95_gap_m"} (the gaps: plan samples to the nearest ink at the best shift) or None without ink."""
    import cv2

    plan = plan_samples(program)
    if not len(plan):
        return None
    dark, valid = print_mask(page_image, px_per_mm, darkness=0.15)
    hx, hy = clear_m[0] / 2.0 - 0.001, clear_m[1] / 2.0 - 0.001
    gx, gy = page_grid(extent_m, px_per_mm)
    reach = search_m + 2 * near_m
    lo, hi = plan.min(axis=0) - reach, plan.max(axis=0) + reach
    ink = (dark & (np.abs(gx) < hx) & (np.abs(gy) < hy)
           & (gx >= lo[0]) & (gx <= hi[0]) & (gy >= lo[1]) & (gy <= hi[1]))
    if ink.sum() < 20:
        return None
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5) / (px_per_mm * 1000.0)
    shifts = np.arange(-search_m, search_m + 1e-12, step_m)
    best = None
    h, w = dist.shape
    for dx in shifts:
        for dy in shifts:
            px = to_image_px(plan + [dx, dy], extent_m, px_per_mm)
            c, r = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
            inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
            if inside.mean() < 0.9:
                continue
            seen = inside.copy()
            seen[inside] = valid[r[inside], c[inside]]
            if seen.mean() < 0.7:
                continue
            d = dist[r[seen], c[seen]]
            score = float(np.mean(np.minimum(d, 0.004)))
            if best is None or score < best[0]:
                best = (score, dx, dy, float((d <= near_m).mean()), float(np.percentile(d, 50)),
                        float(np.percentile(d, 95)))
    if best is None:
        return None
    # Ink the plan cannot reach inside the search, or too little of the plan covered, is not the drawing:
    # the pen drew in the air, or the marks are touches and strays (bench 2026-09-26).
    edge = max(abs(best[1]), abs(best[2])) >= search_m - step_m / 2
    return {"dx_m": float(best[1]), "dy_m": float(best[2]), "coverage": best[3], "mean_gap_m": best[0],
            "p50_gap_m": best[4], "p95_gap_m": best[5], "found": bool(not edge and best[3] >= min_coverage)}


def analyse(views, program: dict, clear_m, extent_m, px_per_mm: float, *, floor=(0.0, 0.0, 0.0),
            ink_prior_m=None, ink_gate_m: float = 0.003, inner_m=None, size_m=PAGE_SIZE, centre_m=(0.0, 0.0)) -> dict:
    """Post-draw analysis of per-pose page images: where the print is against the page the run used
    (the border fit, combined over poses), where the ink is against the plan (the shift: camera mount
    plus arm), the coverage of the plan, and the ink's placement error on the print (shift + border
    correction: what a page trim would take out). The border fit takes the print's inner edges
    `inner_m` and page size `size_m` (border_edges), and its scale about `centre_m`, the page point
    the views aim at (solve_in_plane): the ink's shift is measured there.

    With `ink_prior_m` (the ink-vs-plan shift this rig draws with; a trim moves the ink and the plan
    together, so it does not change it), a view whose shift lies farther than `ink_gate_m` from it is
    not used: a view the wrist registers off by more than that, or whose search matched other ink,
    moved the median by 5-15 mm on a page of small drawings. The placement pairs each used view's
    shift with that view's own border fit, so a view's registration error cancels, and falls back to
    the combined border for a view without one. No view near the prior: every found view is used and
    `ink["gated"]` says so."""
    drawn = drawing_extent(program)
    borders = [solve_in_plane(border_edges(v, extent_m, px_per_mm, clear_m, inner_m=inner_m, size_m=size_m,
                                           drawn_m=drawn), clear_m, inner_m=inner_m, centre_m=centre_m)
               for v in views]
    inks = [ink_alignment(v, program, extent_m, px_per_mm, clear_m) for v in views]
    border = combine_views(borders, floor)
    found = [i for i, k in enumerate(inks) if k is not None and k.get("found", True)]
    used, gated = found, False
    if ink_prior_m is not None:
        prior = np.asarray(ink_prior_m, dtype=float)[:2]
        near = [i for i in found if np.hypot(inks[i]["dx_m"] - prior[0], inks[i]["dy_m"] - prior[1]) <= ink_gate_m]
        if near:
            used, gated = near, len(near) < len(found)
    ink = None
    if used:
        good = [inks[i] for i in used]
        ink = {"dx_m": float(np.median([k["dx_m"] for k in good])), "dy_m": float(np.median([k["dy_m"] for k in good])),
               "coverage": float(np.median([k["coverage"] for k in good])),
               "p50_gap_m": float(np.median([k["p50_gap_m"] for k in good])),
               "p95_gap_m": float(np.median([k["p95_gap_m"] for k in good])), "views": len(good),
               "found": len(found), "used": used, "gated": bool(gated)}
    placement = None
    pairs = [(inks[i], borders[i] or border) for i in used if (borders[i] or border) is not None]
    if pairs:
        placement = {"dx_m": float(np.median([k["dx_m"] + b["tx_m"] for k, b in pairs])),
                     "dy_m": float(np.median([k["dy_m"] + b["ty_m"] for k, b in pairs])),
                     "sigma_m": border["sigma"][:2] if border is not None else [float("nan")] * 2,
                     "views": len(pairs)}
    result = {"border": border, "border_per_pose": borders, "ink": ink, "ink_per_pose": inks, "placement": placement}
    if program.get("research") or program.get('version') == 2:
        from tatbot_session.fidelity import analyse as fidelity

        result["fidelity"] = fidelity(views, program, borders, centre_m, extent_m, px_per_mm)
    return result


def summary(analysis: dict) -> str:
    """One line for the terminal."""
    b, k, p = analysis.get("border"), analysis.get("ink"), analysis.get("placement")
    parts = []
    if b:
        parts.append("print vs used page {:+.1f} {:+.1f} mm {:+.1f} deg (sigma {:.1f} {:.1f} mm, {} poses)".format(
            -b["tx_m"] * 1e3, -b["ty_m"] * 1e3, -np.degrees(b["theta_rad"]), b["sigma"][0] * 1e3, b["sigma"][1] * 1e3,
            b["poses"]))
    if not k:
        parts.append("no ink found along the plan")
    else:
        parts.append("ink vs plan {:+.1f} {:+.1f} mm, coverage {:.0%}, gap p50 {:.2f} p95 {:.2f} mm ({} of {} views)".format(
            k["dx_m"] * 1e3, k["dy_m"] * 1e3, k["coverage"], k.get("p50_gap_m", float("nan")) * 1e3,
            k.get("p95_gap_m", float("nan")) * 1e3, k.get("views", 0), k.get("found", k.get("views", 0))))
    if p:
        parts.append("ink on the print off by {:+.1f} {:+.1f} mm".format(p["dx_m"] * 1e3, p["dy_m"] * 1e3))
    if analysis.get("coded_summary"):
        parts.append(analysis["coded_summary"])
    return "inspect: " + ("; ".join(parts) if parts else "no border or ink found")
