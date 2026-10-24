"""Wrist camera observation, page-depth fitting and print/ink inspection; see ros/README.md."""
from __future__ import annotations

import itertools
import sys
import threading
import time

import numpy as np
from tatbot_description import repo_root
from tatbot_description.transforms import rpy_matrix

sys.path.insert(0, str(repo_root(None) / "scripts/lib"))
import wrist_tap  # noqa: E402,F401  (the executor's tap factory)
from rgbd_geometry import depth_points, page_height_from_depth  # noqa: E402,F401

CAMERA_FRAME = "{arm}/realsense_color_optical_frame"   # optical: z forward, x right, y down


def page_extent(size_m, more_m: float = 0.010) -> tuple[float, float, float, float]:
    """A page image's extent in the page frame (xmin, ymin, xmax, ymax), metres: the page and `more_m` round it."""
    hx, hy = size_m[0] / 2.0 + more_m, size_m[1] / 2.0 + more_m
    return (-hx, -hy, hx, hy)


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


def tcp_target(base_from_camera: np.ndarray, tcp_from_camera: np.ndarray) -> np.ndarray:
    """The tcp pose that puts the camera at base_from_camera (the camera rides the wrist with the tool)."""
    return np.asarray(base_from_camera) @ np.linalg.inv(tcp_from_camera)


REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0])   # the landing sleep pose
JOINT_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0])            # wrist moves cost more


SIDE_LINKS = ("{arm}/link_3", "{arm}/link_4", "{arm}/link_5", "{arm}/link_6")
# The wrist's own bodies, which the pen's clearance says nothing about: the fiducial cube's tag faces, the
# cameras (the lower D405 under the wrist since 2026-10-23) and the tool mount. A view keeps each on the arm's side
# and BODY_CLEARANCE_M over the page: the last post-draw view (roll 90) put the cube's lowest tag face 33 mm over
# the table and 140-180 mm past the camera toward the palette (2026-09-30). The cube's tags are 47 mm, so 70 mm to
# a face centre keeps its corners clear. Over 51 locate views of 2026-10-23 the lower camera stood 141 mm over the
# page at least and up to 69 mm past its centre line: within a top-down view's TOP_DOWN_SIDE_M (41 of 42 were).
WRIST_BODY = ("{arm}/wrist_tag2", "{arm}/wrist_tag3", "{arm}/wrist_tag4", "{arm}/realsense_link",
              "{arm}/realsense_lower_link", "{arm}/tool_mount")
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
# The locate looks straight down on the page first (owner, 2026-10-21: views 35-40 deg off from 0.18-0.22 m saw a
# forearm print's 12 mm markers 20-35 px and squashed, and read none; hovers facing them read 40-53 px). The D405's
# colour fx is ~651 px at 1280x720, so a 12 mm marker spans 41 px from TOP_DOWN_M; the tip, on the camera's axis
# 159 mm out, then stands 31 mm over the target. The D405 sits 48 deg off the pen, so a camera within 10 deg of the
# base's z (top_down: the page levelled) tips the pen 40-55 deg, which aim's 30 deg would refuse; and its wrist cube
# then stands level with the tool mount at best, and reaches up to ~0.1 m across the page centre's y (on a forearm's
# page of 2026-10-21 the ready pose's own reached 59 mm across). The view keeps every other check, its arm joints
# within TOP_DOWN["max_arm_rad"] of the pen-down reference (the straight-down view over that page's centre: 0.71).
TOP_DOWN_M, TOP_DOWN_SIDE_M = 0.19, 0.10
TOP_DOWN = {"tilts_deg": (0, 5, 10), "distances_m": (TOP_DOWN_M,), "max_pen_tilt_deg": 60.0, "cube_up_m": 0.0,
            "max_arm_rad": 0.8}
# A straight-down view also hides the print behind the tool: the pen and the machine stand on the camera's own axis,
# over the middle of the frame. Over the silicone tile (2026-10-22, 13 views) the markers in view read 75% and those
# the tool did not cover 92% (by tool_footprint); the views are chosen by the markers they see (markers_seen) before
# their joint cost, from as high as the print's smallest marker keeps MARKER_PX a side (a 12 mm marker TOP_DOWN_M,
# an 18 mm one up to TOP_DOWN_MAX_M), looking at the print's centre or AROUND_M aside: the pen beside the print, not
# over it. There a view from 0.265 m 40 mm aside read 21 of its 23 markers, 0.75 mm rms; the deployed 50 deg view 18,
# 1.39 mm (3.86 at most).
MARKER_PX, TOP_DOWN_MAX_M, AROUND_M = 40.0, 0.27, 0.04
AROUND = ((0.0, 0.0),) + tuple((AROUND_M * np.cos(a), AROUND_M * np.sin(a)) for a in np.radians(range(0, 360, 45)))
# The datasheet's tool moved onto the gauge's measured tip (the CAD mount puts that 11 mm off it), every radius
# TOOL_PAD_M wider: of the 477 markers in view over 18 views of the silicone tile (2026-10-22) it told the read from the
# missed in 91% (the CAD tool slid a fitted 80 px in the image: 87%), and 94% of those it leaves clear were read.
TOOL_PAD_M = 0.002


def tool_footprint(kin, q, k, profile, frame: str, tip_cam=None, shape=(720, 1280)) -> np.ndarray:
    """The wrist colour image's pixels the tool covers (bool, `shape`): its datasheet profile of revolution about the
    tool mount's z (tool_spec: (z, r) from the mount toward the tip), moved so its tip lies on `tip_cam` (the gauge's
    measured tip in the camera's frame) when given, projected by `k`. The tool rides the camera, so it covers these
    pixels in every view."""
    import cv2
    from tatbot_description import names

    cam_from_mount = np.linalg.inv(kin.frame(q, frame)) @ kin.frame(q, names.tool_mount_frame(kin.arm))
    ring, out = np.linspace(0.0, 2 * np.pi, 24, endpoint=False), np.zeros(shape, np.uint8)
    points = sorted((float(z), float(r)) for z, r in profile)
    if tip_cam is not None and points:
        cam_from_mount[:3, 3] += np.asarray(tip_cam, float) - cam_from_mount[:3, :] @ [0.0, 0.0, points[-1][0], 1.0]
    for (z0, r0), (z1, r1) in zip(points, points[1:], strict=False):
        r = max(r0, r1) + TOOL_PAD_M
        rim = np.concatenate([np.c_[r * np.cos(ring), r * np.sin(ring), np.full(len(ring), z)] for z in (z0, z1)])
        rim = rim @ cam_from_mount[:3, :3].T + cam_from_mount[:3, 3]
        rim = rim[rim[:, 2] > 0.01]
        if len(rim) >= 3:
            uv = rim[:, :2] / rim[:, 2:] * k[[0, 1], [0, 1]] + k[:2, 2]
            cv2.fillConvexPoly(out, cv2.convexHull(np.clip(uv, -1e4, 1e4).astype(np.float32)).astype(np.int32), 1)
    return out.astype(bool)


SEEN_AT = np.array([(s, t) for s in (0.0, 0.5, 1.0) for t in (0.0, 0.5, 1.0)])   # a marker's corners, edges' midpoints, centre


def markers_seen(page, base_from_camera, tile, k, hidden: np.ndarray, min_px: float = MARKER_PX) -> int:
    """How many of the print's markers (fiducials.stencil.Tile, laid on `page`) a camera at base_from_camera sees: its
    corners in the frame, none of its corners, edges' midpoints and centre on `hidden` (tool_footprint: the needle across
    a marker's middle hides it), its sides min_px long on average."""
    c = np.stack([tile.corners[i] for i in sorted(tile.ids)])   # (N, 4, 2): TL, TR, BR, BL
    at = c[:, :1] + SEEN_AT[:, :1] * (c[:, 1:2] - c[:, :1]) + SEEN_AT[:, 1:] * (c[:, 3:4] - c[:, :1])   # (N, 9, 2)
    cam = np.linalg.inv(base_from_camera) @ np.asarray(page, float) @ np.c_[at.reshape(-1, 2), np.zeros(at.size // 2),
                                                                              np.ones(at.size // 2)].T
    if (cam[2] < 0.05).any():
        return 0
    uv = (cam[:2] / cam[2] * k[[0, 1], [0, 1], None] + k[:2, 2, None]).T.reshape(len(c), len(SEEN_AT), 2)
    u, v = uv[..., 0].astype(int), uv[..., 1].astype(int)
    corner = [0, 6, 8, 2]   # SEEN_AT's TL, TR, BR, BL
    inside = ((u >= 0) & (u < hidden.shape[1]) & (v >= 0) & (v < hidden.shape[0]))[:, corner].all(axis=1)
    clear = ~hidden[np.clip(v, 0, hidden.shape[0] - 1), np.clip(u, 0, hidden.shape[1] - 1)].any(axis=1)
    side = np.linalg.norm(uv[:, corner] - np.roll(uv[:, corner], 1, axis=1), axis=2).mean(axis=1)
    return int((inside & clear & (side >= min_px)).sum())


def top_down_distances(tile, fx: float) -> tuple[float, ...]:
    """top_down's camera distances for a print: TOP_DOWN_M, and as high as its smallest marker keeps MARKER_PX a side
    (at most TOP_DOWN_MAX_M) when that is higher."""
    far = min(TOP_DOWN_MAX_M, fx * min(tile.sides_m.values()) / MARKER_PX)
    return (TOP_DOWN_M, far) if far > TOP_DOWN_M + 0.01 else (TOP_DOWN_M,)


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
                 max_pen_tilt_deg: float, around=((0.0, 0.0),)):
    """Each candidate view ({"tilt_deg", "azimuth_deg", "distance_m", "roll_deg", "aside_m"}, base_from_tcp target)
    looking at xy, or at xy moved by each of `around`, whose pen stays within max_pen_tilt_deg of `down`, before any
    IK: most of the grid never needs solving."""
    cos_max = float(np.cos(np.radians(max_pen_tilt_deg)))
    for tilt, d, (dx, dy) in itertools.product(np.asarray(list(tilts_deg), dtype=float), distances_m, around):
        for az in np.asarray(azimuths, dtype=float):
            for roll in np.asarray(list(rolls_deg), dtype=float):
                cam = look_at(page, (xy[0] + dx, xy[1] + dy), tilt_rad=np.radians(tilt), azimuth_rad=az, distance_m=d,
                              roll_rad=np.radians(roll))
                target = tcp_target(cam, tcp_from_cam)
                if float(target[:3, 2] @ down) >= cos_max:
                    yield ({"tilt_deg": float(tilt), "azimuth_deg": float(np.degrees(az)) % 360, "distance_m": float(d),
                            "roll_deg": float(roll), "aside_m": [float(dx), float(dy)]}, target)


def aim(kin, solve_ik, base_from_page, xy, q_seed, *, frame: str, clearance_m: float, azimuths=None,
        tilts_deg=(20, 30, 35, 40, 45, 50), distances_m=(0.18, 0.20, 0.22, 0.24), rolls_deg=range(0, 360, 45),
        rest=REST, max_y: float | None = None, margin_rad: float = 0.0667, body_clearance_m: float = BODY_CLEARANCE_M,
        reference=None, self_gap=None, max_pen_tilt_deg: float = MAX_PEN_TILT_DEG, cube_up_m: float = CUBE_UP_M,
        max_arm_rad: float | None = None, see=None, around=((0.0, 0.0),)):
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
    refused to leave (bench 2026-09-27). With `max_arm_rad`, joint_0-joint_2 within it of the reference too.
    With `see` (base_from_camera -> the print's markers it sees: markers_seen) the views seeing the most that are
    reachable come first, solved in that order (most never are), and `around` moves the point looked at aside.
    Returns (q, {"tilt_deg", "azimuth_deg", "distance_m", "roll_deg", "aside_m", "seen", "cost"}) or None."""
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
                and (max_arm_rad is None or float(np.max(np.abs(q[:3] - seed[:3]))) <= max_arm_rad)
                and (self_gap is None or path_clear(self_gap, q_seed, q)))

    views = list(view_targets(page, xy, tcp_from_cam, kin.fk(seed)[:3, 2], tilts_deg=tilts_deg, distances_m=distances_m,
                              azimuths=azimuths, rolls_deg=rolls_deg, max_pen_tilt_deg=max_pen_tilt_deg, around=around))
    for view, target in views:
        view["seen"] = 0 if see is None else see(target @ tcp_from_cam)
    best = None
    for view, target in sorted(views, key=lambda vt: -vt[0]["seen"]):
        if best is not None and view["seen"] < best[1]["seen"]:
            break   # every view seeing as many as the best reachable one is solved
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


def top_down(kin, solve_ik, base_from_page, xy, q_seed, *, max_y: float | None = None, **kw):
    """aim's view of page point xy looking straight down (TOP_DOWN): about the page levelled (its z turned the least
    onto the base's, its centre kept), the side line TOP_DOWN_SIDE_M farther."""
    from tatbot_session.closeup import _turn

    level = np.array(base_from_page, float)
    level[:3, :3] = _turn(level[:3, 2], [0.0, 0.0, 1.0]) @ level[:3, :3]
    return aim(kin, solve_ik, level, xy, q_seed, max_y=None if max_y is None else max_y + TOP_DOWN_SIDE_M,
               **{**TOP_DOWN, **kw})


# The locate's first look, wide -> narrow: the overhead places the print within a few mm, then the pen goes square
# to the page over its centre, page.locate.over_m up, at the ready move's heading and IK branch (the gauge holds' own arm
# configuration, so the ready move under it only descends), and both wrist D405s read the print there; it rises to the
# next height when they read too little, and the top-down views (top_down, aim) come only after. Over 37 setups'
# overhead pages (2026-10-16..24) the 50 mm view was reachable on every one and stood all six of the 981a tile's
# markers in both cameras (markers_seen); the silicone tile 22 of 23, the forearm's 3x4 five or six of twelve. The
# top-down views took ~30 s of view search and two ~5.5 rad swings (rest -> view, view -> holds) of a 160 s setup.


def over_centre(kin, solve_ik, base_from_page, xy, q_seed, heading, *, frame: str, clearance_m: float,
                height_m: float, seed=REST, margin_rad: float = 0.0667, body_clearance_m: float = BODY_CLEARANCE_M,
                self_gap=None, see=None):
    """Joints with the pen square to the page `height_m` over page point xy at `heading` (a tcp pose whose x axis is
    the pen's heading), solved from `seed`, as a locate view: clear of the limits, the pen and the wrist's bodies over
    the page (view_clear), its straight way from q_seed off the arm itself (path_clear); with `see`, the print's markers
    the camera `frame` sees there (else None). (q, {"over_m", "seen"}) or None. No side line: the ready pose under it
    has none, and a wrist tag stood 1-13 mm past the top-down views' over 5 of those 37 pages."""
    from tatbot_session import geometry

    page = np.asarray(base_from_page, float)
    start = np.asarray(seed, float).copy()
    start[6] = q_seed[6]
    try:
        q = np.asarray(solve_ik(kin, geometry.tool_down_pose(page, xy, height_m, heading), start), float)
    except ValueError:
        return None
    body = frames_of(kin, q_seed, WRIST_BODY)
    if not view_clear(kin, q, page, margin_rad=margin_rad, clearance_m=clearance_m, body=body,
                      body_clearance_m=body_clearance_m, side=[], max_y=None,
                      posture=None if self_gap is None else (lambda v: path_clear(self_gap, q_seed, v))):
        return None
    return q, {"over_m": float(height_m), "seen": None if see is None else int(see(kin.frame(q, frame)))}


SEARCH_BAND_M = 0.05      # the skin a search reads: within this of the last page along its normal (a limb moved since)
SEARCH_MIN_POINTS = 50    # ... and a search point needs this many of its depth points about it


def skin_aims(points, half, *, lift_m: float, band_m: float = SEARCH_BAND_M,
              min_points: int = SEARCH_MIN_POINTS) -> list[dict]:
    """A wrist search's aims over the skin the overhead's depth reads when it reads no print: `points` (N, 3) in the
    last page's frame. Centred on the skin there now (the median of its points within twice `half`, the print's
    extent about its centre, of the last page's: a stale page put the views and holds on a forearm's flank,
    2026-10-21), that centre and the eight points `half` from it, each with min_points of the skin within `half` of
    it, the centre first; each with its rise, `lift_m` over the highest of those points and never below the page,
    which its view keeps its standoff over. No surface is fitted to the skin: each aim stands over its own highest
    point."""
    p = np.asarray(points, float).reshape(-1, 3)
    hx, hy = half
    p = p[np.isfinite(p).all(axis=1) & (np.abs(p[:, 2]) <= band_m) & (np.abs(p[:, 0]) <= 2 * hx) &
          (np.abs(p[:, 1]) <= 2 * hy)]
    at, out = np.median(p[:, :2], axis=0) if len(p) else np.zeros(2), []
    for xy in (at + [i * hx, j * hy] for i in (0, -1, 1) for j in (0, -1, 1)):
        near = p[(np.abs(p[:, 0] - xy[0]) <= hx) & (np.abs(p[:, 1] - xy[1]) <= hy), 2]
        if len(near) >= min_points:
            out.append({"xy": [float(v) for v in xy], "rise_m": max(float(near.max()) + lift_m, 0.0),
                        "points": int(len(near))})
    return sorted(out, key=lambda a: float(np.hypot(*(np.asarray(a["xy"]) - at))))


WIDE_SEARCH_M = 0.15   # with no skin by the last page, the mat this far about it is read for skin (the arm's reach)
RAISED_M = (0.015, 0.12)   # ... skin standing this far over the mat there (its 10th percentile of depth): a
# forearm on its rest, not the mat, nor an arm's wrist or a hand over it


def wide_aims(points, half, *, lift_m: float, reach_m: float = WIDE_SEARCH_M, raised_m=RAISED_M,
              min_points: int = SEARCH_MIN_POINTS) -> list[dict]:
    """skin_aims about the skin nearest the last page anywhere within reach_m of it, when none lies by it: a limb
    put back elsewhere on the mat (2026-10-21: the overhead's depth read no skin within 50 mm of the last page's
    plane about it, the forearm moved since). The skin is what stands raised_m (lowest, highest) over the mat (no
    surface fitted); its min_points nearest the last page's centre centre the search, at their median height. Aims
    as skin_aims', in the last page's frame, "wide" set; none without such skin. They only aim views: a view places
    the page only by markers that agree with their printed layout."""
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1) & (np.abs(p[:, 0]) <= reach_m) & (np.abs(p[:, 1]) <= reach_m)]
    if len(p) < min_points:
        return []
    over = p[:, 2] - np.percentile(p[:, 2], 10)
    up = p[(over > raised_m[0]) & (over < raised_m[1])]
    if len(up) < min_points:
        return []
    nearest = up[np.argsort(np.hypot(up[:, 0], up[:, 1]))[:min_points]]
    shift = np.r_[np.median(nearest[:, :2], axis=0), np.median(nearest[:, 2])]
    return [{**a, "xy": [a["xy"][0] + float(shift[0]), a["xy"][1] + float(shift[1])],
             "rise_m": a["rise_m"] + float(shift[2]), "wide": True}
            for a in skin_aims(up - shift, half, lift_m=lift_m, min_points=min_points)]   # the skin, not the mat


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


SKIN_FACING = 0.3   # a curved page's skin is rectified where it faces the camera within ~73 deg


def page_to_pixel(points_page: np.ndarray, base_from_page: np.ndarray, base_from_camera: np.ndarray,
                  k: np.ndarray, dist: np.ndarray, surface=None) -> np.ndarray:
    """Pixel (u, v) of page-plane points (N, 2), with the Brown-Conrady distortion (k1 k2 p1 p2 k3). With a curved
    page's `surface` (tatbot_motion.surface.PageSurface, its base base_from_page) the points are print points (u, v)
    on it (PageSurface.lift), not on the plane."""
    pts = np.asarray(points_page, dtype=float).reshape(-1, 2)
    if surface is None:
        homog = np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))]
        cam = (np.linalg.inv(base_from_camera) @ base_from_page @ homog.T)[:3].T
    else:   # print points on the skin; where it faces away from the camera (past a forearm's flank), no pixel
        lifted, normals = surface.moved(base_from_page).lift(pts)
        cam_from_base = np.linalg.inv(base_from_camera)
        cam = (cam_from_base @ np.c_[lifted, np.ones(len(lifted))].T)[:3].T
        facing = -np.sum((normals @ cam_from_base[:3, :3].T) * cam, axis=1) / np.linalg.norm(cam, axis=1)
        cam[facing < SKIN_FACING] = np.nan
    x, y = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    k1, k2, p1, p2, k3 = (list(np.asarray(dist, float)) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]]


def rectify(image: np.ndarray, base_from_page, base_from_camera, k, dist, extent_m, px_per_mm: float,
            surface=None) -> np.ndarray:
    """The camera image resampled onto the page plane (page_grid layout); outside the frame is black. With a
    curved page's `surface` onto it, in the print's millimetres (page_to_pixel)."""
    import cv2

    gx, gy = page_grid(extent_m, px_per_mm)
    uv = page_to_pixel(np.c_[gx.ravel(), gy.ravel()], base_from_page, base_from_camera, k, dist, surface)
    uv = np.where(np.isfinite(uv), uv, -1.0)   # unseen skin: outside the image, black
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


def configured_serial() -> str:
    """The serial of the wrist D405 stack.yaml's page.locate.camera names by its vision.toml role, "" when it names none:
    every opener's default, as a second D405 shares the ros node's USB (2026-10-23) and no serial opened whichever
    enumerated first."""
    import wrist_cameras
    from tatbot_description import load_stack

    if not (role := (load_stack()["page"].get("locate") or {}).get("camera")):
        return ""
    for camera in wrist_cameras.registry(repo_root()):
        if camera["role"] == role:
            return camera["serial"]
    raise RuntimeError(f"stack.yaml page.locate.camera {role!r}: no D405 of that role in vision.toml")


class WristCamera:
    """Exclusive ROS wrist D405: colour and optional native or colour-aligned depth in metres; with no `serial`, the one
    stack.yaml names by role (configured_serial). `tap` (wrist_tap) republishes each grab_both's last frames for display; `lock`
    is held while the pipeline is read (the tap's poll). `preview` (wrist_tap.previews) opens display-only D405s beside
    it, closed with it."""

    def __init__(self, serial: str = "", width: int = 1280, height: int = 720, fps: int = 30, depth: bool = False,
                 aligned: bool = False, tap=None, preview=None):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial := serial or configured_serial():
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
        self.depth, self.fps, self.lock, self.stamp_ns = depth, fps, threading.Lock(), None
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
        self.tap = tap(self) if tap else None
        self.preview = preview() if preview else None

    def grab(self, n: int) -> list[np.ndarray]:
        return self.grab_both(n)[0]

    def grab_both(self, n: int, observe=None, fresh: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """n colour frames, and n depth frames in metres (empty without depth); `fresh`: none queued before the call
        (a camera kept open while the arm moved), so they are taken where the arm stands now."""
        colors, depths = [], []
        self.capture_timestamps = []
        self.raw_depth = []
        with self.lock:
            while fresh and self.pipe.poll_for_frames():
                pass
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
        self.stamp_ns = time.time_ns()   # the stamp the tap publishes the last frame at (wrist_tap.show_markers)
        if self.tap is not None and colors:   # references only; the tap thread encodes and publishes
            self.tap.put(colors[-1], self.raw_depth[-1] if self.depth else None, self.stamp_ns)
        return colors, depths

    def close(self) -> None:
        """The previews close, the tap is told to stop (it waits for nothing and never touches a stopped pipeline),
        then the pipeline."""
        if self.preview is not None:
            self.preview.close()
        if self.tap is not None:
            self.tap.close()
        with self.lock:
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


def print_artwork(pattern_id: str | None):
    """The installed fiducial-tile print's markers (scripts/vision/fiducials/stencil.Tile, from the tile.json beside
    its tracking.json); None without one."""
    from stencil_reference import observer_references

    if not pattern_id:
        return None
    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    from fiducials import stencil

    return stencil.load(observer_references() / pattern_id)


def snap_to_plane(cam_from_print: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """The pose moved onto a measured plane n.x = d: its origin along its camera ray, its x projected into the plane."""
    o, x = cam_from_print[:3, 3], cam_from_print[:3, 0]
    zc = n if cam_from_print[:3, 2] @ n >= 0 else -n
    x2 = x - (x @ zc) * zc
    x2 /= np.linalg.norm(x2)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x2, np.cross(zc, x2), zc, o * (d / (n @ o))
    return out


def tip_on_print(cam_from_print: np.ndarray, tip_cam) -> np.ndarray:
    """A camera-frame point in the print's frame: (x, y, height over the paper)."""
    pfc = np.linalg.inv(cam_from_print)
    return pfc[:3, :3] @ np.asarray(tip_cam, float) + pfc[:3, 3]


def view_fit(page_image: np.ndarray, tile, extent_m, px_per_mm: float) -> dict | None:
    """One locate view's in-plane correction by the print's markers in its page image (fiducials.stencil.fit); None
    without a print or when its markers are not accepted."""
    if tile is None:
        return None
    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    from fiducials import stencil

    fit = stencil.fit(page_image, tile, extent_m, px_per_mm)
    return fit if fit is not None and fit["accepted"] else None


def hold_tip(colors, frames, meta: dict, tip_cam, cam_from_page: np.ndarray, page: dict, tile,
             surface=None, markers: dict | None = None, every=None) -> dict | None:
    """Where the pen's tip is on the print in one gauge hold's wrist frames, from the camera's own geometry: the
    surface the gauge fitted (gauge.Frame, the paper's plane by the pen: a whole-frame plane fit takes the mat under
    a 5 mm silicone skin, 3.8 mm off), the page believed there (`cam_from_page`, FK: it only seeds the fit) carried
    onto it, the print's markers read in the frames (tile_fit), and the gauge's tip point `tip_cam` carried into the
    print's frame. Neither FK nor the camera's CAD mount enters. {"tip_m": (x, y, height over the print), "score",
    "margin", "markers", "ids" (theirs), "rms_m", "accepted", "cam_from_print"}, or None when the gauge saw no surface or no marker
    of the print keeps its shape. On a curved page (`surface`, a PageSurface: its shape, carried by cam_from_page)
    the page is the skin in the print's millimetres, moved along its normal onto the skin the gauge measured under
    the tip, and the tip's place is the print point under it. `markers` (a dict) takes every marker read (tile_fit).
    `every`: every colour frame the hold took, its markers read in all of them (else in the gauge frames' own). A
    curved hold's 25 frames read in fives (gauge.WristGauge._pooled) left 20 unread: on a faint bicep transfer
    (2026-10-24) a frame decoded 0-4 of its six markers, a different few each time, and three of five holds read
    none or one in their five; in all 25 they read 4-5, two of them keeping their printed shape (OpenCV 4.6, the
    ros node's)."""
    seen = [(c, f) for c, f in zip(colors, frames, strict=True) if f is not None]
    if not seen:
        return None
    measured = seen[0][1]
    k, dist = (np.asarray(meta["color_intrinsics"][key], float) for key in ("k", "coeffs"))
    extent = page_extent(page["size_m"])
    if surface is None:
        n = -np.asarray(measured.n, float)   # away from the camera
        snapped = snap_to_plane(np.asarray(cam_from_page, float), n, float(n @ np.asarray(measured.c, float)))
        skin = None
    else:   # the skin moved along its normal onto the skin the gauge saw about the tip (else its point under it)
        from tatbot_session import curved, gauge

        placed = surface.moved(cam_from_page)
        gap, normal = placed.height_over(measured.c)
        about = curved.skin_gap(placed, gauge.ring(measured))
        snapped = np.asarray(cam_from_page, float).copy()
        snapped[:3, 3] += (gap if about is None else about) * normal
        skin = surface.moved(snapped)
    fit = tile_fit(colors=[c for c, _ in seen] if every is None else list(every), tile=tile, cam_from_page=snapped,
                   k=k, dist=dist, extent_m=extent, surface=skin, markers=markers)
    if fit is None:
        return None
    correction = correction_matrix(fit["tx_m"], fit["ty_m"], fit["theta_rad"])
    on_print = snapped @ np.linalg.inv(correction)
    tip = tip_on_print(on_print, tip_cam)
    if skin is not None:   # the print point under the tip, in the print's frame
        uv = skin.foot(tip_cam)[0]
        tip[:2] = (correction @ [uv[0], uv[1], 0.0, 1.0])[:2]
    return {"tip_m": [float(v) for v in tip], "score": fit["score"], "margin": fit["margin"], "markers": fit["markers"],
            "ids": fit["ids"], "rms_m": fit["rms_m"], "accepted": fit["accepted"], "cam_from_print": on_print.tolist(),
            "turn_rad": float(fit["theta_rad"])}


def tile_fit(colors, tile, cam_from_page, k, dist, extent_m, surface=None, grid_px_per_mm: float = 2.5,
             reach_px: float = 12.0, markers: dict | None = None) -> dict | None:
    """A fiducial-tile print's in-plane correction (fiducials.stencil.fit_points) from its markers found in the
    camera's own frames, each corner carried onto the page through the map rectify samples by (page_to_pixel, onto
    a curved page's skin), inverted by an affine fit to the eight map points nearest it. Over the curved holder
    (2026-10-18) the raw frames held 9 markers a view, the rectified page 5: a curved page's resampling lost the
    markers its field bends. A corner farther than reach_px from the map (skin facing away) drops its marker.
    `markers` (a dict) takes every marker read, kept or dropped: id -> its corners' median in the frames' pixels."""
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    from fiducials import stencil

    gx, gy = page_grid(extent_m, grid_px_per_mm)
    page = np.c_[gx.ravel(), gy.ravel()]
    uv = page_to_pixel(page, cam_from_page, np.eye(4), k, dist, surface)
    ok = np.isfinite(uv).all(axis=1)
    page, uv = page[ok], uv[ok]
    if len(uv) < 8:
        return None
    tree, det, found = cKDTree(uv), stencil.detector(tile), {}
    for color in colors:
        for tag_id, corners in stencil.find(color, tile, det).items():
            found.setdefault(tag_id, []).append(corners)
    seen = {}
    for tag_id, reads in found.items():
        corners = np.median(np.stack(reads), axis=0)
        if markers is not None:
            markers[tag_id] = corners
        dists, near = tree.query(corners, k=8)
        if float(dists.max()) > reach_px:
            continue
        seen[tag_id] = np.array([np.r_[c, 1.0] @ np.linalg.lstsq(np.c_[uv[j], np.ones(len(j))], page[j], rcond=None)[0]
                                 for c, j in zip(corners, near, strict=True)])
    return stencil.fit_points(seen, tile)


READ_MARGIN = 0.5   # a reading's margin to be taken: a tile reading's is 1 when its markers were accepted, else 0


def holds_shift(holds, *, outlier_m: float = 0.003, min_holds: int = 3, min_margin: float = READ_MARGIN,
                turned: bool = False, turn_holds: int = 5, span_m: float = 0.005,
                max_turn_rad: float = 0.087) -> dict | None:
    """The page's in-plane move from the gauge holds: each hold's believed page point "xy" and the tip's measured
    point on the print "tip_m" (hold_tip), the believed point moved = print. With `turned` (a curved page) the
    median of each hold's own marker fit's turn ("turn_rad", hold_tip; 0.1-0.4 deg apart over 9-10 holds on a
    forearm, 2026-10-19) within max_turn_rad turns the believed points first, and the holds within `outlier_m` of
    the median of tip - xy turn them on about their centre (a 2-D Kabsch fit of their places); the median of what
    they leave shifts them. Else the turn stays the locate's. A hold whose "margin" is under `min_margin` (its
    markers not accepted) is left out, as the track leaves out such a reading. {"shift_m" at "about_m" (the holds'
    centre), "turn_rad" (taken in all), "fit_turn_rad" (the places' in all), "move" (as correction_matrix's tx, ty,
    theta), "spread_m" (1.4826 MAD per axis about it), "off_m", "places" (below), "holds", "dropped"}, or None with
    fewer than `min_holds`."""
    # "off_m": the farthest from the shift of the holds kept and of every one read by two "markers" or more, dropped or
    # not; "places": the kept ones' read by two markers or more. A curved page's holds past their limit move it only
    # on these (executor._hold_page): three lone markers outvoted two holds by three, or one place read again counted
    # as agreeing readings, under a MAD spread and a count of records (2026-10-20 review).
    # The places' turn is what the track's residual sees: on a forearm print (2026-10-19) the holds' places put the
    # page 2-3 deg off the locate's while their markers read it within 1 deg (a camera mount's yaw turns the markers'
    # reading, not the places'), a spot read again later agreed to 0.04-0.23 mm, and the track carried the turn as
    # corrections of up to 2.4 mm. It is not taken from fewer than turn_holds places (one misread then turns the page
    # and nothing is left to show it: four holds turned one 3.3 deg the track read as 0.7; holds nearer each other
    # than span_m are one place: a hold made again is recorded again, 2026-10-19's runs), from holds spanning under
    # span_m one way (the track's Field takes no turn there either), or past max_turn_rad in all: the locate holds the
    # yaw to 1-1.5 deg and a curved page is set up again past 5 deg (page.surface.rigid.max_rad), so it is a misread.
    holds = [h for h in holds if h.get("margin") is None or h["margin"] >= min_margin]
    turns = [h["turn_rad"] for h in holds if turned and h.get("turn_rad") is not None]
    marks = float(np.median(turns)) if len(turns) >= min_holds and abs(np.median(turns)) <= max_turn_rad else 0.0
    xy = np.array([h["xy"] for h in holds], float).reshape(-1, 2) @ correction_matrix(0.0, 0.0, marks)[:2, :2].T
    r = np.array([h["tip_m"][:2] for h in holds], float).reshape(-1, 2) - xy
    if len(r) < min_holds:
        return None
    keep = np.linalg.norm(r - np.median(r, axis=0), axis=1) <= outlier_m
    if keep.sum() < min_holds:
        return None
    sure = np.array([h.get("markers", 0) >= 2 for h in holds])
    about = xy[keep].mean(axis=0)
    a, b = xy[keep] - about, (xy + r)[keep] - (xy + r)[keep].mean(axis=0)
    fit = float(np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())) if turned else 0.0
    wide = _places(xy[keep], span_m) >= turn_holds and np.linalg.eigvalsh(a.T @ a / len(a))[0] >= span_m ** 2
    turn = fit if wide and abs(marks + fit) <= max_turn_rad else 0.0
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    left = r + (xy - about) - (xy - about) @ rot.T   # each hold's shift at the centre once the turn is out
    shift = np.median(left[keep], axis=0)
    spread, off = 1.4826 * np.median(np.abs(left[keep] - shift), axis=0), np.linalg.norm(left - shift, axis=1)
    return {"shift_m": shift.tolist(), "about_m": about.tolist(), "turn_rad": marks + turn, "fit_turn_rad": marks + fit,
            "move": [*(about + shift - rot @ about).tolist(), marks + turn], "spread_m": spread.tolist(),
            "off_m": float(off[keep | sure].max()), "places": _places(xy[keep & sure], span_m),
            "holds": int(keep.sum()), "dropped": int((~keep).sum())}


def _places(xy: np.ndarray, span_m: float) -> int:   # points nearer an earlier one than span_m are its place again
    return int(sum(np.linalg.norm(xy[:k] - p, axis=1).min(initial=np.inf) >= span_m for k, p in enumerate(xy)))


def combine_views(solutions: list[dict | None], floor=(0.0, 0.0, 0.0)) -> dict | None:
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


def analyse(views, program: dict, clear_m, extent_m, px_per_mm: float, *, tile, floor=(0.0, 0.0, 0.0),
            ink_prior_m=None, ink_gate_m: float = 0.003, centre_m=(0.0, 0.0)) -> dict:
    """Post-draw analysis of per-pose page images: where the print is against the page the run used (each view's
    markers of the fiducial `tile`, fiducials.stencil.fit, combined over poses: "border", for its readers), where
    the ink is against the plan (the shift: camera mount plus arm), the coverage of the plan, and the ink's
    placement error on the print (shift + the print's correction: what a page trim would take out). The ink's shift
    is measured about `centre_m`, the page point the views aim at.

    With `ink_prior_m` (the ink-vs-plan shift this rig draws with; a trim moves the ink and the plan
    together, so it does not change it), a view whose shift lies farther than `ink_gate_m` from it is
    not used: a view the wrist registers off by more than that, or whose search matched other ink,
    moved the median by 5-15 mm on a page of small drawings. The placement pairs each used view's
    shift with that view's own print fit, so a view's registration error cancels, and falls back to
    the combined fit for a view without one. No view near the prior: every found view is used and
    `ink["gated"]` says so."""
    sys.path.insert(0, str(repo_root(None) / "scripts/vision"))
    from fiducials import stencil

    fits = [stencil.fit(v, tile, extent_m, px_per_mm) for v in views]
    borders = [f if f is not None and f["accepted"] else None for f in fits]
    inks = [ink_alignment(v, program, extent_m, px_per_mm, clear_m) for v in views]
    border = combine_views(borders, floor)
    found = [i for i, k in enumerate(inks) if k is not None and k.get("found", True)]
    used, gated = found, False
    if ink_prior_m is not None:
        prior = np.asarray(ink_prior_m, dtype=float)[:2]
        near = [i for i in found if (k := inks[i]) is not None and np.hypot(k["dx_m"] - prior[0], k["dy_m"] - prior[1]) <= ink_gate_m]
        if near:
            used, gated = near, len(near) < len(found)
    ink = None
    if used:
        good = [k for i in used if (k := inks[i]) is not None]
        if good:
            ink = {"dx_m": float(np.median([k["dx_m"] for k in good])), "dy_m": float(np.median([k["dy_m"] for k in good])),
                   "coverage": float(np.median([k["coverage"] for k in good])),
                   "p50_gap_m": float(np.median([k["p50_gap_m"] for k in good])),
                   "p95_gap_m": float(np.median([k["p95_gap_m"] for k in good])), "views": len(good),
                   "found": len(found), "used": used, "gated": bool(gated)}
    placement = None
    pairs = [(k, b) for i in used if (k := inks[i]) is not None and (b := (borders[i] or border)) is not None]
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
    return "inspect: " + ("; ".join(parts) if parts else "no print or ink found")


def write_inspection(out_dir, views, analysis: dict, program: dict, clear_m, extent_m, px_per_mm: float) -> None:
    """The inspection's page images (page<n>.png), the plan over each (overlay<n>.png), analysis.json and the line."""
    import json

    import cv2

    for n, view in enumerate(views):
        cv2.imwrite(str(out_dir / f"page{n}.png"), view)
        cv2.imwrite(str(out_dir / f"overlay{n}.png"), overlay(view, program, clear_m, extent_m, px_per_mm))
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(summary(analysis), flush=True)
