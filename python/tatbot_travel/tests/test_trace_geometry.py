"""Scan → surface → ink strokes → pen plan on a ray-cast forearm: a cylinder lying on a table, a curve
drawn on top, seen by the wrist camera from poses the arm's own FK puts it at."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tatbot_travel import inkmap, pen_path, surface
from tatbot_travel.camera import Intrinsics

AXIS_X, AXIS_Z, RADIUS = 0.33, 0.035, 0.035  # forearm: a cylinder along y on the table (z = 0)
FIST, FIST_R = np.array([AXIS_X, 0.10, 0.045]), 0.045  # and a fist at one end, so it is not symmetric
SKIN, INK, TABLE = (225, 205, 120), (25, 25, 30), (150, 120, 90)
REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])


def ink_curve(y: np.ndarray) -> np.ndarray:
    """The drawn line's angle around the cylinder (0 = the top) at each y: a gentle S."""
    return 0.25 * np.sin(y / 0.03)


def render(cam_p: np.ndarray, cam_r: np.ndarray, intr: Intrinsics) -> surface.Frame:
    """Ray-cast the table and the cylinder; ink is a 1.2 mm band around the curve."""
    v, u = np.mgrid[0:intr.height, 0:intr.width].astype(float)
    rays = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=-1) @ cam_r.T
    o = cam_p
    t_table = np.where(rays[..., 2] < 0, -o[2] / np.minimum(rays[..., 2], -1e-9), np.inf)
    # cylinder (x - AXIS_X)^2 + (z - AXIS_Z)^2 = R^2, |y| < 0.12
    dx, dz = rays[..., 0], rays[..., 2]
    ox, oz = o[0] - AXIS_X, o[2] - AXIS_Z
    a, b, c = dx ** 2 + dz ** 2, 2 * (ox * dx + oz * dz), ox ** 2 + oz ** 2 - RADIUS ** 2
    disc = b ** 2 - 4 * a * c
    t_cyl = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12)), np.inf)
    y_hit = o[1] + t_cyl * rays[..., 1]
    t_cyl = np.where((t_cyl > 0) & (np.abs(y_hit) < 0.12), t_cyl, np.inf)
    # the fist: a sphere
    oc = o - FIST
    bs, cs = (rays * oc).sum(axis=-1), oc @ oc - FIST_R ** 2
    a2 = (rays * rays).sum(axis=-1)
    disc_s = bs ** 2 - a2 * cs
    t_fist = np.where(disc_s > 0, (-bs - np.sqrt(np.maximum(disc_s, 0))) / a2, np.inf)
    t_fist = np.where(t_fist > 0, t_fist, np.inf)
    t_cyl = np.minimum(t_cyl, t_fist)
    t = np.minimum(t_table, t_cyl)
    seen = np.isfinite(t)
    hit = o + np.where(seen, t, 0.0)[..., None] * rays
    rgb = np.empty((*t.shape, 3), np.uint8)
    rgb[:] = TABLE
    on_cyl = t_cyl <= t_table
    rgb[on_cyl] = SKIN
    angle = np.arctan2(hit[..., 0] - AXIS_X, hit[..., 2] - AXIS_Z)
    on_ink = on_cyl & (np.abs(angle - ink_curve(hit[..., 1])) * RADIUS < 0.0006) & (np.abs(hit[..., 1]) < 0.05)
    rgb[on_ink] = INK
    cam_z = (hit - o) @ cam_r[:, 2]
    depth = np.where(seen, cam_z, 0.0)
    return surface.Frame(rgb=rgb, depth_m=depth, intr=intr, cam_p=cam_p, cam_r=cam_r)


@pytest.fixture(scope="module")
def kin():
    return pen_path.arm_kinematics()


@pytest.fixture(scope="module")
def scan(kin):
    """Three views looking down the pen axis from above the forearm."""
    intr = replace(Intrinsics.left_wrist(), distortion=(0.0,) * 5)  # the ray caster is a pinhole
    frames = []
    for y in (-0.03, 0.0, 0.03):
        result = kin.solve(REST, np.array([AXIS_X, y, AXIS_Z + RADIUS + 0.06]), np.array([0.3, 0.0, -1.0]),
                           REST, iters=200)
        assert result.ok()
        frames.append(render(*kin.camera_pose(result.q), intr))
    return frames


def test_the_scan_finds_the_table_and_the_forearm_surface(scan):
    points, _ = surface.fuse(scan)
    n, d = surface.table_plane(points)
    assert n @ surface.UP > 0.999 and abs(d) < 0.001
    arm = points[surface.above(points, (n, d))]
    skin = surface.Surface(arm)
    p, normal, count = skin.query(np.array([AXIS_X + 0.01, 0.0, AXIS_Z + RADIUS + 0.01]))
    assert count > 0
    radial = np.array([p[0] - AXIS_X, 0.0, p[2] - AXIS_Z])
    assert abs(np.linalg.norm(radial) - RADIUS) < 0.0005
    assert normal @ radial / np.linalg.norm(radial) > 0.995


def test_ink_strokes_land_on_the_drawn_curve(scan):
    points, _ = surface.fuse(scan)
    plane = surface.table_plane(points)
    skin = surface.Surface(points[surface.above(points, plane)])
    view = scan[1]
    strokes, mask = inkmap.strokes(view, skin, inkmap.forearm_pixels(view, plane))
    assert mask.any() and strokes
    longest = strokes[0]
    assert longest.length_m > 0.05
    angle = np.arctan2(longest.points[:, 0] - AXIS_X, longest.points[:, 2] - AXIS_Z)
    lateral_mm = np.abs(angle - ink_curve(longest.points[:, 1])) * RADIUS * 1000
    assert np.median(lateral_mm) < 0.5 and lateral_mm.max() < 1.5


def test_the_plan_hovers_the_working_point_on_the_ink(kin, scan):
    points, _ = surface.fuse(scan)
    plane = surface.table_plane(points)
    skin = surface.Surface(points[surface.above(points, plane)])
    stroke = inkmap.strokes(scan[1], skin, inkmap.forearm_pixels(scan[1], plane))[0][0]
    plan = pen_path.plan_trace(kin, REST, REST, stroke.points, stroke.normals)
    assert plan.peak_joint_speed <= pen_path.MAX_JOINT_SPEED
    tracing = plan.phase == 1
    reached = np.array([kin.gap_pose(q)[0] for q in plan.q[tracing]])
    nearest = np.min(np.linalg.norm(reached[:, None] - stroke.points[None], axis=2), axis=1)
    assert nearest.max() < 0.002
    # The lens face is the standoff above the skin: the working point sits on it.
    radial = np.linalg.norm(reached[:, [0, 2]] - [AXIS_X, AXIS_Z], axis=1)
    assert np.abs(radial - RADIUS).max() < 0.0015


def test_views_from_a_seed_look_down_on_the_forearm_and_a_scan_round_trips(kin, scan, tmp_path):
    from tatbot_travel import trace

    seed = kin.solve(REST, np.array([AXIS_X, 0.0, 0.15]), np.array([0.0, 0.0, -1.0]), REST, iters=200).q
    views = trace.scan_views(kin, scan[1], seed)
    assert len(views) == 3
    for q in views:
        gap, axis = kin.gap_pose(q)
        assert axis[2] < -0.99 and abs(gap[0] - AXIS_X) < 0.02  # over the forearm, biased to the seen side
        assert abs(gap[2] - (max(AXIS_Z + RADIUS, FIST[2] + FIST_R) + trace.VIEW_HEIGHT_M)) < 0.004  # over the top
    qs = [kin.solve(REST, *kin.gap_pose(q), REST, iters=5).q for q in views]
    trace.save_scan(tmp_path / "scan.npz", scan, qs)
    frames, q_loaded = trace.load_scan(tmp_path / "scan.npz", kin)
    assert np.allclose(q_loaded, qs) and len(frames) == len(scan)


def test_score_reads_a_faithful_trace_as_on_the_line(kin, scan):
    from tatbot_travel import trace

    skin, found, _, _ = trace.strokes_from_scan(scan)
    plan = pen_path.plan_trace(kin, REST, REST, found[0].points, found[0].normals)
    result = trace.score(kin, skin, [found[0]], plan.q[plan.phase == 1])
    assert result["cross_track_mm_mean"] < 0.5 and result["hover_error_mm_mean"] < 1.0
    assert result["ink_covered_mm"] > 0.95 * result["ink_mm"]


def test_astra_pixels_lift_onto_the_skin_and_carry_on_from_the_hover(kin, scan):
    from tatbot_travel import astra, trace

    skin, found, _, view = trace.strokes_from_scan(scan)
    stroke = found[0]
    frame = scan[view]
    uv, _ = frame.project(stroke.points[::8])  # a decider's pixels along the ink
    points, normals = astra.lift(frame, skin, uv)
    assert len(points) > 20
    radial = np.linalg.norm(points[:, [0, 2]] - [AXIS_X, AXIS_Z], axis=1)
    assert np.abs(radial - RADIUS).max() < 0.001
    start = pen_path.plan_trace(kin, REST, REST, points[:10], normals[:10])
    q_hover = start.q[start.phase == 1][-1]
    follow = pen_path.plan_follow(kin, q_hover, points[10:], normals[10:])
    assert follow.peak_joint_speed <= pen_path.MAX_JOINT_SPEED
    reached = np.array([kin.gap_pose(q)[0] for q in follow.q[follow.phase == 1]])
    assert np.min(np.linalg.norm(reached[:, None] - points[None, 10:], axis=2), axis=1).max() < 0.002
    marked = astra.annotate(frame.rgb, uv[0], uv[:5])
    assert marked.shape == frame.rgb.shape and not np.array_equal(marked, frame.rgb)


def test_a_perturbed_demonstration_starts_off_the_line_and_settles_onto_it(scan):
    from tatbot_travel import trace

    _, found, _, _ = trace.strokes_from_scan(scan)
    stroke = found[0]
    pts = trace.perturbed(stroke, np.random.default_rng(3), max_offset_m=0.006)
    off = np.linalg.norm(pts - stroke.points, axis=1)
    assert 0.0005 < off[0] <= 0.006
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(stroke.points, axis=0), axis=1))])
    assert off[s > 0.015].max() < 1e-9


def test_the_overhead_view_aligns_onto_the_wrist_scan(scan):
    """A D555-like camera 0.7 m up, yawed and tilted, recovered from the forearm alone."""
    from tatbot_travel import overhead, trace

    yaw, tilt = np.radians(120), np.radians(8)
    look = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # optical z down, y flipped
    tip = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    cam_r = look @ down @ tip
    cam_p = np.array([AXIS_X + 0.05, 0.02, 0.7])
    intr = replace(Intrinsics.left_wrist(), width=640, height=360, fx=323.0, fy=323.0, cx=320.0, cy=180.0,
                   distortion=(0.0,) * 5)
    seen = render(cam_p, cam_r, intr)
    over = surface.Frame(rgb=seen.rgb, depth_m=seen.depth_m, intr=intr, cam_p=np.zeros(3), cam_r=np.eye(3),
                         range_m=overhead.OVERHEAD_RANGE_M)
    over_arm, over_plane = overhead.forearm_cloud(over, overhead.CAMERA_UP)
    skin, _, plane, _ = trace.strokes_from_scan(scan)
    tf, rms, paired = overhead.align(skin.points, plane, over_arm, over_plane)
    assert paired > 0.8
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = cam_r.T, -cam_r.T @ cam_p  # overhead_from_base
    err = tf @ np.linalg.inv(truth)
    assert rms < 0.002
    assert np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 1.0
    # A cylinder slides along its own axis; across it and in height the pose is pinned.
    axis_in_overhead = cam_r.T @ np.array([0.0, 1.0, 0.0])
    off = err[:3, 3] - (err[:3, 3] @ axis_in_overhead) * axis_in_overhead
    assert np.linalg.norm(off) < 0.003
