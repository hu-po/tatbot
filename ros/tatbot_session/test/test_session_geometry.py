"""Page plane from touches, trim, drift and re-plan decisions, arc bookkeeping."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_session import geometry as g


def test_fit_plane_exact_and_oriented():
    pts = [[0.3, -0.2, 0.03], [0.34, -0.2, 0.031], [0.32, -0.16, 0.029]]
    n, d = g.fit_plane(pts, [0, 0, 1])
    assert n[2] > 0 and abs(np.linalg.norm(n) - 1) < 1e-12
    for p in pts:
        assert abs(n @ p + d) < 1e-12
    n2, d2 = g.fit_plane(pts, [0, 0, -1])
    assert np.allclose(n2, -n) and math.isclose(d2, -d)
    with pytest.raises(ValueError):
        g.fit_plane(pts[:2], [0, 0, 1])


def test_touched_page_takes_xy_yaw_from_camera_and_height_tilt_from_touches():
    camera = g.rpy_matrix([0.371, -0.231, 0.045], [0.0, 0.0, 0.3])  # camera says 45 mm, level
    tilt = g.rpy_matrix([0, 0, 0], [0.02, -0.01, 0.0])
    normal = tilt[:3, 2]
    d = -normal @ np.array([0.371, -0.231, 0.0334])  # the touched paper passes 33.4 mm under the centre
    page = g.touched_page(camera, normal, d)
    assert np.allclose(page[:3, 2], normal)
    assert abs(normal @ page[:3, 3] + d) < 1e-12  # origin on the touched plane
    assert np.allclose(page[:2, 3], camera[:2, 3], atol=1e-3)  # x, y from the camera (projection along n)
    assert np.allclose(page[:3, :3].T @ page[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(page[:3, :3]) > 0
    # yaw: page x is the camera x projected onto the plane
    x_proj = camera[:3, 0] - (camera[:3, 0] @ normal) * normal
    assert np.allclose(page[:3, 0], x_proj / np.linalg.norm(x_proj))


def test_page_used_applies_correction_then_trim_in_page_frame():
    camera = g.rpy_matrix([0.37, -0.23, 0.04], [0, 0, math.pi / 2])
    touched = camera.copy()
    touched[2, 3] = 0.0334
    correction = np.linalg.inv(camera) @ touched
    used = g.page_used(camera, correction, [0.001, -0.002])
    # trim is in the page frame: page x is base +y after the 90 degree yaw
    assert np.allclose(used[:3, 3], [0.37 + 0.002, -0.23 + 0.001, 0.0334])
    # the correction rides with the page: a moved page keeps the touched height offset
    moved = g.rpy_matrix([0.38, -0.23, 0.04], [0, 0, math.pi / 2])
    assert np.allclose(g.page_used(moved, correction, [0, 0])[:3, 3], [0.38, -0.23, 0.0334])


def test_page_moved_thresholds():
    a = g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0])
    assert not g.page_moved(a, g.rpy_matrix([0.3709, -0.23, 0.03], [0, 0, 0]), 0.001, 0.0087)
    assert g.page_moved(a, g.rpy_matrix([0.3711, -0.23, 0.03], [0, 0, 0]), 0.001, 0.0087)
    assert not g.page_moved(a, g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0.0086]), 0.001, 0.0087)
    assert g.page_moved(a, g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0.0088]), 0.001, 0.0087)


def test_touch_points_are_a_triangle_about_the_page_centre():
    pts = g.touch_points(0.012)
    assert np.allclose(pts, [[-0.012, -0.012], [0.012, -0.012], [0.0, 0.012]])
    assert np.all(np.abs(pts) <= 0.031 - 0.005)  # well inside the 62 x 112 mm clear centre


def test_arc_at_latch_uses_the_measured_tip_and_falls_back_to_the_start():
    arc = np.array([np.nan, np.nan, 0.010, 0.011, 0.012, 0.013, np.nan])
    tip = np.array([[0, 0, 0.01], [0, 0, 0.005], [0.010, 0, 0], [0.011, 0, 0], [0.012, 0, 0], [0.013, 0, 0],
                    [0.013, 0, 0.01]])
    traj = SimpleNamespace(arc_m=arc, tip=tip)
    assert g.arc_at(traj, 1) == 0.010  # nothing drawn yet: resume where this plan starts
    assert g.arc_at(traj, 4) == 0.012
    assert g.arc_at(traj, 5, tip_measured=[0.0111, 0, 0]) == 0.011  # measured lags the command
    assert g.arc_at(traj, 6) == 0.013  # latched in the lift: the whole stroke was drawn


def test_stroke_length_closes_closed_strokes():
    pts = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01]]
    assert g.stroke_length(pts) == pytest.approx(0.02)
    assert g.stroke_length(pts, closed=True) == pytest.approx(0.02 + math.hypot(0.01, 0.01))
    assert g.stroke_length([[0.0, 0.0]]) == 0.0


def test_tool_down_pose_points_the_tool_into_the_page_without_wrist_spin():
    page = g.rpy_matrix([0.37, -0.23, 0.03], [0.05, 0.0, 0.4])
    current = g.rpy_matrix([0.3, 0, 0.1], [math.pi, 0, 0.2])
    pose = g.tool_down_pose(page, [0.01, 0.02], 0.01, current)
    assert np.allclose(pose[:3, 2], -page[:3, 2])
    assert np.allclose(pose[:3, 3], (page @ [0.01, 0.02, 0.01, 1])[:3])
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    assert pose[:3, 0] @ current[:3, 0] > 0.99


def test_quaternion_round_trip():
    m = g.rpy_matrix([1, 2, 3], [0.3, -0.7, 2.9])
    back = g.quat_matrix([1, 2, 3], g.matrix_quat(m))
    assert np.allclose(m, back)
    assert g.rotation_angle(m, back) < 1e-6


def test_first_contact_extrapolates_the_force_ramp_to_its_start():
    # A hold and fast leg at -7 N, then a 0.5 mm/s descent from 6 mm above the page to 1.7 mm into a
    # 1.5 N/mm give with the in-air force wandering around -1 N (bench 2026-09-26).
    rng = np.random.default_rng(1)
    hold_h, hold_f = np.full(200, 0.020), np.full(200, -7.0)
    h = np.linspace(0.006, -0.0017, 800)
    f = -1.0 + 0.3 * np.sin(h * 900.0) * (h > 0) + 1500.0 * np.clip(-h, 0, None) + rng.normal(0, 0.05, h.size)
    fit = g.first_contact(np.concatenate([hold_h, h]), np.concatenate([hold_f, f]))
    assert fit is not None and fit["method"] == "fit" and abs(fit["offset_m"] - 0.0017) < 0.0002
    assert abs(fit["stiffness_n_m"] - 1500.0) < 150.0 and abs(fit["base_n"] + 1.0) < 0.3
    # A noisy ramp whose fit is implausible falls back to the give model: rise / 1500 N/m.
    wobbly = f + 1.5 * np.sin(h * 4000.0) * (h < 0)
    model = g.first_contact(h, wobbly)
    assert model is not None and abs(model["offset_m"] - model["rise_n"] / 1500.0) < 1e-12 or model["method"] == "fit"
    assert g.first_contact(h[:30], f[:30]) is None                  # too few samples
    assert g.first_contact(h, np.full(h.size, -1.0)) is None        # no rise: the trip pose stands


def test_height_plane_keeps_the_normal_and_takes_the_median_height():
    n = np.array([0.0, 0.03, 1.0]) / np.linalg.norm([0.0, 0.03, 1.0])
    pts = [[0.30, 0.06, 0.0450], [0.31, 0.04, 0.0410], [0.32, 0.05, 0.0440]]
    normal, d = g.height_plane(pts, n)
    assert np.allclose(normal, n) and np.isclose(-d, np.median(np.asarray(pts) @ n))


def test_fiducial_contact_time_finds_where_the_gap_opens():
    # 16.5 Hz fiducials over 20 s; the EE stops at t = 12 s while the encoders descend 0.2 mm/s.
    rng = np.random.default_rng(5)
    t = np.arange(0, 20, 1 / 16.5)
    gap = 0.004 + 0.0002 * np.clip(t - 12.0, 0, None) + rng.normal(0, 0.0003, t.size)
    fit = g.fiducial_contact_time(t, gap)
    assert fit is not None and abs(fit["t_c"] - 12.0) < 1.0 and abs(fit["rate_m_s"] - 0.0002) < 0.00006
    assert g.fiducial_contact_time(t, 0.004 + rng.normal(0, 0.0003, t.size)) is None   # no contact: flat
    assert g.fiducial_contact_time(t[:16], gap[:16]) is None   # twice min_after: no hinge to try, no crash


def test_a_lost_page_is_stale_past_max_lost_s_or_when_never_measured():
    assert g.measured_age(None, 100.0) is None
    assert g.measured_age(90.0, 100.0) == 10.0
    assert g.stale_page({"source": "measured", "measured_age_s": 0.0}, 5.0) is None
    assert g.stale_page({"source": "fixed", "measured_age_s": None}, 5.0) is None
    assert g.stale_page({"source": "lost", "measured_age_s": 1.2}, 5.0) is None   # a flicker stands
    assert "lost for 3600 s" in g.stale_page({"source": "lost", "measured_age_s": 3600.0}, 5.0)
    assert "since the stack started" in g.stale_page({"source": "lost", "measured_age_s": None}, 5.0)


def test_touches_sit_about_the_drawing_and_inside_a_small_one():
    program = {"ops": [{"op": "stroke", "points_m": [[0.002, 0.029], [0.028, 0.055]]}]}   # 26 mm, centre (15, 42)
    pts = g.touch_layout(program, 0.012)
    assert np.allclose(pts.mean(axis=0), [0.015, 0.042 - 0.0104 / 3])   # spread 0.8 x the 13 mm half-size
    assert np.all(pts >= [0.002, 0.029]) and np.all(pts <= [0.028, 0.055])
    tiny = {"ops": [{"op": "stroke", "points_m": [[0.0, 0.0], [0.010, 0.010]]}]}
    assert np.ptp(g.touch_layout(tiny, 0.012)[:, 0]) == pytest.approx(0.008)   # 0.8 of the 5 mm half-size
    line = {"ops": [{"op": "stroke", "points_m": [[-0.01, 0.0], [0.01, 0.0]]}]}
    assert np.ptp(g.touch_layout(line, 0.012)[:, 1]) == pytest.approx(0.008)   # a line: the 4 mm floor
    assert np.allclose(g.touch_layout({"ops": []}, 0.012), g.touch_points(0.012))


def test_a_drawing_must_stay_inside_the_prints_clear_centre():
    """The session refuses, before the arm moves, a program whose lines (their width included) leave the clear
    centre of the print it draws on, however the program's own page was sized."""
    half = {"size_m": [0.083, 0.127], "clear_m": [0.045, 0.089], "geometry": "print stencil-half",
            "inner_edges_m": {"left": -0.0216, "right": 0.0216, "bottom": -0.0439, "top": 0.0439}}

    def program(half_width_m, line_width_m=0.0005):
        square = [[-half_width_m, -half_width_m], [half_width_m, -half_width_m], [half_width_m, half_width_m]]
        return {"resources": [{"id": "pen", "tool": {"line_width_m": line_width_m}}],
                "ops": [{"op": "tool_change", "id": "t", "resource_id": "pen"},
                        {"op": "stroke", "id": "s", "resource_id": "pen", "points_m": square,
                         "generation_width_m": 0.0003}]}

    assert g.off_print(program(0.020), half) is None
    assert g.off_print({"ops": []}, half) is None
    why = g.off_print(program(0.025), half)
    assert why and "x -25.2 to 25.2" in why and "x -21.6 to 21.6" in why and "83 x 127 mm" in why
    assert g.off_print(program(0.0213, line_width_m=0.0004), half) is None
    assert g.off_print(program(0.0213, line_width_m=0.0008), half)   # the pen's own line reaches the border
    assert g.off_print(program(0.025), {"size_m": [0.100, 0.150], "clear_m": [0.062, 0.112]}) is None
