"""The time laws ported from scripts/lib/pen_path.py keep their tested behaviour (scripts/tests/test_pen_path.py)."""
import math

import numpy as np
import pytest
from tatbot_motion import timelaw as tl

PERIOD = 0.0025


def _off_polyline(points, poly):
    """Largest distance from any point to the polyline (m)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    worst = 0.0
    for p in points:
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0.0, 1.0)
        worst = max(worst, float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min()))
    return worst


def _hatch_stroke():
    return np.array([[0.0, 0.0], [0.006, 0.0], [0.006, 0.0005], [0.0, 0.0005]])


def test_time_law_totals_and_continuity():
    length = 0.0572
    t, s, sdot = tl.time_law(length, 120.0, 2.0, PERIOD)
    assert len(t) == 48000
    assert t[0] == pytest.approx(PERIOD) and t[-1] == pytest.approx(120.0)
    assert s[-1] == pytest.approx(length, abs=1e-12)
    assert np.all(np.diff(s) >= -1e-15)
    cruise = length / 118.0
    assert sdot.max() == pytest.approx(cruise, rel=1e-12)
    assert sdot[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.abs(np.diff(sdot)).max() < cruise * 0.01
    assert np.abs(np.diff(s) / PERIOD - 0.5 * (sdot[1:] + sdot[:-1])).max() < cruise * 0.01
    with pytest.raises(ValueError):
        tl.time_law(length, 3.0, 2.0, PERIOD)


def test_time_law_acceleration_stays_within_the_quintic_peak():
    length, duration, ease = 0.02, 8.0, 2.0
    _, _, sdot = tl.time_law(length, duration, ease, PERIOD)
    cruise = length / (duration - ease)
    accel = np.abs(np.diff(sdot)) / PERIOD
    assert accel.max() <= tl.QUINTIC_PEAK * cruise / ease * 1.01


def test_resample_by_arclength_hits_vertices_and_trim_keeps_the_rest():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    points, tangents = tl.resample_polyline_by_arclength(poly, np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert np.allclose(points, [[0, 0], [0.5, 0], [1, 0], [1, 1], [1, 2]])
    assert np.allclose(tangents[1], [1, 0]) and np.allclose(tangents[-1], [0, 1])
    trimmed = tl.trim_polyline(poly, 1.5)
    assert np.allclose(trimmed, [[1.0, 0.5], [1.0, 2.0]])
    assert tl.polyline_length(trimmed) == pytest.approx(1.5)
    assert tl.trim_polyline(poly, 0.0) is poly or np.array_equal(tl.trim_polyline(poly, 0.0), poly)


def test_corner_speed_limits_isolated_corner_and_dense_arc():
    retime = tl.Retime()
    poly = np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.005]])
    arc, turn = tl.polyline_turn_angles(poly)
    caps = tl.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    assert caps[0] == caps[2] == 0.003
    assert caps[1] == pytest.approx(retime.corner_accel_m_s2 * retime.window_s / (math.pi / 2), rel=1e-6)
    radius = 0.0002
    angles = np.radians(np.arange(0.0, 180.0, 2.0))
    arc_poly = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    poly = np.concatenate([[[radius + 0.005, 0.0]], arc_poly[::-1], [[-radius - 0.005, 0.0]]])
    arc, turn = tl.polyline_turn_angles(poly)
    caps = tl.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    middle = caps[len(caps) // 2]
    assert 0.5 * math.sqrt(retime.corner_accel_m_s2 * radius) < middle < 1.5 * math.sqrt(retime.corner_accel_m_s2 * radius)
    doubled = np.repeat(np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.005]]), 2, axis=0)
    _, turn2 = tl.polyline_turn_angles(doubled)
    assert np.count_nonzero(turn2) == 1 and turn2.max() == pytest.approx(math.pi / 2)


def test_accel_limited_speeds_respect_budget_and_rest_ends():
    arc = np.linspace(0.0, 0.03, 3001)
    caps = np.full(len(arc), 0.010)
    v = tl.accel_limited_speeds(arc, caps, 0.010, v_start=0.0, v_end=0.0)
    assert v[0] == 0.0 and v[-1] == 0.0 and v.max() == pytest.approx(0.010)
    implied = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * np.diff(arc))
    assert np.abs(implied).max() <= 0.010 + 1e-9
    s_curve = tl.accel_limited_speeds(arc, caps, 0.010, v_start=0.0, v_end=0.0, jerk=0.05)
    assert np.all(s_curve <= v + 1e-12)
    implied_s = (s_curve[1:] ** 2 - s_curve[:-1] ** 2) / (2.0 * np.diff(arc))
    plateau = np.flatnonzero(s_curve >= 0.010 - 1e-12)
    assert len(plateau) and abs(implied_s[plateau[0] - 1]) < 0.002


def test_corner_time_law_keeps_the_plain_law_on_a_straight_stroke():
    straight = np.array([[0.0, 0.0], [0.012, 0.0]])
    t, s, _ = tl.time_law(0.012, 0.012 / 0.003 + 2.0, 2.0, PERIOD)
    t2, s2, _, info = tl.corner_time_law(straight, 0.003, 2.0, PERIOD, tl.Retime())
    np.testing.assert_array_equal(t2, t)
    np.testing.assert_array_equal(s2, s)
    assert info["added_s"] == 0.0 and info["limited_vertices"] == 0


def test_corner_time_law_slows_corners_preserves_geometry_and_respects_the_budget():
    stroke = _hatch_stroke()
    retime = tl.Retime()
    length = tl.polyline_length(stroke)
    t, s, _, info = tl.corner_time_law(stroke, 0.003, 2.0, PERIOD, retime)
    plain_t, _, _ = tl.time_law(length, length / 0.003 + 2.0, 2.0, PERIOD)
    assert info["limited_vertices"] == 2 and info["added_s"] == pytest.approx(t[-1] - plain_t[-1])
    assert s[-1] == length and np.all(np.diff(s) >= 0.0)
    points, _ = tl.resample_polyline_by_arclength(stroke, s)
    assert _off_polyline(points, stroke) < 1e-12
    speed = np.gradient(s, PERIOD)
    arc, turn = tl.polyline_turn_angles(stroke)
    caps = tl.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    for vertex in (1, 2):
        row = int(np.argmin(np.abs(s - arc[vertex])))
        assert speed[row] < caps[vertex] * 1.5 + 5e-5
    assert speed[0] < 1e-4 and abs(speed[-1]) < 1e-4
    assert speed.max() <= 0.003 + 1e-5
    accel = np.gradient(speed, PERIOD)
    away = np.ones(len(s), dtype=bool)
    for vertex in (1, 2):
        away &= np.abs(s - arc[vertex]) > 2e-5
    assert np.abs(accel[away]).max() < retime.corner_accel_m_s2 * 1.2


def test_pen_up_leg_is_one_continuous_profile_on_the_exact_polyline():
    retime = tl.Retime()
    points = np.array([[0.30, 0.0, 0.15], [0.32, 0.01, 0.15], [0.32, 0.01, 0.125], [0.32, 0.01, 0.12]])
    r0 = np.eye(3)
    r1 = tl.axis_rotation([0.0, 0.0, 1.0], 0.2)
    p, r, chord = tl.pen_up_leg(points, r0, r1, [0.020, 0.010, 0.003], retime, PERIOD,
                                gentle_end=True, include_start=True, omega_max=math.radians(8.0))
    assert _off_polyline(p, points) < 1e-12
    np.testing.assert_allclose(p[0], points[0], atol=1e-12)
    np.testing.assert_allclose(p[-1], points[-1], atol=1e-12)
    assert sorted(set(chord.tolist())) == [0, 1, 2]
    speed = np.linalg.norm(np.gradient(p, PERIOD, axis=0), axis=1)
    assert speed[0] < 1e-4 and speed[-1] < 1e-4
    assert speed[chord == 0].max() <= 0.020 + 1e-5
    assert speed[chord == 1].max() <= 0.010 + 1e-5
    assert speed[chord == 2].max() <= 0.003 + 1e-5
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    assert speed[(arc > 0.001) & (arc < arc[-1] - 0.001)].min() > 1e-4
    np.testing.assert_allclose(r[chord > 0], np.repeat(r1[None], np.count_nonzero(chord > 0), axis=0), atol=1e-12)
    omega = np.array([tl.rotation_angle(b @ a.T) for a, b in zip(r[:-1], r[1:], strict=True)]) / PERIOD
    assert omega.max() <= math.radians(8.0) + 1e-6
    accel = np.linalg.norm(np.gradient(np.gradient(p, PERIOD, axis=0), PERIOD, axis=0), axis=1)
    assert accel[chord == 2][10:-3].max() < retime.touchdown_accel_m_s2 * 1.3
    # tangential acceleration within the pen-up budget (a vertex turns the direction inside the feedforward window)
    away = np.ones(len(p), bool)
    for boundary in np.flatnonzero(np.diff(chord)):
        away[max(0, boundary - 3):boundary + 4] = False
    assert np.abs(np.gradient(speed, PERIOD))[away][5:-5].max() < retime.pen_up_accel_m_s2 * 1.3
    with pytest.raises(ValueError):
        tl.pen_up_leg(np.array([points[1], points[1], points[2]]), r0, r1, [0.010, 0.003], retime, PERIOD)


def test_pen_up_turn_keeps_the_rotary_budget():
    """pen_path's rotary budget: a leg that turns 0.3 rad changes its angular velocity at <= alpha."""
    points = np.array([[0.30, 0.0, 0.15], [0.31, 0.0, 0.15]])
    r1 = tl.axis_rotation([0.0, 0.0, 1.0], 0.3)
    loose = tl.Retime(angular_accel_rad_s2=0.0)
    retime = tl.Retime()
    _, r_loose, _ = tl.pen_up_leg(points, np.eye(3), r1, [0.12], loose, PERIOD, omega_max=0.1396)
    _, r, _ = tl.pen_up_leg(points, np.eye(3), r1, [0.12], retime, PERIOD, omega_max=0.1396)
    assert len(r) > len(r_loose)
    omega = np.array([tl.rotation_angle(b @ a.T) for a, b in zip(r[:-1], r[1:], strict=True)]) / PERIOD
    window = tl.FEEDFORWARD_WINDOW_TICKS
    change = np.abs(omega[window:] - omega[:-window]) / (window * PERIOD)
    assert change.max() < retime.angular_accel_rad_s2 * 1.3


def test_feedforward_is_centred_and_continuous():
    t = np.arange(400) * PERIOD
    p = np.stack([0.01 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    v = tl.feedforward(p, PERIOD)
    np.testing.assert_allclose(v[:, 0], 0.01, atol=1e-12)
    step = np.concatenate([np.zeros((50, 3)), p[:350]])
    v = tl.feedforward(step, PERIOD)
    assert np.abs(np.diff(v[:, 0])).max() <= 0.01 / tl.FEEDFORWARD_WINDOW_TICKS + 1e-12


def test_rotation_helpers():
    a, b = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    r = tl.align_rotation(a, b)
    np.testing.assert_allclose(r @ a, b, atol=1e-12)
    np.testing.assert_allclose(tl.align_rotation(a, -a) @ a, -a, atol=1e-12)
    axis, angle = tl.rotation_log(tl.axis_rotation([0.0, 0.6, 0.8], 0.7))
    np.testing.assert_allclose(axis, [0.0, 0.6, 0.8], atol=1e-12)
    assert angle == pytest.approx(0.7)
    mid = tl.rotation_slerp(np.eye(3), tl.axis_rotation([1.0, 0.0, 0.0], 1.0), [0.5])[0]
    assert tl.rotation_angle(mid) == pytest.approx(0.5)
