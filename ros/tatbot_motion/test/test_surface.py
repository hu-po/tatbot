"""tatbot_motion.surface: a page that may bend, flat as the plane, a curve as its arc lengths."""
import numpy as np
import pytest
from tatbot_motion.surface import HeightField, PageSurface, fit_height, unfold

R = 0.040   # a forearm's radius


def cylinder(xy):   # a forearm along page y, its crest along y = 0, falling off across x
    xy = np.atleast_2d(xy)
    return np.sqrt(R ** 2 - xy[:, 0] ** 2) - R


def test_a_flat_surface_is_the_plane_point_for_point():
    base = np.eye(4)
    base[:3, 3] = [0.3, 0.05, 0.004]
    s = PageSurface(base)
    uv = np.array([[0.0, 0.0], [0.012, -0.03], [-0.02, 0.04]])
    points, normals = s.lift(uv, 0.003)
    assert s.flat and np.allclose(points, (base @ np.c_[uv, np.full(3, 0.003), np.ones(3)].T).T[:, :3])
    assert np.allclose(normals, [0.0, 0.0, 1.0]) and np.allclose(s.chart.to_plane(uv), uv)
    gap, n = s.height_over(points[1])
    assert np.isclose(gap, 0.003) and np.allclose(n, [0, 0, 1])


def test_a_fit_recovers_a_forearms_curve_and_drops_a_stray_point():
    rng = np.random.default_rng(1)
    xy = rng.uniform([-0.02, -0.04], [0.02, 0.04], (800, 2))
    pts = np.c_[xy, cylinder(xy) + rng.normal(0, 0.0001, len(xy))]
    pts[5, 2] += 0.004
    field = fit_height(pts, degree=4)
    grid = np.stack(np.meshgrid(np.linspace(-0.018, 0.018, 9), np.linspace(-0.038, 0.038, 9)), -1).reshape(-1, 2)
    assert np.max(np.abs(field(grid) - cylinder(grid))) < 0.00015
    with pytest.raises(ValueError, match="too few"):
        fit_height(pts[:10])


def test_the_chart_unrolls_a_cylinder_to_its_arc_lengths():
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.045, 0.045, 31)), -1).reshape(-1, 2)
    field = fit_height(np.c_[xy, cylinder(xy)], degree=4)
    chart = unfold(field)
    for x in (0.01, 0.018, -0.015):
        u = chart.to_chart([[x, 0.02]])[0]
        assert abs(u[0] - R * np.arcsin(x / R)) < 0.0002 and abs(u[1] - 0.02) < 0.0002   # across: the arc
    back = chart.to_plane([[0.018, -0.03]])[0]
    assert abs(R * np.arcsin(back[0] / R) - 0.018) < 0.0002 and abs(back[1] + 0.03) < 0.0002


def test_a_curved_surface_lifts_print_points_along_their_own_normals():
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.045, 0.045, 31)), -1).reshape(-1, 2)
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)], degree=4))
    arc = 0.016   # 16 mm round the arm from the crest
    points, normals = s.lift([[arc, 0.01]], 0.003)
    angle = arc / R
    expect_n = np.array([np.sin(angle), 0.0, np.cos(angle)])
    expect_p = np.array([R * np.sin(angle), 0.01, R * np.cos(angle) - R]) + 0.003 * expect_n
    assert np.allclose(normals[0], expect_n, atol=0.003) and np.allclose(points[0], expect_p, atol=0.0002)
    gap, n = s.height_over(points[0])
    assert abs(gap - 0.003) < 0.0001 and np.allclose(n, expect_n, atol=0.003)
    assert np.allclose(s.moved(np.eye(4)).lift([[arc, 0.01]], 0.003)[0], points)
    again = PageSurface.from_dict(np.eye(4), s.as_dict())
    assert np.allclose(again.lift([[arc, 0.01]], 0.003)[0], points)
    assert HeightField().flat and not s.flat


def test_past_the_fields_extent_a_point_is_measured_from_the_plane():
    """2026-10-08: on a skin pinned over a holder the field's polynomial, carried out to the arm's rest, put the pen
    under the standoff and a resume tried to lift from the rest pose's joint limit."""
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.045, 0.045, 31)), -1).reshape(-1, 2)
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)], degree=4, extent=(-0.022, -0.045, 0.022, 0.045)))
    gap, n = s.height_over([0.03, -0.12, 0.10])
    assert gap == 0.10 and np.allclose(n, [0.0, 0.0, 1.0])
    inside = s.height_over([0.0, 0.0, 0.003])
    assert abs(inside[0] - 0.003) < 0.0001


def test_far_over_a_curved_skin_a_point_reads_its_height_over_the_nearest_skin():
    """2026-10-23: a tip 42 mm over a forearm's field ran the foot's fixed point off the fitted extent into the
    polynomial's overflow; the NaN height held a landing as one at the skin."""
    xy = np.stack(np.meshgrid(np.linspace(-0.020, 0.020, 21), np.linspace(-0.030, 0.030, 31)), -1).reshape(-1, 2)
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)], degree=4, extent=(-0.020, -0.030, 0.020, 0.030)))
    for x in (-0.019, -0.012, 0.0, 0.015):
        for z in (0.003, 0.020, 0.040, 0.060, 0.100):
            point = [x, 0.01, float(cylinder([x, 0.01])[0]) + z]
            true = np.hypot(point[0], point[2] + R) - R   # to the cylinder's axis, less its radius
            gap, n = s.height_over(point)
            assert np.isfinite(gap) and abs(gap - true) < 0.0003, (x, z, gap, true)
            assert np.isclose(np.linalg.norm(n), 1.0)
