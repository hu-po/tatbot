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
    field = fit_height(pts)
    grid = np.stack(np.meshgrid(np.linspace(-0.018, 0.018, 9), np.linspace(-0.038, 0.038, 9)), -1).reshape(-1, 2)
    assert np.max(np.abs(field(grid) - cylinder(grid))) < 0.00015
    with pytest.raises(ValueError, match="too few"):
        fit_height(pts[:10])


def bends(xy):   # no one shape: a crest that rolls into a shelf on one side, a trough and a second rise on the other
    x, y = np.atleast_2d(xy).T
    return 0.006 * np.exp(-(x / 0.012) ** 2) - 0.003 * np.exp(-((x - 0.022 - 0.1 * y) / 0.007) ** 2) + 0.03 * y * x


def test_a_fit_follows_any_bend_its_points_show():
    """What one polynomial over the whole skin could not: tight bends of both signs side by side (a printed spline
    mount's, a wrist's), met within hundredths of a millimetre, the normals with them."""
    xy = np.random.default_rng(2).uniform(-0.04, 0.04, (6000, 2))
    field = fit_height(np.c_[xy, bends(xy)])
    grid = np.stack(np.meshgrid(np.linspace(-0.036, 0.036, 37), np.linspace(-0.036, 0.036, 37)), -1).reshape(-1, 2)
    assert np.abs(field(grid) - bends(grid)).max() < 3e-5
    step = 1e-6
    slope = np.stack([(bends(grid + [step, 0]) - bends(grid - [step, 0])) / (2 * step),
                      (bends(grid + [0, step]) - bends(grid - [0, step])) / (2 * step)], axis=1)
    assert np.abs(field.gradient(grid) - slope).max() < 0.015   # 0.9 degrees, at the edge of what was measured


def test_a_fit_is_as_stiff_as_its_points_noise_and_no_stiffer_than_the_skin():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-0.04, 0.04, (6000, 2))
    grid = np.stack(np.meshgrid(np.linspace(-0.036, 0.036, 37), np.linspace(-0.036, 0.036, 37)), -1).reshape(-1, 2)
    clean = fit_height(np.c_[xy, bends(xy)])
    noisy = fit_height(np.c_[xy, bends(xy) + rng.normal(0, 0.0003, len(xy))])
    assert noisy.stiffness > clean.stiffness
    assert np.sqrt(np.mean((noisy(grid) - bends(grid)) ** 2)) < 0.0001   # a third of one point's noise
    # holds that each sit their own height off (the arm's error at their pose) are not chased step by step: held
    # out hold by hold, the field through four quadrants 0.2 mm apart stays smoother than through any one of them
    hold = (xy[:, 0] > 0).astype(int) + 2 * (xy[:, 1] > 0)
    stepped = np.c_[xy, cylinder(xy) + np.array([0.0001, -0.0001, -0.0001, 0.0001])[hold]]
    assert fit_height(stepped, groups=hold).stiffness > fit_height(stepped).stiffness
    assert np.abs(fit_height(stepped, groups=hold)(grid) - cylinder(grid)).max() < 0.0003


def test_where_nothing_was_measured_the_field_runs_on_as_a_plane():
    xy = np.random.default_rng(4).uniform([-0.04, -0.04], [0.0, 0.04], (3000, 2))   # the left half only
    field = fit_height(np.c_[xy, cylinder(xy)], extent=(-0.04, -0.04, 0.04, 0.04))
    right = np.stack(np.meshgrid(np.linspace(0.012, 0.04, 15), np.linspace(-0.03, 0.03, 13)), -1).reshape(-1, 2)
    assert np.ptp(field.gradient(right), axis=0).max() < 0.05 and np.abs(field(right)).max() < 0.01
    past = field([[0.06, 0.0], [0.08, 0.0], [0.10, 0.0]])   # and past its extent, along the plane it left on
    assert np.isclose(past[2] - past[1], past[1] - past[0]) and np.allclose(field.gradient([[0.10, 0.0]]),
                                                                            field.gradient([[0.04, 0.0]]))


def test_a_field_is_kept_and_read_back_and_an_older_pages_polynomial_is_read_as_the_same_skin():
    xy = np.random.default_rng(5).uniform(-0.03, 0.03, (2000, 2))
    field = fit_height(np.c_[xy, bends(xy)])
    back = HeightField.from_dict(field.as_dict())
    assert np.array_equal(back(xy), field(xy)) and back.stiffness == field.stiffness and back.extent == field.extent
    assert HeightField().flat and not field.flat and HeightField.from_dict(HeightField().as_dict()).flat
    # a page record written before the spline: h = c . (x/s)^i (y/s)^j, constant and linear terms first
    old = {"degree": 2, "coeffs": [0.001, 0.002, -0.001, -0.004, 0.0005, -0.003], "scale": 0.03,
           "extent": [-0.03, -0.04, 0.03, 0.04]}
    x, y = (xy / 0.03).T
    was = 0.001 + 0.002 * x - 0.001 * y - 0.004 * x * x + 0.0005 * x * y - 0.003 * y * y
    assert np.abs(HeightField.from_dict(old)(xy) - was).max() < 1e-6


def test_the_chart_unrolls_a_cylinder_to_its_arc_lengths():
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.045, 0.045, 31)), -1).reshape(-1, 2)
    field = fit_height(np.c_[xy, cylinder(xy)])
    chart = unfold(field)
    for x in (0.01, 0.018, -0.015):
        u = chart.to_chart([[x, 0.02]])[0]
        assert abs(u[0] - R * np.arcsin(x / R)) < 0.0002 and abs(u[1] - 0.02) < 0.0002   # across: the arc
    back = chart.to_plane([[0.018, -0.03]])[0]
    assert abs(R * np.arcsin(back[0] / R) - 0.018) < 0.0002 and abs(back[1] + 0.03) < 0.0002


def test_a_curved_surface_lifts_print_points_along_their_own_normals():
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.045, 0.045, 31)), -1).reshape(-1, 2)
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)]))
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
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)], extent=(-0.022, -0.045, 0.022, 0.045)))
    gap, n = s.height_over([0.03, -0.12, 0.10])
    assert gap == 0.10 and np.allclose(n, [0.0, 0.0, 1.0])
    inside = s.height_over([0.0, 0.0, 0.003])
    assert abs(inside[0] - 0.003) < 0.0001


def test_far_over_a_curved_skin_a_point_reads_its_height_over_the_nearest_skin():
    """2026-10-23: a tip 42 mm over a forearm's field ran the foot's fixed point off the fitted extent into the
    polynomial's overflow; the NaN height held a landing as one at the skin."""
    xy = np.stack(np.meshgrid(np.linspace(-0.020, 0.020, 21), np.linspace(-0.030, 0.030, 31)), -1).reshape(-1, 2)
    s = PageSurface(np.eye(4), fit_height(np.c_[xy, cylinder(xy)], extent=(-0.020, -0.030, 0.020, 0.030)))
    for x in (-0.019, -0.012, 0.0, 0.015):
        for z in (0.003, 0.020, 0.040, 0.060, 0.100):
            point = [x, 0.01, float(cylinder([x, 0.01])[0]) + z]
            true = np.hypot(point[0], point[2] + R) - R   # to the cylinder's axis, less its radius
            gap, n = s.height_over(point)
            assert np.isfinite(gap) and abs(gap - true) < 0.0003, (x, z, gap, true)
            assert np.isclose(np.linalg.norm(n), 1.0)


def test_many_points_heights_found_together_are_each_points_own():
    """Near the skin, far over it (the fixed point runs away), past the field's extent, and over a flat page."""
    rng = np.random.default_rng(2)
    xy = rng.uniform([-0.02, -0.04], [0.02, 0.04], (800, 2))
    field = fit_height(np.c_[xy, cylinder(xy)], extent=(-0.02, -0.04, 0.02, 0.04))
    base = np.eye(4)
    base[:3, 3] = [0.3, 0.05, 0.004]
    points = np.c_[rng.uniform([-0.03, -0.05], [0.03, 0.05], (60, 2)), rng.uniform(-0.002, 0.06, 60)] + base[:3, 3]
    for s in (PageSurface(base, field), PageSurface(base)):
        assert np.allclose(s.heights_over(points), [s.height_over(p)[0] for p in points], rtol=0, atol=1e-12)
