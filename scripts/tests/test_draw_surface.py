"""Pin scripts/lib/draw_surface.py: the numpy surface the drawing stages trust.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_draw_surface.py

Closed-form oracles wherever one exists (Catmull-Rom by hand, an undisplaced
plane IS its chart, a cylinder point is one radius off its axis, arc length
is arc length), synthetic noisy clouds for the fit/fuse path, and an exact
npz round trip so what the mapper writes the planner reads unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import draw_surface as ds  # noqa: E402


def _rot(normal, u_hint):
    """Rotation with columns (e_u, e_v, n) from a normal and a u direction."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    eu = np.asarray(u_hint, float)
    eu = eu - (eu @ n) * n
    eu /= np.linalg.norm(eu)
    return np.stack([eu, np.cross(n, eu), n], axis=1)


def _deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _bump_surface(chart, w=0.10, h=0.12, rows=25, cols=21, peak=0.004):
    us = np.linspace(-w / 2, w / 2, cols)
    vs = np.linspace(-h / 2, h / 2, rows)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    r = np.hypot(uu / 0.035, vv / 0.045)
    height = np.where(r < 1, 0.5 * peak * (1 + np.cos(np.pi * np.clip(r, 0, 1))), 0.0)
    return ds.HeightFieldSurface(chart, height, w, h)


# --------------------------------------------------------------------------- closed forms


def test_catmull_rom_matches_hand_computation():
    p = np.array([0.0, 1.0, 3.0, 2.0])
    val, der = ds.catmull_rom(p, 0.5)
    # a=2, b=3, c=5, d=-4 -> 0.5*(2 + 1.5 + 1.25 - 0.5), 0.5*(3 + 5 - 3)
    assert val == pytest.approx(2.125)
    assert der == pytest.approx(2.5)
    # endpoints interpolate p1/p2, and the derivative matches a finite difference
    assert ds.catmull_rom(p, 0.0)[0] == pytest.approx(1.0)
    assert ds.catmull_rom(p, 1.0)[0] == pytest.approx(3.0)
    eps = 1e-6
    fd = (ds.catmull_rom(p, 0.3 + eps)[0] - ds.catmull_rom(p, 0.3 - eps)[0]) / (2 * eps)
    assert ds.catmull_rom(p, 0.3)[1] == pytest.approx(fd, abs=1e-8)
    # batched: (..., 4) with t broadcasting on the leading axis
    pp = np.stack([p, 2 * p], axis=0)
    v2, _ = ds.catmull_rom(pp, np.array([0.5, 0.5]))
    assert np.allclose(v2, [2.125, 4.25])


def test_plane_surface_with_zero_height_is_its_chart():
    chart = ds.PlaneChart([0.1, -0.2, 0.3], _rot([0.2, -0.1, 1.0], [1, 0, 0]))
    surf = ds.HeightFieldSurface(chart, np.zeros((5, 7)), 0.1, 0.14)
    uv = np.array([[0.0, 0.0], [0.03, -0.05], [-0.05, 0.07], [0.02, 0.01]])
    p_c, du_c, dv_c, n_c = chart.frame(uv)
    p_s, du_s, dv_s, n_s = surf.frame(uv)
    assert np.allclose(p_s, p_c, atol=1e-12)
    assert np.allclose(du_s, du_c, atol=1e-12)
    assert np.allclose(dv_s, dv_c, atol=1e-12)
    assert np.allclose(n_s, n_c, atol=1e-12)
    assert np.allclose(surf.first_fundamental_form(uv), np.eye(2), atol=1e-12)


def test_cylinder_frame_normal_is_unit_and_outward():
    chart = ds.CylinderChart([0.0, 0.1, 0.2], _rot([0, 0.3, 1.0], [1, 0.2, 0]), 0.04)
    uv = np.array([[0.0, 0.0], [0.02, 0.03], [-0.03, -0.05], [0.01, 0.06]])
    point, d_du, d_dv, n = chart.frame(uv)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)
    rel = point - chart.axis_point
    rel_perp = rel - (rel @ chart.rot[:, 0])[:, None] * chart.rot[:, 0]
    assert np.allclose(np.linalg.norm(rel_perp, axis=-1), 0.04, atol=1e-12)
    assert np.all((n * rel_perp).sum(-1) > 0)
    assert np.allclose(np.cross(d_du, d_dv), n, atol=1e-12)
    assert np.allclose(chart.invert(point), uv, atol=1e-12)
    assert np.allclose(chart.frame([[0.0, 0.0]])[0][0], chart.center)


def test_cylinder_chart_is_isometric():
    chart = ds.CylinderChart([0.0, 0.0, 0.0], np.eye(3), 0.04)
    surf = ds.HeightFieldSurface(chart, np.zeros((4, 4)), 0.1, 0.16)
    a, b = np.array([-0.02, -0.05]), np.array([0.03, 0.06])
    ts = np.linspace(0.0, 1.0, 4001)
    uv = a[None] + ts[:, None] * (b - a)[None]
    pts, _, _, _ = surf.frame(uv)
    poly = float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum())
    chord = float(np.linalg.norm(b - a))  # geodesic length on a cylinder = uv distance
    assert abs(poly - chord) < 1e-7  # faceting error ~ (dv^2 / R) per segment
    # and a v-only step is pure arc length
    pts_v, _, _, _ = surf.frame(np.stack([np.zeros_like(ts), -0.05 + ts * 0.11], -1))
    assert abs(np.linalg.norm(np.diff(pts_v, axis=0), axis=-1).sum() - 0.11) < 1e-7


# --------------------------------------------------------------------------- fit + fuse


U_HINT = [1.0, 0.1, 0.0]


def _plane_cloud(rng, n=30000, noise=1e-4, hole=None):
    rot = _rot([0.15, -0.25, 1.0], U_HINT)
    center = np.array([0.30, -0.10, 0.05])
    u = rng.uniform(-0.05, 0.05, n)
    v = rng.uniform(-0.06, 0.06, n)
    if hole is not None:
        cu, cv, rad = hole
        keep = np.hypot(u - cu, v - cv) > rad
        u, v = u[keep], v[keep]
    pts = center + u[:, None] * rot[:, 0] + v[:, None] * rot[:, 1]
    pts = pts + rng.normal(0.0, noise, (u.size, 1)) * rot[:, 2]
    return pts, center, rot


def test_fuse_recovers_a_tilted_plane():
    rng = np.random.default_rng(0)
    pts, center, rot = _plane_cloud(rng)
    chart = ds.fit_plane(pts, normal_hint=[0, 0, 1], u_hint=U_HINT)
    assert _deg(chart.normal, rot[:, 2]) < 0.1
    assert abs((chart.center - center) @ rot[:, 2]) < 2e-5
    surf = ds.fuse(pts, chart, 0.10, 0.12, cell_m=0.005)
    assert surf.height.shape == (25, 21)
    assert (surf.count > 0).all()
    us, vs = surf.grid_uv()
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    uv = np.stack([uu.ravel(), vv.ravel()], -1)
    p, _, _, n = surf.frame(uv)
    dist = (p - center) @ rot[:, 2]
    assert float(np.sqrt(np.mean(dist**2))) < 1e-4  # height rms < 0.1 mm
    mean_n = n.mean(axis=0)
    assert _deg(mean_n, rot[:, 2]) < 0.1
    assert max(_deg(ni, rot[:, 2]) for ni in n) < 1.0
    assert np.all(surf.residual_m[surf.count > 0] < 3e-4)


def _cylinder_cloud(rng, radius=0.04, n=40000, noise=3e-4):
    axis = np.array([1.0, 0.3, 0.1])
    axis /= np.linalg.norm(axis)
    rot = _rot([-0.1, 0.2, 1.0], axis)  # columns: axis, crest tangent, crest normal
    crest = np.array([0.25, 0.05, 0.10])
    u = rng.uniform(-0.04, 0.04, n)
    th = rng.uniform(-1.0, 1.0, n)  # +-57 deg of arc
    r = radius + rng.normal(0.0, noise, n)
    axis_pt = crest - radius * rot[:, 2]
    pts = (
        axis_pt
        + u[:, None] * rot[:, 0]
        + (r * np.sin(th))[:, None] * rot[:, 1]
        + (r * np.cos(th))[:, None] * rot[:, 2]
    )
    return pts, axis_pt, rot, radius


def test_fit_and_fuse_recover_a_cylinder():
    rng = np.random.default_rng(1)
    pts, axis_pt, rot, radius = _cylinder_cloud(rng)
    chart, rms = ds.fit_cylinder(pts, axis_hint=[1, 0, 0], normal_hint=[0, 0, 1])
    assert abs(chart.radius - radius) < 5e-4
    assert _deg(chart.rot[:, 0], rot[:, 0]) < 1.0
    assert rms < 4e-4
    rel = chart.axis_point - axis_pt
    assert np.linalg.norm(rel - (rel @ rot[:, 0]) * rot[:, 0]) < 5e-4
    assert _deg(chart.normal, rot[:, 2]) < 1.0

    surf = ds.fuse(pts, chart, 0.08, 0.075, cell_m=0.0025)
    assert (surf.count > 0).mean() > 0.95
    us, vs = surf.grid_uv()
    vv, uu = np.meshgrid(vs[2:-2], us[2:-2], indexing="ij")
    p, _, _, _ = surf.frame(np.stack([uu.ravel(), vv.ravel()], -1))
    rel = p - axis_pt
    rho = np.linalg.norm(rel - (rel @ rot[:, 0])[:, None] * rot[:, 0], axis=-1)
    assert float(np.sqrt(np.mean((rho - radius) ** 2))) < 5e-4
    assert abs(float(np.mean(rho)) - radius) < 2e-4


def test_holes_keep_zero_count_but_finite_height():
    rng = np.random.default_rng(2)
    pts, center, rot = _plane_cloud(rng, hole=(0.01, -0.01, 0.015))
    chart = ds.fit_plane(pts, [0, 0, 1], U_HINT)
    surf = ds.fuse(pts, chart, 0.10, 0.12, cell_m=0.005)
    us, vs = surf.grid_uv()
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    inside = np.hypot(uu - 0.01, vv + 0.01) < 0.012
    assert inside.sum() > 5
    assert (surf.count[inside] == 0).all()
    assert (surf.count[~inside & (np.hypot(uu - 0.01, vv + 0.01) > 0.02)] > 0).all()
    assert np.isfinite(surf.height).all()
    assert np.abs(surf.height[inside]).max() < 1e-4  # infilled from the flat surround
    assert (surf.residual_m[surf.count == 0] == 0).all()


# --------------------------------------------------------------------------- projection


def test_project_inverts_frame():
    chart = ds.CylinderChart([0.2, 0.0, 0.1], _rot([0.1, 0.2, 1.0], [1, 0, 0]), 0.05)
    surf = _bump_surface(chart)
    rng = np.random.default_rng(3)
    uv = np.stack([rng.uniform(-0.045, 0.045, 200), rng.uniform(-0.055, 0.055, 200)], -1)
    p, _, _, n = surf.frame(uv)
    uv2, dist = surf.project(p)
    assert surf.unconverged == 0
    assert np.abs(uv2 - uv).max() < 1e-6
    assert np.abs(dist).max() < 1e-9
    # a point lifted 3 mm along the local normal projects back with that distance
    uv3, dist3 = surf.project(p + 0.003 * n)
    assert np.abs(uv3 - uv).max() < 1e-6
    assert np.allclose(dist3, 0.003, atol=1e-8)
    uv4, dist4 = surf.project(p - 0.002 * n)
    assert np.allclose(dist4, -0.002, atol=1e-8)
    assert np.abs(uv4 - uv).max() < 1e-6


def test_anchor_to_zeroes_the_distance_at_the_anchor():
    chart = ds.PlaneChart([0.3, 0.0, 0.05], _rot([0.1, -0.1, 1.0], [1, 0, 0]))
    surf = _bump_surface(chart)
    p0, _, _, n0 = surf.frame([[0.012, -0.02]])
    target = p0[0] + 0.0013 * n0[0] + np.array([0.0, 0.0, 0.0])
    anchored, shift, uv = ds.HeightFieldSurface.anchor_to(surf, target)
    _, dist = anchored.project(target[None])
    assert abs(float(dist[0])) < 1e-9
    assert anchored.anchor_shift_m == pytest.approx(shift)
    assert abs(shift - 0.0013) < 1e-4  # the bump slope makes the chart-normal shift slightly larger
    assert np.allclose(anchored.anchor_point, target)
    assert np.allclose(anchored.anchor_uv, uv)
    assert np.allclose(anchored.height - surf.height, shift)
    # the original is untouched, and count/residual travel with the shift
    assert np.isnan(surf.anchor_uv).all()
    assert surf.anchor_shift_m == 0.0
    assert anchored.count.shape == surf.count.shape


# --------------------------------------------------------------------------- io + selection


def test_npz_round_trip_is_exact(tmp_path):
    chart = ds.CylinderChart([0.2, 0.0, 0.1], _rot([0.1, 0.2, 1.0], [1, 0, 0]), 0.05)
    surf = _bump_surface(chart)
    rng = np.random.default_rng(4)
    surf.count = rng.integers(0, 50, surf.height.shape).astype(np.int32)
    surf.residual_m = rng.uniform(0, 1e-4, surf.height.shape)
    surf, _, _ = surf.anchor_to(surf.frame([[0.01, 0.02]])[0][0] + [0.0, 0.0, 0.001])
    path = surf.to_npz(tmp_path / "surface.npz", extra_json={"draw_dir": "D", "tool": "ballpoint"})
    back = ds.HeightFieldSurface.from_npz(path)
    assert back.chart.kind == "cylinder"
    assert back.chart.radius == chart.radius
    assert np.array_equal(back.chart.center, chart.center)
    assert np.array_equal(back.chart.rot, chart.rot)
    assert np.array_equal(back.height, surf.height)
    assert np.array_equal(back.count, surf.count) and back.count.dtype == np.int32
    assert np.array_equal(back.residual_m, surf.residual_m)
    assert np.array_equal(back.anchor_uv, surf.anchor_uv)
    assert np.array_equal(back.anchor_point, surf.anchor_point)
    assert back.anchor_shift_m == surf.anchor_shift_m
    assert (back.width_m, back.height_m) == (surf.width_m, surf.height_m)
    with np.load(path) as d:
        assert str(d["schema"]) == "tatbot.surface/1"
        assert set(d.files) == {
            "schema", "chart_kind", "center", "rot", "radius_m", "width_m", "height_m",
            "height", "count", "residual_m", "anchor_uv", "anchor_point", "anchor_shift_m",
        }
    side = (tmp_path / "surface.json").read_text()
    assert '"draw_dir": "D"' in side and '"chart_kind": "cylinder"' in side

    plane = ds.HeightFieldSurface(ds.PlaneChart([0, 0, 0], np.eye(3)), np.zeros((3, 4)), 0.1, 0.1)
    plane.to_npz(tmp_path / "plane.npz")
    assert not (tmp_path / "plane.json").exists()
    back = ds.HeightFieldSurface.from_npz(tmp_path / "plane.npz")
    assert back.chart.kind == "plane" and np.isnan(back.anchor_uv).all() and back.anchor_shift_m == 0.0


def test_choose_chart_picks_plane_for_a_plane_and_cylinder_for_a_cylinder():
    rng = np.random.default_rng(5)
    pts, _, rot = _plane_cloud(rng, n=8000, noise=3e-4)
    chart, report = ds.choose_chart(pts, normal_hint=[0, 0, 1], u_hint=U_HINT)
    assert chart.kind == "plane" and report["chart_kind"] == "plane"
    assert report["plane_rms_m"] < 4e-4
    assert ds.choose_chart(pts, [0, 0, 1], U_HINT, prefer="plane")[0].kind == "plane"

    pts, _, rot, radius = _cylinder_cloud(rng, n=8000)
    chart, report = ds.choose_chart(pts, normal_hint=[0, 0, 1], u_hint=[1, 0, 0])
    assert chart.kind == "cylinder"
    assert abs(chart.radius - radius) < 5e-4
    assert report["cylinder_rms_m"] * 3 <= report["plane_rms_m"]
    assert ds.choose_chart(pts, [0, 0, 1], [1, 0, 0], prefer="plane")[0].kind == "plane"


def test_mesh_winding_faces_outward():
    chart = ds.CylinderChart([0.2, 0.0, 0.1], _rot([0.1, 0.2, 1.0], [1, 0, 0]), 0.05)
    surf = _bump_surface(chart)
    verts, faces, normals = ds.mesh(surf, 0.01)
    assert verts.shape[0] == normals.shape[0] == 11 * 13
    assert faces.shape == (2 * 10 * 12, 3)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all((fn * normals[faces[:, 0]]).sum(-1) > 0)


def test_choose_chart_finds_a_cylinder_whose_axis_is_perpendicular_to_the_u_hint():
    rng = np.random.default_rng(7)
    r = 0.045
    # axis along world y; u_hint along world x (perpendicular to the axis), like a pipe across the canvas
    y = rng.uniform(-0.03, 0.03, 4000)
    th = rng.uniform(-0.7, 0.7, 4000)
    pts = np.stack([r * np.sin(th), y, r * np.cos(th) - r], 1) + rng.normal(0, 0.0003, (4000, 3))
    chart, report = ds.choose_chart(pts, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    assert chart.kind == "cylinder", report
    assert abs(chart.radius - r) < 0.0005, report
    assert abs(abs(chart.rot[:, 0] @ np.array([0.0, 1.0, 0.0])) - 1.0) < 0.01, report


def test_offsetting_the_chart_keeps_a_cylinder_isometric_after_anchoring():
    rng = np.random.default_rng(3)
    r = 0.045
    y = rng.uniform(-0.03, 0.03, 6000)
    th = rng.uniform(-0.6, 0.6, 6000)
    pts = np.stack([r * np.sin(th), y, r * np.cos(th) - r], 1) + rng.normal(0, 0.0002, (6000, 3))
    chart, _ = ds.choose_chart(pts, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    assert chart.kind == "cylinder"
    fused = ds.fuse(pts, chart, 0.06, 0.06, 0.001, smooth_m=0.004)
    anchor = np.array([0.0, 0.0, 0.003])  # 3 mm outside the crest
    _, shift, _ = fused.anchor_to(anchor)
    assert abs(shift - 0.003) < 0.0004
    chart2 = chart.offset(shift)
    assert abs(chart2.radius - (chart.radius + shift)) < 1e-12
    fused2 = ds.fuse(pts, chart2, 0.06, 0.06, 0.001, smooth_m=0.004)
    # the cloud now sits 3 mm inside the offset chart; anchoring lifts the height back to ~0,
    # so the chart itself is the anchored surface and its metric is the surface's
    surface, second, _ = fused2.anchor_to(anchor)
    assert abs(second - shift) < 0.0004
    assert abs(np.median(surface.height[surface.count > 0])) < 0.0005
    # canvas v is arc length on the anchored surface: 20 mm of v is 20 mm of surface
    uv = np.array([[0.0, -0.01], [0.0, 0.01]])
    pts_on, _, _, _ = surface.frame(uv)
    theta = 0.02 / surface.chart.radius
    chord = 2.0 * surface.chart.radius * np.sin(theta / 2.0)
    assert abs(np.linalg.norm(pts_on[1] - pts_on[0]) - chord) < 0.0002
