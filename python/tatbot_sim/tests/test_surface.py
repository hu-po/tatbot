"""Invariants of the drawable surface: charts, displacement and the metric.

The surface is what every other layer trusts. The expert points the tool along
its normal, the deposit gate measures distance to it, and kernels are sized by
its metric — so an error here is silent everywhere else. These check the
geometry against closed forms rather than against itself:

  * a plane chart with no displacement must BE the old PlanarSurface, which is
    what lets that class retire;
  * a cylinder must be a cylinder (points one radius from the axis) and must
    be arc-length parameterised, so a 30 mm stroke is 30 mm of skin;
  * displacement must produce the first fundamental form the textbook does,
    including the term a curved chart contributes and a flat one does not;
  * projection must land back on the point it started from.

Catmull-Rom reproduces quadratics exactly, so a ramp and a paraboloid are
exact oracles for height, gradient, normal and metric — no tolerance fudging.

Torch + numpy only, no render device:

    cd python/tatbot_sim && uv run python tests/test_surface.py
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.surface import (
    CylinderChart,
    DisplacedSurface,
    PlanarSurface,
    PlaneChart,
    Surface,
    cylinder_amplitude_ceiling,
    random_height_field,
)
from tatbot_sim.textures import SHEET_H_M, SHEET_W_M
from transforms3d.euler import euler2mat

HR, HC = 41, 33  # displacement grid: coarse on purpose, the surface is smooth


def _identity_frame(b: int = 1, z: float = 0.03):
    center = torch.zeros(b, 3)
    center[:, 2] = z
    rot = torch.eye(3).expand(b, 3, 3).contiguous()
    return center, rot


def _grid_uv(b: int = 1):
    """Canvas coordinates of every displacement node, (B, HR, HC) each."""
    us = torch.linspace(-SHEET_W_M / 2, SHEET_W_M / 2, HC)
    vs = torch.linspace(-SHEET_H_M / 2, SHEET_H_M / 2, HR)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    return uu.expand(b, HR, HC).contiguous(), vv.expand(b, HR, HC).contiguous()


def _height_from(fn, b: int = 1):
    uu, vv = _grid_uv(b)
    return fn(uu, vv)


def _tilted_frames(b: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rots = np.stack([euler2mat(*rng.uniform(-0.09, 0.09, 3), "sxyz") for _ in range(b)])
    center = torch.as_tensor(rng.uniform(-0.05, 0.05, (b, 3)), dtype=torch.float32)
    return center, torch.as_tensor(rots, dtype=torch.float32)


# --- the surface base -------------------------------------------------------


def test_sheet_isotropy_guard_still_fires():
    """Kernels handle a stretched SURFACE; a stretched texel is a different bug."""
    try:
        Surface(width_m=0.2, height_m=0.2, cols=100, rows=200)
    except ValueError as exc:
        assert "anisotropic" in str(exc)
        return
    raise AssertionError("a rectangular texel must raise")


# --- charts -----------------------------------------------------------------


def test_plane_chart_frame_and_inverse_round_trip():
    center, rot = _tilted_frames(4, seed=1)
    chart = PlaneChart(center, rot)
    uv = torch.as_tensor(
        np.random.default_rng(2).uniform(-0.08, 0.08, (4, 2)), dtype=torch.float32
    )
    point, d_du, d_dv, n = chart.frame(uv)
    assert torch.allclose(chart.invert(point), uv, atol=1e-6)
    # orthonormal frame, normal is the pad normal
    assert torch.allclose((d_du * d_dv).sum(-1), torch.zeros(4), atol=1e-6)
    assert torch.allclose(d_du.norm(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(n, rot[:, :, 2], atol=1e-6)


def test_cylinder_points_lie_on_the_cylinder():
    center, rot = _identity_frame(1)
    r = torch.tensor([0.05])
    v = torch.linspace(-0.06, 0.06, 9)
    uv = torch.stack([torch.zeros(9), v], dim=-1)
    chart_b = CylinderChart(center.expand(9, 3), rot.expand(9, 3, 3), r.expand(9))
    point, _, _, n = chart_b.frame(uv)
    axis_point = center.expand(9, 3) - r[:, None] * rot[:, :, 2].expand(9, 3)
    radial = point - axis_point
    radial = radial - (radial * rot[:, :, 0].expand(9, 3)).sum(-1, keepdim=True) * rot[
        :, :, 0
    ].expand(9, 3)
    assert torch.allclose(radial.norm(dim=-1), r.expand(9), atol=1e-6)
    # the normal points straight out from the axis
    assert torch.allclose(n, radial / radial.norm(dim=-1, keepdim=True), atol=1e-5)


def test_cylinder_is_arc_length_parameterised():
    """A 30 mm stroke across the canvas must be 30 mm of skin, or every stroke
    the planner lays down is silently the wrong size on a limb."""
    b = 64
    center, rot = _identity_frame(b)
    chart = CylinderChart(center, rot, torch.full((b,), 0.04))
    v = torch.linspace(-0.05, 0.05, b)
    p0, _, _, _ = chart.frame(torch.stack([torch.zeros(b), v], -1))
    p1, _, _, _ = chart.frame(torch.stack([torch.zeros(b), v + 1e-4], -1))
    assert torch.allclose((p1 - p0).norm(dim=-1), torch.full((b,), 1e-4), atol=1e-8)


def test_cylinder_inverse_round_trips():
    b = 16
    center, rot = _tilted_frames(b, seed=5)
    chart = CylinderChart(center, rot, torch.full((b,), 0.045))
    rng = np.random.default_rng(6)
    uv = torch.as_tensor(rng.uniform(-0.05, 0.05, (b, 2)), dtype=torch.float32)
    point, _, _, _ = chart.frame(uv)
    assert torch.allclose(chart.invert(point), uv, atol=1e-5)


def test_cylinder_rejects_a_non_positive_radius():
    center, rot = _identity_frame(1)
    for bad in (0.0, -0.05):
        try:
            CylinderChart(center, rot, torch.tensor([bad]))
        except ValueError:
            continue
        raise AssertionError(f"radius {bad} must raise")


# --- displacement sampling --------------------------------------------------


def test_height_is_exact_on_the_nodes_it_was_built_from():
    h = _height_from(lambda u, v: 0.004 * torch.sin(30.0 * u) + 0.003 * torch.cos(20.0 * v))
    center, rot = _identity_frame(1)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    us = torch.linspace(-SHEET_W_M / 2, SHEET_W_M / 2, HC)
    vs = torch.linspace(-SHEET_H_M / 2, SHEET_H_M / 2, HR)
    for j in (5, 12, 20, 27):
        for i in (7, 18, 30):
            uv = torch.tensor([[us[j], vs[i]]])
            got, _, _ = surf.sample_height(uv)
            assert abs(float(got) - float(h[0, i, j])) < 1e-7, (i, j)


def test_height_gradient_is_exact_for_a_quadratic():
    """Catmull-Rom reproduces quadratics, so this is an oracle, not a tolerance."""
    c = 2.5
    h = _height_from(lambda u, v: c * (u**2 + v**2))
    center, rot = _identity_frame(1)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    rng = np.random.default_rng(7)
    for _ in range(20):
        u, v = rng.uniform(-0.06, 0.06, 2)
        uv = torch.tensor([[u, v]], dtype=torch.float32)
        got, du, dv = surf.sample_height(uv)
        assert abs(float(got) - c * (u**2 + v**2)) < 1e-6
        assert abs(float(du) - 2 * c * u) < 1e-4, (float(du), 2 * c * u)
        assert abs(float(dv) - 2 * c * v) < 1e-4, (float(dv), 2 * c * v)


def test_normals_are_continuous_across_a_cell_boundary():
    """Bilinear height would step the normal at every cell edge and the wrist
    would chase those steps; C1 interpolation is why this is smooth."""
    h = _height_from(lambda u, v: 0.006 * torch.sin(25.0 * u) * torch.cos(18.0 * v))
    center, rot = _identity_frame(1)
    du_cell = SHEET_W_M / (HC - 1)
    edge = -SHEET_W_M / 2 + 7 * du_cell  # exactly on a node, i.e. a cell boundary
    eps = du_cell * 1e-3
    uvs = torch.tensor([[edge - eps, 0.01], [edge + eps, 0.01]], dtype=torch.float32)
    surf2 = DisplacedSurface(PlaneChart(center.expand(2, 3), rot.expand(2, 3, 3)), h.expand(2, HR, HC))
    _, _, _, n = surf2.frame(uvs)
    angle = torch.arccos((n[0] * n[1]).sum().clamp(-1, 1))
    assert float(angle) < 1e-3, float(angle)


# --- the surface built on top -----------------------------------------------


def test_zero_displacement_reproduces_the_planar_surface():
    """The gate that lets PlanarSurface retire: same uv, same distance, same
    incidence, on tilted pads and arbitrary points."""
    b = 8
    center, rot = _tilted_frames(b, seed=11)
    flat = PlanarSurface(center, rot)
    curved = DisplacedSurface(PlaneChart(center, rot), torch.zeros(b, HR, HC))
    rng = np.random.default_rng(12)
    pts = torch.as_tensor(rng.uniform(-0.08, 0.08, (b, 3)), dtype=torch.float32)
    axis = torch.as_tensor(rng.normal(size=(b, 3)), dtype=torch.float32)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    for a in (None, axis):
        uv_f, d_f, i_f = flat.project(pts, a)
        uv_c, d_c, i_c = curved.project(pts, a)
        assert torch.allclose(uv_f, uv_c, atol=1e-6), (uv_f, uv_c)
        assert torch.allclose(d_f, d_c, atol=1e-6), (d_f, d_c)
        assert torch.allclose(i_f, i_c, atol=1e-6)
    assert curved.unconverged == 0
    eye = torch.eye(2).expand(b, 2, 2)
    assert torch.allclose(curved.first_fundamental_form(torch.zeros(b, 2)), eye, atol=1e-6)


def test_ramp_normal_and_metric_match_the_closed_form():
    a = 0.35  # 19 degrees of slope
    h = _height_from(lambda u, v: a * u)
    b = 1
    center, rot = _identity_frame(b, z=0.0)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    uv = torch.tensor([[0.01, -0.02]])
    point, s_du, s_dv, n = surf.frame(uv)
    want_n = torch.tensor([[-a, 0.0, 1.0]]) / np.sqrt(1 + a**2)
    assert torch.allclose(n, want_n, atol=1e-5), (n, want_n)
    assert torch.allclose(point, torch.tensor([[0.01, -0.02, a * 0.01]]), atol=1e-6)
    m = surf.first_fundamental_form(uv)[0]
    assert abs(float(m[0, 0]) - (1 + a**2)) < 1e-4
    assert abs(float(m[1, 1]) - 1.0) < 1e-4
    assert abs(float(m[0, 1])) < 1e-4


def test_slope_stretches_texel_space_along_the_gradient_only():
    """The reason kernels must be elliptical: a texel spans more world distance
    along the slope than across it, by exactly 1/cos(theta)."""
    for deg in (15.0, 30.0, 45.0):
        a = float(np.tan(np.radians(deg)))
        h = _height_from(lambda u, v, a=a: a * u)
        center, rot = _identity_frame(1, z=0.0)
        surf = DisplacedSurface(PlaneChart(center, rot), h)
        m = surf.first_fundamental_form(torch.tensor([[0.0, 0.0]]))[0]
        along = float(torch.sqrt(m[0, 0]))
        across = float(torch.sqrt(m[1, 1]))
        assert abs(along - 1.0 / np.cos(np.radians(deg))) < 1e-4, (deg, along)
        assert abs(across - 1.0) < 1e-4


def test_a_curved_chart_contributes_its_own_stretch():
    """Displacement on a cylinder sweeps more arc than the base does: G is
    (1 + h/R)^2, not 1. Dropping the chart's normal derivative silently loses
    this and every kernel on a limb comes out the wrong size."""
    r, h0 = 0.05, 0.005
    h = _height_from(lambda u, v: torch.full_like(u, h0))
    center, rot = _identity_frame(1)
    surf = DisplacedSurface(CylinderChart(center, rot, torch.tensor([r])), h)
    m = surf.first_fundamental_form(torch.tensor([[0.0, 0.01]]))[0]
    assert abs(float(m[1, 1]) - (1 + h0 / r) ** 2) < 1e-4, float(m[1, 1])
    assert abs(float(m[0, 0]) - 1.0) < 1e-4
    # and with no displacement the cylinder is isometric
    flat = DisplacedSurface(CylinderChart(center, rot, torch.tensor([r])), torch.zeros(1, HR, HC))
    m0 = flat.first_fundamental_form(torch.tensor([[0.0, 0.01]]))[0]
    assert torch.allclose(m0, torch.eye(2), atol=1e-5)


def test_projection_recovers_a_point_placed_along_the_normal():
    """Walk off the surface by a known distance along the known normal; the
    projection must walk back to where it started."""
    c = 2.5
    b = 12
    h = _height_from(lambda u, v: c * (u**2 + v**2), b)
    center, rot = _tilted_frames(b, seed=21)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    rng = np.random.default_rng(22)
    uv = torch.as_tensor(rng.uniform(-0.05, 0.05, (b, 2)), dtype=torch.float32)
    point, _, _, n = surf.frame(uv)
    dist = torch.as_tensor(rng.uniform(-0.004, 0.02, b), dtype=torch.float32)
    pts = point + dist.unsqueeze(-1) * n
    got_uv, got_dist, _ = surf.project(pts)
    assert surf.unconverged == 0
    assert torch.allclose(got_dist, dist, atol=2e-5), (got_dist, dist)
    assert torch.allclose(got_uv, uv, atol=2e-5)


def test_incidence_is_measured_against_the_local_normal():
    """On a slope, a vertically held tool is already oblique — which is the
    whole reason a curved surface needs the tool to follow the normal."""
    a = float(np.tan(np.radians(40.0)))
    h = _height_from(lambda u, v: a * u)
    center, rot = _identity_frame(1, z=0.0)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    uv = torch.tensor([[0.01, 0.0]])
    point, _, _, n = surf.frame(uv)
    _, _, inc_normal = surf.project(point + 0.001 * n, n)
    up = torch.tensor([[0.0, 0.0, 1.0]])
    _, _, inc_vertical = surf.project(point + 0.001 * n, up)
    assert abs(float(inc_normal) - 1.0) < 1e-5
    assert abs(float(inc_vertical) - np.cos(np.radians(40.0))) < 1e-4


def test_a_point_the_iteration_cannot_place_is_infinitely_far_away():
    """A bad uv would deposit ink somewhere nobody asked for, so an unplaced
    point must fail the deposit gate rather than land plausibly."""
    c = 2.5
    h = _height_from(lambda u, v: c * (u**2 + v**2))
    center, rot = _identity_frame(1, z=0.0)
    # iters=0 leaves the chart's guess, which on a displaced surface is wrong
    surf = DisplacedSurface(PlaneChart(center, rot), h, iters=0)
    pts = torch.tensor([[0.05, 0.05, 0.0]])
    _, dist, _ = surf.project(pts)
    assert bool(torch.isinf(dist).all())
    assert surf.unconverged == 1
    # the deposit gate the env uses must reject it
    assert not bool((dist < 0.0055).any())


def test_height_grid_shape_is_checked():
    center, rot = _identity_frame(1)
    for bad in (torch.zeros(1, 1, 8), torch.zeros(HR, HC)):
        try:
            DisplacedSurface(PlaneChart(center, rot), bad)
        except ValueError:
            continue
        raise AssertionError(f"height {tuple(bad.shape)} must raise")


# --- the generator ----------------------------------------------------------


GEN_HR, GEN_HC = 84, 65


def _generated(b=6, feature=0.05, slope=0.25, amp=0.010, seed=3):
    rng = np.random.default_rng(seed)
    return random_height_field(
        rng, b, GEN_HR, GEN_HC,
        feature_m=np.full(b, feature),
        max_slope_rad=np.full(b, slope),
        amplitude_m=np.full(b, amp),
    )


def test_generated_surfaces_respect_the_slope_they_were_asked_for():
    """Slope is the controlled quantity because it is what costs reach, tilts
    the tool and stretches texel space. Measured through the real bicubic
    surface, not on the grid the generator scaled."""
    b, slope = 6, 0.25
    h = _generated(b=b, slope=slope)
    center, rot = _identity_frame(b)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(300):
        uv = torch.as_tensor(
            np.stack([rng.uniform(-0.09, 0.09, b), rng.uniform(-0.12, 0.12, b)], -1),
            dtype=torch.float32,
        )
        _, du, dv = surf.sample_height(uv)
        worst = max(worst, float(torch.hypot(du, dv).max()))
    assert worst <= np.tan(slope) * 1.15, (worst, np.tan(slope))
    assert worst > np.tan(slope) * 0.4, "asked for slope but got a nearly flat field"


def test_a_generated_skin_rests_on_its_support_and_only_bulges_up():
    """A skin lying on a pad cannot sink into it. Centring the field put half
    the surface below the substrate's top face, where the body underneath poked
    through the sheet and hid the ruling and the ink, and where the flat
    workspace floor clamped motion that was legal."""
    h = _generated(b=6)
    assert float(h.min()) >= 0.0, float(h.min())
    assert abs(float(h.min())) < 1e-9  # it RESTS on zero, not floating above it
    assert float(h.max()) > 0.0


def test_generated_surfaces_respect_the_amplitude_cap():
    b, amp = 6, 0.008
    h = _generated(b=b, amp=amp, slope=1.0)  # slope so loose that amplitude binds
    assert float(h.abs().max()) <= amp * 1.001, float(h.abs().max())
    assert float(h.abs().max()) > amp * 0.5


def test_a_flat_field_is_asked_for_by_asking_for_zero():
    """Amplitude zero is the planar regression, and it must be exactly flat."""
    h = _generated(b=2, amp=0.0)
    assert float(h.abs().max()) == 0.0


def test_generation_is_reproducible_from_its_seed():
    assert torch.equal(_generated(seed=11), _generated(seed=11))
    assert not torch.equal(_generated(seed=11), _generated(seed=12))


def test_generated_surfaces_stay_inside_the_kernel_window():
    """A plane chart can only STRETCH texel space, never compress it, so a
    kernel never needs a bigger window than the flat case — the property that
    lets the field size its margin once."""
    b = 6
    h = _generated(b=b, slope=0.45)
    center, rot = _identity_frame(b)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    rng = np.random.default_rng(13)
    for _ in range(50):
        uv = torch.as_tensor(
            np.stack([rng.uniform(-0.09, 0.09, b), rng.uniform(-0.12, 0.12, b)], -1),
            dtype=torch.float32,
        )
        m = surf.first_fundamental_form(uv)
        e, f, g = m[:, 0, 0], m[:, 0, 1], m[:, 1, 1]
        lam_min = 0.5 * ((e + g) - torch.sqrt((e - g) ** 2 + 4.0 * f**2))
        assert float(lam_min.min()) >= 1.0 - 1e-4, float(lam_min.min())


def test_the_cylinder_ceiling_keeps_compression_bounded():
    """Displaced inward on a limb, texel space compresses by (1 + h/R); the
    ceiling is what keeps that inside the margin the field allocated."""
    b = 4
    radius = np.full(b, 0.05)
    amp = cylinder_amplitude_ceiling(radius)
    h = _generated(b=b, amp=float(amp[0]), slope=1.0)
    center, rot = _identity_frame(b)
    surf = DisplacedSurface(
        CylinderChart(center, rot, torch.as_tensor(radius, dtype=torch.float32)), h
    )
    m = surf.first_fundamental_form(torch.zeros(b, 2))
    e, f, g = m[:, 0, 0], m[:, 0, 1], m[:, 1, 1]
    lam_min = 0.5 * ((e + g) - torch.sqrt((e - g) ** 2 + 4.0 * f**2))
    stretch = float(torch.rsqrt(lam_min).max())
    assert stretch < 1.5, stretch  # InkField's default max_stretch


# --- the mesh the cameras see -----------------------------------------------


def test_the_rendered_mesh_is_the_surface_the_ink_model_uses():
    """The picture and the model must be the same shape. The mesh is cut by
    evaluating the Surface itself, so this checks the writer round-trips those
    vertices rather than that two separate shape formulas happen to agree."""
    import tempfile
    from pathlib import Path

    from tatbot_sim.textures import write_surface_mesh

    rows, cols = 21, 17
    h = random_height_field(
        np.random.default_rng(2), 1, rows, cols,
        feature_m=np.full(1, 0.06), max_slope_rad=np.full(1, 0.25),
        amplitude_m=np.full(1, 0.008),
    )
    center, rot = _identity_frame(1, z=0.0)
    surf = DisplacedSurface(PlaneChart(center, rot), h)
    us = torch.linspace(-SHEET_W_M / 2, SHEET_W_M / 2, cols)
    vs = torch.linspace(-SHEET_H_M / 2, SHEET_H_M / 2, rows)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    point, _, _, normal = surf.env_view(0, uv.shape[0]).frame(uv)

    with tempfile.TemporaryDirectory() as td:
        path = Path(
            write_surface_mesh(
                Path(td) / "skin", "grid_00",
                point.numpy(), normal.numpy(), rows, cols,
            )
        )
        lines = path.read_text().splitlines()
    verts = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("v ")])
    norms = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("vn ")])
    uvs = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("vt ")])
    faces = [ln for ln in lines if ln.startswith("f ")]

    assert len(verts) == rows * cols
    assert len(faces) == 2 * (rows - 1) * (cols - 1)
    assert np.allclose(verts, point.numpy(), atol=1e-6)
    assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)
    # the ruling has to land where it always did: the flat quad's UV mapping,
    # v = 1 at y = -H/2, unchanged by the displacement
    assert np.allclose(uvs[0], [0.0, 1.0], atol=1e-6)
    assert np.allclose(uvs[cols - 1], [1.0, 1.0], atol=1e-6)
    assert np.allclose(uvs[-1], [1.0, 0.0], atol=1e-6)
    # and the mesh really is displaced, or it proves nothing
    assert np.ptp(verts[:, 2]) > 0.002, np.ptp(verts[:, 2])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
