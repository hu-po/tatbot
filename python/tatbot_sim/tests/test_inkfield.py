"""Invariants of the pigment field — the layer the removal tool depends on.

These are the guarantees the rest of the ink/laser design assumes and cannot
check for itself: that texel space agrees with the paper the sheet generator
bakes, that a kernel asked for in millimetres stays that size in millimetres,
and above all that removal actually inverts deposition rather than
approximately undoing it.

Torch + numpy only, no render device:

    cd python/tatbot_sim && uv run python tests/test_inkfield.py
    # or: uv run python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.inkfield import (
    InkField,
    kernel_half,
    laser_eta,
    resample_polyline,
    splat,
    stamp_kernels,
)
from tatbot_sim.strokes import Stroke
from tatbot_sim.surface import PlanarSurface
from tatbot_sim.textures import (
    SHEET_H_M,
    SHEET_W_M,
    SIZE_X,
    SIZE_Y,
    _rule_px,
    grid_paper_sheets,
)
from transforms3d.euler import euler2mat


def _flat_surface(b: int = 2, z: float = 0.03) -> PlanarSurface:
    center = torch.zeros(b, 3)
    center[:, 2] = z
    rot = torch.eye(3).expand(b, 3, 3).contiguous()
    return PlanarSurface(center, rot)


def _field(surface, b: int = 2, pen_r: float = 0.0015, laser_r: float = 0.0015) -> InkField:
    return InkField(
        b,
        surface,
        pen_radius_m=torch.full((b,), pen_r),
        laser_radius_m=torch.full((b,), laser_r),
        ink_rgb=torch.zeros(b, 3),
    )


def _canvas_at(surface, col, row, b: int = 1):
    """Canvas metres at a texel centre — the inverse of canvas_to_px, so a test
    can still say where it wants a stamp in pixels."""
    x = (col + 0.5) * surface.m_per_px_x - surface.width_m / 2
    y = (row + 0.5) * surface.m_per_px_y - surface.height_m / 2
    return torch.tensor([[x, y]], dtype=torch.float32).expand(b, 2).contiguous()


def test_pixel_mapping_matches_the_baked_sheet():
    """A ruling's canvas coordinate must land on the pixel textures.py drew it at."""
    sheet = grid_paper_sheets(1, seed=3)[0]
    surface = _flat_surface(1)
    # xs/ys are canvas coords of the rules; _rule_px is the pixel oracle
    off_x = sheet["xs"][0] + SHEET_W_M / 2
    off_y = sheet["ys"][0] + SHEET_H_M / 2
    want_cols = _rule_px(off_x, SHEET_W_M, SIZE_X)
    want_rows = _rule_px(off_y, SHEET_H_M, SIZE_Y)
    xy = torch.tensor([[sheet["xs"][3], sheet["ys"][5]]], dtype=torch.float32)
    px = surface.canvas_to_px(xy)[0]
    assert abs(float(px[0]) - want_cols[3]) <= 1.0, (float(px[0]), want_cols[3])
    assert abs(float(px[1]) - want_rows[5]) <= 1.0, (float(px[1]), want_rows[5])


def test_mapping_corners_and_centre():
    surface = _flat_surface(1)
    corners = torch.tensor(
        [[-SHEET_W_M / 2, -SHEET_H_M / 2], [SHEET_W_M / 2, SHEET_H_M / 2], [0.0, 0.0]]
    )
    px = surface.canvas_to_px(corners)
    assert np.allclose(px[0].numpy(), [-0.5, -0.5], atol=1e-3)
    assert np.allclose(px[1].numpy(), [SIZE_X - 0.5, SIZE_Y - 0.5], atol=1e-3)
    assert np.allclose(px[2].numpy(), [SIZE_X / 2 - 0.5, SIZE_Y / 2 - 0.5], atol=1e-3)


def test_projection_matches_the_env_distance_gate():
    """signed_dist must be what env._after_control_step computes, on tilted pads."""
    rng = np.random.default_rng(0)
    b = 4
    quats = np.stack([euler2mat(*rng.uniform(-0.09, 0.09, 3), "sxyz") for _ in range(b)])
    center = torch.as_tensor(rng.uniform(-0.05, 0.05, (b, 3)), dtype=torch.float32)
    rot = torch.as_tensor(quats, dtype=torch.float32)
    surface = PlanarSurface(center, rot)
    pts = torch.as_tensor(rng.uniform(-0.1, 0.1, (b, 3)), dtype=torch.float32)
    _, dist, inc = surface.project(pts)
    want = ((pts - center) * rot[:, :, 2]).sum(-1)  # the env's formula
    assert torch.allclose(dist, want, atol=1e-6)
    assert torch.allclose(inc, torch.ones(b))


def test_incidence_falls_off_with_obliquity():
    surface = _flat_surface(1)
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    tilted = torch.tensor([[np.sin(np.pi / 3), 0.0, np.cos(np.pi / 3)]], dtype=torch.float32)
    _, _, inc_n = surface.project(torch.zeros(1, 3), normal)
    _, _, inc_t = surface.project(torch.zeros(1, 3), tilted)
    assert abs(float(inc_n) - 1.0) < 1e-6
    assert abs(float(inc_t) - 0.5) < 1e-5


def test_kernel_size_is_specified_in_millimetres():
    """Doubling the texel density must double the PIXEL radius, not the world one."""
    r = torch.tensor([0.002])
    k1 = stamp_kernels(r, 2400.0, "disc")
    k2 = stamp_kernels(r, 4800.0, "disc")
    # count covered texels, convert back to a world area
    a1 = float(k1.sum()) / 2400.0**2
    a2 = float(k2.sum()) / 4800.0**2
    assert abs(a1 - a2) / a1 < 0.05, (a1, a2)
    assert k2.shape[-1] > k1.shape[-1]


def test_kernel_profiles():
    k = stamp_kernels(torch.tensor([0.002]), 2400.0, "disc")
    c = k.shape[-1] // 2
    assert abs(float(k[0, c, c]) - 1.0) < 1e-6  # flat top
    assert float(k[0, 0, 0]) == 0.0  # corner outside the disc
    g = stamp_kernels(torch.tensor([0.002]), 2400.0, "gaussian")
    gc = g.shape[-1] // 2
    assert abs(float(g[0, gc, gc]) - 1.0) < 1e-6
    assert float(g[0, gc, gc + 1]) < 1.0  # falls off from the centre
    assert float(g[0, 0, 0]) < 0.01  # window holds the tail


def _slope_metric(deg: float) -> torch.Tensor:
    """First fundamental form of a ramp at ``deg``: stretched along u only."""
    a = float(np.tan(np.radians(deg)))
    return torch.tensor([[[1.0 + a * a, 0.0], [0.0, 1.0]]])


def test_identity_metric_is_the_plain_radial_kernel():
    """The flat case must be untouched by the metric machinery."""
    r = torch.tensor([0.002])
    plain = stamp_kernels(r, 2400.0, "disc")
    ident = stamp_kernels(r, 2400.0, "disc", _slope_metric(0.0))
    assert torch.equal(plain, ident)


def test_a_kernel_is_elliptical_on_a_slope():
    """A round spot in world space is an ellipse in texel space: along the
    gradient each texel spans more skin, so the stamp covers fewer of them."""
    r = torch.tensor([0.002])
    half = kernel_half(r, 2400.0, "disc", 1.5)
    flat = stamp_kernels(r, 2400.0, "disc", _slope_metric(0.0), half)[0]
    tilted = stamp_kernels(r, 2400.0, "disc", _slope_metric(45.0), half)[0]
    c = half
    along = float((tilted[c, :] > 0.5).sum())  # across columns = canvas u = the slope
    across = float((tilted[:, c] > 0.5).sum())
    assert along < across, (along, across)
    # across the slope nothing changed
    assert abs(across - float((flat[:, c] > 0.5).sum())) <= 1.0
    # and the narrowing is the 1/cos(45) the metric says it is
    assert abs(along / across - np.cos(np.radians(45.0))) < 0.12, along / across


def test_a_kernel_asked_for_in_millimetres_stays_that_size_on_a_slope():
    """The invariant the whole metric exists for: world area is the same on a
    45-degree flank as on the flat, so line weight does not drift over a limb."""
    r_m, tpm = 0.002, 2400.0
    half = kernel_half(torch.tensor([r_m]), tpm, "disc", 1.5)
    for deg in (0.0, 20.0, 35.0, 45.0):
        m = _slope_metric(deg)
        k = stamp_kernels(torch.tensor([r_m]), tpm, "disc", m, half)
        det = float(m[0, 0, 0] * m[0, 1, 1] - m[0, 0, 1] ** 2)
        world_area = float(k.sum()) / tpm**2 * np.sqrt(det)
        assert abs(world_area - np.pi * r_m**2) / (np.pi * r_m**2) < 0.06, (deg, world_area)


def test_a_kernel_too_big_for_its_window_raises():
    """Silently clipping a kernel would thin every line on a steep flank, so a
    surface that outgrows the field's margin has to say so."""
    r = torch.tensor([0.002])
    half = kernel_half(r, 2400.0, "disc", 1.0)
    try:
        stamp_kernels(r, 2400.0, "disc", torch.tensor([[[0.25, 0.0], [0.0, 1.0]]]), half)
    except ValueError as exc:
        assert "max_stretch" in str(exc)
        return
    raise AssertionError("an over-stretched kernel must raise")


def test_per_env_radius_shares_one_window():
    ks = stamp_kernels(torch.tensor([0.001, 0.003]), 2400.0, "disc")
    assert ks.shape[0] == 2 and ks.shape[1] == ks.shape[2]
    assert float(ks[0].sum()) < float(ks[1].sum())


def test_where_a_stamp_lands_does_not_depend_on_the_margin():
    """The field carries a margin sized from the tool radii. Rounding a stamp
    AFTER that offset is added moves a point sitting exactly on a texel
    boundary by a texel whenever the margin is odd, because ties round to even
    — ink placement quietly coupled to an allocation detail."""
    surface = _flat_surface(1)
    centre = torch.tensor([[0.0, 0.0]])  # the canvas centre falls on a boundary
    assert float(surface.canvas_to_px(centre)[0, 0]) % 1.0 == 0.5
    seen = {}
    for laser_r in (0.0015, 0.0019):  # different radii, different margin
        fld = _field(surface, 1, laser_r=laser_r)
        fld.deposit(surface, centre, torch.tensor([1.0]))
        ys, xs = np.nonzero(fld.field[0].numpy() > 0.5)
        seen[fld.pad] = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    assert len(seen) == 2, "these radii must give different margins to test anything"
    assert len(set(seen.values())) == 1, seen


def test_removal_inverts_deposition():
    """The whole point: erode(deposit(f), c) == deposit(f) * (1 - c*K)."""
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    uv = _canvas_at(surface, 200, 300)
    fld.deposit(surface, uv, torch.tensor([0.8]))
    before = fld.field.clone()
    c = 0.3
    fld.remove(surface, uv, torch.tensor([c]))
    kern = stamp_kernels(
        fld.laser_radius_m, fld.texel_per_m, "gaussian",
        surface.first_fundamental_form(uv), fld.laser_half,
    )[0]
    h = kern.shape[-1] // 2
    r, col = 300, 200
    patch_before = before[0, r - h : r + h + 1, col - h : col + h + 1]
    patch_after = fld.field[0, r - h : r + h + 1, col - h : col + h + 1]
    assert torch.allclose(patch_after, patch_before * (1 - c * kern), atol=1e-6)


def test_repeated_passes_fade_asymptotically_and_never_go_negative():
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    uv = _canvas_at(surface, 200, 300)
    fld.deposit(surface, uv, torch.tensor([1.0]))
    assert float(fld.field.max()) > 0.99
    last = float(fld.field[0, 300, 200])
    for _ in range(25):
        fld.remove(surface, uv, torch.tensor([0.25]))
        now = float(fld.field[0, 300, 200])
        assert now < last  # strictly fading under the beam centre
        assert float(fld.field.min()) >= 0.0  # never negative
        last = now
    assert last < 0.02  # (1 - 0.25)^25 at the peak


def test_gaussian_spot_leaves_a_halo_a_single_pass_cannot_clear():
    """Beam profile, not a cookie cutter: one stationary spot clears its centre
    long before its rim, so removing a stroke needs the tool to travel over it.
    A disc-profile laser would erase a perfect circle and hide that."""
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    uv = _canvas_at(surface, 200, 300)
    fld.deposit(surface, uv, torch.tensor([1.0]))
    for _ in range(25):
        fld.remove(surface, uv, torch.tensor([0.25]))
    centre = float(fld.field[0, 300, 200])
    rim = float(fld.field.max())
    assert centre < 0.02 < rim, (centre, rim)


def test_deposit_saturates_at_opaque():
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    uv = _canvas_at(surface, 200, 300)
    for _ in range(10):
        fld.deposit(surface, uv, torch.tensor([0.6]))
    assert float(fld.field.max()) <= 1.0
    assert abs(float(fld.field.max()) - 1.0) < 1e-6


def test_edge_stamps_do_not_wrap_or_throw():
    """A stamp at the sheet border must fall off, not appear on the far side."""
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    fld.deposit(surface, _canvas_at(surface, 0, 300), torch.tensor([1.0]))  # left edge
    assert float(fld.field[0, 300, -1]) == 0.0  # nothing on the right edge
    assert float(fld.field[0, 300, 0]) > 0.0
    # far outside: dropped entirely, and quietly
    fld.reset()
    fld.deposit(surface, _canvas_at(surface, -5000, -5000), torch.tensor([1.0]))
    assert float(fld.field.sum()) == 0.0


def test_splat_is_per_env_independent():
    surface = _flat_surface(3)
    fld = _field(surface, 3)
    active = torch.tensor([True, False, True])
    fld.deposit(surface, _canvas_at(surface, 100, 100, 3), torch.tensor([1.0, 1.0, 1.0]), active)
    cov = fld.coverage()
    assert float(cov[0]) > 0 and float(cov[2]) > 0
    assert float(cov[1]) == 0.0
    assert bool(fld.dirty[0]) and not bool(fld.dirty[1])


def test_splat_rejects_unknown_op():
    f = torch.zeros(1, 40, 40)
    try:
        splat(f, torch.tensor([[20.0, 20.0]]), torch.ones(1, 5, 5), torch.ones(1), op="bleach")
    except ValueError:
        return
    raise AssertionError("unknown op must raise")


def test_composite_endpoints():
    surface = _flat_surface(1)
    ink = torch.tensor([[0.05, 0.05, 0.065]])
    fld = InkField(1, surface, torch.tensor([0.0015]), torch.tensor([0.0015]), ink)
    base = torch.full((1, surface.rows, surface.cols, 3), 0.9)
    bare = fld.composite_rgba(base)
    assert int(bare[0, 0, 0, 0]) == round(0.9 * 255)
    assert int(bare[0, 0, 0, 3]) == 255
    fld._f.fill_(1.0)
    full = fld.composite_rgba(base)
    assert int(full[0, 5, 5, 0]) == round(0.05 * 255)


def test_laser_eta_scales_with_incidence():
    e = laser_eta(torch.tensor([0.2, 0.2]), torch.tensor([1.0, 0.5]))
    assert abs(float(e[0]) - 0.2) < 1e-6
    assert abs(float(e[1]) - 0.1) < 1e-6
    assert float(laser_eta(torch.tensor([5.0]), torch.tensor([1.0]))) == 1.0  # clamped


def test_resample_polyline_spacing():
    pts = np.array([[0.0, 0.0], [0.01, 0.0]])
    out = resample_polyline(pts, 0.001)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert d.max() <= 0.001 + 1e-6
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
    assert len(resample_polyline(np.array([[0.0, 0.0]]), 0.001)) == 1


def test_ink_lands_where_the_strokes_say_it_does():
    """An OFF-CENTRE motif must rasterize to the pixels its canvas coordinates
    predict, bar the line's own half-width.

    Centred marks pass under a mapping that is offset or has a flipped axis;
    this is the case that does not. Checked against a real generated episode
    on 2026-08-25 (recorded strokes 366.6-395.0 px, ink 363-399 px).
    """
    surface = _flat_surface(1)
    r = 0.0015
    fld = _field(surface, 1, pen_r=r)
    box = np.array([[0.02, 0.03], [0.05, 0.03], [0.05, 0.06], [0.02, 0.06], [0.02, 0.03]])
    fld.rasterize(surface, [[Stroke(box)]], torch.tensor([1.0]))
    ys, xs = np.nonzero(fld.field[0].numpy() > 0.5)
    want = surface.canvas_to_px(torch.as_tensor(box, dtype=torch.float32)).numpy()
    half_px = r * surface.texel_per_m
    for got, exp in ((xs, want[:, 0]), (ys, want[:, 1])):
        assert abs(got.min() - (exp.min() - half_px)) < 2.0, (got.min(), exp.min())
        assert abs(got.max() - (exp.max() + half_px)) < 2.0, (got.max(), exp.max())


def test_rasterize_pre_inks_each_env_with_its_own_program():
    surface = _flat_surface(2)
    fld = _field(surface, 2)
    big = Stroke(np.array([[-0.03, 0.0], [0.03, 0.0]]))
    small = Stroke(np.array([[-0.005, 0.0], [0.005, 0.0]]))
    fld.rasterize(surface, [[big], [small]], torch.tensor([1.0, 1.0]))
    cov = fld.coverage()
    assert float(cov[0]) > float(cov[1]) > 0
    # a drawn line is continuous: no bare texels along its span
    row = SIZE_Y // 2
    span = fld.field[0, row, SIZE_X // 2 - 30 : SIZE_X // 2 + 30]
    assert float(span.min()) > 0.5


def test_erase_prompts_are_definite_and_name_the_verb():
    """An erase episode acts on ink already there, so its prompt must say so —
    "remove THE star", never "remove A star"."""
    from tatbot_sim.language import LEXICON_VERSION, _definite, sample_scene
    from tatbot_sim.textures import grid_paper_sheets

    assert _definite("a small star") == "the small star"
    assert _definite("an oval") == "the oval"
    assert _definite("the letter R") == "the letter R"  # not "the the letter R"
    assert _definite("two circles") == "the two circles"

    sheet = grid_paper_sheets(1, seed=1)[0]
    rng = np.random.default_rng(3)
    _, prog = sample_scene(rng, sheet, 12.0, verb="erase")
    assert prog["verb"] == "erase"
    assert prog["lexicon"] == LEXICON_VERSION
    assert prog["prompt"].startswith("remove the ")
    assert " a " not in prog["prompt"], prog["prompt"]


def test_a_pre_inked_scene_is_the_thing_the_laser_clears():
    """End to end on the field: rasterize a scene, run the laser along the
    same strokes, and the coverage it started with must come down."""
    surface = _flat_surface(1)
    fld = _field(surface, 1)
    box = np.array([[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01]])
    strokes = [Stroke(box)]
    fld.rasterize(surface, [strokes], torch.tensor([0.9]))
    start = float(fld.coverage()[0])
    assert start > 0
    pts = resample_polyline(box, 0.0005)
    for _pass in range(4):
        for p in pts:
            uv = torch.as_tensor(p, dtype=torch.float32).unsqueeze(0)
            fld.remove(surface, uv, torch.tensor([0.25]))
    end = float(fld.coverage()[0])
    assert end < 0.25 * start, (start, end)
    assert float(fld.field.min()) >= 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
