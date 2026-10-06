"""What the tools work on: three substrates, and the texture each one gets.

A tool and its substrates are a pair. The ballpoint draws on the two gridded
paper fixtures — the 7.5 x 11 in pad and the 85 mm paper cylinder — and the
laser and the 3RL only ever work on the silicone skin, which is smaller,
thinner, pink, and has nothing printed on it. The sim sizes both its geometry
and its texture from one record, so those cannot drift apart — and the thing
most likely to drift silently is the texture, because a sheet at the wrong
resolution composites against the wrong field and a ruling drawn on a skin
would give a policy a stencil that does not exist.

Needs cv2 and the asset dir, no render device:

    cd python/tatbot_sim && uv run --with pytest pytest -q tests/test_substrate.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tatbot_sim import textures, tools
from tatbot_sim.textures import grid_paper_sheets, skin_sheets

REPO = Path(__file__).resolve().parents[3]


def _sub(name):
    return tools.registry().load_substrate(name, REPO)


def _quad_extent(obj_path):
    """Half-extent of the sheet quad, read back out of the OBJ."""
    verts = np.array([[float(v) for v in ln.split()[1:]]
                      for ln in Path(obj_path).read_text().splitlines()
                      if ln.startswith("v ")])
    return abs(verts[:, 0]).max(), abs(verts[:, 1]).max()


def test_the_skin_is_the_size_its_datasheet_says():
    sub = _sub("silicon_skin")
    sheet = skin_sheets(1, sub, seed=0)[0]
    img = cv2.imread(sheet["png"])
    assert img.shape[:2] == (sub.texel_rows, sub.texel_cols), img.shape
    hx, hy = _quad_extent(sheet["obj"])
    assert abs(hx - sub.width_m / 2) < 1e-6 and abs(hy - sub.height_m / 2) < 1e-6


def test_the_skin_is_skin_coloured():
    sub = _sub("silicon_skin")
    for i, sheet in enumerate(skin_sheets(3, sub, seed=4)):
        b, g, r = cv2.imread(sheet["png"]).reshape(-1, 3).mean(0)
        assert r > g > b, (i, r, g, b)          # pink/peach, not grey
        assert 140 < r < 255, (i, r)


def test_nothing_is_printed_on_the_skin():
    """A ruling on a skin would hand the policy a stencil that is not there.
    Paper has hard dark lines; silicone has only a slow mottle and grain, so
    the darkest row is nowhere near as dark as the sheet's own average."""
    skin = cv2.imread(skin_sheets(1, _sub("silicon_skin"), seed=1)[0]["png"])
    paper = cv2.imread(grid_paper_sheets(1, seed=1)[0]["png"])
    for img, name, ruled in ((paper, "paper", True), (skin, "skin", False)):
        grey = img.mean(-1)
        # how far below the sheet's mean its darkest pixels sit
        contrast = (grey.mean() - np.percentile(grey, 0.5)) / 255.0
        if ruled:
            assert contrast > 0.05, (name, contrast)
        else:
            assert contrast < 0.03, (name, contrast)


def test_the_skin_still_offers_a_placement_lattice():
    """Motifs that want to sit on a grid still need somewhere to sit when
    nothing is printed; the lattice is a layout device, not a ruling."""
    sub = _sub("silicon_skin")
    sheet = skin_sheets(1, sub, seed=2)[0]
    assert sheet["ruled"] is False
    assert len(sheet["xs"]) > 4 and len(sheet["ys"]) > 4
    assert max(abs(x) for x in sheet["xs"]) <= sub.width_m / 2 + 1e-9
    assert max(abs(y) for y in sheet["ys"]) <= sub.height_m / 2 + 1e-9


def test_the_paper_pad_is_the_operators_pad():
    """7.5 x 11 in, 1 cm thick, 1/4 in grid — as measured 2026-09-11. The
    module constants Surface defaults to must say the same thing as the record
    the env builds from, or the two would size the sheet differently."""
    sub = _sub("paper_pad")
    assert (sub.width_m, sub.height_m, sub.thickness_m) == (0.1905, 0.2794, 0.010)
    assert sub.grid_pitch_m == 0.00635 and sub.shape == "pad" and sub.radius_m is None
    assert (sub.width_m, sub.height_m) == (textures.SHEET_W_M, textures.SHEET_H_M)
    assert (sub.texel_cols, sub.texel_rows) == (textures.SIZE_X, textures.SIZE_Y)
    assert sub.grid_pitch_m == textures.GRID_PITCH_M
    sheet = grid_paper_sheets(1, seed=1)[0]
    img = cv2.imread(sheet["png"])
    assert img.shape[:2] == (sub.texel_rows, sub.texel_cols) == (662, 452)
    hx, hy = _quad_extent(sheet["obj"])
    assert abs(hx - sub.width_m / 2) < 1e-6 and abs(hy - sub.height_m / 2) < 1e-6
    # the rules land on an exact 1/4 in pitch, phase aside
    xs = np.diff(sheet["xs"])
    assert np.allclose(xs, 0.00635, atol=1e-9), xs[:3]


def test_the_paper_is_white_with_faint_blue_rules():
    for i, sheet in enumerate(grid_paper_sheets(3, seed=2)):
        img = cv2.imread(sheet["png"]).astype(float)
        b, g, r = np.median(img.reshape(-1, 3), axis=0)  # the paper between the rules
        assert min(r, g, b) > 225, (i, r, g, b)
        grey = img.mean(-1)
        rows = img[grey < np.percentile(grey, 2)]          # on the rules
        rb, rg, rr = rows.mean(0)
        assert rb > rr + 15, (i, rr, rg, rb)                # blue, not grey or black
        assert rb > 140, (i, rb)                            # faint, not saturated


def test_a_flat_sheet_does_not_bake_in_a_surface_profile():
    """Flat/cylinder is a scenario axis for a sheet of silicone or paper; the
    record says how it is presented (a pad), not how the scenario bends it."""
    for name in ("silicon_skin", "paper_pad"):
        sub = _sub(name)
        assert sub.shape == "pad" and sub.radius_m is None
        assert not hasattr(sub, "mound_peak_m")


def test_the_paper_cylinder_is_a_rigid_fixture_of_one_radius():
    """85 mm diameter, 7.5 in long, gridded all round; the drawable band is
    the outer surface less the bottom quarter, and the actor origin is the axis."""
    sub = _sub("paper_cylinder")
    assert sub.shape == "cylinder" and sub.radius_m == 0.0425
    assert sub.height_m == 0.1905 and sub.thickness_m == 0.085
    assert abs(sub.width_m - 1.5 * np.pi * sub.radius_m) < 1e-4
    assert sub.grid_pitch_m == 0.00635 and sub.ruled
    sheet = grid_paper_sheets(1, sub, seed=1)[0]
    img = cv2.imread(sheet["png"])
    assert img.shape[:2] == (sub.texel_rows, sub.texel_cols)
    hx, hy = _quad_extent(sheet["obj"])
    assert abs(hx - sub.width_m / 2) < 1e-6 and abs(hy - sub.height_m / 2) < 1e-6
    assert sheet["radius_m"] == sub.radius_m
    # the fixture: a closed tube of the right radius and length, along +y
    verts = np.array([[float(v) for v in ln.split()[1:]]
                      for ln in Path(sheet["fixture_obj"]).read_text().splitlines()
                      if ln.startswith("v ")])
    radial = np.hypot(verts[:, 0], verts[:, 2])
    assert abs(radial.max() - sub.radius_m) < 1e-6
    assert abs(verts[:, 1].max() - sub.height_m / 2) < 1e-6 and abs(verts[:, 1].min() + sub.height_m / 2) < 1e-6
    wrap = cv2.imread(sheet["fixture_png"])
    assert wrap.shape[0] == sub.texel_rows
    assert abs(wrap.shape[1] / (2 * np.pi * sub.radius_m) - sub.texel_rows / sub.height_m) < 3.0


def test_the_fixture_grid_continues_the_bands_rules():
    """A rule on the crest runs straight down the side of the tube: the wrap's
    circumferential lines are phase-locked to the band's."""
    sub = _sub("paper_cylinder")
    sheet = grid_paper_sheets(1, sub, seed=3)[0]
    off_band = sheet["xs"][0] + sub.width_m / 2            # first band rule from the band's left edge
    off_wrap = textures.fixture_seam_offset_m(sub, off_band)
    pitch = sub.grid_pitch_m
    # arc from the seam to the band's left edge, then to its first rule
    expect = (np.pi * sub.radius_m - sub.width_m / 2 + off_band) % pitch
    assert abs(off_wrap - expect) < 1e-9


def test_a_rigid_cylinder_pins_the_profile_to_its_own_radius():
    from tatbot_sim.config import DRConfig

    sub = _sub("paper_cylinder")
    dr = DRConfig().resolve_for(sub)
    assert dr.surface.profile == "cylinder"
    assert dr.surface.cylinder_radius_m == (0.0425, 0.0425)
    assert dr.pad.z_range == tuple(sub.rest_z_m)
    bad = DRConfig()
    bad.surface.profile = "balanced"
    try:
        bad.resolve_for(sub)
    except ValueError:
        pass
    else:
        raise AssertionError("a balanced profile was accepted on a rigid cylinder")


def test_the_ballpoint_admits_both_paper_fixtures_and_nothing_else():
    reg = tools.registry()
    pen = reg.load_tool("lutin-ballpoint-dot", REPO)
    assert pen.substrates == ("paper_pad", "paper_cylinder")
    assert reg.substrate_for(pen, REPO, name="paper_cylinder").name == "paper_cylinder"
    try:
        reg.substrate_for(pen, REPO, name="silicon_skin")
    except ValueError:
        pass
    else:
        raise AssertionError("the ballpoint was allowed onto the skin")


def test_balanced_profile_sampling_contains_flat_and_cylinder():
    from tatbot_sim.env import sample_surface_profiles

    profiles, radii = sample_surface_profiles(
        np.random.default_rng(5), 9, "balanced", (0.075, 0.110)
    )
    assert abs(profiles.count("flat") - profiles.count("cylinder")) <= 1
    assert set(profiles) == {"flat", "cylinder"}
    for profile, radius in zip(profiles, radii, strict=True):
        assert np.isinf(radius) if profile == "flat" else 0.075 <= radius <= 0.110


def test_profile_sampling_refuses_unknown_profiles_and_bad_radii():
    from tatbot_sim.env import sample_surface_profiles

    for profile, radius in (("mound", (0.075, 0.110)), ("cylinder", (0.1, 0.05))):
        try:
            sample_surface_profiles(np.random.default_rng(0), 4, profile, radius)
        except ValueError:
            continue
        raise AssertionError(f"accepted profile={profile!r}, radius={radius!r}")


def test_a_shaped_substrate_is_one_solid_not_a_sheet_over_a_box():
    """A flat box behind a curved profile would show through it. The mesh
    carries its own underside instead, so there is no body to poke out."""
    import tempfile

    from tatbot_sim.textures import write_surface_mesh

    rows, cols = 9, 7
    xs = np.linspace(-0.07, 0.07, cols)
    ys = np.linspace(-0.09, 0.09, rows)
    vv, uu = np.meshgrid(ys, xs, indexing="ij")
    z = 0.025 * np.cos(np.pi * np.clip(np.hypot(uu / 0.06, vv / 0.08), 0, 1)) * 0.5 + 0.0125
    verts = np.stack([uu.ravel(), vv.ravel(), z.ravel()], 1)
    nrm = np.tile([0.0, 0.0, 1.0], (len(verts), 1))
    with tempfile.TemporaryDirectory() as td:
        thin = Path(write_surface_mesh(Path(td) / "a", "m", verts, nrm, rows, cols))
        thin_v = thin.read_text().count("\nv ")
        solid = Path(write_surface_mesh(Path(td) / "b", "m", verts, nrm, rows, cols,
                                        thickness_m=0.0025))
        text = solid.read_text()
    zs = np.array([float(ln.split()[3]) for ln in text.splitlines() if ln.startswith("v ")])
    assert len(zs) == 2 * thin_v                       # a top and an underside
    assert abs(zs.min() - (z.min() - 0.0025)) < 1e-6   # exactly one thickness below
    # top, underside, and a wall joining their rims
    faces = sum(1 for ln in text.splitlines() if ln.startswith("f "))
    assert faces > 4 * (rows - 1) * (cols - 1)


def test_each_substrate_owns_how_it_rests_but_not_its_profile():
    from tatbot_sim.config import DRConfig

    skin, pad = _sub("silicon_skin"), _sub("paper_pad")
    assert skin.rest_z_m[1] <= 0.005 < pad.rest_z_m[1]
    assert pad.rest_z_m[1] < _sub("paper_cylinder").rest_z_m[0]  # a tube's crest is a diameter up

    for sub in (skin, pad):
        dr = DRConfig().resolve_for(sub)
        assert dr.pad.z_range == tuple(sub.rest_z_m)
        assert dr.surface.profile == "flat"
        # resolving again must not move a value that is already settled
        other = pad if sub is skin else skin
        assert dr.resolve_for(other).pad.z_range == tuple(sub.rest_z_m)


def test_an_explicit_range_survives_the_substrate_default():
    from tatbot_sim.config import DRConfig

    dr = DRConfig()
    dr.pad.z_range = (0.02, 0.02)          # what fiducial_benchmark pins
    dr.surface.profile = "cylinder"
    dr.resolve_for(_sub("silicon_skin"))
    assert dr.pad.z_range == (0.02, 0.02)
    assert dr.surface.profile == "cylinder"


def test_a_reversed_range_is_refused_rather_than_sampled_empty():
    import tool_spec

    for bad in ([0.01, 0.0], [0.0], 0.02):
        try:
            tool_spec._pair({"rest_z_m": bad}, "rest_z_m", (0.0, 0.05), Path("x"))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
