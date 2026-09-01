"""What the tools work on: two substrates, and the texture each one gets.

A tool and a substrate are a pair. The ballpoint only ever draws on the ruled
paper pad; the laser and the 3RL only ever work on the silicone skin, which is
smaller, thinner, pink, and has nothing printed on it. The sim sizes both its
geometry and its texture from one record, so those cannot drift apart — and
the thing most likely to drift silently is the texture, because a sheet at the
wrong resolution composites against the wrong field and a ruling drawn on a
skin would give a policy a stencil that does not exist.

Needs cv2 and the asset dir, no render device:

    cd python/tatbot_sim && uv run python tests/test_substrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from tatbot_sim import tools
from tatbot_sim.textures import grid_paper_sheets, skin_sheets

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "lib"))


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


def test_the_paper_pad_is_untouched():
    """Every existing dataset was drawn on this sheet; it must not move."""
    sub = _sub("paper_pad")
    sheet = grid_paper_sheets(1, seed=1)[0]
    img = cv2.imread(sheet["png"])
    assert img.shape[:2] == (sub.texel_rows, sub.texel_cols) == (662, 512)
    hx, hy = _quad_extent(sheet["obj"])
    assert abs(hx - sub.width_m / 2) < 1e-6 and abs(hy - sub.height_m / 2) < 1e-6


def test_the_skin_is_shaped_by_the_mound_its_datasheet_measured():
    """The skin has its mound in the room, so the sim starts from the measured
    25 mm rather than waiting to be asked for domain randomisation."""
    from tatbot_sim.surface import drape_height_field

    sub = _sub("silicon_skin")
    assert sub.shape == "draped" and sub.mound_peak_m > 0.02
    h = drape_height_field(None, 1, 111, 84, [sub.mound_peak_m], [0.057], [0.075],
                           width_m=sub.width_m, height_m=sub.height_m)
    a = h[0].numpy()
    # the summit IS the measurement, up to where the grid samples it
    assert a.max() <= sub.mound_peak_m + 1e-9
    assert a.max() > sub.mound_peak_m * 0.995
    assert abs(a[0, :]).max() == 0.0                    # edges lie flat
    assert abs(a[:, 0]).max() == 0.0
    # ONE rise, not a field of them: the high ground is a single run in each
    # axis, and it does not span the whole skin
    high = a > sub.mound_peak_m / 2
    for axis in (0, 1):
        line = high.any(axis=axis)
        idx = np.flatnonzero(line)
        assert len(idx) == idx[-1] - idx[0] + 1, "the high ground is not contiguous"
        assert len(idx) < len(line) * 0.9, "the mound covers the whole skin"


def test_a_shaped_substrate_is_one_solid_not_a_sheet_over_a_box():
    """A flat box body under a 25 mm mound shows through it. The mesh carries
    its own underside instead, so there is no body to poke out."""
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


def test_each_substrate_owns_how_it_rests_and_how_much_its_shape_varies():
    from tatbot_sim.config import DRConfig

    skin, pad = _sub("silicon_skin"), _sub("paper_pad")
    # the skin's shape IS the training signal for contouring, so it is drawn at
    # the measured mound; the pad's millimetre of lift is incidental
    assert skin.peak_scale == (0.95, 1.00)
    assert pad.peak_scale == (0.85, 1.10)
    # and the skin has to rest low, because its mound spends the arm's height
    assert skin.rest_z_m[1] <= 0.005 < pad.rest_z_m[1]

    for sub in (skin, pad):
        dr = DRConfig().resolve_for(sub)
        assert dr.pad.z_range == tuple(sub.rest_z_m)
        assert dr.surface.peak_scale == tuple(sub.peak_scale)
        # resolving again must not move a value that is already settled
        other = pad if sub is skin else skin
        assert dr.resolve_for(other).pad.z_range == tuple(sub.rest_z_m)


def test_an_explicit_range_survives_the_substrate_default():
    from tatbot_sim.config import DRConfig

    dr = DRConfig()
    dr.pad.z_range = (0.02, 0.02)          # what fiducial_benchmark pins
    dr.surface.peak_scale = (0.5, 0.6)
    dr.resolve_for(_sub("silicon_skin"))
    assert dr.pad.z_range == (0.02, 0.02)
    assert dr.surface.peak_scale == (0.5, 0.6)


def test_a_reversed_range_is_refused_rather_than_sampled_empty():
    import tool_spec

    for bad in ([0.01, 0.0], [0.0], 0.02):
        try:
            tool_spec._pair({"rest_z_m": bad}, "rest_z_m", (0.0, 0.05), Path("x"))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
