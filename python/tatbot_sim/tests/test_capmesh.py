"""Ink caps render at the size config/palette.yaml specifies: the outside for
the cup, the derived bore and depth for where the tool goes and the ink sits.

The spec is the cap's OUTSIDE; the sim once took it for the bore, rendering
every cap a wall too wide, floating a wall above its support floor, with a
full cap's ink a millimetre short of the brim.
"""

from __future__ import annotations

import math

import pytest
from tatbot_sim import capmesh, tools
from tatbot_sim import palette as sim_palette


def _sizes():
    return {slot.size.size_id: slot.size for slot in sim_palette.load(tools.REPO).palette.values()}


@pytest.mark.parametrize("size_id", ["large", "medium", "small"])
def test_the_cap_mesh_has_the_specified_outside_and_bore(tmp_path, size_id):
    trimesh = pytest.importorskip("trimesh")
    size = _sizes()[size_id]
    mesh = trimesh.load(capmesh.cap_mesh_path(tmp_path, size), force="mesh")
    (x0, _, z0), (x1, _, z1) = mesh.bounds
    assert x1 - x0 == pytest.approx(size.diameter_m, abs=1e-6)
    assert (z0, z1) == pytest.approx((-size.height_m, 0.0), abs=1e-6)
    radii = [math.hypot(x, y) for x, y, _ in mesh.vertices]
    assert min(r for r in radii if r > 1e-6) == pytest.approx(size.bore_diameter_m / 2, abs=1e-6)


def test_the_ink_surface_is_the_one_the_dip_plunges_below():
    for size in _sizes().values():
        half_ul = size.area_m2 * size.depth_m / 2 * 1e9
        assert capmesh.ink_level_z(size, half_ul) == pytest.approx(-size.depth_m / 2)
        assert capmesh.ink_level_z(size, 0.0) == pytest.approx(-size.depth_m)
        assert capmesh.ink_level_z(size, 2 * size.capacity_ul) == -capmesh.BRIM_M
