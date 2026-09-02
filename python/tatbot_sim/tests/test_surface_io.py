"""Parity between scripts/lib/draw_surface.py and tatbot_sim.surface on one surface.npz.

The mapper (NumPy-only) and the sim (torch) must agree on what the file
means, or a design rehearsed in sim lands somewhere else on the pad. Frame
points within 1e-6 m and normals within 1e-6 on a grid of uv, in float64;
float32 (the env's working dtype) is held to a looser bound.

    cd python/tatbot_sim && uv run --with pytest pytest -q tests/test_surface_io.py
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tatbot_sim.repo import repo_root  # noqa: E402
from tatbot_sim.surface_io import displaced_surface_from_npz  # noqa: E402

sys.path.insert(0, str(repo_root() / "scripts" / "lib"))

import draw_surface as ds  # noqa: E402


def _rot(normal, u_hint):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    eu = np.asarray(u_hint, float)
    eu = eu - (eu @ n) * n
    eu /= np.linalg.norm(eu)
    return np.stack([eu, np.cross(n, eu), n], axis=1)


def _surface(chart, w=0.10, h=0.12, rows=25, cols=21, peak=0.004):
    us = np.linspace(-w / 2, w / 2, cols)
    vs = np.linspace(-h / 2, h / 2, rows)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    r = np.hypot(uu / 0.035, vv / 0.045)
    height = np.where(r < 1, 0.5 * peak * (1 + np.cos(np.pi * np.clip(r, 0, 1))), 0.0)
    height += 0.0005 * np.sin(uu * 90.0) * np.cos(vv * 70.0)
    return ds.HeightFieldSurface(chart, height, w, h)


def _grid(w=0.10, h=0.12, n=37):
    us = np.linspace(-w / 2 - 0.005, w / 2 + 0.005, n)  # a little outside too: border replicate
    vs = np.linspace(-h / 2 - 0.005, h / 2 + 0.005, n + 4)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    return np.stack([uu.ravel(), vv.ravel()], -1)


CHARTS = {
    "plane": ds.PlaneChart([0.3, -0.05, 0.05], _rot([0.1, -0.2, 1.0], [1, 0.1, 0])),
    "cylinder": ds.CylinderChart([0.25, 0.05, 0.10], _rot([-0.1, 0.2, 1.0], [1, 0.3, 0.1]), 0.04),
}


@pytest.mark.parametrize("kind", list(CHARTS))
def test_sim_loads_the_mapper_surface_within_a_micrometre(tmp_path, kind):
    surf = _surface(CHARTS[kind])
    path = surf.to_npz(tmp_path / "surface.npz")
    sim = displaced_surface_from_npz(path, dtype=torch.float64)
    assert sim.batch_size == 1 and sim.chart.__class__.__name__.lower().startswith(kind)
    assert (sim.cols, sim.rows) == (200, 240)

    uv = _grid()
    p_np, du_np, dv_np, n_np = surf.frame(uv)
    view = sim.env_view(0, uv.shape[0])
    p_t, du_t, dv_t, n_t = view.frame(torch.as_tensor(uv, dtype=torch.float64))
    assert np.abs(p_t.numpy() - p_np).max() < 1e-6
    assert np.abs(n_t.numpy() - n_np).max() < 1e-6
    assert np.abs(du_t.numpy() - du_np).max() < 1e-6
    assert np.abs(dv_t.numpy() - dv_np).max() < 1e-6

    # projection agrees too: the sim places the mapper's own surface points back on uv
    inside = (np.abs(uv[:, 0]) < 0.045) & (np.abs(uv[:, 1]) < 0.055)
    uv_in = uv[inside]
    pts = torch.as_tensor(surf.frame(uv_in)[0], dtype=torch.float64)
    view_in = sim.env_view(0, uv_in.shape[0])
    uv_back, dist, _ = view_in.project(pts)
    assert view_in.unconverged == 0
    assert np.abs(uv_back.numpy() - uv_in).max() < 1e-6
    assert np.abs(dist.numpy()).max() < 1e-8


def test_float32_load_is_the_env_dtype_and_close_enough(tmp_path):
    surf = _surface(CHARTS["cylinder"])
    path = surf.to_npz(tmp_path / "surface.npz")
    sim = displaced_surface_from_npz(path)
    assert sim.height.dtype == torch.float32
    uv = _grid()
    p_np, _, _, n_np = surf.frame(uv)
    p_t, n_t = sim.frame_np(0, uv)
    assert np.abs(p_t - p_np).max() < 1e-5
    assert np.abs(n_t - n_np).max() < 1e-4
    assert np.allclose(sim.base_normal_np(0), CHARTS["cylinder"].normal, atol=1e-6)


def test_loader_refuses_the_wrong_schema(tmp_path):
    np.savez(tmp_path / "bad.npz", schema=np.str_("tatbot.surface/0"), chart_kind=np.str_("plane"))
    with pytest.raises(ValueError, match="schema"):
        displaced_surface_from_npz(tmp_path / "bad.npz")
