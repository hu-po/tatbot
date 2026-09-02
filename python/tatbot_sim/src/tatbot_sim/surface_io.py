"""Load a mapped ``surface.npz`` (``docs/draw.md``) into the sim's ``DisplacedSurface``.

The drawing stages map a real pad or limb with ``scripts/lib/draw_surface.py``
(numpy) and write ``tatbot.surface/1``. This is the sim side of that contract:
the same chart and the same height grid, as a batch-1 ``DisplacedSurface``, so
a policy can be rehearsed on the surface that was actually measured. The file
carries no texel resolution -- that is a rendering choice -- so ``cols``/``rows``
are derived here from the canvas extent at a fixed texel pitch, which keeps the
sheet's pixels isotropic (``Surface`` refuses more than 1 % skew).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tatbot_sim.surface import Chart, CylinderChart, DisplacedSurface, PlaneChart

SCHEMA = "tatbot.surface/1"
TEXEL_M = 0.0005  # sheet texel pitch used for cols/rows; ~2 px/mm


def displaced_surface_from_npz(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    texel_m: float = TEXEL_M,
    iters: int = 3,
    tol_m: float = 1e-5,
) -> DisplacedSurface:
    """``surface.npz`` -> batch-1 ``DisplacedSurface`` on ``device``.

    ``cols = round(width_m / texel_m)``, ``rows = round(height_m / texel_m)``.
    ``iters``/``tol_m`` are the sim's projection settings, not part of the file.
    """
    with np.load(Path(path)) as d:
        schema = str(d["schema"])
        if schema != SCHEMA:
            raise ValueError(f"{path}: schema {schema!r} != {SCHEMA!r}")
        kind = str(d["chart_kind"])
        center = torch.as_tensor(np.asarray(d["center"], dtype=np.float64), dtype=dtype, device=device)
        rot = torch.as_tensor(np.asarray(d["rot"], dtype=np.float64), dtype=dtype, device=device)
        radius = float(d["radius_m"])
        height = torch.as_tensor(np.asarray(d["height"], dtype=np.float64), dtype=dtype, device=device)
        width_m, height_m = float(d["width_m"]), float(d["height_m"])
    if center.shape != (3,) or rot.shape != (3, 3) or height.ndim != 2:
        shapes = (tuple(center.shape), tuple(rot.shape), tuple(height.shape))
        raise ValueError(f"{path}: malformed center/rot/height shapes {shapes}")

    chart: Chart
    if kind == "plane":
        chart = PlaneChart(center[None], rot[None])
    elif kind == "cylinder":
        chart = CylinderChart(center[None], rot[None], torch.tensor([radius], dtype=dtype, device=device))
    else:
        raise ValueError(f"{path}: unknown chart_kind {kind!r}")

    cols = max(2, int(round(width_m / texel_m)))
    rows = max(2, int(round(height_m / texel_m)))
    return DisplacedSurface(chart, height[None], width_m, height_m, cols, rows, iters=iters, tol_m=tol_m)
