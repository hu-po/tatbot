"""The wrist depth corruptor, against what the real D405 does.

The 2026-08-26 comparison against the bench found the dropped FRACTION was
about right (35.3% sim vs 42.8% real on matched frames) while the holes
themselves were wrong: axis-aligned rectangles on a coarse lattice, scattered
without regard for what was behind them. These pin both properties, because
the fraction alone is the part that already looked fine.

No render device needed -- the corruptor is tensor math over a depth map:

    cd python/tatbot_sim && uv run python tests/test_depth_noise.py
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.depth_noise import DepthCorruptor, DepthNoiseConfig

B, H, W = 3, 120, 160
DEV = torch.device("cpu")


def _flat(depth_mm=300.0, b=B):
    """A featureless wall: no edges, no slope, nothing but the blob term."""
    return torch.full((b, H, W, 1), float(depth_mm), dtype=torch.float32)


def _corruptor(**kw):
    cfg = DepthNoiseConfig(**{
        # isolate the blob term unless a test asks otherwise
        "sigma_at_ref_mm": (0.0, 0.0), "warp_mm": (0.0, 0.0),
        "min_z_mm": (0.0, 0.0), "edge_drop_prob": (0.0, 0.0),
        **kw})
    return DepthCorruptor(B, DEV, cfg, seed=0)


def test_the_dropped_fraction_is_the_one_that_was_sampled():
    """Weighting moves WHERE holes go; it must not move how many."""
    for target in (0.10, 0.25, 0.40):
        c = _corruptor(blob_drop_frac=(target, target))
        out = c(_flat()).numpy()
        for i in range(B):
            got = float((out[i] == 0).mean())
            assert abs(got - target) < 0.02, (target, got)


def test_holes_are_not_rectangles_on_a_lattice():
    """The failure this replaced: a coarse grid upsampled nearest-neighbour.

    Such a mask can only change value at a multiple of the cell width, so every
    horizontal transition lands on the lattice. A thresholded smooth field puts
    them anywhere.
    """
    cells = 24
    c = _corruptor(blob_drop_frac=(0.25, 0.25), blob_cells=cells)
    m = (c(_flat()).numpy()[0, :, :, 0] == 0)
    step = W / cells
    cols = np.flatnonzero(np.diff(m.astype(np.int8), axis=1).any(axis=0) != 0)
    assert cols.size > cells, f"only {cols.size} transition columns"
    # distance from each transition to the nearest lattice line
    off = np.abs(cols[:, None] + 1 - np.arange(cells + 1)[None, :] * step).min(axis=1)
    assert np.median(off) > 1.0, f"transitions hug the lattice: median {np.median(off):.2f} px"


def test_holes_have_curved_boundaries_not_straight_ones():
    """A level set of a smooth field is not made of long axis-aligned runs."""
    c = _corruptor(blob_drop_frac=(0.25, 0.25))
    m = (c(_flat()).numpy()[0, :, :, 0] == 0)
    # the longest straight vertical edge: with rectangles this spans a whole
    # cell (20 px at 24 cells over 480), with blobs it is a few pixels
    edge = np.diff(m.astype(np.int8), axis=1) != 0
    runs = 0
    for col in range(edge.shape[1]):
        r, best = 0, 0
        for row in range(edge.shape[0]):
            r = r + 1 if edge[row, col] else 0
            best = max(best, r)
        runs = max(runs, best)
    assert runs < H // 3, f"a straight edge {runs} px long in a {H}-row frame"


def test_a_surface_turned_away_from_the_camera_loses_more_of_it():
    """Real stereo fails on grazing surfaces. Content-blind dropout does not."""
    ramp = _flat()
    # right half slopes steeply across the pixel grid, left half is flat
    x = torch.arange(W, dtype=torch.float32)
    slope = torch.where(x > W // 2, (x - W // 2) * 8.0, torch.zeros_like(x))
    ramp = ramp + slope.view(1, 1, W, 1)
    c = _corruptor(blob_drop_frac=(0.25, 0.25), blob_grazing_weight=(1.2, 1.2),
                   blob_range_weight=(0.0, 0.0))
    m = (c(ramp).numpy()[:, :, :, 0] == 0)
    flat_side = m[:, :, : W // 2].mean()
    steep_side = m[:, :, W // 2 + 1:].mean()
    assert steep_side > 2 * flat_side, (flat_side, steep_side)


def test_distance_costs_matches_too():
    """Directional, and deliberately gentler than the grazing term.

    Grazing is what the bench frame actually showed -- the real holes gather on
    the pen barrel and the flanks. How dropout scales with range over this
    scene's narrow depth band is not something the recordings pin down, so the
    term is kept modest and asserted as a clear trend rather than a factor
    nobody measured.
    """
    near, far = _flat(200.0, b=1), _flat(900.0, b=1)
    both = torch.cat([near, far], dim=2)  # near on the left, far on the right
    cfg = DepthNoiseConfig(sigma_at_ref_mm=(0.0, 0.0), warp_mm=(0.0, 0.0),
                           min_z_mm=(0.0, 0.0), edge_drop_prob=(0.0, 0.0),
                           blob_drop_frac=(0.25, 0.25), blob_grazing_weight=(0.0, 0.0),
                           blob_range_weight=(0.7, 0.7))
    m = (DepthCorruptor(1, DEV, cfg, seed=1)(both).numpy()[0, :, :, 0] == 0)
    assert m[:, W:].mean() > 1.4 * m[:, :W].mean(), (m[:, :W].mean(), m[:, W:].mean())


def test_zero_weights_reproduce_content_blind_dropout():
    """The weighting is an axis, not a rewrite: turned off, holes go anywhere."""
    ramp = _flat() + (torch.arange(W, dtype=torch.float32) * 6.0).view(1, 1, W, 1)
    c = _corruptor(blob_drop_frac=(0.25, 0.25), blob_grazing_weight=(0.0, 0.0),
                   blob_range_weight=(0.0, 0.0))
    m = (c(ramp).numpy()[:, :, :, 0] == 0)
    left, right = m[:, :, : W // 2].mean(), m[:, :, W // 2:].mean()
    assert abs(left - right) < 0.08, (left, right)


def test_the_static_share_keeps_holes_still_between_frames():
    c = _corruptor(blob_drop_frac=(0.25, 0.25), blob_static_frac=1.0)
    d = _flat()
    a = (c(d).numpy() == 0)
    b = (c(d).numpy() == 0)
    assert (a == b).mean() > 0.99, "a fully static pattern moved between frames"
    c2 = _corruptor(blob_drop_frac=(0.25, 0.25), blob_static_frac=0.0)
    a2 = (c2(d).numpy() == 0)
    b2 = (c2(d).numpy() == 0)
    assert (a2 == b2).mean() < 0.95, "a fully flickering pattern stood still"


def test_invalid_and_blind_zone_still_win_over_everything():
    d = _flat(50.0)                      # nearer than any blind zone draw
    c = _corruptor(blob_drop_frac=(0.0, 0.0), min_z_mm=(70.0, 70.0))
    assert (c(d).numpy() == 0).all()
    d0 = _flat(0.0)
    assert (_corruptor(blob_drop_frac=(0.0, 0.0))(d0).numpy() == 0).all()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
