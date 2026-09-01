"""Invariant tests for the data factory's pure-geometry layer.

These encode the label-integrity guarantees the 2026-08-24 audit added:
what run_meta records must be what the episode drew, and what the prompt
says must be what is on the sheet. They run without SAPIEN or a GPU
(strokes/language/textures are numpy + cv2), so they are cheap to run
before any regeneration:

    cd python/tatbot_sim && .venv/bin/python tests/test_invariants.py
    # or, with pytest available: .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import dataclasses

import numpy as np
from tatbot_sim import language
from tatbot_sim.language import MOTIFS, SIZES, sample_scene
from tatbot_sim.strokes import (
    MazeConfig,
    ShapeConfig,
    Stroke,
    build_ee_trajectory,
    fit_strokes,
    overhead_steps,
    sample_maze,
    sample_shape,
)
from tatbot_sim.textures import grid_paper_sheets

HORIZON = 420  # the production maze horizon


def _maze_cfg():
    return dataclasses.replace(
        ShapeConfig(), draw_speed_range=MazeConfig().draw_speed_range
    )


def test_build_refuses_mid_stroke_truncation():
    rng = np.random.default_rng(0)
    long_line = [Stroke(np.array([[-0.05, 0.0], [0.05, 0.0]]))]
    cfg = dataclasses.replace(ShapeConfig(), draw_speed_range=(0.005, 0.005))
    try:
        build_ee_trajectory(long_line, rng, cfg, horizon=100)
    except ValueError:
        pass
    else:
        raise AssertionError("over-long trajectory must raise, not truncate")


def test_build_pads_exactly_to_horizon():
    rng = np.random.default_rng(0)
    strokes = [Stroke(np.array([[-0.01, 0.0], [0.01, 0.0]]))]
    traj = build_ee_trajectory(strokes, rng, ShapeConfig(), horizon=HORIZON)
    assert len(traj) == HORIZON
    # the padded tail holds the final hover pose, pen up
    assert not traj.pen_down[-1]
    assert np.allclose(traj.positions[-1], traj.positions[-2])


def test_maze_budget_fits_horizon():
    """The 10%-truncation regression: every budgeted walk must fit."""
    sheets = grid_paper_sheets(8)
    rng = np.random.default_rng(1)
    cfg, mc = _maze_cfg(), MazeConfig()
    for i in range(100):
        speed = float(rng.uniform(*mc.draw_speed_range))
        avail = HORIZON - overhead_steps(cfg)
        max_seg = int(avail / 30.0 * speed / sheets[i % 8]["pitch_m"]) - 1
        strokes = sample_maze(rng, sheets[i % 8], mc, max_segments=max_seg)
        assert len(strokes[0].points) - 1 <= max_seg
        pos, drawn, natural, _ = fit_strokes(
            strokes, rng, cfg, speed, HORIZON, max_start_z=0.08, grid_walk=True
        )
        assert pos.shape == (HORIZON, 3)
        # run_meta records `drawn` — it must be what the trajectory traces
        assert np.allclose(pos[np.linalg.norm(pos[:, :2] - drawn[0].points[-1], axis=1).argmin(), :2],
                           drawn[0].points[-1], atol=1e-5)


def test_shapes_fit_by_shrinking():
    rng = np.random.default_rng(2)
    for _ in range(100):
        kind, strokes, _extent = sample_shape(rng, ShapeConfig())
        speed = float(rng.uniform(*ShapeConfig().draw_speed_range))
        pos, drawn, natural, _ = fit_strokes(
            strokes, rng, ShapeConfig(), speed, HORIZON, max_start_z=0.08, grid_walk=False
        )
        assert pos.shape == (HORIZON, 3)


def test_language_scene_invariants():
    """Separation, legibility, stroke bookkeeping, prompt shape."""
    sheets = grid_paper_sheets(8)
    rng = np.random.default_rng(3)
    for i in range(100):
        strokes, prog = sample_scene(rng, sheets[i % 8], 20.0)
        # n_strokes bookkeeping must partition the stroke list exactly
        assert sum(m["n_strokes"] for m in prog["motifs"]) == len(strokes)
        # motifs must not overlap: >= 6 mm of clear sheet between any two
        groups, k = [], 0
        for m in prog["motifs"]:
            groups.append(np.concatenate([s.points for s in strokes[k:k + m["n_strokes"]]]))
            k += m["n_strokes"]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = np.sqrt(((groups[a][:, None] - groups[b][None]) ** 2).sum(-1)).min()
                assert d >= 0.006, f"motifs {d * 1000:.1f} mm apart in: {prog['prompt']}"
        # legibility floor: no motif below its own minimum size bin
        for m in prog["motifs"]:
            if not m["grid_locked"]:
                floor = SIZES[MOTIFS[m["key"]].min_size][0]
                assert m["size_r"] >= floor - 1e-9
        # ink stays inside the reach envelope
        pts = np.concatenate([s.points for s in strokes])
        assert np.abs(pts).max() <= 0.06 + 1e-6
        # prompt shape: single sentence in the slotted frame
        p = prog["prompt"]
        # the surface slot follows the fitted substrate, so read it rather
        # than hardcoding the pad — this file used to fail the moment a
        # skin-bound tool was fitted
        assert p.startswith("draw ") and p.endswith(language.SURFACE_PHRASE)
        assert "  " not in p


def test_letters():
    """All 26 letterforms compile, centred, sized to the bin, legible-ish."""
    from tatbot_sim.language import LETTERS, MOTIFS

    m = MOTIFS["letter"]
    for ch in map(chr, range(65, 91)):
        assert ch in LETTERS, f"missing letterform {ch}"
        strokes = m.compile({"r": 0.018, "rot": np.pi, "slant": 0.0, "char": ch}, {})
        assert 1 <= len(strokes) <= 3
        pts = np.concatenate(strokes)
        c = (pts.min(0) + pts.max(0)) / 2
        assert np.abs(c).max() < 1e-9, f"{ch} not centred"
        h = pts[:, 1].max() - pts[:, 1].min()
        assert 0.030 <= h <= 0.041, f"{ch} height {h}"
    # slant shears, small rot jitter keeps letters upright
    up = m.compile({"r": 0.018, "rot": np.pi, "slant": 0.0, "char": "L"}, {})
    it = m.compile({"r": 0.018, "rot": np.pi, "slant": 0.28, "char": "L"}, {})
    assert not np.allclose(np.concatenate(up), np.concatenate(it))
    # scene-level: letter prompts carry "the letter <X>" and honest style word
    sheets = grid_paper_sheets(8)
    rng = np.random.default_rng(6)
    seen = 0
    for i in range(200):
        _, prog = sample_scene(rng, sheets[i % 8], 15.0)
        for mm in prog["motifs"]:
            if mm["key"] == "letter":
                seen += 1
                assert f"letter {mm['char']}" in prog["prompt"], prog["prompt"]
                assert ("slanted" in prog["prompt"]) == bool(mm["slant"]) or \
                    sum(x["key"] == "letter" for x in prog["motifs"]) > 1
    assert seen > 20, f"letters too rare in the mix: {seen}"


def test_engaged_rejects_an_episode_that_never_touched_the_sheet():
    from tatbot_sim.generate import _engaged

    # an erase that cleared nothing and a draw that deposited nothing both
    # carry a prompt describing work the episode does not show
    assert not _engaged("erase", 0.0044, 0.0044)
    assert not _engaged("language", 0.0, 0.0)
    # moving the pigment the wrong way is just as mislabelled as not at all
    assert not _engaged("erase", 0.004, 0.009)
    assert not _engaged("language", 0.009, 0.004)
    assert _engaged("erase", 0.0066, 0.0020)
    assert _engaged("language", 0.0, 0.0031)


def test_language_budget_respected():
    sheets = grid_paper_sheets(8)
    rng = np.random.default_rng(4)
    for i in range(100):
        _, prog = sample_scene(rng, sheets[i % 8], 12.0)
        assert prog["est_cost_s"] <= 12.0 + 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} invariant tests passed")
