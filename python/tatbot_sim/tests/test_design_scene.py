"""A portable design drawn by the factory: `--design FILE`.

The design is built here the way `tatbot design place` builds one (through the
browser reader, so Node and web/inkmap's dependencies are needed, as for
test_design_build), then mapped onto a substrate record without touching the
process's own fitted substrate. Nothing renders.

    uv run pytest -q tests/test_design_scene.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tatbot_sim import design_scene, tools
from tatbot_sim.config import DRConfig
from tatbot_sim.inkmap.design_build import artwork_from_svg, design_from_artwork, file_source
from tatbot_sim.inkmap.design_strokes import design_strokes

REPO = Path(__file__).resolve().parents[3]
PAD_CANVAS = (0.1905, 0.2794)
CYLINDER_CANVAS = (0.1905, 0.20028)
CYLINDER_RADIUS = 0.0425
# A notched square: no mirror symmetry, so a swapped axis would show as a flipped notch.
NOTCHED = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
           '<path d="M3 1 H9 V9 H1 V4 Z"/></svg>')


def _sub(name):
    return tools.registry().load_substrate(name, REPO)


@pytest.fixture(scope="module")
def record() -> dict:
    return artwork_from_svg(NOTCHED, name="Notched square", size_mm=(12.0, 12.0),
                            source=file_source(NOTCHED, path="notched.svg"))


def _write(tmp_path: Path, design: dict, name: str = "design.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(design))
    return path


def _centre(strokes) -> np.ndarray:
    everything = np.concatenate([np.asarray(s) for s in strokes])
    return (everything.min(axis=0) + everything.max(axis=0)) / 2


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_a_plane_design_lands_where_inkmap_put_it(record, tmp_path):
    design = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS, anchor_uv_m=(0.02, -0.03))
    scene = design_scene.load(_write(tmp_path, design), _sub("paper_pad"))
    assert scene.chart_to_canvas == design_scene.IDENTITY
    assert np.allclose(_centre(scene.strokes_m), [0.02, -0.03], atol=0.0005)
    assert scene.design_sha256 == design["content_sha256"]
    assert scene.artwork_names == ("Notched square",)
    strokes, program = scene.sample(np.random.default_rng(0), None, 1e9)
    assert program["source_sha256"] == design["content_sha256"]
    assert program["family"] == "portable-design"
    assert program["prompt"] == "draw the notched square tattoo design"
    assert program["offset_m"] == [0.0, 0.0]
    assert np.allclose(np.concatenate([s.points for s in strokes]), np.concatenate(scene.strokes_m), atol=1e-6)


def test_a_cylinder_design_turns_a_quarter_and_keeps_its_hand(record, tmp_path):
    design = design_from_artwork(record, kind="cylinder", radius_m=CYLINDER_RADIUS, canvas_m=CYLINDER_CANVAS,
                                 anchor_uv_m=(0.03, 0.01))
    scene = design_scene.load(_write(tmp_path, design), _sub("paper_cylinder"))
    assert scene.chart_to_canvas == design_scene.CYLINDER_QUARTER_TURN
    # u (along the axis) is now +y and v (arc from the crest) is -x
    assert np.allclose(_centre(scene.strokes_m), [-0.01, 0.03], atol=0.0005)
    chart = [np.asarray(s) / 1000 for s in design_strokes(design)["strokes_mm"]]
    for before, after in zip(chart, scene.strokes_m, strict=True):
        assert np.sign(_signed_area(before)) == np.sign(_signed_area(after))
        assert np.allclose(np.linalg.norm(before[1:] - before[:-1], axis=1),
                           np.linalg.norm(after[1:] - after[:-1], axis=1))


def test_the_surface_kinds_and_radius_must_match(record, tmp_path):
    cylinder = design_from_artwork(record, kind="cylinder", radius_m=CYLINDER_RADIUS, canvas_m=CYLINDER_CANVAS)
    with pytest.raises(design_scene.DesignSceneError, match="flat sheet"):
        design_scene.load(_write(tmp_path, cylinder, "c.json"), _sub("paper_pad"))
    plane = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS)
    with pytest.raises(design_scene.DesignSceneError, match="rigid cylinder"):
        design_scene.load(_write(tmp_path, plane, "p.json"), _sub("paper_cylinder"))
    thinner = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    with pytest.raises(design_scene.DesignSceneError, match="radius"):
        design_scene.load(_write(tmp_path, thinner, "t.json"), _sub("paper_cylinder"))


def test_an_authored_placement_off_the_substrate_is_refused(record, tmp_path):
    # a legal chart in Inkmap's terms, wider than the silicone skin it is asked to go on
    wide = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS, anchor_uv_m=(0.08, 0.0))
    with pytest.raises(design_scene.DesignSceneError, match="leaves"):
        design_scene.load(_write(tmp_path, wide), _sub("silicon_skin"))
    scene = design_scene.load(_write(tmp_path, wide), _sub("silicon_skin"), placement="sampled")
    strokes, program = scene.sample(np.random.default_rng(1), None, 1e9)
    assert np.allclose(_centre([s.points for s in strokes]), program["offset_m"], atol=0.0005)


def test_sampled_placement_is_the_crest_first_then_the_reach_mask(record, tmp_path):
    from tatbot_sim.expert import ReachMask

    design = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS, anchor_uv_m=(0.05, 0.05))
    scene = design_scene.load(_write(tmp_path, design), _sub("paper_pad"), placement="sampled")
    everywhere = ReachMask(np.ones((27, 21), bool), *PAD_CANVAS)
    strokes, program = scene.sample(np.random.default_rng(2), None, 1e9, reachable=everywhere)
    assert program["offset_m"] == [0.0, 0.0]
    assert np.allclose(_centre([s.points for s in strokes]), [0.0, 0.0], atol=0.0005)
    nowhere = ReachMask(np.zeros((27, 21), bool), *PAD_CANVAS)
    with pytest.raises(SystemExit, match="cannot be held normal"):
        scene.sample(np.random.default_rng(2), None, 1e9, reachable=nowhere)
    authored = design_scene.load(_write(tmp_path, design), _sub("paper_pad"))
    with pytest.raises(SystemExit, match="authored placement"):
        authored.sample(np.random.default_rng(2), None, 1e9, reachable=nowhere)


def test_the_episode_budget_and_the_flags_are_checked_before_anything_builds(record, tmp_path):
    from tatbot_sim.generate import Args

    design = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS)
    path = str(_write(tmp_path, design))
    pad = _sub("paper_pad")
    scene = design_scene.from_args(Args(out_dir="x", design=path, horizon=9000), pad)
    assert scene is not None and scene.width_mm == pytest.approx(0.3, abs=0.02)
    assert design_scene.from_args(Args(out_dir="x"), pad) is None
    with pytest.raises(SystemExit, match="--horizon"):
        design_scene.from_args(Args(out_dir="x", design=path, horizon=120), pad)
    with pytest.raises(SystemExit, match="--task spiral"):
        design_scene.from_args(Args(out_dir="x", design=path, task="spiral"), pad)
    with pytest.raises(SystemExit, match="--scenario"):
        design_scene.from_args(Args(out_dir="x", design=path, scenario="s.json"), pad)
    with pytest.raises(SystemExit, match="own stroke width"):
        design_scene.from_args(Args(out_dir="x", design=path, artwork_width_mm=1.0), pad)
    with pytest.raises(SystemExit, match="not a tatbot.inkmap-design"):
        design_scene.from_args(Args(out_dir="x", design=str(_write(tmp_path, {"schema": "x"}, "bad.json"))), pad)
    assert design_scene.width_mm_for(scene, Args(out_dir="x")) == scene.width_mm
    assert design_scene.width_mm_for(None, Args(out_dir="x")) == 0.3
    assert design_scene.sampler(None) is None and design_scene.summary(None) is None


def test_plan_batch_draws_the_design(record, tmp_path):
    import torch
    from tatbot_sim.planning import plan_batch
    from tatbot_sim.surface import PlanarSurface
    from tatbot_sim.textures import grid_paper_sheets

    design = design_from_artwork(record, kind="plane", canvas_m=PAD_CANVAS, anchor_uv_m=(0.01, 0.02))
    scene = design_scene.load(_write(tmp_path, design), _sub("paper_pad"))
    surface = PlanarSurface(torch.tensor([[0.29, 0.0, 0.03]]), torch.eye(3)[None])
    dr = DRConfig()
    dr.approach.prob = 0.0
    plan = plan_batch(np.random.default_rng(0), grid_paper_sheets(1), surface, task="artwork",
                      horizon=6000, num_envs=1, dr=dr, draw_clearance=0.004, task_name="", maze_task_name="",
                      artwork_sampler=scene.sample)
    assert plan.kinds == ["artwork"]
    assert plan.programs[0]["source_sha256"] == design["content_sha256"]
    assert plan.programs[0]["portable_design"]["placement"] == "authored"
    assert plan.tasks[0] == "draw the notched square tattoo design"
    drawn = np.concatenate([np.asarray(s) for s in plan.paths[0]])
    assert np.allclose(drawn, np.concatenate(scene.strokes_m), atol=1e-6)
