"""The judge must move the right way when the drawing gets worse.

A scoreboard nobody has tried to fool is not a scoreboard. These tests degrade
a known design in ways whose sign is not in question — wrong place, wrong size,
stopped early, shaky, strokes missing — and assert the score falls every time.
Absolute values are the simulator's business; the SIGN is the judge's, and it is
the only part a checkpoint ranking depends on.

Torch + numpy + cv2, no render device:

    cd python/tatbot_sim && uv run python -m pytest tests/test_judge.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tatbot_sim.inkfield import InkField
from tatbot_sim.judge import (
    drop_strokes,
    intended_field_like,
    jitter_strokes,
    offset_strokes,
    scale_strokes,
    score_fields,
    strokes_from_plan_paths,
    truncate_strokes,
)
from tatbot_sim.strokes import Stroke
from tatbot_sim.surface import PlanarSurface

PEN_R = 0.0015  # 1.5 mm pen radius; the default tolerance follows it


def _surface(b: int = 1) -> PlanarSurface:
    center = torch.zeros(b, 3)
    center[:, 2] = 0.03
    return PlanarSurface(center, torch.eye(3).expand(b, 3, 3).contiguous())


def _field(surface, b: int = 1) -> InkField:
    return InkField(
        b, surface,
        pen_radius_m=torch.full((b,), PEN_R),
        laser_radius_m=torch.full((b,), PEN_R),
        ink_rgb=torch.zeros(b, 3),
    )


def _square(half: float = 0.03, n: int = 200) -> list[Stroke]:
    """A closed square, densely sampled — one continuous pen-down polyline."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    corners = np.array([[-half, -half], [half, -half], [half, half],
                        [-half, half], [-half, -half]], dtype=np.float32)
    seg = (t[:, None] * (len(corners) - 1))
    idx = np.clip(seg.astype(int), 0, len(corners) - 2)
    frac = seg - idx
    pts = corners[idx][:, 0] + frac * (corners[idx + 1][:, 0] - corners[idx][:, 0])
    return [Stroke(pts)]


def _line(y: float = 0.0, n: int = 200) -> list[Stroke]:
    """One horizontal stroke. Offsetting it in y moves every sample
    perpendicular to the line, which a square does not: two of a square's four
    sides slide ALONG themselves under an x-offset and their ink stays on the
    intended band, so a square understates a translation by about half."""
    return [Stroke(np.stack([np.linspace(-0.04, 0.04, n, dtype=np.float32),
                             np.full(n, y, dtype=np.float32)], axis=1))]


def _multi_stroke_design() -> list[Stroke]:
    """Four separate strokes, so dropping whole strokes is meaningful."""
    ys = [-0.03, -0.01, 0.01, 0.03]
    return [Stroke(np.stack([np.linspace(-0.04, 0.04, 120, dtype=np.float32),
                             np.full(120, y, dtype=np.float32)], axis=1)) for y in ys]


def _raster(surface, field, strokes: list[Stroke]) -> torch.Tensor:
    """Rasterize one env's strokes through the field's own kernel."""
    return intended_field_like(field, surface, [strokes], torch.ones(1)).clone()


def _score(surface, field, drawn: list[Stroke], intended: list[Stroke], tol_m=PEN_R):
    return score_fields(
        _raster(surface, field, drawn), _raster(surface, field, intended),
        texel_per_m=surface.texel_per_m, tolerance_m=tol_m,
    )[0]


@pytest.fixture
def rig():
    surface = _surface()
    return surface, _field(surface)


def test_perfect_drawing_scores_one(rig):
    surface, field = rig
    design = _square()
    score = _score(surface, field, design, design)
    assert score.f1 == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.chamfer_drawn_to_intended_mm == pytest.approx(0.0)
    assert score.coverage_ratio == pytest.approx(1.0)
    assert not score.blank and not score.degenerate


def test_blank_sheet_scores_zero_and_says_so(rig):
    surface, field = rig
    intended = _raster(surface, field, _square())
    score = score_fields(torch.zeros_like(intended), intended,
                         texel_per_m=surface.texel_per_m, tolerance_m=PEN_R)[0]
    assert score.f1 == 0.0
    assert score.blank and not score.degenerate
    assert score.as_dict()["chamfer_drawn_to_intended_mm"] is None
    assert score.as_dict()["chamfer_intended_to_drawn_mm"] is None


def test_empty_design_is_degenerate_not_a_crash(rig):
    surface, field = rig
    drawn = _raster(surface, field, _square())
    score = score_fields(drawn, torch.zeros_like(drawn),
                         texel_per_m=surface.texel_per_m, tolerance_m=PEN_R)[0]
    assert score.degenerate
    assert np.isnan(score.f1)


def test_shape_mismatch_is_rejected(rig):
    surface, field = rig
    drawn = _raster(surface, field, _square())
    with pytest.raises(ValueError, match="field shapes differ"):
        score_fields(drawn, drawn[:, :10, :10],
                     texel_per_m=surface.texel_per_m, tolerance_m=PEN_R)


# --- monotonicity: the whole point ------------------------------------------

def test_score_falls_monotonically_with_offset(rig):
    surface, field = rig
    design = _square()
    scores = [_score(surface, field, offset_strokes(design, d, 0.0), design).f1
              for d in (0.0, 0.001, 0.002, 0.004, 0.008)]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > scores[-1]


def test_score_falls_monotonically_with_scale_error(rig):
    surface, field = rig
    design = _square()
    scores = [_score(surface, field, scale_strokes(design, f), design).f1
              for f in (1.0, 1.05, 1.15, 1.35)]
    assert scores == sorted(scores, reverse=True), scores


def test_recall_falls_monotonically_with_truncation(rig):
    surface, field = rig
    design = _square()
    results = [_score(surface, field, truncate_strokes(design, k), design)
               for k in (1.0, 0.75, 0.5, 0.25)]
    recalls = [r.recall for r in results]
    assert recalls == sorted(recalls, reverse=True), recalls
    # a truncated drawing is still ON the line where it exists — precision
    # must NOT collapse, or the judge is punishing the wrong failure
    assert min(r.precision for r in results) > 0.9


def test_score_falls_monotonically_with_jitter(rig):
    surface, field = rig
    design = _square()
    scores = [_score(surface, field, jitter_strokes(design, s, seed=7), design).f1
              for s in (0.0, 0.0005, 0.0015, 0.004)]
    assert scores == sorted(scores, reverse=True), scores


def test_recall_falls_when_whole_strokes_go_missing(rig):
    surface, field = rig
    design = _multi_stroke_design()
    recalls = [_score(surface, field, drop_strokes(design, k, seed=3), design).recall
               for k in (1.0, 0.75, 0.5, 0.25)]
    assert recalls == sorted(recalls, reverse=True), recalls


def test_overdrawing_shows_up_as_lost_precision_not_lost_recall(rig):
    """Drawing the design plus a stray scribble keeps recall and costs precision."""
    surface, field = rig
    design = _square()
    stray = Stroke(np.stack([np.linspace(-0.05, 0.05, 150, dtype=np.float32),
                             np.full(150, 0.06, dtype=np.float32)], axis=1))
    score = _score(surface, field, design + [stray], design)
    assert score.recall == pytest.approx(1.0)
    assert score.precision < 0.9
    assert score.coverage_ratio > 1.0


# --- the numbers have to mean what they say ---------------------------------

def test_chamfer_reports_the_offset_it_was_given(rig):
    """A perpendicular translation must read as the distance it really is.

    Chamfer measures distance to the intended BAND, not to its centreline, so a
    stroke of radius r pushed d away sits a mean (d - r) off it. Both terms are
    stated here because a judge that silently reported centreline distance would
    overstate every offset by one line radius.
    """
    surface, field = rig
    design = _line()
    texel_mm = 1000.0 / surface.texel_per_m
    measured = [_score(surface, field, offset_strokes(design, 0.0, d), design)
                .chamfer_drawn_to_intended_mm for d in (0.004, 0.006, 0.008)]
    assert measured == sorted(measured), measured
    for d, got in zip((0.004, 0.006, 0.008), measured, strict=True):
        expected_mm = (d - PEN_R) * 1000.0
        assert got == pytest.approx(expected_mm, abs=3 * texel_mm), (d, got, expected_mm)


def test_tolerance_is_the_knob_it_claims_to_be(rig):
    """Tightening the tolerance on a fixed drawing must lower the score.

    The ladder, not a single threshold: for a d-offset stroke of radius r the
    score at tolerance t is analytically (r + t - d/2) / r-ish, so a lone
    assertion is easy to pin exactly on the boundary value and learn nothing.
    """
    surface, field = rig
    design = _line()
    drawn = offset_strokes(design, 0.0, 0.003)
    scores = [_score(surface, field, drawn, design, tol_m=t).f1
              for t in (0.006, 0.003, 0.0015, 0.0005)]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > 0.9, "a tolerance wider than the error should forgive it"
    assert scores[-1] < 0.3, "a tolerance far below the error should not"


def test_reference_uses_the_fields_own_kernel(rig):
    """intended_field_like must not invent its own line weight."""
    surface, _ = rig
    thin, thick = _field(surface), _field(surface)
    thick.pen_radius_m = torch.full((1,), PEN_R * 3)
    thick = InkField(1, surface, pen_radius_m=torch.full((1,), PEN_R * 3),
                     laser_radius_m=torch.full((1,), PEN_R),
                     ink_rgb=torch.zeros(1, 3))
    design = _square()
    assert _raster(surface, thick, design).sum() > _raster(surface, thin, design).sum() * 1.5


def test_scores_are_per_env_across_a_batch():
    """A batch must not average: env 0 drew it, env 1 did not."""
    surface = _surface(2)
    field = _field(surface, 2)
    design = _square()
    intended = intended_field_like(field, surface, [design, design], torch.ones(2))
    drawn = intended.clone()
    drawn[1] = 0.0
    scores = score_fields(drawn, intended,
                          texel_per_m=surface.texel_per_m, tolerance_m=PEN_R)
    assert scores[0].f1 == pytest.approx(1.0)
    assert scores[1].f1 == 0.0 and scores[1].blank


# --- plan.paths is three shapes, not one ------------------------------------

def test_plan_paths_list_of_strokes_round_trips():
    design = _multi_stroke_design()
    path = [[[float(v) for v in pt] for pt in s.points] for s in design]
    strokes = strokes_from_plan_paths(path)
    assert len(strokes) == len(design)
    for got, want in zip(strokes, design, strict=True):
        np.testing.assert_allclose(got.points, want.points)


def test_plan_paths_flat_polyline_is_one_stroke_not_many():
    """The maze/squiggle form. Read as a list of strokes it would become a pile
    of two-point fragments that still scores, wrongly."""
    design = _line()[0]
    path = [[float(v) for v in pt] for pt in design.points]
    strokes = strokes_from_plan_paths(path)
    assert len(strokes) == 1
    np.testing.assert_allclose(strokes[0].points, design.points)


def test_plan_paths_empty_means_nothing_was_meant_to_be_drawn():
    assert strokes_from_plan_paths([]) == []
    assert strokes_from_plan_paths(None) == []


def test_flat_and_nested_forms_score_identically(rig):
    """Whatever the planner wrote, the same design must score the same."""
    surface, field = rig
    design = _line()
    flat = strokes_from_plan_paths([[float(v) for v in pt] for pt in design[0].points])
    nested = strokes_from_plan_paths([[[float(v) for v in pt] for pt in design[0].points]])
    assert _score(surface, field, flat, design).f1 == pytest.approx(
        _score(surface, field, nested, design).f1)
