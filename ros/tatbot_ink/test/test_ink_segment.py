"""Corner cuts and arc addresses (ported from scripts/tests/test_pen_path.py and test_stroke_material.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from tatbot_ink.errors import CompileError
from tatbot_ink.segment import CUT_WINDOW_FRACTION, arc_lengths, cut_points, resample, split

# 5 mm legs with 90-degree corners every 5 mm of arc; 40 mm long.
ZIGZAG = np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.005], [0.010, 0.005], [0.010, 0.010],
                   [0.015, 0.010], [0.015, 0.015], [0.020, 0.015], [0.020, 0.020]])


def test_cuts_land_on_corners_without_changing_count_or_budget():
    arc = arc_lengths(ZIGZAG)
    length = float(arc[-1])
    for budget in (0.0195, 0.009, 0.0065):
        count = int(math.ceil(length / budget))
        cuts = cut_points(ZIGZAG, arc, count, budget)
        uniform = np.linspace(0.0, length, count + 1)
        assert len(cuts) == count + 1 and cuts[0] == 0.0 and cuts[-1] == pytest.approx(length)
        assert np.all(np.diff(cuts) > 0) and np.all(np.diff(cuts) <= budget + 1e-12)
        assert np.all(np.abs(cuts - uniform) <= CUT_WINDOW_FRACTION * budget + 1e-12)
        moved = [c for c, u in zip(cuts[1:-1], uniform[1:-1], strict=True) if not np.isclose(c, u, rtol=0, atol=1e-12)]
        assert moved, f"no cut moved at budget {budget}"
        assert all(np.isclose(arc[1:-1], c).any() for c in moved), "a moved cut lies on a corner"
        np.testing.assert_array_equal(cuts, cut_points(ZIGZAG.copy(), arc.copy(), count, budget))
    line = np.array([[0.0, 0.0], [0.03, 0.0]])
    np.testing.assert_allclose(cut_points(line, np.array([0.0, 0.03]), 3, 0.011), np.linspace(0.0, 0.03, 4))
    # A tessellated arc turns under 5 degrees per vertex: no corner, uniform cuts.
    angles = np.linspace(0.0, np.pi / 2, 64)
    curve = np.column_stack([0.02 * np.cos(angles), 0.02 * np.sin(angles)])
    arc = arc_lengths(curve)
    np.testing.assert_allclose(cut_points(curve, arc, 3, 0.012), np.linspace(0.0, arc[-1], 4))


def test_split_keeps_addresses_and_shares_cut_points():
    stroke = np.array([[0., 0.], [.005, 0.], [.005, 0.], [.005, .005], [.01, .005]])
    whole = split(stroke, 0.02)
    assert len(whole) == 1 and whole[0][1] == (0.0, pytest.approx(.015))
    pieces = split(stroke, .006)
    assert len(pieces) == 3
    assert pieces[0][1][0] == 0.0 and pieces[-1][1][1] == pytest.approx(.015)
    for (a, (_, a_hi)), (b, (b_lo, _)) in zip(pieces[:-1], pieces[1:], strict=True):
        np.testing.assert_array_equal(a[-1], b[0])
        assert a_hi == b_lo
    for points, (lo, hi) in pieces:
        assert float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) == pytest.approx(hi - lo)
        assert hi - lo <= .006 + 1e-12
        # The piece's ends are the source polyline at its arc address.
        np.testing.assert_allclose(points[[0, -1]], resample(stroke[[0, 1, 3, 4]], [lo, hi]), atol=1e-15)
    # Nested addresses: a piece of a piece keeps the source arc.
    nested = split(pieces[1][0], .002, pieces[1][1][0])
    assert nested[0][1][0] == pieces[1][1][0] and nested[-1][1][1] == pytest.approx(pieces[1][1][1])
    assert [p[1] for p in split(stroke, .006)] == [p[1] for p in pieces]


def test_split_refuses_what_it_cannot_address():
    with pytest.raises(CompileError, match="no length"):
        split(np.array([[0., 0.], [0., 0.]]), .01)
    with pytest.raises(CompileError, match="positive"):
        split(np.array([[0., 0.], [.01, 0.]]), 0.0)
