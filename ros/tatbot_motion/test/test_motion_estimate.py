"""Duration estimates use the drawing law and never hide unmeasured workflow time."""
from copy import deepcopy

import numpy as np
import pytest
from tatbot_motion import load_motion
from tatbot_motion.estimate import estimate_duration
from tatbot_motion.plan import _draw_leg


def stroke(points, **extra):
    return {"op": "stroke", "points_m": points, "continues": False, **extra}


def test_estimate_accounts_for_corner_easing_and_exact_planner_ticks():
    motion = load_motion()
    points = np.array([[0, 0, 0], [.01, 0, 0], [.01, .01, 0]])
    estimate = estimate_duration([stroke(points.tolist())], .0035, 0, motion)
    leg = _draw_leg(points, 0, np.eye(3), .0035, motion)
    assert estimate["components_s"]["drawing"] == pytest.approx(len(leg.p) / motion["control_rate_hz"])
    assert estimate["components_s"]["drawing"] > .02 / .0035
    assert estimate["total_s"] == estimate["modeled_s"]
    assert estimate["unknown_operations"] == {}


def test_unmeasured_dips_and_colour_changes_are_unknown_not_free():
    ops = [stroke([[0, 0], [.02, 0]]), {"op": "dip"}, {"op": "dip"},
           {"op": "pause", "reason": "swap pen: red"}]
    motion = load_motion()
    estimate = estimate_duration(ops, .0035, 0, motion)
    assert estimate["unknown_operations"] == {"dip": 2, "pen_change": 1}
    assert estimate["total_s"] is None
    calibrated = deepcopy(motion)
    calibrated["duration_estimate"] = {"dip_s": 7, "pen_change_s": 11}  # synthetic measured-input fixture
    known = estimate_duration(ops, .0035, 0, calibrated)
    assert known["total_s"] == pytest.approx(estimate["modeled_s"] + 25)
    assert known["motion_sha256"] != estimate["motion_sha256"]


def test_closed_paths_include_closing_edge_and_chunks_include_each_ease():
    motion = load_motion()
    points = [[0, 0], [.01, 0], [.01, .01]]
    a = estimate_duration([stroke(points, closed=True)], .0035, 0, motion)
    b = estimate_duration([stroke(points + [points[0]])], .0035, 0, motion)
    assert a == b
    whole = estimate_duration([stroke([[0, 0], [.2, 0]])], .0035, 0, motion)
    chunks = [stroke([[i*.02, 0], [(i+1)*.02, 0]], continues=i > 0) for i in range(10)]
    split = estimate_duration(chunks, .0035, 0, motion)
    assert split["components_s"]["drawing"] > whole["components_s"]["drawing"]
    assert split["components_s"]["settle"] == whole["components_s"]["settle"]


def test_the_touchdown_dwell_is_the_pen_modes():
    """Each touchdown dwells as its pen mode says: the press settles, a riding pen does not."""
    motion = load_motion()
    motion["pen"]["mode"] = "press"
    ops = [stroke([[0, 0], [.01, 0]]), stroke([[0, .01], [.01, .01]])]
    riding = deepcopy(motion)
    riding["pen"]["mode"] = "ride"
    pressed, ridden = (estimate_duration(ops, .0035, 0, m)["components_s"]["settle"] for m in (motion, riding))
    assert pressed == pytest.approx(2 * motion["pen"]["press"]["settle_s"])
    assert ridden == pytest.approx(2 * motion["pen"]["ride"]["settle_s"])
