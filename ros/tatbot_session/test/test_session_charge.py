"""A dipping cartridge's charge (tatbot_session.charge): the numpad's steps of the dip scale and their range, the
scales a resume starts from, source-path contact and the charge boundaries. Compiler/executor agreement and
changed scales are exercised through ArmExecutor in test_session_executor."""
import pytest
from tatbot_session import charge


def test_plus_shortens_and_minus_lengthens_the_line_per_dip_a_step_within_the_range():
    assert charge.stepped(1.0, "dip") == 0.9 and charge.stepped(1.0, "fewer_dips") == 1.1
    scale = 1.0
    for _ in range(20):
        scale = charge.stepped(scale, "dip")
    assert scale == charge.LOWEST
    for _ in range(40):
        scale = charge.stepped(scale, "fewer_dips")
    assert scale == charge.HIGHEST


def test_a_resume_starts_each_resource_at_its_last_scale_on_its_own_arm():
    rows = [{"event": "dip_scale", "arm": "right", "resource": "liner", "scale": 0.9},
            {"event": "pen_trim", "arm": "right", "resource": "liner", "trim_m": 0.0001},
            {"event": "dip_scale", "arm": "left", "resource": "liner", "scale": 1.5},
            {"event": "dip_scale", "arm": "right", "resource": "liner", "scale": 0.8},
            {"event": "dip_scale", "arm": "right", "resource": "mag", "scale": 1.2}]
    assert charge.scales(rows, "right") == {"liner": 0.8, "mag": 1.2}
    assert charge.budget_m({"dip": {"mm_per_dip": 200.0}}, 0.8) == pytest.approx(0.16)
    assert charge.budget_m({"dip": None}) is None


def test_a_stroke_that_starts_a_path_commits_to_all_of_it_and_one_entered_partway_to_its_own_rest():
    line = [[0.0, 0.0], [0.02, 0.0]]

    def stroke(path, lo, **extra):
        return {"op": "stroke", "points_m": line, "src": {"artwork_sha256": "a", "path": path, "arc_m": [lo, lo + 0.02]},
                **extra}

    ops = [stroke("p0", 0.0), stroke("p0", 0.02, continues=True), {"op": "dip", "reason": "replenish"},
           stroke("p0", 0.04), stroke("p1", 0.0), stroke("p0", 0.0), {"op": "stroke", "points_m": line}]
    assert charge.need_m(ops, 0) == pytest.approx(0.06)          # its sections across the dip
    assert charge.need_m(ops, 0, 0.005) == pytest.approx(0.015)  # resumed partway
    assert charge.need_m(ops, 1) == pytest.approx(0.02)          # pen-down from the stroke before
    assert charge.need_m(ops, 3) == pytest.approx(0.02)          # another path follows
    assert charge.need_m(ops, 5) == pytest.approx(0.02)          # a repeat of p0 starts again at its arc 0
    assert charge.need_m(ops, 6) == pytest.approx(0.02)          # no source: its own line


def test_charge_boundaries_keep_the_rounding_slack_and_never_dip_a_full_charge():
    budget = 0.030
    # A fresh charge may enter a long path; only an already used charge dips before it.
    assert not charge.dip_first(0.0, 0.040, budget)
    assert not charge.dip_first(0.010, 0.0209, budget)
    assert charge.dip_first(0.010, 0.0211, budget)
    assert charge.covered(0.010, 0.0189, budget)
    assert not charge.covered(0.010, 0.0191, budget)
    assert not charge.spent(0.010, 0.0209, budget)
    assert charge.spent(0.010, 0.0211, budget)
