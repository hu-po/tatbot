"""The inspection's lift off the page before its views and close-up (client._inspect_lift), its holds' wrist, and the
guard its views are chosen by (client._inspect_self_gap)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tatbot_motion
from tatbot_description import arm_model
from tatbot_session import client, config, inspect, ready


def _lift(monkeypatch, height_m):
    planned = []
    monkeypatch.setattr(tatbot_motion, "plan_lift", lambda **kw: planned.append(kw["distance_m"]) or "traj")
    monkeypatch.setattr(tatbot_motion, "to_knots", lambda traj, rate: traj)
    tcp = np.eye(4)
    tcp[2, 3] = height_m
    kin = SimpleNamespace(fk=lambda q: tcp)
    io = SimpleNamespace(execute=lambda knots: ("ok", None, ""))
    client._inspect_lift(io, kin, np.zeros(7), np.eye(4), {}, SimpleNamespace(lift=0.040))
    return planned


def test_a_pen_already_at_the_lift_height_is_not_lifted_again(monkeypatch):
    """2026-10-18: after the views the pen stood at the lift height to IK round-off, and the close-up's lift of
    a few micrometres stalled the pen-up profile ("pen-up profile stalls"): no close-up was taken."""
    assert _lift(monkeypatch, 0.040 - 2e-6) == []
    assert _lift(monkeypatch, 0.0395) == []


def test_a_pen_under_the_lift_height_is_lifted_to_it(monkeypatch):
    assert np.allclose(_lift(monkeypatch, 0.010), [0.030])


def test_a_run_whose_page_set_up_never_finished_is_not_looked_at(tmp_path, monkeypatch, capsys):
    (tmp_path / "page.json").write_text("{}")
    (tmp_path / "program.json").write_text("{}")
    monkeypatch.setattr(inspect, "newest_run", lambda root, name: tmp_path)
    assert client._inspect(None, SimpleNamespace(logs=str(tmp_path), run="", arm="right")) == 2
    assert "never finished" in capsys.readouterr().err


def test_a_closeup_hold_turning_the_wrist_out_of_range_moves_nothing(monkeypatch):
    """2026-11-05: a failed run's page stood on its edge; its hold would have turned the wrist into its cables."""
    monkeypatch.setattr(ready, "solve_ik_seeded", lambda kin, target, q: np.array([0, 0, 0, -0.7, -0.6, -1.8, 0.0]))
    sent = []
    io = SimpleNamespace(measured=lambda: np.zeros(7), execute=lambda knots: sent.append(knots))
    with pytest.raises(RuntimeError, match="outside"):
        client._closeup_hold(io, SimpleNamespace(arm_model=arm_model.load("trossen-wxai")), None, np.eye(4), (0.0, 0.0),
                             np.eye(4), {}, None, {}, {}, None, None, 0)
    assert sent == []


def test_the_views_guard_is_the_inspected_arms_not_the_installed_stacks(tmp_path, monkeypatch):
    """2026-11-10: the xArm's look at the chicano TATBOT read the installed stack.yaml (`arms: [right]`, both arms'
    registrations on the ros node), built the WXAI follower's guard and failed on the xArm's rest (KeyError 'xarm')."""
    pytest.importorskip("coal")
    registrations = {}
    for arm in ("right", "xarm"):
        path = tmp_path / f"arm-registration-{arm}-current.json"
        path.write_text(json.dumps({"world_from_arm_base": np.eye(4).tolist()}))
        registrations[arm] = str(path)
    monkeypatch.setattr(config, "load_stack", lambda path=None: {"arms": ["right"], "registration": registrations})
    self_gap = client._inspect_self_gap("xarm")
    assert self_gap(np.array(arm_model.for_arm("xarm").rest)) > 0.0
    assert self_gap(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.31, 0.0])) < -0.05   # R7's core folded into the base
