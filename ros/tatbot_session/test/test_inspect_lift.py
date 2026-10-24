"""The inspection's lift off the page before its views and close-up (client._inspect_lift)."""
from types import SimpleNamespace

import numpy as np
import tatbot_motion
from tatbot_session import client


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
