"""The bridge's page_at: a choice it cannot make is refused, and otherwise the sheet nearest the point is the page from
the turn it is read there on, recorded so that a restart starts there."""
from __future__ import annotations

import json

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter  # noqa: E402
from tatbot_bridge import node as bridge_node  # noqa: E402
from tatbot_bridge import page, stack  # noqa: E402


def at(x, y=0.0):
    pose = np.eye(4)
    pose[:2, 3] = [x, y]
    return pose


def sample(*xs):
    """One measured turn reading a sheet at each x, the first the largest."""
    copies = [{"world_from_target": at(x).tolist(), "translation_sigma_m": 0.002, "rotation_sigma_rad": 0.02,
               "support": {"markers": {"read": 9}}} for x in xs[1:]]
    return {"source": "measured", "world_from_target": at(xs[0]), "stamp_ns": 5, "translation_sigma_m": 0.002,
            "rotation_sigma_rad": 0.02, "support": {"markers": {"read": 12}, "copies": copies}}


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "log_root", lambda: tmp_path)
    monkeypatch.setattr(stack, "world_from_arm_base", lambda cfg, arm: at(0.0, 0.2) if arm == "right" else None)
    monkeypatch.setattr(stack, "open_bus", lambda endpoint: (_ for _ in ()).throw(OSError("no bus in a test")))
    rclpy.init(args=["--ros-args", "-p", "page_source:=stencil", "-p", "arms:=right,left", "-p",
                     "bus_endpoint:=tcp/127.0.0.1:1"])
    node = bridge_node.Bridge()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def choose(node, text):
    return node._on_parameters([Parameter("page_at", Parameter.Type.STRING, text)])


def test_a_choice_names_a_registered_arm_and_a_point(bridge):
    assert "not an arm" in choose(bridge, "left,400,-100").reason
    assert not choose(bridge, "right,400").successful
    assert choose(bridge, "").successful and bridge.chooser.wanted is None   # '' chooses nothing


def test_the_chosen_sheet_is_the_page_from_the_turn_it_is_read_and_a_restart_starts_there(bridge, tmp_path):
    assert bridge._page_of(sample(0.3, 0.5)) is None   # two sheets, none chosen yet
    assert choose(bridge, "right,500,-190").successful   # the sheet at x 0.5 in the world, in right/base_link
    chosen = bridge._page_of(sample(0.3, 0.5))
    assert np.allclose(page.centre(chosen), [0.5, 0.0, 0.0]) and chosen["support"]["other_copies_m"] == [0.2]
    record = json.loads((tmp_path / "page-chosen-right.json").read_text())
    assert record["arm"] == "right" and np.allclose(page.page_place(tmp_path / "page-chosen-right.json", at(0.0, 0.2)),
                                                    [0.5, 0.0, 0.0])
    assert np.allclose(stack.page_chooser(bridge.stack, bridge.registrations).place, [0.5, 0.0, 0.0])
