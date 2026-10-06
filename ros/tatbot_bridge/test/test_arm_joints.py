"""/joint_states on the tatbot bus as `tatbot.arm-joints/1`, held to what stencild parses.

The example (joints 0.1..0.6 rad, carriage 0.012 m at 1.5 N, mode Position, calibration id "bundle"), and the
checks mirror stencild's decode: tatbot_bus::Envelope, Producer and Stamp deny unknown fields,
`transport::decode` requires the schema, and `JointsSample::parse` wants a configured arm, a
positive stamp, six finite joints and a finite carriage.
"""
import json
import re
import types
from pathlib import Path

import pytest
from tatbot_bridge import joints

REPO = Path(__file__).resolve().parents[3]
STENCILD = REPO / "rust" / "trackd" / "src" / "bin" / "stencild.rs"
NAMES = [f"right/joint_{i}" for i in range(6)] + ["right/left_carriage_joint"]
WALL = 1_790_000_000_123_456_789


def stencild_accepts(data: bytes, expected_schema: str = joints.SCHEMA) -> dict:
    """stencild's decode of one sample, as far as it reads it; raises AssertionError on a refusal."""
    envelope = json.loads(data)
    assert set(envelope) == {"schema", "producer", "stamp", "seq", "payload"}
    assert envelope["schema"] == expected_schema
    assert set(envelope["producer"]) == {"node", "pid", "sha", "run_id"}
    assert isinstance(envelope["producer"]["pid"], int) and envelope["producer"]["pid"] >= 0
    assert set(envelope["stamp"]) == {"mono_ns", "wall_ns", "basis"}
    assert all(isinstance(envelope["stamp"][k], int) and envelope["stamp"][k] >= 0 for k in ("mono_ns", "wall_ns"))
    assert isinstance(envelope["seq"], int) and envelope["seq"] >= 0
    p = envelope["payload"]
    arms = json.loads((REPO / "config" / "arms.json").read_text())["arms"] if (REPO / "config").is_dir() \
        else {"left": {}, "right": {}}
    assert p["arm"] in arms
    assert isinstance(p["measured_wall_ns"], int) and p["measured_wall_ns"] > 0
    assert len(p["joints"]) == 6 and all(isinstance(v, float) for v in p["joints"])
    assert p["carriage"] is not None   # pose_joints refuses a view without one
    assert set(p["carriage"]) == {"position_m", "effort_n"}
    assert p["calibration_id"] is None or isinstance(p["calibration_id"], str)
    return envelope


def sample(positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.012), efforts=(0.0,) * 6 + (1.5,), names=NAMES, **kw):
    return joints.payload("right", names, positions, efforts, kw.get("wall", WALL), kw.get("mode", "Position"),
                          kw.get("calibration", "bundle"))


def test_reference_example_encodes_identically():
    body = sample()
    assert body == {"arm": "right", "measured_wall_ns": WALL, "joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "carriage": {"position_m": 0.012, "effort_n": 1.5}, "mode": "Position",
                    "calibration_id": "bundle"}
    data = joints.envelope(body, joints.producer("arm-node", 4242, "0123abcd"), 7, 99)
    envelope = stencild_accepts(data)
    assert envelope["stamp"] == {"mono_ns": 99, "wall_ns": WALL, "basis": "host"}
    assert envelope["seq"] == 7 and envelope["payload"]["measured_wall_ns"] == envelope["stamp"]["wall_ns"]
    assert envelope["producer"] == {"node": "arm-node", "pid": 4242, "sha": "0123abcd", "run_id": "ros"}
    assert joints.topic("right") == "tatbot/session/ros/arm/right/joints"


def test_joint_order_follows_names_not_message_order():
    order = [6, 3, 0, 5, 1, 4, 2]      # JSB orders by interface, not by the arm's chain
    positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.012]
    efforts = [0.0] * 6 + [1.5]
    left = [f"left/joint_{i}" for i in range(6)] + ["left/left_carriage_joint"]
    body = sample(positions=[positions[i] for i in order] + [9.0] * 7,
                  efforts=[efforts[i] for i in order] + [9.0] * 7, names=[NAMES[i] for i in order] + left)
    assert body["joints"] == positions[:6] and body["carriage"] == {"position_m": 0.012, "effort_n": 1.5}


def test_incomplete_or_non_finite_states_do_not_ride():
    assert sample(names=NAMES[:6], positions=(0.0,) * 6) is None                # no carriage
    assert sample(efforts=()) is None                                             # no effort interface
    assert sample(positions=(0.0,) * 5 + (float("nan"), 0.0)) is None
    assert sample(wall=0) is None
    assert sample(calibration=None)["calibration_id"] is None


def test_mode_from_safety_state():
    state = types.SimpleNamespace
    assert joints.mode(None) == "Position"
    assert joints.mode(state(controller_error=False, landed=False)) == "Position"
    assert joints.mode(state(controller_error=False, landed=True)) == "Idle"
    assert joints.mode(state(controller_error=True, landed=True)) == "Fault"


def test_throttle_publishes_only_advanced_measurements():
    t = joints.Throttle()
    assert [t.take(ns) for ns in (5, 5, 4, 6, 10)] == [1, None, None, 2, 3]
    assert joints.PERIOD_S == 0.1    # <= 10 Hz


def test_calibration_id_from_registration(tmp_path):
    reg = tmp_path / "arm-registration-right-current.json"
    assert joints.calibration_id(reg) is None and joints.calibration_id(None) is None
    reg.write_text(json.dumps({"schema": "tatbot.arm-registration/1", "calibration_id": "b" * 64}))
    assert joints.calibration_id(reg) == "b" * 64
    reg.write_text("not json")
    assert joints.calibration_id(reg) is None


@pytest.mark.skipif(not STENCILD.is_file(), reason="stencild source is not in this tree (a deployed ros/ copy)")
def test_stencild_subscribes_this_key_and_schema():
    source = STENCILD.read_text()
    assert f'const JOINTS_SCHEMA: &str = "{joints.SCHEMA}";' in source
    key = re.search(r'const JOINTS_KEY: &str = "([^"]+)";', source).group(1)
    pattern = "^" + re.escape(key).replace(r"\*", "[^/]+") + "$"
    assert re.match(pattern, joints.topic("right")) and re.match(pattern, joints.topic("left"))
    assert "joints.len() == 6" in source
