"""The arm's measured joints on the tatbot bus as `tatbot.arm-joints/1`, the topic stencild poses its
wrist views by (rust/trackd stencild: JOINTS_KEY `tatbot/session/*/arm/*/joints`, JointsSample).

The payload: the six revolute joints in driver order (rad), the carriage's position (m) and
external effort (N), the driver mode, the adopted registration's calibration id (null without one),
and the measurement's wall-clock stamp, both in the payload and as the envelope stamp. At most one
sample per PERIOD_S per arm, and only when the measurement advanced: stencild keeps 64 samples per
arm (6.4 s at this rate) and pairs a wrist exposure with the nearest one.

Pure Python, no ROS: the node and the tests share it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tatbot_description import names

SCHEMA = "tatbot.arm-joints/1"
SESSION = "ros"          # the key's session segment: stencild subscribes every session
PERIOD_S = 0.1           # the topic moves at most ten times a second
STAMP_BASIS = "host"
# SafetyState -> the driver mode stencild reads (tatbot_arm::Mode): the stack's driver runs
# position mode while active and idles only when landed.
MODE_FAULT, MODE_IDLE, MODE_POSITION = "Fault", "Idle", "Position"


def topic(arm: str, session: str = SESSION) -> str:
    return f"tatbot/session/{session}/arm/{arm}/joints"


def producer(node: str, pid: int, sha: str, run_id: str = SESSION) -> dict:
    """tatbot_bus::Producer: exactly these four fields (the bus refuses any other)."""
    return {"node": node, "pid": int(pid), "sha": sha, "run_id": run_id}


def mode(safety) -> str:
    """The driver mode from the arm's SafetyState (any object with controller_error and landed), or
    Position before the first one."""
    if safety is None:
        return MODE_POSITION
    if safety.controller_error:
        return MODE_FAULT
    return MODE_IDLE if safety.landed else MODE_POSITION


def calibration_id(path) -> str | None:
    """The adopted registration's calibration_id, or None when there is no readable one."""
    if path is None:
        return None
    try:
        value = json.loads(Path(path).expanduser().read_text()).get("calibration_id")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def payload(arm: str, joint_names, positions, efforts, measured_wall_ns: int, mode_: str,
            calibration: str | None) -> dict | None:
    """One arm's `tatbot.arm-joints/1` payload from a sensor_msgs/JointState's name, position and
    effort arrays, or None when the message does not carry every joint of the arm finitely."""
    index = {name: i for i, name in enumerate(joint_names)}
    wanted = names.joint_names(arm)
    if measured_wall_ns <= 0 or any(name not in index for name in wanted):
        return None
    try:
        q = [float(positions[index[name]]) for name in wanted]
        carriage_effort = float(efforts[index[wanted[-1]]])
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (*q, carriage_effort)):
        return None
    return {
        "arm": arm,
        "measured_wall_ns": int(measured_wall_ns),
        "joints": q[:6],
        "carriage": {"position_m": q[6], "effort_n": carriage_effort},
        "mode": mode_,
        "calibration_id": calibration,
    }


def envelope(body: dict, producer_: dict, seq: int, mono_ns: int) -> bytes:
    """tatbot_bus::Envelope as JSON bytes; the stamp's wall_ns is the measurement's."""
    return json.dumps({
        "schema": SCHEMA,
        "producer": producer_,
        "stamp": {"mono_ns": int(mono_ns), "wall_ns": body["measured_wall_ns"], "basis": STAMP_BASIS},
        "seq": int(seq),
        "payload": body,
    }).encode()


class Throttle:
    """At most one sample per PERIOD_S, and only a newer measurement than the last."""

    def __init__(self):
        self.published_ns = 0
        self.seq = 0

    def take(self, measured_wall_ns: int) -> int | None:
        """The next seq when this measurement should ride, else None."""
        if measured_wall_ns <= self.published_ns:
            return None
        self.published_ns = measured_wall_ns
        self.seq += 1
        return self.seq
