"""What ArmExecutor reaches outside itself, as ports: the arm (ArmPort, tatbot_session.arm.ArmIO under ROS), the node
(NodePort), the clock (Clock) and the wrist's sensing of the page (Sensing). The rig and fake hardware run them on
ROS and the wall clock; a run that is not ROS drives the same executor through its own ports. Pure Python: no ROS,
no OpenCV.

Event and Phase carry tatbot_interfaces' Event.KIND_* and Draw.Feedback.PHASE_* values, which a ROS-suite test pins
to the generated messages (test_ports.py).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tatbot_session import rules


class Event:
    """tatbot_interfaces/msg/Event KIND_*."""
    KIND_RUN_START, KIND_RUN_END = "run_start", "run_end"
    KIND_OP_SENT, KIND_OP_DONE, KIND_OP_ABORTED = "op_sent", "op_done", "op_aborted"
    KIND_REPLAN, KIND_PAUSE, KIND_LATCHED, KIND_RELEASED = "replan", "pause", "latched", "released"
    KIND_UNCERTAIN, KIND_DECISION, KIND_TOUCH, KIND_PAGE = "uncertain", "decision", "touch", "page"
    KIND_LANDED, KIND_MACHINE, KIND_PEN_TRIM, KIND_DIP, KIND_ERROR = "landed", "machine", "pen_trim", "dip", "error"


class Phase:
    """tatbot_interfaces/action/Draw Feedback PHASE_*."""
    PHASE_PLAN, PHASE_TRAVEL, PHASE_DESCEND, PHASE_SETTLE, PHASE_DRAW = 1, 2, 3, 4, 5
    PHASE_LIFT, PHASE_TOUCH, PHASE_PAUSED, PHASE_LATCHED, PHASE_LANDING = 6, 7, 8, 9, 10


class Clock(Protocol):
    """The session's time: now() in seconds on its own base, utc() in Unix seconds, sleep, and a wait on a held
    threading.Condition until `predicate` or `timeout_s`, polling every `poll_s` where the clock cannot be woken."""
    def now(self) -> float: ...
    def utc(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def wait(self, cv, predicate, timeout_s: float, poll_s: float = 0.05) -> bool: ...


class WallClock:
    """The rig's and fake hardware's clock: monotonic now, Unix utc, the condition's own timed wait."""

    def now(self) -> float:
        return time.monotonic()

    def utc(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def wait(self, cv, predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
        return cv.wait_for(predicate, timeout=timeout_s)


WALL = WallClock()


@dataclass(frozen=True)
class GaugeRead:
    """One gauge hold's reading: each frame's height of the tip over the surface along normal_base (m), the surface's
    point under the tip and its normal (base), the skin about it (N x 3 base; empty on a flat page), the raw
    frames (real wrist only; None otherwise): ((colours, depths, meta[, every colour taken]), the gauge's frames, its
    calibration), the print's markers read in every colour taken, and each further camera's
    (page.locate.cameras: {view, camera, shots, frames, heights_m, used}, used when its heights joined these)."""
    heights_m: list
    point_base: np.ndarray
    normal_base: np.ndarray
    patch_base: np.ndarray
    raw: tuple | None = None
    further: tuple = ()


class Sensing(Protocol):
    """The wrist's sensing of the page: `wrist` real (the D405, gauge.WristGauge), sim (a truth surface) or none."""
    wrist: str
    def ready(self, arm: str) -> bool: ...
    def gauge(self, q, tcp, prior, *, curved: bool, count: int, name: str = "") -> GaugeRead | None: ...


class NoSensing:
    """No wrist sensing: no gauge, no wrist camera."""
    wrist = "none"

    def ready(self, arm: str) -> bool:
        return False

    def gauge(self, q, tcp, prior, *, curved: bool, count: int, name: str = "") -> GaugeRead | None:
        return None


class NodePort(Protocol):
    guard: object
    def wait_camera_page(self, arm: str, timeout: float, max_lost_s=None, should_stop=None, since=None): ...
    def measured_pages(self, arm: str, since: float) -> list: ...
    def measured_marks(self, since: float) -> list: ...
    def base_from_world(self, arm: str) -> np.ndarray | None: ...
    def camera_page(self, arm: str): ...
    def event(self, kind: str, *, arm: str = "", op_id: str = "", text: str = "") -> None: ...


class ArmPort(Protocol):
    """The arm members ArmExecutor uses (ArmIO's), and goal_log: the run's goals.jsonl writer, or None."""
    arm: str
    q: np.ndarray | None
    safety: dict
    active: object
    op_id: str
    goal_log: object
    latched: bool        # read-only (properties on ArmIO)
    estop_ok: bool
    latch_reason: int
    def latch_text(self) -> str: ...
    def flag(self, name: str, default: float = 0.0) -> bool: ...
    def trip_joints(self) -> np.ndarray | None: ...
    def measured(self, timeout: float = 5.0) -> np.ndarray: ...
    def seed(self) -> np.ndarray: ...
    def record(self, on: bool) -> list: ...
    def wait(self, predicate, timeout: float) -> bool: ...
    def write(self, **values: float) -> None: ...
    def unlatch(self) -> tuple[bool, str]: ...
    def land(self, budget_s: float, on_text=None) -> tuple[bool, str]: ...
    def execute(self, traj, *, watch_latch: bool = True, on_tick=None, should_stop=None, poll: float = 0.02,
                retarget=None): ...


class SafetyReadings:
    """The ArmPort members read off its <arm>_safety states (`safety`), and the seed: ArmIO's and the batch driver's."""

    def flag(self, name: str, default: float = 0.0) -> bool:
        value = self.safety.get(name, default)
        return bool(value) and not math.isnan(value) and value >= 0.5

    @property
    def latched(self) -> bool:
        return self.flag("latched")

    @property
    def estop_ok(self) -> bool:
        return self.flag("estop_ok", 1.0)

    @property
    def latch_reason(self) -> int:
        value = self.safety.get("latch_reason", 0.0)
        return 0 if math.isnan(value) else int(value)

    def trip_joints(self) -> np.ndarray | None:
        q = np.array([self.safety.get(f"trip_q{i}", math.nan) for i in range(7)])
        return None if np.isnan(q).any() else q

    def seed(self) -> np.ndarray:
        """Measured joints with the carriage from the last command (its encoder rests 10-20 um off)."""
        q = self.measured()
        if self.last_command is not None:
            q[6] = self.last_command[6]
        return q


def goal_tag(op_id: str, traj, first: int = 0) -> str:
    """The /tatbot/goals header.frame_id: JSON of the op and where each planned phase starts, in s on the goal's
    time base (the Draw PHASE_* numbers), so a bag splits the goal into travel, descent, drawing and lift."""
    phase = np.asarray(traj.phase)[first:]
    t = np.asarray(traj.t)[first:]
    starts = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]]) if len(phase) else []
    return json.dumps({"op": op_id, "phases": [[round(float(t[i]), 4), int(phase[i])] for i in starts]},
                      separators=(",", ":"))


# A goal sent from its start sends no knot before this (but its last): JTC splines from the measured state to the
# first knot sent, and an arm one encoder count (0.38 mrad) off its seed crossed it in the 10 ms to the first 100 Hz
# knot at 0.06 rad/s, a step on 20-40% of calibration goals and up to 0.4 rad/s on joint 3 (2026-10-22). A goal
# starts at rest, so its first 50 ms hardly move.
START_BRIDGE_S = 0.05


def goal_points(traj, first: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The knots a goal sends (t s, q, qd) from `first` on: the seed knot (t <= 0) is dropped, the controller starts
    from its own state; from the start, the knots before START_BRIDGE_S too."""
    t = np.asarray(traj.t, float)[first:]
    keep = t > (START_BRIDGE_S if first == 0 else 0.0)
    keep[-1:] = t[-1:] > 0.0
    return t[keep], np.asarray(traj.q, float)[first:][keep], np.asarray(traj.qd, float)[first:][keep]


# A goal that may be replaced mid-way (a pen trim) starts on an explicit stamp this far ahead and its replacements
# share it, each starting this far in the past: JTC runs one from the knot that holds now, never from the measured
# pose, which lags the command ~70 ms while tracking (README section 1).
STAMP_AHEAD_S = REPLACE_BACK_S = 0.05


def stop_reason(io, watch_latch: bool, should_stop) -> str | None:
    """Why a running goal stops now: the arm latched (when watched), a stop requested, or None."""
    if watch_latch and io.latched:
        return "latched"
    return "cancelled" if should_stop and should_stop() else None


def execute(io, traj, run, refused=None):
    """ArmPort.execute around `run()`, the goal itself: a landed arm is sent nothing (failed; its driver would ignore
    the goal and JTC report a success), nor a way io.clearance refuses (the node's collision guard; `refused(why)`
    hears it). io.active is the goal's from here."""
    io.active = traj
    if io.flag("landed"):
        return "failed", 0, rules.landed_refusal(io.arm)
    why = None if io.clearance is None else io.clearance(io.arm, traj.q)
    if why is not None:
        if refused is not None:
            refused(why)
        return "failed", 0, why
    return run()
