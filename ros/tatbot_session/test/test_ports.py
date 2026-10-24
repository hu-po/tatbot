"""Executor imports, generated message constants, injected clock behavior and the ports' helpers."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_session import ports


def test_the_executor_imports_with_ros_and_opencv_blocked():
    """sys.modules[name] = None makes `import name` raise: the robot path's executor, its draw-run steps and the
    ports load without rclpy, tatbot_interfaces or cv2."""
    code = ("import sys\n"
            "for name in ('rclpy', 'tatbot_interfaces', 'cv2'):\n"
            "    sys.modules[name] = None\n"
            "import tatbot_session.executor, tatbot_session.drawrun, tatbot_session.ports, tatbot_session.track\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_constants_are_the_generated_messages():
    action = pytest.importorskip("tatbot_interfaces.action")
    msg = pytest.importorskip("tatbot_interfaces.msg")
    srv = pytest.importorskip("tatbot_interfaces.srv")
    kinds = {k: v for k, v in vars(msg.Event).items() if k.startswith("KIND_")}
    phases = {k: v for k, v in vars(action.Draw.Feedback).items() if k.startswith("PHASE_")}
    assert kinds == {k: v for k, v in vars(ports.Event).items() if k.startswith("KIND_")}
    assert phases == {k: v for k, v in vars(ports.Phase).items() if k.startswith("PHASE_")}
    assert {value: name.lower() for name, value in vars(srv.Decide.Request).items() if name in ("CONTINUE", "LAND", "REDRAW", "SKIP")} == ports.rules.NAMES


def _traj(t):
    n = len(t)
    return SimpleNamespace(t=np.asarray(t, float), q=np.arange(n * 7, dtype=float).reshape(n, 7),
                           qd=np.ones((n, 7)), phase=np.array([2] * (n - 1) + [5], np.uint8))


def test_a_goal_sends_its_knots_after_the_seed():
    traj = _traj([0.0, 0.1, 0.2, 0.3])
    t, q, qd = ports.goal_points(traj)
    assert t.tolist() == [0.1, 0.2, 0.3] and (q == traj.q[1:]).all() and qd.shape == (3, 7)
    t, q, _ = ports.goal_points(traj, first=2)
    assert t.tolist() == [0.2, 0.3] and (q == traj.q[2:]).all()
    assert json.loads(ports.goal_tag("s0001", traj)) == {"op": "s0001", "phases": [[0.0, 2], [0.3, 5]]}


def test_a_goal_from_rest_crosses_an_encoder_count_to_its_first_knot_over_the_bridge_not_10_ms():
    """2026-10-22: 20-40% of calibration goals began with a one-tick 0.06 rad/s step (0.4 rad/s on joint 3 at worst):
    the arm one encoder count off its plan's seed, which JTC (sim.JtcModel, pinned to the real one) crossed to the
    first 100 Hz knot in 10 ms. From its start a goal sends no knot before START_BRIDGE_S; a replacement keeps its own."""
    from tatbot_session.sim import JtcModel

    t = np.round(np.arange(0.0, 0.5001, 0.01), 4)
    still = SimpleNamespace(t=t, q=np.zeros((len(t), 7)), qd=np.zeros((len(t), 7)), phase=np.full(len(t), 2, np.uint8))
    sent = ports.goal_points(still)
    assert sent[0][0] >= ports.START_BRIDGE_S and sent[0][-1] == 0.5
    assert ports.goal_points(still, first=1)[0][0] == 0.01
    assert ports.goal_points(SimpleNamespace(t=np.array([0.0, 0.02]), q=np.zeros((2, 7)), qd=np.zeros((2, 7))))[0] \
        .tolist() == [0.02]   # a goal shorter than the bridge still sends its last knot
    count = 2.0 * np.pi / 16471

    def peak(points) -> float:
        jtc = JtcModel(np.zeros(7))
        jtc.send(*points)
        return max(abs(jtc.sample(int(k * 2.5e6), np.full(7, count), np.zeros(7))[1][0]) for k in range(1, 100))

    assert peak(sent) < 0.015 and peak((t[1:], still.q[1:], still.qd[1:])) > 0.05


class StepClock:
    """A clock that moves only when slept."""

    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def utc(self):
        return 1e9 + self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds

    def wait(self, cv, predicate, timeout_s, poll_s=0.05):
        return predicate()


def test_the_ledger_stamps_rows_on_the_session_clock_and_logs_goals(tmp_path):
    from tatbot_session.ledger import GoalLog, Ledger

    clock = StepClock()
    clock.t = 2.5
    row = Ledger(tmp_path / "ledger.jsonl", clock).append("sent", "right", op="s0001", index=1, arc_m=0.0)
    assert row["t"] == 1e9 + 2.5 and row["t_clock"] == 2.5
    log, traj = GoalLog(tmp_path / "goals.jsonl"), _traj([0.0, 0.1, 0.2])
    log.append("s0001", ports.goal_tag("s0001", traj), 2.5, None, *ports.goal_points(traj))
    goal = json.loads((tmp_path / "goals.jsonl").read_text())
    assert goal["start"] is None and goal["t"] == [0.1, 0.2] and goal["q"] == traj.q[1:].tolist()


def test_the_real_arm_senses_the_page_with_its_own_wrist_only():
    """sensing.wrist: auto is the D405 on real hardware and nothing elsewhere; the real arm refuses a simulator's
    gauge and none (a curved page would be set up from no measurement of it, or drawn flat), as does a typo."""
    from tatbot_session import gauge

    assert gauge.wrist({"hardware": "real"}) == "real"
    assert gauge.wrist({"hardware": "real", "sensing": {"wrist": "real"}}) == "real"
    assert [gauge.wrist({"hardware": hw}) for hw in ("mock", "fake", "sim")] == ["none"] * 3
    assert gauge.wrist({"hardware": "fake", "sensing": {"wrist": "sim"}}) == "sim"
    for kind in ("sim", "none", "Real"):
        with pytest.raises(ValueError, match="sensing.wrist"):
            gauge.wrist({"hardware": "real", "sensing": {"wrist": kind}})
    with pytest.raises(ValueError, match="sensing.wrist"):
        gauge.wrist({"hardware": "fake", "sensing": {"wrist": "d405"}})


def test_a_sim_switch_tells_its_sink_and_its_endpoint_each_change():
    from tatbot_session import machine

    clock, heard = StepClock(), []
    clock.t = 1.25
    switch = machine.SimSwitch(sink=lambda on, t_ns: heard.append((on, t_ns)), clock=clock)
    assert switch.wait(True, 1.0) and switch.wait(True, 1.0) and not switch.is_off()
    switch.set(False)
    assert heard == [(True, 1_250_000_000), (False, 1_250_000_000)] and switch.is_off()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(2.0)
        stack = {"machine": {"switch": "sim", "arm": "right", "sim_endpoint": f"udp://127.0.0.1:{sock.getsockname()[1]}"}}
        switch = machine.from_stack(stack, "right", clock=clock)
        threading.Thread(target=switch.set, args=(True,)).start()
        assert sock.recv(64) == b"machine 1 1 1250000000"
        switch.close()
        assert sock.recv(64) == b"machine 0 2 1250000000"
    assert machine.SimSwitch().wait(True, 0.1)   # mock and fake hardware: nobody to tell
