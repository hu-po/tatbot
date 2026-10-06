"""The blue arm when the controller wedges: every driver call bounded, and a landing never reported as a success.

No hardware. The vendor SDK is a stand-in whose driver is the one a wedged controller leaves behind: its
``configure()`` answers and its next read never returns. The e-stop is a relay stream on loopback, read by the real
monitor, and the driver lease is the real one. The runner must never open that driver in its own process, where
nothing could end the read; the recovery landing opens it in a process of its own, and the runner has to end that
process at its budget and call the arm state unknown.

The runner's own session is bounded by the watchdog: ``wedged_runner.py`` is a runner, in a process of its own,
whose chosen driver call blocks in recv() holding the GIL, as the vendor binding does. The watchdog has to end it at
the call's budget and hand the arm to the same recovery landing, and never act on anything else.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from tatbot_travel import hardware, watchdog

STAGED = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966]
CARRIAGE = 0.0123  # where the run found the carriage; the landing keeps it there
BUDGET_S = 2.5
CALL_BUDGET_S = 1.0  # the stand-in runner's, for every call on its driver
RUNNER = Path(__file__).with_name("wedged_runner.py")

SDK = '''"""A stand-in vendor SDK: the driver a wedged controller leaves, configure() answers and the next read never."""
import json, os, sys, time, types

Model = types.SimpleNamespace(wxai_v0="wxai_v0")
Mode = types.SimpleNamespace(idle="idle", position="position")
StandardEndEffector = types.SimpleNamespace(wxai_v0_leader="wxai_v0_leader", wxai_v0_follower="wxai_v0_follower")


class TrossenArmDriver:
    def configure(self, *args):
        print("Driver version: 'v1.8.5'", file=sys.stderr, flush=True)
        with open(os.environ["FAKE_SDK_TRACE"], "a") as trace:
            trace.write(json.dumps({"pid": os.getpid(), "at": time.monotonic(), "argv": sys.argv}) + "\\n")

    def get_is_configured(self):
        return True

    def get_error_information(self):
        time.sleep(10.0)  # "never", for a test: a read in the test's own process fails it instead of hanging it
        raise RuntimeError("TCP connection closed unexpectedly")

    def cleanup(self):
        pass
'''


class IdledSession:
    """The session's driver after the controller dropped it: every command and the cleanup raise."""

    def __init__(self):
        self.commanded_at: list[float] = []

    def set_all_positions(self, *args):
        self.commanded_at.append(time.monotonic())
        raise RuntimeError("Requested to set joint 0 position but it is in mode idle")

    def set_all_modes(self, *args):
        raise RuntimeError("TCP connection closed unexpectedly")

    def cleanup(self):
        raise RuntimeError("TCP connection closed unexpectedly")


class Relay:
    """The e-stop relay on loopback: EST1 frames at 100 Hz from the address the monitor trusts, the button
    pressed until ``released_at``."""

    def __init__(self, pressed_s: float = 0.0):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.device = f"udp://127.0.0.1:{self.port}?from=127.0.0.1"
        self.released_at = time.monotonic() + pressed_s
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.send, daemon=True)
        self.thread.start()

    def send(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", 0))
            seq = 0
            while not self.stop.wait(0.01):
                state = int(time.monotonic() >= self.released_at)
                with contextlib.suppress(OSError):  # nobody bound: the port changes hands during the landing
                    sock.sendto(f"EST1 {seq} {state}\n".encode(), ("127.0.0.1", self.port))
                seq += 1

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """driver_lease, estop and recovery from the checkout over the stand-in SDK, here and in the recovery's
    process; the e-stop status file and the budget are the test's own."""
    sdk_dir = tmp_path / "sdk"
    sdk_dir.mkdir()
    (sdk_dir / "trossen_arm.py").write_text(SDK)
    spec = importlib.util.spec_from_file_location("trossen_arm", sdk_dir / "trossen_arm.py")
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    monkeypatch.setitem(sys.modules, "trossen_arm", sdk)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(sdk_dir), os.environ.get("PYTHONPATH")) if p))
    monkeypatch.setenv("FAKE_SDK_TRACE", str(tmp_path / "sdk.jsonl"))
    monkeypatch.setenv("TATBOT_ESTOP_STATUS", str(tmp_path / "estop-status.json"))
    modules = types.SimpleNamespace(**{name: hardware.plugin(name) for name in ("driver_lease", "estop", "recovery")})
    monkeypatch.setattr(modules.recovery, "trossen_arm", sdk)  # recovery may be imported already
    monkeypatch.setattr(modules.recovery, "LANDING_DEADLINE_S", BUDGET_S)
    monkeypatch.setattr(hardware, "RECOVERY_KILL_AFTER_S", 0.2)
    monkeypatch.setattr(hardware, "RECOVERY_SLACK_S", 0.3)
    # The landing process takes the lease at its one fixed path, as every arm process does.
    if not hardware.lease_free(modules.driver_lease.HARDWARE_LEASE):
        pytest.skip("the arm-driver lease is held on this node")
    return modules


@pytest.fixture
def relay():
    relay = Relay()
    yield relay
    relay.close()


@pytest.fixture
def blue_arm(plugin):
    """BlueArms as connect() leaves them, no config read: the e-stop monitor (which takes a share of the lease),
    the lease, and a session whose controller has dropped it. Whatever a test leaves held is released after it."""
    made = []

    def make(relay: Relay) -> hardware.BlueArm:
        arm = hardware.BlueArm.__new__(hardware.BlueArm)
        arm.estop_required = True
        arm.trossen = sys.modules["trossen_arm"]
        arm.ip, arm.staged, arm.carriage = "192.0.2.7", list(STAGED), CARRIAGE
        arm.estop_device, arm.lease_path = relay.device, plugin.driver_lease.HARDWARE_LEASE
        arm.estop = plugin.estop.acquire_estop(relay.device, required=True)
        arm.estop.wait_for_initial_state()
        arm.lease = plugin.driver_lease.acquire(arm.lease_path)
        arm.driver, arm.watch, arm.summary = IdledSession(), None, None
        made.append(arm)
        return arm

    yield make
    for arm in made:
        arm._release(plugin.estop)


def sessions(tmp_path) -> list[dict]:
    """Every driver session the stand-in SDK opened, in any process."""
    trace = tmp_path / "sdk.jsonl"
    return [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []


def exited(pid: int, within_s: float = 2.0) -> bool:
    """The process is gone (a zombie waiting for its new parent to reap it has exited too)."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return True
        if state == "Z":
            return True
        time.sleep(0.02)
    return False


class Runner:
    """``wedged_runner.py`` in a process of its own (a wedged call holds its GIL), its output read line by line.

    The watchdog and the recovery landing it starts write to the same pipe, so the output ends only once every one of
    them has exited.
    """

    def __init__(self, tmp_path: Path, relay: Relay, scenario: str, call_budget_s: float = CALL_BUDGET_S):
        golden = tmp_path / "leader.yaml"
        golden.write_text("{}\n")
        self.summary, self.stderr = tmp_path / "summary.json", tmp_path / "runner.stderr"
        env = {**os.environ, "CALL_BUDGET_S": str(call_budget_s), "LANDING_BUDGET_S": str(BUDGET_S)}
        with self.stderr.open("w") as stderr:
            self.process = subprocess.Popen(
                [sys.executable, str(RUNNER), scenario, relay.device, str(self.summary), str(golden)],
                stdout=subprocess.PIPE, stderr=stderr, text=True, env=env)
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def expect(self, prefix: str, within_s: float = 30.0) -> str:
        while (line := self.lines.get(timeout=within_s)) is not None:
            if line.startswith(prefix):
                return line
        raise AssertionError(f"the runner's output ended before {prefix!r}: {self.stderr.read_text()}")

    def finish(self, within_s: float = 30.0) -> list[str]:
        """The rest of the output, once the runner, the watchdog and any recovery landing have all exited."""
        rest = []
        while (line := self.lines.get(timeout=within_s)) is not None:
            rest.append(line)
        self.process.wait(timeout=within_s)
        return rest


@pytest.mark.parametrize("ends_it", ["timeout", "runner"])
def test_a_controller_wedged_after_configure_ends_the_landing_at_its_budget_as_unknown(
        tmp_path, monkeypatch, plugin, relay, blue_arm, caplog, ends_it):
    if ends_it == "runner":  # a timeout that never fires: the runner's own cap has to end the landing
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "timeout").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_SDK_TRACE.timeout"\nshift 3\n'
                                         'exec "$@"\n')
        (bin_dir / "timeout").chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    arm = blue_arm(relay)

    started = time.monotonic()
    assert arm.land() == "unknown"
    elapsed = time.monotonic() - started

    assert elapsed < BUDGET_S + 0.2 + 0.3 + 1.0, elapsed  # the budget, the SIGKILL grace, the runner's slack
    assert "arm state UNKNOWN" in caplog.text
    # The recovery bound the e-stop port and took the lease, so the runner let go of both, and it reached the
    # wedged driver; the runner itself never opened one, where nothing could have ended the read.
    (session,) = sessions(tmp_path)
    assert session["pid"] != os.getpid()
    assert exited(session["pid"])  # no orphan left holding the lease and the e-stop
    assert arm.estop is None and arm.lease is None
    assert hardware.lease_free(plugin.driver_lease.HARDWARE_LEASE)
    if ends_it == "runner":
        assert (tmp_path / "sdk.jsonl.timeout").read_text().splitlines()[:3] == [
            "--signal=TERM", "--kill-after=0.2s", f"{BUDGET_S:g}s"]


def test_the_recovery_lands_this_arm_under_its_estop_at_the_carriage_the_run_found(plugin, relay, blue_arm):
    arm = blue_arm(relay)
    arm._release(plugin.estop)  # as land() does before the handover
    *_, module, ip, staged_flag, staged, estop_flag, device = arm._recovery_command(BUDGET_S)
    assert (module, ip, staged_flag, estop_flag, device) == ("tatbot_travel.hardware", "192.0.2.7", "--staged",
                                                             "--estop", relay.device)
    assert [float(v) for v in staged.split(",")] == [*STAGED, CARRIAGE]
    arm.estop_required = False  # a supervised run without the e-stop lands without one, as it ran
    assert "--estop" not in arm._recovery_command(BUDGET_S)


@pytest.mark.parametrize(("code", "outcome"), [(0, "recovered"), (1, "unknown"), (3, "unknown"), (6, "unknown")])
def test_nothing_moves_the_arm_while_the_estop_is_engaged_and_only_a_measured_landing_counts(
        tmp_path, monkeypatch, blue_arm, code, outcome):
    relay = Relay(pressed_s=0.3)
    try:
        arm = blue_arm(relay)
        session = arm.driver  # land() lets go of it
        assert arm.estopped
        started = tmp_path / "recovery-started"
        monkeypatch.setattr(hardware.BlueArm, "_recovery_command", lambda self, budget_s: [
            sys.executable, "-c", f"import time; open({str(started)!r}, 'w').write(repr(time.monotonic()));"
                                  f"raise SystemExit({code})"])
        assert arm.land() == outcome
    finally:
        relay.close()
    assert session.commanded_at and min(session.commanded_at) >= relay.released_at
    assert float(started.read_text()) >= relay.released_at


def test_a_lease_still_held_here_is_never_handed_to_the_recovery(tmp_path, plugin, relay, blue_arm, caplog):
    arm = blue_arm(relay)
    stuck = plugin.driver_lease.acquire(arm.lease_path)  # a share that did not let go, e.g. a stuck reader
    try:
        assert arm.land() == "unknown"
    finally:
        stuck.close()
    assert "still held" in caplog.text
    assert sessions(tmp_path) == []  # the recovery was never started
    assert hardware.lease_free(arm.lease_path)


@pytest.mark.parametrize(("scenario", "call", "carriage"), [
    ("connect", "get_error_information", 0.0),  # before connect() measured the carriage
    ("measured", "get_all_positions", CARRIAGE),
    ("land", "set_all_modes", CARRIAGE),
])
def test_a_driver_call_that_never_returns_ends_the_runner_at_its_budget_and_hands_the_arm_over(
        tmp_path, plugin, relay, scenario, call, carriage):
    runner = Runner(tmp_path, relay, scenario)
    watchdog_pid = int(runner.expect("watchdog ").split()[1])
    runner.expect(f"wedging {call}")
    wedged = time.monotonic()
    runner.process.wait(timeout=CALL_BUDGET_S + 10.0)
    ended = time.monotonic() - wedged
    rest = runner.finish()

    assert runner.process.returncode == -signal.SIGTERM
    assert CALL_BUDGET_S - 0.2 < ended < CALL_BUDGET_S + 1.5, ended
    summary = json.loads(runner.summary.read_text())
    assert summary["landing"] == "unknown" and call in summary["ended"] and summary["carriage"] == carriage
    assert json.loads(rest[-1]) == {"out": str(tmp_path), **summary}
    assert "arm state UNKNOWN" in runner.stderr.read_text()
    # The recovery landing reached the stand-in's wedged driver in a process of its own, for this run's arm, pose
    # and e-stop, and ended at its budget; nothing is left holding the lease or the e-stop.
    (session,) = sessions(tmp_path)
    assert session["pid"] not in (runner.process.pid, watchdog_pid)
    argv = session["argv"]
    assert [float(v) for v in argv[argv.index("--staged") + 1].split(",")] == [*STAGED, carriage]
    assert argv[argv.index("--estop") + 1] == relay.device
    assert exited(watchdog_pid) and exited(session["pid"])
    assert hardware.lease_free(plugin.driver_lease.HARDWARE_LEASE)


def test_neither_a_wait_on_the_estop_nor_a_long_blocking_move_is_a_wedge(tmp_path, plugin, relay):
    runner = Runner(tmp_path, relay, "estop")
    runner.expect("connected")
    pressed = time.monotonic()
    relay.released_at = pressed + 2 * CALL_BUDGET_S  # pressed until then
    assert runner.expect("landed") == "landed"
    assert time.monotonic() - pressed >= 2 * CALL_BUDGET_S + 4.0  # the wait, then the staged landing's 4 s move
    assert runner.finish() == [] and runner.process.returncode == 0
    assert not runner.summary.exists() and sessions(tmp_path) == []  # the watchdog never acted


def test_the_watchdog_never_acts_once_the_runner_is_gone(tmp_path, plugin, relay):
    runner = Runner(tmp_path, relay, "land", call_budget_s=30.0)
    watchdog_pid = int(runner.expect("watchdog ").split()[1])
    runner.expect("wedging set_all_modes")
    runner.process.kill()  # mid-call, long before its budget: whoever ended it owns what happens next
    assert runner.finish(within_s=10.0) == []
    assert exited(watchdog_pid)
    assert not runner.summary.exists() and sessions(tmp_path) == []
    assert hardware.lease_free(plugin.driver_lease.HARDWARE_LEASE)


def test_a_wedge_after_the_runner_ruled_the_recovery_out_ends_it_and_starts_nothing(tmp_path, plugin, relay):
    runner = Runner(tmp_path, relay, "beyond")
    watchdog_pid = int(runner.expect("watchdog ").split()[1])
    runner.expect("wedging cleanup")
    runner.finish()
    assert runner.process.returncode == -signal.SIGTERM
    summary = json.loads(runner.summary.read_text())
    assert summary["landing"] == "unknown" and "cleanup" in summary["ended"]
    assert "ruled out the recovery landing: the blue arm measures joint 5 at -4.796" in runner.stderr.read_text()
    assert sessions(tmp_path) == [] and exited(watchdog_pid)  # the recovery's clamp is the move refused
    assert hardware.lease_free(plugin.driver_lease.HARDWARE_LEASE)


def test_no_driver_is_opened_without_its_watchdog(tmp_path, plugin, relay):
    runner = Runner(tmp_path, relay, "unwatched")
    assert "watchdog did not come up" in runner.expect("refused")
    assert runner.finish() == ["sessions 0", "None"]  # no session opened, so nothing to land
    assert runner.process.returncode == 0
    assert hardware.lease_free(plugin.driver_lease.HARDWARE_LEASE)


def test_a_call_may_take_its_budget_plus_a_blocking_moves_goal_time_or_configures_connect_timeout():
    budget, q = watchdog.CALL_BUDGET_S, [0.0] * 7
    assert watchdog.call_budget("get_error_information", (), {}) == budget
    assert watchdog.call_budget("set_all_positions", (q, 4.0, True), {}) == budget + 4.0
    assert watchdog.call_budget("set_all_positions", (q, 4.0, False), {}) == budget
    assert watchdog.call_budget("set_all_positions", (q,), {"goal_time": 3.0}) == budget + 3.0
    assert watchdog.call_budget("set_all_positions", (q,), {}) == budget + 2.0  # the vendor's defaults
    assert watchdog.call_budget("configure", ("wxai_v0", "leader", "192.0.2.7", True), {}) == budget + 20.0
    assert watchdog.call_budget("configure", ("wxai_v0", "leader", "192.0.2.7", True, 5.0), {}) == budget + 5.0


def test_a_budget_runs_from_its_announcements_arrival_so_a_backlog_never_shortens_it():
    said = watchdog.Announcements()
    said.feed(b'{"call": "set_all_modes", "budget_s": 1.0}\n{"call": null}\n{"call": "cleanup", "bud', 100.0)
    assert said.call is None and said.remaining(100.0) is None  # half an announcement is none
    said.feed(b'get_s": 1.0}\n', 250.0)  # read long after it was written
    assert said.call == "cleanup" and said.remaining(250.25) == pytest.approx(0.75)
    said.feed(b'{"handover": {"ip": "192.0.2.7"}}\n', 251.0)
    assert said.call == "cleanup" and said.handover == {"ip": "192.0.2.7"}


def test_every_driver_call_and_the_drivers_destructor_run_inside_a_bounded_call():
    class Watch:
        def __init__(self):
            self.log, self.in_flight = [], None

        @contextlib.contextmanager
        def bounded(self, name, budget_s):
            self.log.append((name, budget_s))
            self.in_flight = name
            try:
                yield
            finally:
                self.in_flight = None

        def close(self):
            self.log.append("closed")

    watch, seen = Watch(), []

    class Vendor:
        def set_all_positions(self, q, goal_time=2.0, blocking=True):
            seen.append(watch.in_flight)

        def __del__(self):  # the vendor's destructor runs cleanup(), a TCP round trip unless cleanup() ran
            seen.append(watch.in_flight)

    arm = hardware.BlueArm.__new__(hardware.BlueArm)
    arm.driver, arm.watch = watchdog.WatchedDriver(Vendor(), watch), watch
    arm.driver.set_all_positions([0.0] * 7, 4.0, True)
    arm._close_driver()
    assert seen == ["set_all_positions", watchdog.DESTRUCTOR]
    assert watch.log == [("set_all_positions", watchdog.CALL_BUDGET_S + 4.0),
                         (watchdog.DESTRUCTOR, watchdog.CALL_BUDGET_S), "closed"]
    assert arm.driver is None and arm.watch is None


def test_the_watchdog_ends_a_process_through_the_raw_syscalls_where_python_lacks_the_pidfd_wrappers(monkeypatch):
    # uv's standalone interpreters are built against an old glibc: no os.pidfd_open, no signal.pidfd_send_signal.
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pidfd = watchdog.pidfd_open(child.pid)
        try:
            assert watchdog.end(pidfd, 5.0)  # SIGTERM through the pidfd, then its exit makes the pidfd readable
        finally:
            os.close(pidfd)
        assert child.wait(timeout=5.0) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    with pytest.raises(ProcessLookupError):
        watchdog.pidfd_open(child.pid)  # reaped: the pid names no process
