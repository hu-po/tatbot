"""The relay against an emulated Pico on a pseudo-terminal and an emulated GPIO line. Stdlib +
pytest, no ROS:

    python3 -m pytest -q ros/tatbot_estop_relay/test

End to end into the driver's UDP reader (loss, delay, reorder, wrong source) is
ros/tatbot_hardware/test/test_hw_relay.cpp."""
import argparse
import contextlib
import importlib.util
import itertools
import os
import pty
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tatbot_estop_relay.py"
spec = importlib.util.spec_from_file_location("tatbot_estop_relay", SCRIPT)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def test_split_lines_keeps_frames_and_drops_overlong_lines():
    lines, rest = relay.split_lines(b"", b"EST1 1 1\nEST1 2 1\nEST1 3")
    assert lines == [b"EST1 1 1\n", b"EST1 2 1\n"] and rest == b"EST1 3"
    lines, rest = relay.split_lines(rest, b" 0\n" + b"x" * 200 + b"\nEST1 4 1\n")
    assert lines == [b"EST1 3 0\n", b"EST1 4 1\n"] and rest == b""
    lines, rest = relay.split_lines(b"", b"y" * 600)
    assert lines == [] and rest == b""
    # never parsed: anything line-shaped under 128 bytes passes unchanged
    assert relay.split_lines(b"", b"PRB1 7 1 123\n")[0] == [b"PRB1 7 1 123\n"]


def test_dest_parsing():
    assert relay.parse_dest("192.0.2.5:7640") == ("192.0.2.5", 7640)
    with pytest.raises(argparse.ArgumentTypeError):
        relay.parse_dest("7640")


class Pico:
    """A pty whose slave is reachable through a stable symlink, like /dev/tatbot-estop."""

    def __init__(self, link: Path):
        self.link = link
        self.plug()

    def plug(self):
        self.master, self.slave = pty.openpty()
        if self.link.is_symlink():
            self.link.unlink()
        self.link.symlink_to(os.ttyname(self.slave))

    def unplug(self):
        self.link.unlink()
        os.close(self.master)
        os.close(self.slave)

    def put(self, data: bytes):
        os.write(self.master, data)


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.05)
    yield sock
    sock.close()


def drain(sock, seconds):
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        with contextlib.suppress(socket.timeout):
            out.append(sock.recvfrom(512))
    return out


def test_forwards_each_line_unchanged_and_reopens(tmp_path, receiver):
    pico = Pico(tmp_path / "tatbot-estop")
    stop = threading.Event()
    thread = threading.Thread(
        target=relay.run, args=(str(pico.link), [receiver.getsockname()]), kwargs={"stop": stop.is_set}, daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        pico.put(b"EST1 1 1\nEST1 2 1\n" + b"z" * 300 + b"\nEST1 3 0\nEST1 4")
        got = drain(receiver, 0.3)
        assert [data for data, _ in got] == [b"EST1 1 1\n", b"EST1 2 1\n", b"EST1 3 0\n"]
        assert all(addr[0] == "127.0.0.1" for _, addr in got)
        pico.put(b" 1\n")
        assert [data for data, _ in drain(receiver, 0.2)] == [b"EST1 4 1\n"]
        # Delay passes through as delay: nothing is synthesized in the gap.
        assert drain(receiver, 0.3) == []
        # Unplugged: silence; replugged (a new pty behind the same path): forwarding resumes.
        pico.unplug()
        assert drain(receiver, 0.3) == []
        pico.plug()
        time.sleep(1.2)
        pico.put(b"EST1 0 1\n")
        assert [data for data, _ in drain(receiver, 0.3)] == [b"EST1 0 1\n"]
    finally:
        stop.set()
        thread.join(2)
        pico.unplug()


def test_cli_runs_as_a_process(tmp_path, receiver):
    pico = Pico(tmp_path / "tatbot-estop")
    host, port = receiver.getsockname()
    proc = subprocess.Popen([sys.executable, str(SCRIPT), "--device", str(pico.link), "--dest", f"{host}:{port}"],
                            stderr=subprocess.PIPE)
    try:
        time.sleep(0.5)
        for seq in range(10):
            pico.put(b"EST1 %d 1\n" % seq)
            time.sleep(0.01)
        got = [data for data, _ in drain(receiver, 0.3)]
        assert got == [b"EST1 %d 1\n" % seq for seq in range(10)]
    finally:
        proc.terminate()
        proc.wait(5)
        pico.unplug()


def test_gpio_ioctl_numbers_match_linux_gpio_h():
    assert relay.GPIO_GET_CHIPINFO_IOCTL == 0x8044B401
    assert relay.GPIO_V2_GET_LINEINFO_IOCTL == 0xC100B405
    assert relay.GPIO_V2_GET_LINE_IOCTL == 0xC250B407
    assert relay.GPIO_V2_LINE_GET_VALUES_IOCTL == 0xC010B40E


class Line:
    """An emulated GPIO line: `high` is the level (an open contact); `fail` makes the next read fail."""

    def __init__(self):
        self.high, self.fail, self.opens = False, False, 0

    def open(self, name):
        assert name == "GPIO17"
        self.opens += 1
        return self.read, lambda: None

    def read(self):
        if self.fail:
            self.fail = False
            raise OSError(5, "Input/output error")
        return self.high


def frames(got):
    return [tuple(int(v) for v in data.split()[1:]) for data, _ in got]


def test_gpio_sends_the_contact_at_100_hz_and_goes_silent_on_failure(receiver):
    line = Line()
    stop = threading.Event()
    thread = threading.Thread(target=relay.run_gpio, args=("GPIO17", [receiver.getsockname()]),
                              kwargs={"stop": stop.is_set, "open_line": line.open}, daemon=True)
    thread.start()
    try:
        drain(receiver, 0.2)
        released = frames(drain(receiver, 0.5))
        assert 35 <= len(released) <= 65 and {state for _, state in released} == {1}
        line.high = True   # pressed, or a broken wire
        drain(receiver, 0.05)
        assert {state for _, state in frames(drain(receiver, 0.2))} == {0}
        # A failed read is silence until the line is reopened; the sequence keeps advancing.
        line.high, line.fail = False, True
        time.sleep(0.05)
        drain(receiver, 0.05)
        assert drain(receiver, 0.3) == []
        resumed = frames(drain(receiver, 0.5))
        assert resumed and {state for _, state in resumed} == {1} and line.opens == 2
        seqs = [seq for seq, _ in released] + [seq for seq, _ in resumed]
        assert seqs == sorted(set(seqs))
    finally:
        stop.set()
        thread.join(2)


def test_every_reader_gets_the_same_frames_and_one_sequence(receiver):
    """One button stops both arms: the ros node's driver and the arm node's monitor each get every
    frame, and a reader that stops receiving goes silent alone."""
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    other.bind(("127.0.0.1", 0))
    other.settimeout(0.05)
    line = Line()
    stop = threading.Event()
    thread = threading.Thread(target=relay.run_gpio, args=("GPIO17", [receiver.getsockname(), other.getsockname()]),
                              kwargs={"stop": stop.is_set, "open_line": line.open}, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        line.high = True   # pressed: both readers see the stop in the same frames
        time.sleep(0.1)
        stop.set()
        thread.join(2)
        first, second = frames(drain(receiver, 0.1)), frames(drain(other, 0.1))
        assert first and first == second and 0 in {state for _, state in first}
    finally:
        stop.set()
        thread.join(2)
        other.close()


def test_one_source_per_relay_and_no_repeated_destination():
    with pytest.raises(SystemExit):
        relay.main(["--gpio", "GPIO17", "--device", "/dev/null", "--dest", "127.0.0.1:1"])
    with pytest.raises(SystemExit):
        relay.main(["--probe", "GPIO27", "--gpio", "GPIO17", "--dest", "127.0.0.1:1"])
    with pytest.raises(SystemExit):   # the same reader twice would double its frames
        relay.main(["--gpio", "GPIO17", "--dest", "127.0.0.1:1", "--dest", "127.0.0.1:1"])


def event(stamp_ns: int, rising: bool) -> bytes:
    """One struct gpio_v2_line_event as the kernel writes it."""
    return struct.pack("QIIII24x", stamp_ns, 1 if rising else 2, 27, 1, 1)


def test_probe_event_layout():
    assert len(event(1, True)) == relay.LINE_EVENT_SIZE
    assert relay.parse_events(event(123, True) + event(456, False) + b"\0" * 7) == [(123, True), (456, False)]


class ProbeLine:
    """An emulated probe line: a pipe stands in for the line fd that the kernel's edge events arrive on."""

    def __init__(self):
        self.high, self.opens, self.writer = False, 0, None

    def open(self, name):
        assert name == "GPIO27"
        self.opens += 1
        reader, self.writer = os.pipe()
        return (reader, lambda: self.high), (lambda: os.close(reader))

    def edge(self, stamp_ns: int, high: bool):
        self.high = high
        os.write(self.writer, event(stamp_ns, high))

    def unplug(self):
        os.close(self.writer)


def probe_frames(got):
    return [tuple(int(v) for v in data.split()[1:]) for data, _ in got if data.startswith(b"PRB1 ")]


def test_probe_sends_each_edge_at_once_and_its_level_at_50_hz(receiver):
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    other.bind(("127.0.0.1", 0))
    other.settimeout(0.05)
    line = ProbeLine()
    stop = threading.Event()
    thread = threading.Thread(target=relay.run_probe, args=("GPIO27", [receiver.getsockname(), other.getsockname()]),
                              kwargs={"stop": stop.is_set, "open_line": line.open}, daemon=True)
    thread.start()
    try:
        drain(receiver, 0.2)
        rest = probe_frames(drain(receiver, 0.5))
        assert 18 <= len(rest) <= 32 and {(state, edge) for _, state, edge, _ in rest} == {(0, 0)}
        before = time.monotonic_ns()
        line.edge(987654321, True)   # touched
        first = probe_frames(drain(receiver, 0.01))
        assert first and first[0][1:3] == (1, 987654321) and before <= first[0][3] <= time.monotonic_ns()
        line.edge(987700000, False)  # released
        time.sleep(0.05)
        after = probe_frames(drain(receiver, 0.1))
        assert after[-1][1:3] == (0, 987700000)
        seqs = [f[0] for f in rest + first + after]
        assert seqs == sorted(set(seqs))
        assert probe_frames(drain(other, 0.1)), "every --dest gets the frames"
        # A line that closes is silence until it is reopened; the sequence keeps advancing.
        line.unplug()
        time.sleep(0.05)
        drain(receiver, 0.05)
        assert drain(receiver, 0.3) == []
        resumed = probe_frames(drain(receiver, 0.6))
        assert resumed and line.opens == 2 and resumed[0][0] > seqs[-1]
    finally:
        stop.set()
        thread.join(2)
        other.close()


def test_service_unit_template():
    unit = (SCRIPT.parent / "tatbot-estop-relay.service.in").read_text()
    for token in ("@USER@", "@DIR@", "@ARGS@", "Restart=always"):
        assert token in unit


def test_machine_gate_needs_a_fresh_command_and_a_fresh_released_estop_and_never_restarts_by_itself():
    gate = relay.MachineGate(timeout_s=0.2)
    assert gate.command(b"MCH1 1 1\n", 0.0) == 1 and not gate.powered(0.0)   # no e-stop frame yet
    assert gate.estop(0.0) == relay.ESTOP_SILENT
    gate.take_estop(b"EST1 7 1\n", 0.0)
    assert not gate.powered(0.0)   # asked on before the e-stop was known: it must be asked off first
    gate.command(b"MCH1 2 0\n", 0.01)
    gate.command(b"MCH1 3 1\n", 0.02)
    assert gate.powered(0.1) and not gate.powered(0.16)   # the e-stop frame went stale first
    gate.take_estop(b"EST1 8 1\n", 0.17)
    assert not gate.powered(0.17)   # a silent e-stop disarmed it: released again, it stays off
    gate.command(b"MCH1 4 0\n", 0.18)
    gate.command(b"MCH1 5 1\n", 0.19)
    assert gate.powered(0.19)
    gate.take_estop(b"EST1 9 0\n", 0.2)
    assert not gate.powered(0.2) and gate.estop(0.2) == relay.ESTOP_PRESSED
    gate.take_estop(b"EST1 10 1\n", 0.21)
    assert not gate.powered(0.21)   # released, still asked on: no restart
    assert gate.command(b"MCH1 5 0\n", 0.22) is None and gate.asked[1]   # a repeat or reordered frame is not taken
    assert gate.command(b"MCH1 x 1\n", 0.22) is None and gate.command(b"EST1 2 1\n", 0.22) is None
    gate.take_estop(b"EST1 11 1\n", 0.6)
    assert gate.command(b"MCH1 1 0\n", 0.6) == 1   # a restarted session, after silence: taken, and it starts off
    gate.command(b"MCH1 2 1\n", 0.61)
    assert gate.powered(0.61)


def test_machine_powers_its_line_only_while_asked_and_released_and_answers_each_command():
    probe = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
    for sock in probe:
        sock.bind(("127.0.0.1", 0))
    listen, estop_port = (sock.getsockname()[1] for sock in probe)
    for sock in probe:
        sock.close()
    writes, closed = [], []
    stop = threading.Event()
    thread = threading.Thread(target=relay.run_machine, args=("GPIO22", listen, "127.0.0.1", estop_port, 0.2),
                              kwargs={"stop": stop.is_set, "open_line": lambda name: (writes.append, lambda: closed.append(1))},
                              daemon=True)
    thread.start()
    session, button, stranger = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3))
    stranger.bind(("127.0.0.2", 0))
    session.settimeout(0.05)
    seq = itertools.count(1)

    def run(seconds, on, released=True, estop=True, sender=session):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if estop:
                button.sendto(b"EST1 %d %d\n" % (next(seq), released), ("127.0.0.1", estop_port))
            sender.sendto(b"MCH1 %d %d\n" % (next(seq), on), ("127.0.0.1", listen))
            time.sleep(0.02)
        return [tuple(int(v) for v in data.split()[1:]) for data, _ in drain(session, 0.05)]

    try:
        time.sleep(0.1)
        assert writes == [False] and {a[1:] for a in run(0.2, 0)} == {(0, 0, relay.ESTOP_RELEASED)}
        assert run(0.3, 1)[-1][1:] == (1, 1, relay.ESTOP_RELEASED) and writes[-1] is True
        assert {a[1:] for a in run(0.2, 1, released=False)} == {(1, 0, relay.ESTOP_PRESSED)} and writes[-1] is False
        assert {a[1:] for a in run(0.2, 1)} == {(1, 0, relay.ESTOP_RELEASED)}   # released: still off, no restart
        run(0.1, 0)
        run(0.2, 1)
        assert writes[-1] is True   # asked off and on again
        assert run(0.3, 1, estop=False)[-1][1:] == (1, 0, relay.ESTOP_SILENT) and writes[-1] is False
        run(0.1, 0)
        run(0.2, 1)
        time.sleep(0.3)   # the session went silent
        assert writes[-1] is False
        assert run(0.2, 1, sender=stranger) == [] and writes[-1] is False   # commands from elsewhere are ignored
    finally:
        stop.set()
        thread.join(2)
        for sock in (session, button, stranger):
            sock.close()
    assert closed == [1]


def test_set_values_ioctl_matches_linux_gpio_h():
    assert relay.GPIO_V2_LINE_SET_VALUES_IOCTL == 0xC010B40F
