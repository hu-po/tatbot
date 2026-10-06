import importlib.util
import json
import os
import pty
import socket
import threading
import time
from pathlib import Path

import pytest

ESTOP_PATH = Path(__file__).parents[1] / "src" / "lerobot_robot_tatbot" / "estop.py"
SPEC = importlib.util.spec_from_file_location("tatbot_estop_test_module", ESTOP_PATH)
assert SPEC is not None and SPEC.loader is not None
ESTOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ESTOP)
EstopMonitor = ESTOP.EstopMonitor
EstopState = ESTOP.EstopState
acquire_estop = ESTOP.acquire_estop
release_estop = ESTOP.release_estop


@pytest.fixture(autouse=True)
def isolated_status_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_ESTOP_STATUS", str(tmp_path / "status.json"))


def _pty():
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    os.close(slave)
    return master, path


def _frames(master: int, start: int, state: int, count: int = 3) -> int:
    seq = start
    for _ in range(count):
        os.write(master, f"EST1 {seq} {state}\n".encode())
        seq += 1
        time.sleep(0.01)
    return seq


def _wait(monitor: EstopMonitor, state: EstopState, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if monitor.state is state:
            return
        time.sleep(0.005)
    assert monitor.state is state


def test_protocol_debounce_malformed_reset_and_timeout():
    master, path = _pty()
    monitor = EstopMonitor(path, required=True)
    try:
        assert monitor.state is EstopState.FAULT
        seq = _frames(master, 0, 1)
        _wait(monitor, EstopState.OK)

        os.write(master, b"EST1 99 1 trailing\nnot-a-frame\n")
        time.sleep(0.03)
        assert monitor.state is EstopState.OK

        seq = _frames(master, seq, 0)
        _wait(monitor, EstopState.PRESSED)
        seq = _frames(master, seq, 1)
        _wait(monitor, EstopState.OK)

        # Sender reboot/non-monotonic sequence must re-establish debounce.
        _frames(master, 0, 0, 2)
        assert monitor.state is EstopState.OK
        _frames(master, 2, 0, 1)
        _wait(monitor, EstopState.PRESSED)

        _wait(monitor, EstopState.FAULT, timeout=0.4)
    finally:
        monitor.close()
        os.close(master)


def test_one_process_wide_reader_is_reference_counted():
    master, path = _pty()
    first = acquire_estop(path, required=True)
    second = acquire_estop(path, required=True)
    try:
        assert first is second
        _frames(master, 0, 1)
        _wait(first, EstopState.OK)
        release_estop(first)
        _frames(master, 3, 0)
        _wait(second, EstopState.PRESSED)
    finally:
        release_estop(second)
        os.close(master)


def test_monitor_publishes_and_removes_passive_status(tmp_path, monkeypatch):
    status_path = tmp_path / "status.json"
    monkeypatch.setenv("TATBOT_ESTOP_STATUS", str(status_path))
    master, path = _pty()
    monitor = EstopMonitor(path, required=True)
    try:
        _frames(master, 0, 1)
        _wait(monitor, EstopState.OK)
        deadline = time.monotonic() + 1
        payload = {}
        while time.monotonic() < deadline:
            if status_path.exists():
                payload = json.loads(status_path.read_text())
                if payload.get("state") == "ok":
                    break
            time.sleep(0.01)
        assert payload["schema"] == "tatbot.estop-status/1"
        assert payload["state"] == "ok" and payload["engaged"] is False
        assert time.time() - payload["updated_unix"] < 1
        assert status_path.stat().st_mode & 0o777 == 0o600
        assert payload["last_sequence"] >= 2
        assert payload["heartbeat_age_ms"] < 100
    finally:
        monitor.close()
        os.close(master)
    assert not status_path.exists()


def test_unplug_fault_and_replug_recovery(tmp_path):
    link = tmp_path / "tatbot-estop"
    first_master, first_path = _pty()
    link.symlink_to(first_path)
    monitor = acquire_estop(str(link), required=True)
    try:
        _frames(first_master, 0, 1)
        _wait(monitor, EstopState.OK)
        os.close(first_master)
        _wait(monitor, EstopState.FAULT, timeout=0.4)

        second_master, second_path = _pty()
        replacement = tmp_path / "replacement"
        replacement.symlink_to(second_path)
        replacement.replace(link)

        stop = threading.Event()

        def pump():
            seq = 0
            while not stop.is_set():
                try:
                    os.write(second_master, f"EST1 {seq} 1\n".encode())
                except OSError:
                    return
                seq += 1
                time.sleep(0.01)

        writer = threading.Thread(target=pump)
        writer.start()
        try:
            _wait(monitor, EstopState.OK, timeout=1.5)
        finally:
            stop.set()
            writer.join()
            os.close(second_master)
    finally:
        release_estop(monitor)


def test_blocked_snapshot_writer_does_not_delay_press_or_timeout(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = EstopMonitor._publish_status

    def blocked(self, payload):
        entered.set()
        release.wait(timeout=3)
        original(self, payload)

    monkeypatch.setattr(EstopMonitor, "_publish_status", blocked)
    master, path = _pty()
    monitor = EstopMonitor(path, required=True)
    try:
        seq = _frames(master, 0, 1)
        _wait(monitor, EstopState.OK)
        assert entered.wait(timeout=1)
        seq = _frames(master, seq, 0)
        _wait(monitor, EstopState.PRESSED, timeout=0.2)
        _frames(master, seq, 1)
        _wait(monitor, EstopState.OK, timeout=0.2)
        _wait(monitor, EstopState.FAULT, timeout=0.3)
    finally:
        release.set()
        monitor.close()
        os.close(master)


def test_snapshot_write_failure_does_not_stop_monitor(tmp_path, monkeypatch):
    parent = tmp_path / "not-a-directory"
    parent.write_text("blocked")
    monkeypatch.setenv("TATBOT_ESTOP_STATUS", str(parent / "status.json"))
    master, path = _pty()
    monitor = EstopMonitor(path, required=True)
    try:
        seq = _frames(master, 0, 1)
        _wait(monitor, EstopState.OK)
        _frames(master, seq, 0)
        _wait(monitor, EstopState.PRESSED)
        _wait(monitor, EstopState.FAULT, timeout=0.3)
    finally:
        monitor.close()
        os.close(master)


def _udp_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _datagrams(sock: socket.socket, port: int, start: int, state: int, count: int = 3) -> int:
    seq = start
    for _ in range(count):
        sock.sendto(f"EST1 {seq} {state}\n".encode(), ("127.0.0.1", port))
        seq += 1
        time.sleep(0.01)
    return seq


def test_udp_reads_only_the_relay_drops_reordered_frames_and_faults_on_silence():
    """The palette relay's datagrams, read as the ROS driver reads them, so one button stops both arms."""
    port = _udp_port()
    monitor = EstopMonitor(f"udp://127.0.0.1:{port}?from=127.0.0.1", required=True)
    relay = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    relay.bind(("127.0.0.1", 0))
    stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stranger.bind(("127.0.0.2", 0))
    try:
        assert monitor.state is EstopState.FAULT
        seq = _datagrams(relay, port, 0, 1)
        _wait(monitor, EstopState.OK)
        # Another address is not the relay: neither its press nor its release counts.
        for _ in range(4):
            stranger.sendto(f"EST1 {seq + 100} 0\n".encode(), ("127.0.0.1", port))
            seq = _datagrams(relay, port, seq, 1, 1)
        assert monitor.state is EstopState.OK
        relay.sendto(b"EST1 x 0\n", ("127.0.0.1", port))
        relay.sendto(b"EST1 " + b"9" * 200 + b" 0\n", ("127.0.0.1", port))
        seq = _datagrams(relay, port, seq, 1, 1)
        assert monitor.state is EstopState.OK
        # Older frames on a fresh stream are reordered or replayed: dropped, even three pressed ones.
        _datagrams(relay, port, seq - 10, 0)
        seq = _datagrams(relay, port, seq, 1, 1)
        assert monitor.state is EstopState.OK
        seq = _datagrams(relay, port, seq, 0)
        _wait(monitor, EstopState.PRESSED)
        seq = _datagrams(relay, port, seq, 1)
        _wait(monitor, EstopState.OK)
        # Silence past the relay path's budget is a fault; after it a restarted relay's sequence re-seeds.
        _wait(monitor, EstopState.FAULT, timeout=0.5)
        _datagrams(relay, port, 0, 1)
        _wait(monitor, EstopState.OK)
    finally:
        monitor.close()
        relay.close()
        stranger.close()


def test_udp_port_has_one_reader_and_a_bad_source_refuses():
    port = _udp_port()
    first = EstopMonitor(f"udp://127.0.0.1:{port}?from=127.0.0.1", required=True)
    try:
        with pytest.raises(RuntimeError):
            EstopMonitor(f"udp://127.0.0.1:{port}?from=127.0.0.1", required=True)
    finally:
        first.close()
    for bad in ("udp://127.0.0.1:7640", "udp://127.0.0.1?from=127.0.0.1"):
        with pytest.raises(RuntimeError):
            EstopMonitor(bad, required=True)


def test_udp_source_resolves_the_relay_role_through_nodes_json(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "nodes.json").write_text(json.dumps({
        "pi": {"roles": ["estop-relay"], "lan": "192.0.2.98"}, "arm": {"roles": ["arm", "estop"]}}))
    monkeypatch.setattr(ESTOP.paths, "repo_root", lambda: tmp_path)
    assert ESTOP.udp_source("udp://:7640?from=estop-relay") == (("0.0.0.0", 7640), "192.0.2.98")
    assert ESTOP.udp_source("udp://127.0.0.1:7640?from=192.0.2.5") == (("127.0.0.1", 7640), "192.0.2.5")
    assert ESTOP.udp_source("/dev/tatbot-estop") is None
    for bad in ("udp://:7640", "udp://:7640?from=nobody"):
        with pytest.raises(ValueError):
            ESTOP.udp_source(bad)
