import importlib.util
import os
import pty
import threading
import time
from pathlib import Path

ESTOP_PATH = Path(__file__).parents[1] / "src" / "lerobot_robot_tatbot" / "estop.py"
SPEC = importlib.util.spec_from_file_location("tatbot_estop_test_module", ESTOP_PATH)
assert SPEC is not None and SPEC.loader is not None
ESTOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ESTOP)
EstopMonitor = ESTOP.EstopMonitor
EstopState = ESTOP.EstopState
acquire_estop = ESTOP.acquire_estop
release_estop = ESTOP.release_estop


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
