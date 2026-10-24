"""The station keyboard's relay against an emulated evdev device (a pipe) and a UDP reader. Stdlib + pytest, no ROS:

    python3 -m pytest -q ros/tatbot_keys_relay/test
"""
import importlib.util
import os
import socket
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tatbot_keys_relay.py"
spec = importlib.util.spec_from_file_location("tatbot_keys_relay", SCRIPT)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def _event(kind, code, value):
    return relay.EVENT.pack(0, 0, kind, code, value)


def test_only_drawing_key_presses_are_sent_once_each():
    held_f13 = [_event(4, 4, 458856), _event(1, 183, 1), _event(0, 0, 0), _event(1, 183, 2), _event(1, 183, 0)]
    typing = [_event(1, code, 1) for code in (30, 28, 103, 108, 12, 13, 66, 68)]   # a, Enter, arrows, -, =, F8, F10
    others = [_event(1, code, 1) for code in (184, 185, 186, 187, 188)]
    data = b"".join(held_f13 + typing + others) + b"\0" * 5
    assert relay.presses(data) == ["up", "down", "reset", "dip", "fewer_dips", "enter"]


def test_the_keys_are_the_ones_the_session_takes():
    assert set(relay.KEYS.values()) == {"up", "down", "reset", "dip", "fewer_dips", "enter"}


def test_a_press_reaches_the_reader_and_a_keyboard_appearing_later_is_read(tmp_path):
    reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reader.bind(("127.0.0.1", 0))
    reader.settimeout(5)
    done = threading.Event()
    worker = threading.Thread(target=relay.run, args=(reader.getsockname(), str(tmp_path / "*-event-kbd"), done.is_set))
    worker.start()
    try:
        time.sleep(0.3)
        keyboard = tmp_path / "usb-test-event-kbd"
        os.mkfifo(keyboard)
        fd = os.open(keyboard, os.O_RDWR)   # held open so the relay's read end never sees EOF
        os.write(fd, _event(1, 30, 1) + _event(1, 186, 1))
        assert reader.recvfrom(64)[0] == b"TBK1 dip\n"
        os.close(fd)
    finally:
        done.set()
        worker.join(5)
