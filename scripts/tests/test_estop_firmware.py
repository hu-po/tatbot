"""Run the actual Pico loop against synthetic clocks and USB, with no devices."""

import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
from estop_check import Sample, validate_sample  # noqa: E402


class EndSimulation(BaseException):
    pass


class CoarseFloat(float):
    """Model the loss of fractional precision at long CircuitPython uptimes."""

    def __add__(self, other):
        value = float(self) + other
        return CoarseFloat(round(value * 8) / 8) if value >= 2**18 else CoarseFloat(value)


@pytest.mark.parametrize("uptime_s", [0, 2**18, 2**22])
def test_firmware_keeps_100hz_after_long_uptime(monkeypatch, uptime_s):
    now_ns = uptime_s * 1_000_000_000
    frames = []
    runtime = SimpleNamespace(autoreload=True)

    def monotonic():
        seconds = now_ns / 1_000_000_000
        return CoarseFloat(round(seconds * 8) / 8) if uptime_s else CoarseFloat(seconds)

    def sleep(seconds):
        nonlocal now_ns
        assert math.isfinite(seconds) and 0 < seconds <= 0.01
        now_ns += round(seconds * 1_000_000_000)

    def write(frame):
        nonlocal now_ns
        # A host filesystem notification halfway through a run must not
        # terminate this VM and restart the heartbeat counter.
        if len(frames) == 50 and runtime.autoreload:
            raise EndSimulation
        if len(frames) == 100:
            raise EndSimulation
        frames.append((now_ns, frame))
        now_ns += 500_000  # GPIO/USB/interpreter work within each iteration
        return len(frame)

    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(
        monotonic=monotonic, monotonic_ns=lambda: now_ns, sleep=sleep))
    monkeypatch.setitem(sys.modules, "board", SimpleNamespace(GP2=2, LED=25))
    monkeypatch.setitem(sys.modules, "digitalio", SimpleNamespace(
        DigitalInOut=lambda pin: SimpleNamespace(value=False),
        Direction=SimpleNamespace(INPUT=0, OUTPUT=1), Pull=SimpleNamespace(UP=1)))
    monkeypatch.setitem(sys.modules, "usb_cdc", SimpleNamespace(data=SimpleNamespace(write=write)))
    monkeypatch.setitem(sys.modules, "supervisor", SimpleNamespace(runtime=runtime))
    with pytest.raises(EndSimulation):
        runpy.run_path(str(REPO / "firmware/estop_pico/code.py"))
    assert len(frames) == 100
    intervals = [b[0] - a[0] for a, b in zip(frames, frames[1:], strict=False)]
    assert all(interval == 10_000_000 for interval in intervals)
    assert [frame for _, frame in frames] == [f"EST1 {seq} 1\n".encode() for seq in range(100)]


def test_bench_check_rejects_live_observed_heartbeat_flood():
    sample = Sample(3.0, tuple(range(5268)), (1,) * 5268, 0)
    assert any("above" in error for error in validate_sample(sample, "released", 80))


def test_bench_check_accepts_nominal_heartbeat():
    sample = Sample(3.0, tuple(range(299)), (1,) * 299, 0)
    assert validate_sample(sample, "released", 80) == []


@pytest.mark.parametrize("usb_stalled", [False, True])
def test_nc_input_and_stop_led_without_usb_progress(monkeypatch, usb_stalled):
    """Exercise closed -> open -> closed and an open input with blocked USB."""
    now_ns = 0
    frames = []
    led_samples = []
    nc_pin = SimpleNamespace(value=usb_stalled)
    led = SimpleNamespace(value=False)

    def sleep(seconds):
        nonlocal now_ns
        led_samples.append(led.value)
        now_ns += round(seconds * 1_000_000_000)
        if len(led_samples) == 30:
            raise EndSimulation
        nc_pin.value = usb_stalled or 10 <= len(led_samples) < 20

    def write(frame):
        if usb_stalled:
            return 0
        frames.append(frame)
        return len(frame)

    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(
        monotonic_ns=lambda: now_ns, sleep=sleep))
    monkeypatch.setitem(sys.modules, "board", SimpleNamespace(GP2=2, LED=25))
    monkeypatch.setitem(sys.modules, "digitalio", SimpleNamespace(
        DigitalInOut=lambda pin: {2: nc_pin, 25: led}[pin],
        Direction=SimpleNamespace(INPUT=0, OUTPUT=1), Pull=SimpleNamespace(UP=1)))
    monkeypatch.setitem(sys.modules, "usb_cdc", SimpleNamespace(data=SimpleNamespace(write=write)))
    monkeypatch.setitem(sys.modules, "supervisor", SimpleNamespace(
        runtime=SimpleNamespace(autoreload=True)))
    with pytest.raises(EndSimulation):
        runpy.run_path(str(REPO / "firmware/estop_pico/code.py"))

    assert nc_pin.direction == 0 and nc_pin.pull == 1
    if usb_stalled:
        assert led_samples == ([True] * 5 + [False] * 5) * 3
    else:
        states = [1] * 10 + [0] * 10 + [1] * 10
        assert frames == [f"EST1 {seq} {state}\n".encode() for seq, state in enumerate(states)]
        assert led_samples == [True] * 15 + [False] * 5 + [True] * 10
