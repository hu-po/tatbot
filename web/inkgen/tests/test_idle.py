"""Idle-stop timer, driven by an injected clock. No model, no GPU, no network."""
from __future__ import annotations

import pytest
from idle import DEFAULT_IDLE_STOP_S, IdleStop, idle_stop_seconds  # noqa: E402


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def timer(timeout_s: int = 900, *, clock: Clock | None = None) -> tuple[IdleStop, Clock, list[int]]:
    clock = clock or Clock()
    stops: list[int] = []
    return IdleStop(timeout_s, monotonic=clock, wall=lambda: 1_700_000_000 + clock.t,
                    stop=lambda: stops.append(1)), clock, stops


def test_default_is_fifteen_minutes_and_zero_gpu_disables_it():
    assert idle_stop_seconds({}) == DEFAULT_IDLE_STOP_S == 900
    assert idle_stop_seconds({"INKGEN_IDLE_STOP_S": "120"}) == 120
    assert idle_stop_seconds({"INKGEN_IDLE_STOP_S": " 0 "}) == 0
    assert idle_stop_seconds({"INKGEN_IDLE_STOP_S": ""}) == DEFAULT_IDLE_STOP_S
    # The Space owns the lifecycle on the Hub, whatever the environment says.
    assert idle_stop_seconds({"INKGEN_IDLE_STOP_S": "120"}, zero_gpu=True) == 0


@pytest.mark.parametrize("value", ["-1", "later", "9.5"])
def test_unusable_timeout_is_refused_not_defaulted(value):
    with pytest.raises(ValueError):
        idle_stop_seconds({"INKGEN_IDLE_STOP_S": value})


def test_stops_only_after_the_whole_timeout_with_no_work():
    idle, clock, stops = timer(900)
    clock.advance(899)
    assert idle.check() is False and stops == []
    clock.advance(1)
    assert idle.check() is True and stops == [1]


def test_work_restarts_the_countdown():
    idle, clock, stops = timer(900)
    clock.advance(800)
    idle.touch()
    clock.advance(800)
    assert idle.check() is False and stops == []
    clock.advance(100)
    assert idle.check() is True


def test_a_request_in_flight_is_never_cut_in_half():
    idle, clock, stops = timer(60)
    with idle.hold():
        clock.advance(600)  # a render slower than the whole timeout
        assert idle.check() is False and stops == []
        assert idle.remaining_s() == 60
    assert idle.check() is False  # leaving the hold counts as work
    clock.advance(60)
    assert idle.check() is True


def test_disabled_timer_never_stops_and_starts_no_thread():
    idle, clock, stops = timer(0)
    assert idle.enabled is False
    clock.advance(10_000)
    assert idle.check() is False and stops == []
    assert idle.start() is None
    assert idle.remaining_s() is None
    assert idle.state()["idle_stop_s"] is None


def test_it_stops_once():
    idle, clock, stops = timer(10)
    clock.advance(100)
    assert idle.check() is True
    assert idle.check() is False
    assert stops == [1]


def test_health_fields_report_the_last_request_and_the_countdown():
    idle, clock, _ = timer(900)
    assert idle.state()["last_request_unix"] is None  # nothing has been generated yet
    assert idle.state()["idle_stop_in_s"] == 900
    idle.touch()
    clock.advance(300)
    state = idle.state()
    assert state["last_request_unix"] == 1_700_001_000
    assert state["idle_stop_in_s"] == 600
    assert state["idle_stop_s"] == 900
    assert state["requests_in_flight"] == 0
    with idle.hold():
        assert idle.state()["requests_in_flight"] == 1
