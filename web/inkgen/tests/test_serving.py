"""Admission policy and inference gate, with an injected clock and no model."""
from __future__ import annotations

import threading
import time

import pytest
from contracts import GenerationSettings, normalize_request  # noqa: E402
from engine import check_request  # noqa: E402
from fake_engine import FakeEngine  # noqa: E402
from serving import AdmissionPolicy, InferenceGate, RefusalError, quota  # noqa: E402


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_quota_defaults_are_public_only():
    assert quota("INKGEN_PER_IP_PER_MIN", 6, public=True, env={}) == 6
    assert quota("INKGEN_PER_IP_PER_MIN", 6, public=False, env={}) == 0
    assert quota("INKGEN_PER_IP_PER_MIN", 6, public=False, env={"INKGEN_PER_IP_PER_MIN": "2"}) == 2
    assert quota("INKGEN_PER_IP_PER_MIN", 6, public=True, env={"INKGEN_PER_IP_PER_MIN": "0"}) == 0
    with pytest.raises(ValueError):
        quota("INKGEN_PER_IP_PER_MIN", 6, public=True, env={"INKGEN_PER_IP_PER_MIN": "-1"})


def test_public_policy_from_env_and_private_off():
    public = AdmissionPolicy.from_env(public=True, env={})
    assert (public.per_ip_per_min, public.per_ip_per_day, public.daily_budget_s) == (6, 60, 1800)
    private = AdmissionPolicy.from_env(public=False, env={})
    assert not private.enabled
    for _ in range(100):  # a batch worker's own process never inherits the public cap
        private.admit("198.51.100.1", 12)


def test_per_minute_cap_refuses_with_a_wait_and_recovers():
    clock = Clock()
    policy = AdmissionPolicy(per_ip_per_min=2, clock=clock)
    policy.admit("a", 12)
    clock.t += 10
    policy.admit("a", 12)
    with pytest.raises(RefusalError) as refused:
        policy.admit("a", 12)
    assert refused.value.status == 429
    assert refused.value.retry_after_s == 50
    policy.admit("b", 12)  # another address is not the same visitor
    clock.t += 51
    policy.admit("a", 12)


def test_daily_budget_is_shared_and_reports_spend():
    clock = Clock()
    policy = AdmissionPolicy(daily_budget_s=30, clock=clock)
    policy.admit("a", 12)
    policy.admit("b", 12)
    assert policy.describe()["budget_spent_s"] == 24
    with pytest.raises(RefusalError) as refused:
        policy.admit("c", 12)
    assert refused.value.status == 503
    assert refused.value.retry_after_s >= 60
    clock.t += 86_401
    policy.admit("c", 12)
    assert policy.spent_s() == 12


def test_gate_runs_one_render_at_a_time_and_bounds_the_line():
    gate = InferenceGate(max_waiting=1, wait_s=5)
    engine = FakeEngine(GenerationSettings())
    release = threading.Event()
    inside = threading.Event()
    engine.render_hook = lambda _request: (inside.set(), release.wait(5))
    request = normalize_request({"subject": "a swallow", "seed": 1})
    outcomes: list[str] = []

    def worker(name: str) -> None:
        try:
            with gate.acquire():
                engine.render(request)
            outcomes.append(f"{name}:ok")
        except RefusalError as exc:
            outcomes.append(f"{name}:{exc.status}")

    first = threading.Thread(target=worker, args=("first",))
    first.start()
    assert inside.wait(5)
    assert gate.state()["inference_in_flight"] == 1
    second = threading.Thread(target=worker, args=("second",))
    second.start()
    for _ in range(100):
        if gate.state()["inference_waiting"] == 1:
            break
        time.sleep(0.01)
    assert gate.state()["inference_waiting"] == 1
    # The line is full: a third caller is refused at once, not parked.
    with pytest.raises(RefusalError) as refused, gate.acquire():
        pass
    assert refused.value.status == 503
    release.set()
    first.join(5)
    second.join(5)
    assert sorted(outcomes) == ["first:ok", "second:ok"]
    assert engine.max_concurrent == 1
    assert gate.state() == {"inference_in_flight": 0, "inference_running_s": None,
                            "inference_waiting": 0, "inference_max_waiting": 1}


def test_gate_wait_is_bounded():
    gate = InferenceGate(max_waiting=2, wait_s=0.05)
    held = threading.Event()
    go = threading.Event()

    def hold() -> None:
        with gate.acquire():
            held.set()
            go.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(5)
    with pytest.raises(RefusalError) as refused, gate.acquire():
        pass
    assert refused.value.status == 503 and refused.value.retry_after_s >= 1
    go.set()
    thread.join(5)


def test_model_and_revision_mismatch_are_refused_before_drawing():
    loaded = GenerationSettings(model="Tongyi-MAI/Z-Image-Turbo", model_revision="a" * 40)
    engine = FakeEngine(loaded)
    ok = normalize_request({"subject": "a swallow", "seed": 0}, defaults=loaded)
    engine.render(ok)
    assert engine.calls == [ok]
    other_model = normalize_request({"subject": "a swallow", "seed": 0, "model": "someone/else"}, defaults=loaded)
    with pytest.raises(ValueError, match="serves 'Tongyi-MAI/Z-Image-Turbo', not 'someone/else'"):
        engine.render(other_model)
    other_revision = normalize_request({"subject": "a swallow", "seed": 0, "model_revision": "b" * 40}, defaults=loaded)
    with pytest.raises(ValueError, match="at revision " + "a" * 40):
        check_request(loaded, other_revision)
    assert engine.calls == [ok], "a refused request must not reach the model"
    # An engine that could not resolve its own revision cannot confirm one either.
    unresolved = GenerationSettings(model="Tongyi-MAI/Z-Image-Turbo", model_revision=None)
    pinned = normalize_request({"subject": "a swallow", "seed": 0, "model_revision": "b" * 40}, defaults=unresolved)
    with pytest.raises(ValueError, match="cannot confirm"):
        check_request(unresolved, pinned)
    # An unpinned request against pinned weights is fine: it asked for nothing else.
    check_request(loaded, normalize_request({"subject": "a swallow", "seed": 0}))


def test_fake_engine_follows_the_seed():
    engine = FakeEngine(GenerationSettings())
    a = engine.render(normalize_request({"subject": "x", "seed": 1}))
    b = engine.render(normalize_request({"subject": "x", "seed": 2}))
    again = engine.render(normalize_request({"subject": "x", "seed": 1}))
    assert a.data.startswith(b"\x89PNG") and a.data != b.data and a.data == again.data
