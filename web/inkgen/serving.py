"""Serving policy: who is admitted to the GPU, and how many draw at once.

Stdlib only and engine-free, like `contracts.py`, so the rules a public
deployment lives by can be exercised on a laptop with no model in sight.

Two boundaries, kept apart on purpose:

* `AdmissionPolicy` is the visitor quota — a few per minute per address, a
  daily per-address cap and a daily GPU-seconds budget for the whole service.
  It protects the owner's shared allowance from strangers, so it belongs to
  *every* public entry (the HTTP API and the Gradio page alike) and to none
  of the private ones: a batch worker pointed at its own process owns its own
  job limits and configures the quota off.
* `InferenceGate` is the engine's concurrency: one render at a time, and a
  bounded number of callers allowed to wait for it. Health probes are cheap
  and are never queued behind a render.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

__all__ = ["AdmissionPolicy", "InferenceGate", "RefusalError", "quota"]

DEFAULT_PUBLIC_PER_IP_PER_MIN = 6
DEFAULT_PUBLIC_PER_IP_PER_DAY = 60
DEFAULT_PUBLIC_DAILY_BUDGET_S = 1800
DEFAULT_MAX_WAITING = 3
DEFAULT_QUEUE_WAIT_S = 120.0


class RefusalError(Exception):
    """A request this boundary will not serve, with the status the caller gets."""

    def __init__(self, status: int, detail: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(detail)
        self.status = int(status)
        self.detail = detail
        self.retry_after_s = None if retry_after_s is None else max(1, int(round(retry_after_s)))


def quota(name: str, public_default: int, *, public: bool, env: dict[str, str] | os._Environ | None = None) -> int:
    """A limit's configured value, or 0 for "no limit here".

    Off the Hub the caps default to off; any deploy that wants them states the
    numbers explicitly. A stated value wins everywhere.
    """
    env = os.environ if env is None else env
    raw = env.get(name)
    if raw not in (None, ""):
        value = int(raw)
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")
        return value
    return public_default if public else 0


class AdmissionPolicy:
    """Per-address and whole-service quotas, in memory, per replica."""

    def __init__(self, *, per_ip_per_min: int = 0, per_ip_per_day: int = 0, daily_budget_s: int = 0,
                 clock: Callable[[], float] = time.time) -> None:
        for name, value in (("per_ip_per_min", per_ip_per_min), ("per_ip_per_day", per_ip_per_day),
                            ("daily_budget_s", daily_budget_s)):
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
        self.per_ip_per_min = int(per_ip_per_min)
        self.per_ip_per_day = int(per_ip_per_day)
        self.daily_budget_s = int(daily_budget_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._per_ip: dict[str, deque[float]] = defaultdict(deque)
        self._budget: deque[tuple[float, float]] = deque()  # (t, seconds)

    @classmethod
    def from_env(cls, *, public: bool, env: dict[str, str] | os._Environ | None = None,
                 clock: Callable[[], float] = time.time) -> AdmissionPolicy:
        return cls(per_ip_per_min=quota("INKGEN_PER_IP_PER_MIN", DEFAULT_PUBLIC_PER_IP_PER_MIN, public=public, env=env),
                   per_ip_per_day=quota("INKGEN_PER_IP_PER_DAY", DEFAULT_PUBLIC_PER_IP_PER_DAY, public=public, env=env),
                   daily_budget_s=quota("INKGEN_DAILY_BUDGET_S", DEFAULT_PUBLIC_DAILY_BUDGET_S, public=public, env=env),
                   clock=clock)

    @property
    def enabled(self) -> bool:
        return bool(self.per_ip_per_min or self.per_ip_per_day or self.daily_budget_s)

    def admit(self, ip: str, cost_s: float) -> None:
        """Charge one request to `ip`, or refuse it with the reason and a wait."""
        now = self._clock()
        with self._lock:
            q = self._per_ip[ip]
            while q and now - q[0] > 86400:
                q.popleft()
            if self.per_ip_per_min:
                recent = [t for t in q if now - t < 60]
                if len(recent) >= self.per_ip_per_min:
                    raise RefusalError(429, "slow down — a few per minute is plenty",
                                       retry_after_s=60 - (now - recent[0]))
            if self.per_ip_per_day and len(q) >= self.per_ip_per_day:
                raise RefusalError(429, "daily limit for this address reached", retry_after_s=86400 - (now - q[0]))
            while self._budget and now - self._budget[0][0] > 86400:
                self._budget.popleft()
            if self.daily_budget_s and sum(s for _, s in self._budget) + cost_s > self.daily_budget_s:
                oldest = self._budget[0][0] if self._budget else now
                raise RefusalError(503, "the generator's daily budget is spent — try again tomorrow",
                                   retry_after_s=max(60.0, 86400 - (now - oldest)))
            q.append(now)
            self._budget.append((now, cost_s))

    def spent_s(self) -> float:
        now = self._clock()
        with self._lock:
            return sum(s for t, s in self._budget if now - t <= 86400)

    def describe(self) -> dict[str, object]:
        """The health document's quota fields."""
        return {"budget_s": self.daily_budget_s, "budget_spent_s": round(self.spent_s(), 1),
                "per_ip_per_min": self.per_ip_per_min, "per_ip_per_day": self.per_ip_per_day}


class InferenceGate:
    """One render at a time per engine, with a bounded waiting line.

    A caller past the line is refused at once (503, busy) rather than parked
    for an unbounded time; one inside the line waits at most `wait_s` for the
    engine before it is refused the same way. Nothing here touches the GPU.
    """

    def __init__(self, *, max_waiting: int = DEFAULT_MAX_WAITING, wait_s: float = DEFAULT_QUEUE_WAIT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_waiting < 0:
            raise ValueError(f"max_waiting must not be negative, got {max_waiting}")
        if wait_s < 0:
            raise ValueError(f"wait_s must not be negative, got {wait_s}")
        self.max_waiting = int(max_waiting)
        self.wait_s = float(wait_s)
        self._clock = clock
        self._engine = threading.Lock()
        self._state = threading.Lock()
        self._waiting = 0
        self._running = 0
        self._started: float | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | os._Environ | None = None) -> InferenceGate:
        env = os.environ if env is None else env
        return cls(max_waiting=int(env.get("INKGEN_MAX_WAITING") or DEFAULT_MAX_WAITING),
                   wait_s=float(env.get("INKGEN_QUEUE_WAIT_S") or DEFAULT_QUEUE_WAIT_S))

    @contextmanager
    def acquire(self) -> Iterator[None]:
        with self._state:
            if self._waiting >= self.max_waiting:
                raise RefusalError(503, "the generator is busy — try again in a moment", retry_after_s=self.wait_s / 4)
            self._waiting += 1
        try:
            got = self._engine.acquire(timeout=self.wait_s)
        finally:
            with self._state:
                self._waiting -= 1
        if not got:
            raise RefusalError(503, "the generator is busy — try again in a moment", retry_after_s=self.wait_s / 4)
        with self._state:
            self._running = 1
            self._started = self._clock()
        try:
            yield
        finally:
            with self._state:
                self._running = 0
                self._started = None
            self._engine.release()

    def state(self) -> dict[str, object]:
        """The health document's scheduling fields."""
        with self._state:
            running = self._running
            started = self._started
            waiting = self._waiting
        return {"inference_in_flight": running,
                "inference_running_s": round(self._clock() - started, 1) if started is not None else None,
                "inference_waiting": waiting, "inference_max_waiting": self.max_waiting}
