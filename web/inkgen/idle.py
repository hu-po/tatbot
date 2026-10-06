"""Idle stop: a locally hosted generator that goes away when nobody is using it.

The model is ~13 GB of VRAM on a node that also runs simulation, and a warm
start is about half a minute, so the generator is meant to be started on
demand and to leave by itself. Only real work counts as use: `/api/generate`
touches the timer, health probes and page loads do not.

Disabled (`timeout_s == 0`) under ZeroGPU, where the Space owns the lifecycle.

Stdlib only, and no model import, so `web/inkgen/tests/test_idle.py` can drive
the whole thing with an injected clock on any node.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

DEFAULT_IDLE_STOP_S = 900
POLL_S = 5.0


def idle_stop_seconds(env: dict[str, str] | os._Environ = os.environ, *, zero_gpu: bool = False) -> int:
    """Seconds of idleness before the process stops; 0 never stops.

    ZeroGPU forces 0: on the Hub the Space, not this timer, owns the lifecycle.
    """
    if zero_gpu:
        return 0
    raw = (env.get("INKGEN_IDLE_STOP_S") or "").strip()
    if not raw:
        return DEFAULT_IDLE_STOP_S
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"INKGEN_IDLE_STOP_S must be an integer number of seconds, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"INKGEN_IDLE_STOP_S must not be negative, got {value}")
    return value


def _terminate() -> None:
    """Ask this process to shut down the way `inkgen_ctl.sh stop` would."""
    os.kill(os.getpid(), signal.SIGTERM)


class IdleStop:
    """Stop the process after `timeout_s` with no generation.

    Work in flight holds the timer open, so a render slower than the timeout is
    never cut in half. The clocks are injectable: `monotonic` drives expiry (no
    wall-clock jumps), `wall` is only reported to callers.
    """

    def __init__(self, timeout_s: int, *, monotonic: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, stop: Callable[[], None] = _terminate,
                 poll_s: float = POLL_S) -> None:
        if timeout_s < 0:
            raise ValueError(f"idle timeout must not be negative, got {timeout_s}")
        if poll_s <= 0:
            raise ValueError(f"poll interval must be positive, got {poll_s}")
        self.timeout_s = int(timeout_s)
        self.poll_s = float(poll_s)
        self._monotonic = monotonic
        self._wall = wall
        self._stop = stop
        self._lock = threading.Lock()
        self._last = monotonic()
        self._last_wall: float | None = None
        self._busy = 0
        self._fired = False
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0

    @property
    def fired(self) -> bool:
        with self._lock:
            return self._fired

    def touch(self) -> None:
        """Record real work. Health probes must not call this."""
        with self._lock:
            self._last = self._monotonic()
            self._last_wall = self._wall()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Keep the generator alive for the length of one request."""
        with self._lock:
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1
                self._last = self._monotonic()
                self._last_wall = self._wall()

    def idle_s(self) -> float:
        with self._lock:
            return 0.0 if self._busy else max(0.0, self._monotonic() - self._last)

    def remaining_s(self) -> float | None:
        """Seconds until the stop, or None when the timer is disabled."""
        if not self.enabled:
            return None
        return round(max(0.0, self.timeout_s - self.idle_s()), 1)

    def state(self) -> dict[str, object]:
        """The health document's idle fields."""
        with self._lock:
            last_wall = self._last_wall
            busy = self._busy
        return {"idle_stop_s": self.timeout_s if self.enabled else None,
                "idle_stop_in_s": self.remaining_s(),
                "last_request_unix": round(last_wall, 1) if last_wall is not None else None,
                "requests_in_flight": busy}

    def expired(self) -> bool:
        return self.enabled and not self.fired and self.idle_s() >= self.timeout_s

    def check(self) -> bool:
        """Stop once when idle long enough. Returns whether it fired."""
        if not self.expired():
            return False
        with self._lock:
            if self._fired:
                return False
            self._fired = True
        print(f"inkgen: idle for {self.timeout_s} s — stopping (INKGEN_IDLE_STOP_S=0 keeps it running)", flush=True)
        self._stop()
        return True

    def start(self) -> threading.Thread | None:
        """Run the watchdog in the background. A disabled timer starts nothing."""
        if not self.enabled or self._thread is not None:
            return self._thread
        thread = threading.Thread(target=self._run, name="inkgen-idle-stop", daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def _run(self) -> None:  # pragma: no cover - exercised through check() in tests
        while not self.check():
            time.sleep(self.poll_s)
