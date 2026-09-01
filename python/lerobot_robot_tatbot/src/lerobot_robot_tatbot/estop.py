"""Monitor for the tatbot hardware e-stop (firmware/estop_pico/).

The box is a latching NC mushroom button wired to a Pico that streams
``EST1 <seq> <0|1>\\n`` heartbeat frames over USB CDC at 100 Hz. The stream
itself is the safety signal: a press, an unplugged cable, wedged firmware, or
a dead board all stop producing state-1 frames and read as ENGAGED. Mirrors
the C++ monitor in cpp/teleop/estop_monitor.cpp — keep protocol constants in
sync with it and with the firmware.
"""

import logging
import os
import select
import termios
import threading
import time
from enum import Enum

from lerobot_robot_tatbot import paths

logger = logging.getLogger(__name__)

# From the hardware profile / TATBOT_ESTOP_DEVICE; the private rig maps its
# board to a stable name via config/udev/99-tatbot-estop.rules.
DEFAULT_DEVICE = paths.driver_default("estop_device", "TATBOT_ESTOP_DEVICE")
DEBOUNCE_FRAMES = 3  # 30 ms at the 100 Hz frame rate
HEARTBEAT_TIMEOUT_S = 0.100
REOPEN_PERIOD_S = 0.5

_registry_lock = threading.Lock()
_registry: dict[str, tuple["EstopMonitor", int]] = {}


class EstopState(Enum):
    DISABLED = "disabled"  # no device configured; hardware e-stop not in play
    OK = "ok"
    PRESSED = "pressed"  # debounced button press (NC circuit open)
    FAULT = "fault"  # no valid heartbeat within the timeout


class EstopMonitor:
    """Background reader owning one atomic-ish state the control path polls.

    ``state`` reads are a single attribute load (GIL-atomic); the caller's
    control loop pays nothing for the safety check.
    """

    def __init__(self, device: str = DEFAULT_DEVICE, required: bool = False):
        self.device = device
        self.state = EstopState.DISABLED
        self._fd: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._fd = self._open_device()
        if self._fd is None:
            if required:
                raise RuntimeError(f"cannot open e-stop device: {device}")
            logger.warning(
                "e-stop device %s not found — running WITHOUT hardware e-stop "
                "(plug it in and restart, or set estop_required)", device
            )
            return
        self.state = EstopState.FAULT  # engaged until the first healthy frames
        self._thread = threading.Thread(target=self._run, name="estop-monitor", daemon=True)
        self._thread.start()

    @property
    def engaged(self) -> bool:
        return self.state in (EstopState.PRESSED, EstopState.FAULT)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def wait_for_initial_state(self, timeout_s: float = 0.5) -> EstopState:
        """Wait until healthy frames establish OK/PRESSED, or timeout.

        FAULT is deliberately retained on timeout: silence and an undecided
        startup state are both stop conditions.
        """
        deadline = time.monotonic() + timeout_s
        while self.state is EstopState.FAULT and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.state

    def _open_device(self) -> int | None:
        try:
            fd = os.open(self.device, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError:
            return None
        try:
            if os.isatty(fd):
                attrs = termios.tcgetattr(fd)
                # cfmakeraw equivalent: no line buffering / translation.
                attrs[0] = attrs[1] = attrs[3] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except termios.error:
            pass
        return fd

    def _run(self) -> None:
        buffer = b""
        last_frame = time.monotonic()
        last_reopen = time.monotonic()
        last_seq = -1
        raw_state = -1
        stable_state = -1  # debounced; -1 = no reading believed yet
        stable_count = 0

        while not self._stop.is_set():
            if self._fd is None:
                # Device vanished (USB unplug): stay engaged, retry so
                # replugging the box recovers without a restart.
                self.state = EstopState.FAULT
                if time.monotonic() - last_reopen > REOPEN_PERIOD_S:
                    last_reopen = time.monotonic()
                    self._fd = self._open_device()
                    if self._fd is not None:
                        buffer = b""
                        last_seq = raw_state = stable_state = -1
                        stable_count = 0
                        last_frame = time.monotonic()
                time.sleep(0.05)
                continue

            readable, _, _ = select.select([self._fd], [], [], 0.02)
            if readable:
                try:
                    chunk = os.read(self._fd, 256)
                except BlockingIOError:
                    chunk = None  # spurious wakeup; nothing to parse
                except OSError:
                    chunk = b""
                if chunk == b"":  # EOF or hard error
                    os.close(self._fd)
                    self._fd = None
                    continue
                if chunk:
                    buffer += chunk
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    parts = line.split()
                    if (
                        len(parts) == 3
                        and parts[0] == b"EST1"
                        and parts[1].isdigit()
                        and parts[2] in (b"0", b"1")
                    ):
                        seq, state = int(parts[1]), int(parts[2])
                        if last_seq >= 0 and seq <= last_seq:
                            raw_state = stable_state = -1  # sender rebooted
                            stable_count = 0
                        last_seq = seq
                        last_frame = time.monotonic()
                        # Debounce contact bounce around press/twist-release.
                        if state == raw_state:
                            stable_count = min(stable_count + 1, DEBOUNCE_FRAMES)
                        else:
                            raw_state = state
                            stable_count = 1
                        if stable_count >= DEBOUNCE_FRAMES:
                            stable_state = state
                if len(buffer) > 1024:
                    buffer = b""  # garbage stream; resync

            if time.monotonic() - last_frame > HEARTBEAT_TIMEOUT_S:
                self.state = EstopState.FAULT
            elif stable_state == 0:
                self.state = EstopState.PRESSED
            elif stable_state == 1:
                self.state = EstopState.OK
            # stable_state == -1: frames flowing but nothing debounced yet —
            # keep the current (engaged) state until a reading is believed.


def acquire_estop(
    device: str = DEFAULT_DEVICE, *, required: bool = True
) -> EstopMonitor | None:
    """Acquire the one process-wide reader for ``device``.

    Serial heartbeat bytes must have exactly one reader. Leader, follower,
    coordinated lifecycle motion, and recovery therefore share this monitor
    rather than opening the tty independently.
    """
    if not device:
        # A REQUIRED monitor with no device is a refusal, never a silent
        # skip: after the profile refactor an unresolvable profile yields
        # device="" with estop_required still True, and the old behavior
        # (adversarial audit 2026-08-31, finding 1) let arms move
        # unmonitored. Empty device + required fails closed, loudly.
        if required:
            raise RuntimeError(
                "e-stop required but no device resolved — the hardware "
                "profile did not supply driver.estop_device (check "
                "TATBOT_PROFILE / config/profiles/). An explicit "
                "hardware-free bench run must opt out via estop_required.")
        return None
    with _registry_lock:
        entry = _registry.get(device)
        if entry is not None:
            monitor, refs = entry
            if required and monitor.state is EstopState.DISABLED:
                raise RuntimeError(f"cannot open e-stop device: {device}")
            _registry[device] = (monitor, refs + 1)
            return monitor
        monitor = EstopMonitor(device, required=required)
        _registry[device] = (monitor, 1)
        return monitor


def release_estop(monitor: EstopMonitor | None) -> None:
    """Release a shared monitor acquired with :func:`acquire_estop`."""
    if monitor is None:
        return
    close = False
    with _registry_lock:
        for device, (candidate, refs) in list(_registry.items()):
            if candidate is not monitor:
                continue
            if refs <= 1:
                del _registry[device]
                close = True
            else:
                _registry[device] = (candidate, refs - 1)
            break
    if close:
        monitor.close()
