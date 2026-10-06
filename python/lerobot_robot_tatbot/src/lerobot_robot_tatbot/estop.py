"""Monitor for the tatbot hardware e-stop (firmware/estop_pico/, or the palette Pi's relay).

The box is a latching NC mushroom button wired to a Pico that streams
``EST1 <seq> <0|1>\\n`` heartbeat frames over USB CDC at 100 Hz. The stream
itself is the safety signal: a press, an unplugged cable, wedged firmware, or
a dead board all stop producing state-1 frames and read as ENGAGED. Mirrors
the C++ monitor in cpp/teleop/estop_monitor.cpp — keep protocol constants in
sync with it and with the firmware.

A device ``udp://[HOST]:PORT?from=SOURCE`` reads the same frames from the palette
Pi's relay instead (ros/tatbot_estop_relay), one per datagram, so one button
stops the ROS stack's arm and this one. SOURCE is the relay's IPv4 address or
the fleet role that names it (``estop-relay``, its ``lan`` in config/nodes.json);
datagrams from anywhere else are ignored, which is silence, never a release.
That path mirrors the ROS driver's UDP reader (ros/tatbot_hardware/src/estop.cpp).
"""

import contextlib
import ipaddress
import json
import logging
import os
import select
import socket
import tempfile
import termios
import threading
import time
import urllib.parse
from enum import Enum
from pathlib import Path

from lerobot_robot_tatbot import paths
from lerobot_robot_tatbot.driver_lease import acquire as acquire_driver_lease

logger = logging.getLogger(__name__)

# From the hardware profile / TATBOT_ESTOP_DEVICE; the private rig maps its
# board to a stable name via config/udev/99-tatbot-estop.rules.
DEFAULT_DEVICE = paths.driver_default("estop_device", "TATBOT_ESTOP_DEVICE")
DEBOUNCE_FRAMES = 3  # 30 ms at the 100 Hz frame rate
HEARTBEAT_TIMEOUT_S = 0.100
UDP_HEARTBEAT_TIMEOUT_S = 0.150  # the relay path's budget: ros stack.yaml estop.udp_timeout_s
UDP_MAX_FRAME = 128  # bytes, newline included; the relay and the ROS driver hold the same limit
REOPEN_PERIOD_S = 0.5
STATUS_PUBLISH_PERIOD_S = 0.25


def udp_source(device: str) -> tuple[tuple[str, int], str] | None:
    """(bind address, the relay's IPv4 address) of a ``udp://[HOST]:PORT?from=SOURCE`` e-stop, None for a
    serial device. Raises ValueError for a UDP device it cannot resolve."""
    if not device.startswith("udp://"):
        return None
    url = urllib.parse.urlsplit(device)
    source = (urllib.parse.parse_qs(url.query).get("from") or [""])[0]
    if url.port is None or not source:
        raise ValueError(f"e-stop {device!r}: want udp://[HOST]:PORT?from=<relay IPv4 or role>")
    try:
        relay = str(ipaddress.IPv4Address(source))
    except ValueError:
        relay = _role_lan(source)
    return (url.hostname or "0.0.0.0", url.port), relay


def _role_lan(role: str) -> str:
    """The `lan` of the single node carrying `role` in config/nodes.json."""
    nodes = json.loads((paths.repo_root() / "config" / "nodes.json").read_text())
    lans = [rec.get("lan") for rec in nodes.values() if isinstance(rec, dict) and role in rec.get("roles", [])]
    if len(lans) != 1 or not lans[0]:
        raise ValueError(f"e-stop relay: config/nodes.json names {len(lans)} node(s) with role {role!r} and a lan")
    return str(ipaddress.IPv4Address(lans[0]))


def _parse_frame(data: bytes) -> tuple[int, int] | None:
    """(seq, state) of one ``EST1 <seq> <0|1>`` frame, as estop.cpp parse_frame reads it."""
    parts = data.removesuffix(b"\n").removesuffix(b"\r").split(b" ")
    if len(parts) == 3 and parts[0] == b"EST1" and parts[1].isdigit() and len(parts[1]) <= 18 and parts[2] in (b"0", b"1"):
        return int(parts[1]), int(parts[2])
    return None


def _status_path() -> Path:
    override = os.environ.get("TATBOT_ESTOP_STATUS")
    if override:
        return Path(override).expanduser()
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(runtime) if runtime else Path("/tmp") / f"tatbot-{os.getuid()}"
    return root / "tatbot" / "estop-status.json"

_registry_lock = threading.Lock()
_registry: dict[str, tuple["EstopMonitor", int]] = {}



class EstopState(Enum):
    DISABLED = "disabled"  # no device configured; hardware e-stop not in play
    OK = "ok"
    PRESSED = "pressed"  # debounced button press (NC circuit open)
    FAULT = "fault"  # no valid heartbeat within the timeout


class _RelayFrames:
    """The relay stream's reading, kept as ros/tatbot_hardware/src/estop.cpp's Reader keeps it: a reordered
    or replayed frame is dropped until silence re-seeds the sequence, three frames decide, and silence past
    the relay path's budget is a fault."""

    def __init__(self):
        self.last_frame: float | None = None
        self.last_seq = -1
        self.raw = self.stable = -1
        self.count = 0

    def accept(self, seq: int, state: int, now: float) -> None:
        stale = self.last_frame is None or now - self.last_frame > UDP_HEARTBEAT_TIMEOUT_S
        if not stale and seq <= self.last_seq:
            return  # reordered or replayed
        if stale:  # after silence the sequence re-seeds
            self.raw = self.stable = -1
            self.count = 0
        self.last_seq, self.last_frame = seq, now
        if state == self.raw:
            self.count = min(self.count + 1, DEBOUNCE_FRAMES)
        else:
            self.raw, self.count = state, 1
        if self.count >= DEBOUNCE_FRAMES:
            self.stable = state

    def state(self, now: float) -> "EstopState | None":
        """FAULT past the budget; PRESSED or OK once three frames agree; None while undecided."""
        if self.last_frame is None or now - self.last_frame > UDP_HEARTBEAT_TIMEOUT_S:
            return EstopState.FAULT
        return {0: EstopState.PRESSED, 1: EstopState.OK}.get(self.stable)


class EstopMonitor:
    """Background reader owning one atomic-ish state the control path polls.

    ``state`` reads are a single attribute load (GIL-atomic); the caller's
    control loop pays nothing for the safety check.
    """

    def __init__(self, device: str = DEFAULT_DEVICE, required: bool = False):
        self.device = device
        self.state = EstopState.DISABLED
        self._fd: int | None = None
        self._sock: socket.socket | None = None  # a udp:// device's bound socket (it owns self._fd)
        self._relay: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: dict | None = None
        self._status_thread: threading.Thread | None = None

        self._driver_lease = acquire_driver_lease()
        self._fd = self._open_device()
        if self._fd is None:
            self._driver_lease.close()
            if required:
                raise RuntimeError(f"cannot open e-stop device: {device}")
            logger.warning(
                "e-stop device %s not found — running WITHOUT hardware e-stop "
                "(plug it in and restart, or set estop_required)", device
            )
            return
        self.state = EstopState.FAULT  # engaged until the first healthy frames
        self._thread = threading.Thread(target=self._run if self._sock is None else self._run_udp,
                                        name="estop-monitor", daemon=True)
        self._thread.start()
        self._status_thread = threading.Thread(target=self._status_loop, name="estop-status", daemon=True)
        try:
            self._status_thread.start()
        except RuntimeError:
            self._status_thread = None
            logger.warning("e-stop status writer unavailable; serial monitoring continues")

    @property
    def engaged(self) -> bool:
        return self.state in (EstopState.PRESSED, EstopState.FAULT)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        elif self._fd is not None:
            os.close(self._fd)
        self._fd = None
        if self._status_thread is not None:
            self._status_thread.join(timeout=2.0)
        path = _status_path()
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and payload.get("pid") == os.getpid() and payload.get("device") == self.device:
                path.unlink()
        except (OSError, ValueError, TypeError):
            pass

        lease = getattr(self, "_driver_lease", None)
        if lease is not None and (self._thread is None or not self._thread.is_alive()):
            lease.close()

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
        if self.device.startswith("udp://"):
            return self._open_udp()
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

    def _open_udp(self) -> int | None:
        """Bind the relay's port. No SO_REUSEADDR: a second reader of the port is refused rather than
        silently handed part of the frames."""
        try:
            bind, relay = udp_source(self.device)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(bind)
            except OSError:
                sock.close()
                raise
        except (OSError, ValueError) as exc:
            logger.warning("e-stop %s unavailable: %s", self.device, exc)
            return None
        sock.setblocking(False)
        self._sock, self._relay = sock, relay
        return sock.fileno()

    def _run_udp(self) -> None:
        """The relay's datagrams, read as ros/tatbot_hardware/src/estop.cpp reads them (_RelayFrames)."""
        frames = _RelayFrames()
        while not self._stop.is_set():
            if select.select([self._sock], [], [], 0.02)[0]:
                for seq, state in self._relay_datagrams():
                    frames.accept(seq, state, time.monotonic())
            state = frames.state(time.monotonic())
            if state is not None:  # undecided with frames flowing: keep the current (engaged) state
                self.state = state
            self._capture_status(frames.last_seq, frames.last_frame)

    def _relay_datagrams(self):
        """The frames waiting on the socket that came from the relay's address and parse, one per datagram."""
        while True:
            try:
                data, (source, _) = self._sock.recvfrom(512)
            except OSError:  # drained (BlockingIOError) or failing: silence, which times out
                return
            frame = _parse_frame(data) if len(data) <= UDP_MAX_FRAME and source == self._relay else None
            if frame is not None:
                yield frame

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
                self._capture_status(-1, None)
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
            self._capture_status(last_seq, last_frame if last_seq >= 0 else None)

    def _capture_status(self, sequence: int, last_frame: float | None) -> None:
        # One immutable snapshot assignment; the serial reader never performs disk I/O.
        state = self.state
        self._snapshot = {
            "schema": "tatbot.estop-status/1", "pid": os.getpid(), "device": self.device,
            "state": state.value, "engaged": state is not EstopState.OK,
            "updated_unix": time.time(),
            "heartbeat_age_ms": None if last_frame is None else
                round(max(0.0, time.monotonic() - last_frame) * 1000, 1),
            "last_sequence": None if sequence < 0 else sequence,
        }

    def _status_loop(self) -> None:
        last_publish, last_state = 0.0, None
        while not self._stop.wait(0.02):
            payload = self._snapshot
            now = time.monotonic()
            if payload and (payload["state"] != last_state or now - last_publish >= STATUS_PUBLISH_PERIOD_S):
                self._publish_status(payload)
                last_publish, last_state = now, payload["state"]

    def _publish_status(self, payload: dict) -> None:
        path = _status_path()
        tmp = None
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                             delete=False) as output:
                tmp = Path(output.name)
                output.write(json.dumps(payload, sort_keys=True) + "\n")
            if not self._stop.is_set():
                os.replace(tmp, path)
        except OSError as exc:
            logger.debug("cannot publish e-stop status to %s: %s", path, exc)
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)


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
