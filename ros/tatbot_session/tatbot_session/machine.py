"""The tattoo machine's power switch (ros/README.md 4.4): the palette Pi's machine relay
(tatbot_estop_relay.py --machine), or a simulated one on mock and fake hardware.

The session sends `MCH1 <seq> <on>` every period_s, off as well as on. The Pi powers the machine only
while those frames keep coming and ask for it and its e-stop relay reads released, and after the e-stop
reads anything else only once it has been asked off again: a dead session, a pressed e-stop and its
release never leave the machine running by themselves. The Pi answers each frame with
`MCS1 <seq> <on> <powered> <estop>`, and a wait counts only answers to the frames sent since it asked.
"""
from __future__ import annotations

import contextlib
import select
import socket
import threading
import time
from dataclasses import dataclass

from tatbot_session.ports import WALL

ESTOP_PRESSED, ESTOP_RELEASED, ESTOP_SILENT = 0, 1, 2   # MCS1's e-stop field (the relay's)


def require_off(switch, powered, timeout_s, activity):
    """Require the current off command's fresh reply for powered non-drawing motion."""
    if switch is None and powered:
        raise RuntimeError(f'{activity} has no switch to establish machine off')
    if switch is not None and (not switch.wait(False, timeout_s) or not switch.is_off()):
        raise RuntimeError(f'{activity} requires a fresh reply to the current machine-off command')


@dataclass(frozen=True)
class Answer:
    seq: int
    on: bool
    powered: bool
    estop: int
    at: float   # monotonic


def reply(switch) -> tuple[bool, float]:
    """SafetyState's (machine_powered, machine_reply_age_s): the newest reply's powered field and its age in s;
    (False, -1) with no switch or before its first reply. Display only: nothing gates on it."""
    answer = getattr(switch, "answer", None)
    if answer is None:
        return False, -1.0
    return answer.powered, max(0.0, time.monotonic() - answer.at)


class PiSwitch:
    def __init__(self, addr: str, port: int, period_s: float):
        self.dest, self.period_s = (addr, int(port)), float(period_s)
        self.on = False
        self.answer: Answer | None = None
        self._seq = 0      # the last frame sent
        self._since = 1    # a wait counts only answers to frames from this one on
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", 0))
        self._thread = threading.Thread(target=self._run, name=f"machine switch {addr}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        due = time.monotonic()
        while not self._stop.is_set():
            with self._cv:
                self._seq += 1
                frame = b"MCH1 %d %d\n" % (self._seq, self.on)
            with contextlib.suppress(OSError):
                self._sock.sendto(frame, self.dest)
            due = max(due + self.period_s, time.monotonic())
            while (wait := due - time.monotonic()) > 0:
                if select.select([self._sock], [], [], wait)[0]:
                    with contextlib.suppress(OSError, ValueError):
                        self._take(self._sock.recv(128))

    def _take(self, data: bytes) -> None:
        tag, seq, on, powered, estop = data.split()
        if tag == b"MCS1":
            with self._cv:
                self.answer = Answer(int(seq), on == b"1", powered == b"1", int(estop), time.monotonic())
                self._cv.notify_all()

    def _fresh(self) -> Answer | None:
        answer = self.answer
        return answer if answer is not None and time.monotonic() - answer.at <= 3 * self.period_s else None

    def set(self, on: bool) -> None:
        with self._cv:
            if bool(on) != self.on:
                self.on, self._since = bool(on), self._seq + 1

    def wait(self, on: bool, timeout_s: float) -> bool:
        """Ask for `on`; True once the Pi reports the machine so, answering a frame sent since."""
        self.set(on)

        def done():
            answer = self._fresh()
            return answer is not None and answer.seq >= self._since and answer.powered == bool(on)

        with self._cv:
            return self._cv.wait_for(done, timeout_s)

    def why_off(self) -> str:
        """Why the machine is not running, from the newest answer."""
        answer = self._fresh()
        if answer is None:
            return f"its switch at {self.dest[0]}:{self.dest[1]} does not answer (tatbot ros relay-install --machine)"
        if answer.estop == ESTOP_PRESSED:
            return "the e-stop is pressed"
        if answer.estop == ESTOP_SILENT:
            return "its switch hears no e-stop frames (tatbot ros relay-install feeds it)"
        return "its switch holds it off until it is asked off after the e-stop" if answer.on else "it was asked off"

    def is_off(self) -> bool:
        """Only a fresh reply to the current off command establishes service power off."""
        with self._cv:
            answer = self._fresh()
            return (not self.on and answer is not None and answer.seq >= self._since
                    and not answer.on and not answer.powered)

    def close(self) -> None:
        self.set(False)
        time.sleep(3 * self.period_s)
        self._stop.set()
        self._thread.join(1.0)
        self._sock.close()


class SimSwitch:
    """A switch that does what it is asked at once (mock, fake and simulated hardware). A simulated machine hears each
    change: `sink(on, t_ns)` in process, or to `endpoint` (udp://host:port, stack.yaml machine.sim_endpoint) the
    datagram `machine <0|1> <seq> <t_ns>`; t_ns is `clock`'s (the session's) now."""

    def __init__(self, sink=None, endpoint=None, clock=None):
        self.on, self.sink, self.clock, self._seq, self._dest = False, sink, clock, 0, None
        if endpoint:
            host, port = str(endpoint).removeprefix("udp://").rsplit(":", 1)
            self._dest, self._sock = (host, int(port)), socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def answer(self) -> Answer:
        """It answers at once: the machine powered as asked, now."""
        return Answer(0, self.on, self.on, ESTOP_RELEASED, time.monotonic())

    def set(self, on: bool) -> None:
        changed, self.on = bool(on) != self.on, bool(on)
        if not changed or (self.sink is None and self._dest is None):
            return
        self._seq += 1
        t_ns = int(round((self.clock or WALL).now() * 1e9))
        if self.sink is not None:
            self.sink(self.on, t_ns)
        if self._dest is not None:
            with contextlib.suppress(OSError):
                self._sock.sendto(b"machine %d %d %d" % (self.on, self._seq, t_ns), self._dest)

    def wait(self, on: bool, timeout_s: float) -> bool:
        self.set(on)
        return True

    def why_off(self) -> str:
        return "it was asked off"

    def is_off(self) -> bool:
        return not self.on

    def close(self) -> None:
        self.set(False)
        if self._dest is not None:
            self._sock.close()


def from_stack(stack: dict, arm: str, clock=None, sink=None):
    """stack.yaml `machine`: the switch of the arm it names, or None. A sim switch tells `sink` (an in-process
    simulator), else machine.sim_endpoint, each change on `clock`."""
    cfg = stack.get("machine") or {}
    kind = cfg.get("switch", "none")
    if kind == "none" or cfg.get("arm") != arm:
        return None
    if kind == "sim":
        return SimSwitch(sink, None if sink is not None else cfg.get("sim_endpoint"), clock)
    if kind != "pi":
        raise ValueError(f"stack.yaml machine.switch is none, pi or sim, not {kind!r}")
    if not cfg.get("addr"):
        raise ValueError("stack.yaml machine.switch is pi but machine.addr is unset (tatbot ros up resolves it)")
    return PiSwitch(cfg["addr"], cfg.get("port", 7642), cfg.get("period_s", 0.02))
