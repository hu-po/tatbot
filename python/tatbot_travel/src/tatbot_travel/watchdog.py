"""A bound on every vendor driver call the travel runner makes, kept by a process of its own.

What blocks. The trossen_arm driver (1.8.5, for controller firmware 1.8.x) sends motion over UDP and the
rest over TCP, request then reply: configure()'s handshake, every configuration get and set (modes, end
effector, joint limits, characteristics, motor parameters, the error state and log), and cleanup(), which
idles the joints. Its TCP socket gets TCP_NODELAY and a send timeout, never a receive timeout, and the reply
is read with a blocking recv() that is neither retried nor timed: once the controller's TCP server wedges with
the connection still open, that read never returns. configure()'s own timeout bounds only its connect.
Position reads and motion commands wait on the driver's data mutex, which its UDP thread holds through each
exchange. The UDP read is bounded, but when the controller reports an error that thread fetches the error
log over TCP with the mutex held, and a position read or command then waits as long. The Python binding holds
the GIL through all of it: no thread, timer or signal handler in the process runs, so nothing there can end
the call. Another process can end the process.

What bounds it. The runner announces every driver call to this watchdog, before and after, with a budget:
``CALL_BUDGET_S``, plus a blocking move's goal time, plus configure()'s connect timeout. The watchdog times
each budget from the announcement's arrival, so a backlog never shortens one. A call still in flight at its
deadline means a wedged driver: the watchdog ends the runner (SIGTERM, SIGKILL after a grace), which lets go
of the driver lease and the e-stop as it exits, and hands the arm to the recovery landing
(``hardware.hand_over``) that the runner itself uses when its own landing fails -- unless the runner withdrew
that (a joint measured past its limits, whose clamp into them is the move refused), when it starts nothing.
The controller idles an arm whose driver connection drops, so an unsupported arm can fall when the runner
ends. Only a landing the recovery measured is ``recovered``; an expiry, a refusal (an engaged e-stop: nothing
moves) or a failure is ``unknown``, and the watchdog records which in the run's summary.

The watchdog takes neither the lease nor the e-stop and never opens a driver. It acts on nothing but its own
expired deadline: whenever the runner's end of the socket closes, for any reason, it exits. Waits between
calls -- on a pressed e-stop, on the recovery landing -- are not driver calls and have no budget.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("travel.watchdog")

# A driver call that has not returned this long after it began is wedged: a healthy controller answers in
# milliseconds. A blocking move adds its goal time, configure() its connect timeout.
CALL_BUDGET_S = 10.0
# The watchdog is up and watching before the runner opens a driver, or the runner does not take the arm.
READY_S = 10.0
# The vendor's defaults for the arguments a caller leaves out (trossen_arm.hpp).
VENDOR_CONNECT_TIMEOUT_S = 20.0
VENDOR_GOAL_TIME_S = 2.0
# The destructor of the vendor's driver runs cleanup(): a TCP round trip unless cleanup() already ran.
DESTRUCTOR = "~TrossenArmDriver"


def call_budget(name: str, args: tuple, kwargs: dict) -> float:
    """How long one driver call may take before it counts as wedged."""
    if name == "configure":
        connect = kwargs.get("timeout", args[4] if len(args) > 4 else VENDOR_CONNECT_TIMEOUT_S)
        return CALL_BUDGET_S + float(connect)
    if name == "set_all_positions" and kwargs.get("blocking", args[2] if len(args) > 2 else True):
        return CALL_BUDGET_S + float(kwargs.get("goal_time", args[1] if len(args) > 1 else VENDOR_GOAL_TIME_S))
    return CALL_BUDGET_S


class DriverWatch:
    """The runner's end: starts the watchdog, announces each driver call, says how to land the arm."""

    def __init__(self, sock: socket.socket, process: subprocess.Popen):
        self.sock, self.process, self.alive = sock, process, True

    @classmethod
    def start(cls) -> DriverWatch:
        """The watchdog process, up and watching this one, else a RuntimeError: no driver is opened unwatched."""
        ours, theirs = socket.socketpair()
        try:
            process = subprocess.Popen(_command(os.getpid(), theirs.fileno()), stdin=subprocess.DEVNULL,
                                       pass_fds=(theirs.fileno(),), process_group=0)
        except OSError as failed:
            ours.close()
            raise RuntimeError(f"the driver watchdog did not start: {failed}") from failed
        finally:
            theirs.close()
        ours.settimeout(READY_S)
        ready = b""
        try:
            with contextlib.suppress(OSError):  # the timeout included
                ready = ours.recv(16)
        finally:
            if ready != b"ready\n":  # not up, or Ctrl+C while waiting: the watchdog goes too
                ours.close()
                _stop(process)
        if ready != b"ready\n":
            raise RuntimeError("the driver watchdog did not come up: no driver is opened without it")
        ours.settimeout(None)
        return cls(ours, process)

    @contextlib.contextmanager
    def bounded(self, name: str, budget_s: float):
        """One driver call in flight, for at most ``budget_s``."""
        self._send({"call": name, "budget_s": round(budget_s, 3)})
        try:
            yield
        finally:
            self._send({"call": None})

    def handover(self, **recovery) -> None:
        """How to land the arm if the watchdog has to end this process: ``hardware.recovery_command``'s arguments,
        the recovery's ``budget_s``, and ``summary``, where the run's summary goes."""
        self._send({"handover": recovery})

    def withdraw(self, why: str, summary: str | None) -> None:
        """The arm must not go to the recovery landing, ``why``: if the watchdog has to end this process, it only
        records the arm state as unknown in ``summary``."""
        self._send({"handover": {"withdrawn": why, "summary": summary}})

    def close(self) -> None:
        """No driver left to watch: the watchdog exits on the closed socket."""
        self.sock.close()
        _stop(self.process)

    def _send(self, message: dict) -> None:
        if not self.alive:
            return
        try:  # blocks only while the watchdog does not read, which it stops doing only to end this process
            self.sock.sendall(json.dumps(message).encode() + b"\n")
        except OSError as gone:
            self.alive = False
            log.error("the driver watchdog is gone (%s): driver calls are unbounded from here", gone)


class WatchedDriver:
    """The vendor driver with each call announced to the watchdog, with its budget."""

    def __init__(self, driver, watch: DriverWatch):
        self._driver, self._watch = driver, watch

    def __getattr__(self, name: str):
        if name.startswith("_"):  # the vendor's names are public; this one is ours, not set yet (copy, pickle)
            raise AttributeError(name)
        method = getattr(self._driver, name)
        if not callable(method):
            return method

        def bounded(*args, **kwargs):
            with self._watch.bounded(name, call_budget(name, args, kwargs)):
                return method(*args, **kwargs)

        return bounded


def _command(runner_pid: int, fd: int) -> list[str]:
    return [sys.executable, "-m", "tatbot_travel.watchdog", str(runner_pid), str(fd)]


def _stop(process: subprocess.Popen) -> None:
    """Reap the watchdog, which exits on its own once the socket closes; end it if it lingers."""
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class Announcements:
    """What the runner said last: the call in flight and its deadline (timed from arrival), how to land the arm."""

    def __init__(self):
        self.partial = b""
        self.call: str | None = None
        self.budget_s = 0.0
        self.deadline: float | None = None
        self.handover: dict | None = None

    def feed(self, data: bytes, now: float) -> None:
        *lines, self.partial = (self.partial + data).split(b"\n")
        for line in lines:
            message = json.loads(line)
            if "handover" in message:
                self.handover = message["handover"]
            elif message["call"] is None:
                self.call, self.deadline = None, None
            else:
                self.call, self.budget_s = message["call"], float(message["budget_s"])
                self.deadline = now + self.budget_s

    def remaining(self, now: float) -> float | None:
        """Seconds until the call in flight counts as wedged; None with no call in flight."""
        return None if self.deadline is None else max(0.0, self.deadline - now)


def watch(sock: socket.socket) -> Announcements | None:
    """The runner's announcements until a call outlives its budget (returned), or None once the runner is gone."""
    said = Announcements()
    while True:
        readable, _, _ = select.select([sock], [], [], said.remaining(time.monotonic()))
        if not readable:
            return said
        try:
            data = sock.recv(65536)
        except OSError:
            return None
        if not data:
            return None
        said.feed(data, time.monotonic())


# Linux's pidfd syscalls, the same numbers on x86_64 and aarch64. Interpreters built against an old glibc
# (python-build-standalone, as uv installs it, which the arm node's runner uses) lack os.pidfd_open and
# signal.pidfd_send_signal though the kernel has both.
SYS_PIDFD_SEND_SIGNAL, SYS_PIDFD_OPEN = 424, 434


def _syscall(number: int, *args: int) -> int:
    import ctypes
    import platform

    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise OSError(38, f"no pidfd syscall numbers for {sys.platform} {platform.machine()}")  # ENOSYS
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(a) for a in args))
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))  # ESRCH comes back as ProcessLookupError, as from os
    return result


def pidfd_open(pid: int) -> int:
    """A file descriptor for process ``pid`` that turns readable when it exits: ``os.pidfd_open``, or the syscall."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _syscall(SYS_PIDFD_OPEN, pid, 0)


def pidfd_send_signal(pidfd: int, sig: int) -> None:
    """``signal.pidfd_send_signal``, or the syscall: no pid reuse can redirect it."""
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, sig)
    else:
        _syscall(SYS_PIDFD_SEND_SIGNAL, pidfd, int(sig), 0, 0)  # no siginfo, no flags


def end(pidfd: int, grace_s: float) -> bool:
    """SIGTERM the runner, SIGKILL it after the grace: True once it has exited, its files (lease, e-stop) closed."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            pidfd_send_signal(pidfd, sig)
        if select.select([pidfd], [], [], grace_s)[0]:
            return True
    return False


def end_wedged_runner(pidfd: int, said: Announcements) -> int:
    """End the wedged runner, hand the arm to the recovery landing, record the landing in the run's summary.

    Exits 0 only when the recovery measured the sleep pose.
    """
    from tatbot_travel import hardware

    log.error("the driver call %s has not returned in %.1f s: the vendor driver is wedged in a read that never "
              "returns. Ending the runner; the controller idles an arm whose driver connection drops, so an "
              "unsupported arm can fall", said.call, said.budget_s)
    recovery = dict(said.handover or {})
    summary, budget, withdrawn = (recovery.pop(key, None) for key in ("summary", "budget_s", "withdrawn"))
    if not end(pidfd, hardware.RECOVERY_KILL_AFTER_S):
        landing = hardware.unknown("the runner outlived SIGKILL, so its lease and e-stop may still be held; "
                                   "the arm was not handed to the recovery landing")
    elif withdrawn:
        landing = hardware.unknown(f"the runner ruled out the recovery landing: {withdrawn}")
    elif budget is None:
        landing = hardware.unknown("the runner never said how to land the arm")
    else:
        landing = hardware.hand_over(lambda: hardware.recovery_command(budget_s=budget, **recovery), budget)
    pose = recovery.get("pose") or [None]  # six joints, then the carriage
    record(summary, {"ended": f"wedged: the driver call {said.call} outlived its {said.budget_s:g} s budget, and "
                              "the watchdog ended the runner", "carriage": pose[-1], "landing": landing})
    return 0 if landing == "recovered" else 1


def record(path: str | None, summary: dict) -> None:
    """The run's summary, as the runner would have written it; never over one the runner did write."""
    if path is None:
        return
    try:
        with open(path, "x") as out:
            out.write(json.dumps(summary, indent=1) + "\n")
    except OSError as failed:
        log.error("the run's summary was not written: %s", failed)
    with contextlib.suppress(OSError):
        print(json.dumps({"out": str(Path(path).parent), **summary}), flush=True)


def main(argv: list[str] | None = None) -> int:
    """``python -m tatbot_travel.watchdog RUNNER_PID FD``: watch the runner's driver calls on the socket FD."""
    runner, fd = (int(v) for v in (argv if argv is not None else sys.argv[1:]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sock = socket.socket(fileno=fd)
    sock.set_inheritable(False)  # the recovery landing it may start holds nothing of the runner's
    try:
        pidfd = pidfd_open(runner)
    except OSError as failed:
        log.error("cannot watch the runner (pid %d): %s", runner, failed)
        return 1
    if os.getppid() != runner:  # the runner exited before the pidfd was open: nothing to watch
        return 0
    try:
        sock.sendall(b"ready\n")
    except OSError:  # the runner stopped waiting for this watchdog
        return 0
    said = watch(sock)
    return 0 if said is None else end_wedged_runner(pidfd, said)


if __name__ == "__main__":
    sys.exit(main())
