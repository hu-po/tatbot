"""Who has the wrist D405s on the ros node: a reading's camera, or the demo viewer's idle feed.

A process that opens a wrist D405 for a reading (tatbot_session.inspect.WristCamera, in the session or in any tool:
calib, the `ros` client verbs, `vision capture`) claims the cameras for as long as it holds one open. The session's
IdleFeed shows both on the viewer only while nobody has claimed them for `quiet_s` and the rig is awake, and lets go
the moment a claim appears; the claim waits for that, so no reading finds a device busy. Display only: the feed
never delays a claim by more than its poll and a pipeline stop.

Two flock(2) files under ROOT, symmetric: a claim holds `claims` shared and probes `feed` exclusively; a feed holds
`feed` shared and probes `claims` exclusively. Each side takes its own lock before it probes the other's, so a claim
and a feed that start together cannot both open the cameras. Locks end with their process: a crash holds nothing.
"""
from __future__ import annotations

import fcntl
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(f"/tmp/tatbot-wrist-d405-{os.getuid()}")   # one per node and login user, whatever a process's env says
POLL_S = 0.5      # how often the feed looks for a claim (and how long, at most, a claim waits for it to look)
WAIT_S = 15.0     # a claim stops waiting for a feed that has not let go this long (a stuck close) and opens anyway
QUIET_S = 30.0    # the cameras free this long before the feed opens them: a calibration's touches ~10 s apart keep it off
JOIN_S = 15.0     # a feed camera still opening (~2 s each, 30 frames of auto exposure) is closed once it has opened


def _stderr(message: str) -> None:
    print(f"wrist_claim: {message}", file=sys.stderr, flush=True)


def _lock(path: Path, mode: int):
    """`path` open with an flock in `mode`, or None when LOCK_NB finds it held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")  # noqa: SIM115  (the lock: held past this call, let go when closed or collected)
    try:
        fcntl.flock(handle.fileno(), mode)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _free(path: Path) -> bool:
    """Whether nobody holds `path` shared: an exclusive probe, let go at once."""
    handle = _lock(path, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if handle is not None:
        handle.close()
    return handle is not None


def claimed(root: Path = ROOT) -> bool:
    """Whether a reading holds the cameras now."""
    return not _free(root / "claims")


class Claim:
    """The cameras for a reading, from construction to release(): waits up to `wait_s` for an idle feed to let go."""

    def __init__(self, root: Path = ROOT, wait_s: float = WAIT_S, log=None):
        self._handle = _lock(root / "claims", fcntl.LOCK_SH)
        deadline = time.monotonic() + wait_s
        while not _free(root / "feed"):
            if time.monotonic() >= deadline:
                (log or _stderr)(f"the idle feed kept the wrist D405s {wait_s:g} s; opening anyway")
                break
            time.sleep(0.05)

    def release(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class IdleFeed:
    """The wrist D405s on the demo viewer between readings. A daemon thread calls `start()` (wrist_tap.previews: the
    cameras opened display-only, each tap polling it at 2 Hz) once they have been free of claims for `quiet_s` while
    `asleep()` says no, and closes what it returned (its `close()`, then its `thread` joined, so a camera still opening
    is closed too) at the next claim or sleep, and at close()."""

    def __init__(self, start, *, root: Path = ROOT, quiet_s: float = QUIET_S, poll_s: float = POLL_S,
                 asleep=lambda: False, log=None):
        self.start, self.root, self.quiet_s, self.poll_s = start, root, quiet_s, poll_s
        self.asleep, self.log = asleep, log or _stderr
        self.preview = self._handle = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="wrist-idle-feed", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        free_since = None
        while not self._stop.wait(self.poll_s):
            try:
                busy = self.asleep() or claimed(self.root)
                if self.preview is not None:
                    if busy:
                        self._close()
                    continue
                free_since = None if busy else free_since or time.monotonic()
                if free_since is not None and time.monotonic() - free_since >= self.quiet_s:
                    self._open()
                    free_since = None
            except Exception as error:   # display only: the loop goes on
                self.log(f"idle feed ({type(error).__name__}: {error})")
        self._close()

    def _open(self) -> None:
        self._handle = _lock(self.root / "feed", fcntl.LOCK_SH)
        if claimed(self.root):   # a claim that came in after the last look: it is waiting for this lock
            self._release()
            return
        self.preview = self.start()

    def _close(self) -> None:
        preview, self.preview = self.preview, None
        if preview is not None:
            preview.close()
            thread = getattr(preview, "thread", None)
            if thread is not None:
                thread.join(JOIN_S)
        self._release()

    def _release(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        self._stop.set()
        self.thread.join(JOIN_S + 2 * self.poll_s)
