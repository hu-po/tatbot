#!/usr/bin/env python3
"""Run a lerobot client entry point with a graceful-Ctrl+C shield.

    il_client_shield.py <module[:func]> [args...]

A terminal Ctrl+C delivers SIGINT to the whole foreground process group, and
`timeout --signal=INT` forwards a second one to the child. The first
KeyboardInterrupt starts the client's teardown (disconnect -> staged -> sleep
landing moves, several seconds of blocking driver calls); any later SIGINT
would abort that mid-motion and leave the arm frozen in position mode,
holding its last target. This wrapper turns the first SIGINT into the normal
KeyboardInterrupt and swallows repeats while the landing runs.

If the landing genuinely hangs, a SIGINT more than HANG_ESCAPE_S seconds
after the first is allowed through (and the hardware e-stop is always live).
"""

import contextlib
import ctypes
import importlib
import os
import runpy
import signal
import sys
import time

HANG_ESCAPE_S = 10.0

_first_sigint = 0.0


def _install_parent_death_signal() -> None:
    """Ask Linux to SIGINT this client if its owning launcher disappears.

    This closes the SSH-loss path from the 2026-08-27 incident: the launcher
    shell died, Python was reparented to PID 1, and policy inference continued.
    The post-prctl parent check closes the race between reading the parent PID
    and installing the kernel contract.
    """

    parent = os.getppid()
    if parent == 1:
        raise RuntimeError("refusing to start an unowned rollout client under PID 1")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGINT, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGINT)


def _handler(signum, frame):
    global _first_sigint
    now = time.monotonic()
    if _first_sigint == 0.0:
        _first_sigint = now
        raise KeyboardInterrupt
    if now - _first_sigint > HANG_ESCAPE_S:
        print(
            "shield: landing appears hung — allowing Ctrl+C through "
            "(recover with scripts/il_recover_arm.sh)",
            file=sys.stderr,
        )
        signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt
    print(
        "shield: shutdown already in progress — landing the arm "
        "(further Ctrl+C ignored for a few seconds)",
        file=sys.stderr,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    signal.signal(signal.SIGINT, _handler)
    _install_parent_death_signal()
    # SA_RESTART: a swallowed SIGINT must not EINTR-abort the driver's
    # blocking socket reads mid-landing (the Trossen SDK does not retry
    # EINTR). time.sleep/select still interrupt promptly for the first press.
    with contextlib.suppress(Exception):
        signal.siginterrupt(signal.SIGINT, False)
    try:
        if ":" in target:
            module_name, func_name = target.split(":", 1)
            getattr(importlib.import_module(module_name), func_name)()
        else:
            runpy.run_module(target, run_name="__main__")
    except KeyboardInterrupt:
        # The client's own teardown (finally blocks) has already run by the
        # time the interrupt propagates here; exit with the conventional code.
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
