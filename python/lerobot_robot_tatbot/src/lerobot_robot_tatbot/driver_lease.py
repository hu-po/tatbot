"""Cross-language, process-lifetime driver ownership shared by a session's arms.

The inode is never unlinked. C++ teleop and the ROS 2 hardware plugin use the same flock.
Ownership grants no motion authority and never changes e-stop handling.
"""
from __future__ import annotations

import fcntl
import functools
import os
import stat
import threading
import weakref
from pathlib import Path

HARDWARE_LEASE = Path('/tmp/tatbot-arm-driver.lock')
_mutex = threading.RLock()
_owners: weakref.WeakValueDictionary[str, _Owner] = weakref.WeakValueDictionary()


class DriverBusyError(RuntimeError):
    """Another process owns the driver, or ownership cannot be established."""


class _Owner:
    def __init__(self, path: Path):
        self.fd = -1
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise DriverBusyError('driver lease must be a regular file owned by this user')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd

    def __del__(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class DriverLease:
    """One reference to a session owner; releasing one arm retains other arms."""
    def __init__(self, owner: _Owner):
        self._owner: _Owner | None = owner

    def close(self) -> None:
        with _mutex:
            self._owner = None

    def __enter__(self) -> DriverLease:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def acquire(path: Path = HARDWARE_LEASE) -> DriverLease:
    with _mutex:
        key = str(path.absolute())
        owner = _owners.get(key)
        if owner is None:
            try:
                owner = _Owner(path)
            except (OSError, DriverBusyError) as error:
                raise DriverBusyError(f'driver busy or lease unavailable: {error}') from error
            _owners[key] = owner
        return DriverLease(owner)


def _after_fork() -> None:
    # Close only the child's descriptor copies, never LOCK_UN the parent's
    # shared open-file description. A child must acquire its own ownership.
    global _mutex
    for owner in list(_owners.values()):
        if owner.fd >= 0:
            os.close(owner.fd)
            owner.fd = -1
    _owners.clear()
    _mutex = threading.RLock()


os.register_at_fork(after_in_child=_after_fork)


def owned(function):
    """Reserve ownership for a standalone recovery call or nested landing."""
    @functools.wraps(function)
    def run(*args, **kwargs):
        with acquire():
            return function(*args, **kwargs)
    return run
