#!/usr/bin/env python3
"""Make a window fullscreen or maximized through the X11 window manager, found by its WM_CLASS.

What `tatbot viewer open` runs once it has started the window: rerun 0.36 has
neither flag. It sends the EWMH `_NET_WM_STATE` request that `wmctrl -b add`
sends (the station has no wmctrl, and its xdotool predates `windowstate`), so it
types no key and needs no focus. With --keep-focus it then hands the focus back
to the window that held it when this started (`_NET_ACTIVE_WINDOW`, asked as a
pager), so the station's demo page opens behind the agent terminal and the
operator alt-tabs to it. Idempotent: a window already in the state is not asked
again. Display only. Stdlib (ctypes on libX11).

Exit 0 once the window is in the state (or there is no X display to ask),
1 when no such window is managed within --wait seconds or the state never shows.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
import time

STATES = {"fullscreen": ("_NET_WM_STATE_FULLSCREEN",),
          "maximized": ("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ")}
CLIENT_MESSAGE = 33
REDIRECT = (1 << 19) | (1 << 20)  # SubstructureNotifyMask | SubstructureRedirectMask: the WM's ears
ADD, PAGER = 1, 2
POLL_S = 0.1


class _ClientMessage(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
                ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
                ("format", ctypes.c_int), ("l", ctypes.c_long * 5)]


class _Event(ctypes.Union):
    _fields_ = [("xclient", _ClientMessage), ("pad", ctypes.c_long * 24)]  # an XEvent is 24 longs


_ERRORS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_IGNORE = _ERRORS(lambda _display, _error: 0)  # a window that closes mid-read is no reason to exit


def _bind(lib):
    c = ctypes
    lib.XOpenDisplay.restype, lib.XOpenDisplay.argtypes = c.c_void_p, [c.c_char_p]
    lib.XDefaultRootWindow.restype, lib.XDefaultRootWindow.argtypes = c.c_ulong, [c.c_void_p]
    lib.XInternAtom.restype, lib.XInternAtom.argtypes = c.c_ulong, [c.c_void_p, c.c_char_p, c.c_int]
    lib.XGetWindowProperty.argtypes = [c.c_void_p, c.c_ulong, c.c_ulong, c.c_long, c.c_long, c.c_int, c.c_ulong,
                                       c.POINTER(c.c_ulong), c.POINTER(c.c_int), c.POINTER(c.c_ulong),
                                       c.POINTER(c.c_ulong), c.POINTER(c.POINTER(c.c_ubyte))]
    lib.XSendEvent.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_long, c.c_void_p]
    lib.XFree.argtypes = [c.c_void_p]
    lib.XSync.argtypes = [c.c_void_p, c.c_int]
    lib.XCloseDisplay.argtypes = [c.c_void_p]
    lib.XSetErrorHandler.argtypes = [_ERRORS]
    return lib


class Display:
    """The few EWMH reads and requests this needs, on $DISPLAY."""

    def __init__(self, lib, dpy):
        self.lib, self.dpy = lib, dpy
        self.root = lib.XDefaultRootWindow(dpy)

    @classmethod
    def open(cls) -> Display | None:
        name = ctypes.util.find_library("X11")
        if not name:
            return None
        lib = _bind(ctypes.CDLL(name))
        dpy = lib.XOpenDisplay(None)
        if not dpy:
            return None
        lib.XSetErrorHandler(_IGNORE)
        return cls(lib, dpy)

    def close(self) -> None:
        self.lib.XCloseDisplay(self.dpy)

    def atom(self, name: str) -> int:
        return self.lib.XInternAtom(self.dpy, name.encode(), 0)

    def prop(self, window: int, name: str) -> list[int] | bytes:
        """A property: format-32 items (atoms, windows) as ints, text as bytes; empty when absent."""
        kind, fmt, n, after = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_ulong(), ctypes.c_ulong()
        data = ctypes.POINTER(ctypes.c_ubyte)()
        if self.lib.XGetWindowProperty(self.dpy, window, self.atom(name), 0, 1 << 16, 0, 0, ctypes.byref(kind),
                                       ctypes.byref(fmt), ctypes.byref(n), ctypes.byref(after),
                                       ctypes.byref(data)) != 0 or not data:
            return []
        try:
            if fmt.value == 32:  # Xlib hands 32-bit items back as C longs
                return list(ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))[:n.value])
            return bytes(data[:n.value])
        finally:
            self.lib.XFree(data)

    def ask(self, window: int, message: str, *data: int) -> None:
        """A request to the window manager about `window`, sent to the root as EWMH says."""
        event = _Event()
        msg = event.xclient
        msg.type, msg.window, msg.message_type, msg.format = CLIENT_MESSAGE, window, self.atom(message), 32
        msg.l[:] = (list(data) + [0] * 5)[:5]
        self.lib.XSendEvent(self.dpy, self.root, 0, REDIRECT, ctypes.byref(event))
        self.lib.XSync(self.dpy, 0)

    def clients(self) -> list[int]:
        return list(self.prop(self.root, "_NET_CLIENT_LIST"))

    def active(self) -> int:
        return next(iter(self.prop(self.root, "_NET_ACTIVE_WINDOW")), 0)

    def wm_class(self, window: int) -> list[str]:
        return [part.decode(errors="replace") for part in bytes(self.prop(window, "WM_CLASS")).split(b"\0") if part]

    def states(self, window: int) -> set[int]:
        return set(self.prop(window, "_NET_WM_STATE"))


def find(x: Display, wm_class: str, wait_s: float) -> int | None:
    """The first managed window whose WM_CLASS instance or class is `wm_class`, waiting for it to map."""
    deadline = time.monotonic() + wait_s
    while True:
        hit = next((w for w in x.clients() if wm_class in x.wm_class(w)), None)
        if hit is not None or time.monotonic() >= deadline:
            return hit
        time.sleep(POLL_S)


def apply(x: Display, window: int, state: str, wait_s: float = 3.0) -> bool | None:
    """Ask for the state unless the window has it. False: it already had it; True: it has it now;
    None: the window manager did not grant it within `wait_s`."""
    want = {x.atom(name) for name in STATES[state]}
    if want <= x.states(window):
        return False
    atoms = [x.atom(name) for name in STATES[state]] + [0]
    x.ask(window, "_NET_WM_STATE", ADD, atoms[0], atoms[1], PAGER)
    deadline = time.monotonic() + wait_s
    while not want <= x.states(window):
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_S)
    return True


def took_focus(x: Display, window: int, wait_s: float = 1.0) -> bool:
    """Whether the window manager focuses `window` within `wait_s` (it may, just after a map)."""
    deadline = time.monotonic() + wait_s
    while x.active() != window:
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)
    return True


def run(x: Display, wm_class: str, state: str, keep_focus: bool, wait_s: float) -> int:
    before = x.active()
    window = find(x, wm_class, wait_s)
    if window is None:
        print(f"x11_window: no window of class {wm_class} mapped within {wait_s:g} s", file=sys.stderr)
        return 1
    changed = apply(x, window, state)
    if changed is None:
        print(f"x11_window: the window manager did not make {wm_class} {state}", file=sys.stderr)
        return 1
    # Becoming fullscreen can raise a window, and the WM may focus a window that just mapped
    # (one that restored its own fullscreen too): the window that had the focus when this
    # started gets it back, raised, as a pager would ask.
    if keep_focus and before not in (0, window) and before in x.clients() and (changed or took_focus(x, window)):
        x.ask(before, "_NET_ACTIVE_WINDOW", PAGER)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wm_class", help="the window's WM_CLASS instance or class (rerun's is `rerun`)")
    ap.add_argument("state", choices=sorted(STATES))
    ap.add_argument("--keep-focus", action="store_true",
                    help="hand the focus back to the window that held it when this started")
    ap.add_argument("--wait", type=float, default=20.0, help="seconds to wait for the window to map (default 20)")
    args = ap.parse_args(argv)
    x = Display.open()
    if x is None:
        return 0  # no X display here (a Wayland session, a headless node): nothing to ask
    try:
        return run(x, args.wm_class, args.state, args.keep_focus, args.wait)
    finally:
        x.close()


if __name__ == "__main__":
    sys.exit(main())
