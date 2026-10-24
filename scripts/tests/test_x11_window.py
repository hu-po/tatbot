"""The viewer's fullscreen step, against an Xvfb display and a stand-in window manager.

The stand-in does what an EWMH window manager (mutter, on the station) does with the
two requests the helper sends: it adds the _NET_WM_STATE atoms asked for, and it moves
_NET_ACTIVE_WINDOW. That proves the requests are well formed and that the helper sends
only what it must, which the live station's display cannot be used for.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts/lib/x11_window.py"
sys.path.insert(0, str(REPO / "scripts/lib"))
import x11_window as xw  # noqa: E402

if not shutil.which("Xvfb") or not ctypes.util.find_library("X11"):
    pytest.skip("needs Xvfb and libX11", allow_module_level=True)

XA_ATOM, XA_STRING, XA_WINDOW = 4, 31, 33


@pytest.fixture
def display(monkeypatch):
    read, write = os.pipe()
    server = subprocess.Popen(["Xvfb", "-displayfd", str(write), "-nolisten", "tcp", "-screen", "0", "640x480x24"],
                              pass_fds=(write,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(write)
    with os.fdopen(read) as fh:
        number = fh.readline().strip()
    if not number:
        server.kill()
        pytest.skip("Xvfb did not start")
    monkeypatch.setenv("DISPLAY", f":{number}")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    yield f":{number}"
    server.kill()
    server.wait()


class WindowManager:
    """Two clients, an agent terminal and the viewer, managed the EWMH way."""

    def __init__(self):
        self.x = xw.Display.open()
        lib, c = self.x.lib, ctypes
        lib.XCreateSimpleWindow.restype = c.c_ulong
        lib.XCreateSimpleWindow.argtypes = [c.c_void_p, c.c_ulong] + [c.c_int] * 2 + [c.c_uint] * 3 + [c.c_ulong] * 2
        lib.XChangeProperty.argtypes = [c.c_void_p, c.c_ulong, c.c_ulong, c.c_ulong, c.c_int, c.c_int,
                                        c.c_void_p, c.c_int]
        lib.XSelectInput.argtypes = [c.c_void_p, c.c_ulong, c.c_long]
        lib.XPending.argtypes = [c.c_void_p]
        lib.XNextEvent.argtypes = [c.c_void_p, c.c_void_p]
        lib.XSelectInput(self.x.dpy, self.x.root, xw.REDIRECT)
        self.terminal, self.viewer = self._client(b"gnome-terminal-server\0Gnome-terminal\0"), self._client(b"\0rerun\0")
        self.requests: list[tuple[str, int, list[int]]] = []

    def _client(self, wm_class: bytes) -> int:
        window = self.x.lib.XCreateSimpleWindow(self.x.dpy, self.x.root, 0, 0, 100, 100, 0, 0, 0)
        self.x.lib.XChangeProperty(self.x.dpy, window, self.x.atom("WM_CLASS"), XA_STRING, 8, 0, wm_class,
                                   len(wm_class))
        return window

    def set(self, window: int, name: str, kind: int, values: list[int]) -> None:
        data = (ctypes.c_ulong * len(values))(*values)
        self.x.lib.XChangeProperty(self.x.dpy, window, self.x.atom(name), kind, 32, 0, data, len(values))
        self.x.lib.XSync(self.x.dpy, 0)

    def manage(self, clients: list[int], active: int) -> None:
        self.set(self.x.root, "_NET_CLIENT_LIST", XA_WINDOW, clients)
        self.set(self.x.root, "_NET_ACTIVE_WINDOW", XA_WINDOW, [active])

    def _handle(self, msg) -> None:
        name = {self.x.atom("_NET_WM_STATE"): "state", self.x.atom("_NET_ACTIVE_WINDOW"): "activate"}.get(
            msg.message_type, "other")
        self.requests.append((name, msg.window, list(msg.l)))
        if name == "state" and msg.l[0] == xw.ADD:
            self.set(msg.window, "_NET_WM_STATE", XA_ATOM, sorted(self.x.states(msg.window) | {a for a in msg.l[1:3] if a}))
        elif name == "activate":
            self.set(self.x.root, "_NET_ACTIVE_WINDOW", XA_WINDOW, [msg.window])

    def serve(self, helper: subprocess.Popen, map_after_s: float | None = None, focus_on_map: bool = True) -> int:
        """Answer the helper's requests until it exits; the viewer maps (and takes the focus) after `map_after_s`."""
        start, event = time.monotonic(), xw._Event()
        while True:
            done = helper.poll() is not None  # what it sent before exiting is still answered
            if map_after_s is not None and time.monotonic() - start >= map_after_s:
                self.manage([self.terminal, self.viewer], self.viewer if focus_on_map else self.terminal)
                map_after_s = None
            self.x.lib.XSync(self.x.dpy, 0)
            while self.x.lib.XPending(self.x.dpy):
                self.x.lib.XNextEvent(self.x.dpy, ctypes.byref(event))
                if event.xclient.type == xw.CLIENT_MESSAGE:
                    self._handle(event.xclient)
            if done:
                return helper.returncode
            time.sleep(0.02)


def _helper(*args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(HELPER), "rerun", *args, "--wait", "5"])


def test_a_new_viewer_goes_fullscreen_behind_the_window_that_had_the_focus(display):
    wm = WindowManager()
    wm.manage([wm.terminal], wm.terminal)
    assert wm.serve(_helper("fullscreen", "--keep-focus"), map_after_s=0.3) == 0
    full = wm.x.atom("_NET_WM_STATE_FULLSCREEN")
    assert wm.requests == [("state", wm.viewer, [xw.ADD, full, 0, xw.PAGER, 0]),
                           ("activate", wm.terminal, [xw.PAGER, 0, 0, 0, 0])]
    assert wm.x.states(wm.viewer) == {full} and wm.x.active() == wm.terminal
    # Again on the running window: it already is fullscreen and behind, so nothing is asked.
    wm.requests.clear()
    assert wm.serve(_helper("fullscreen", "--keep-focus")) == 0
    assert wm.requests == []


def test_a_viewer_that_restored_its_own_fullscreen_still_gives_the_focus_back(display):
    wm = WindowManager()
    wm.manage([wm.terminal], wm.terminal)
    wm.set(wm.viewer, "_NET_WM_STATE", XA_ATOM, [wm.x.atom("_NET_WM_STATE_FULLSCREEN")])
    assert wm.serve(_helper("fullscreen", "--keep-focus"), map_after_s=0.3) == 0
    assert [r[:2] for r in wm.requests] == [("activate", wm.terminal)] and wm.x.active() == wm.terminal


def test_maximize_asks_for_both_axes_and_leaves_the_focus_to_the_window_manager(display):
    wm = WindowManager()
    wm.manage([wm.terminal], wm.terminal)
    assert wm.serve(_helper("maximized"), map_after_s=0.2) == 0
    axes = [wm.x.atom("_NET_WM_STATE_MAXIMIZED_VERT"), wm.x.atom("_NET_WM_STATE_MAXIMIZED_HORZ")]
    assert wm.requests == [("state", wm.viewer, [xw.ADD, *axes, xw.PAGER, 0])] and wm.x.active() == wm.viewer


def test_no_viewer_within_the_wait_is_exit_1_and_no_display_is_exit_0(display):
    wm = WindowManager()
    wm.manage([wm.terminal], wm.terminal)
    started = time.monotonic()
    helper = subprocess.Popen([sys.executable, str(HELPER), "rerun", "fullscreen", "--wait", "0.5"],
                              stderr=subprocess.PIPE, text=True)
    assert wm.serve(helper) == 1 and time.monotonic() - started < 5 and "no window of class rerun" in helper.stderr.read()
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    assert subprocess.run([sys.executable, str(HELPER), "rerun", "fullscreen"], env=env).returncode == 0
