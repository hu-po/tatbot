#!/usr/bin/env python3
"""The station keyboard's relay: the operator's drawing keys, pressed on the keyboard that people also type
the station agent's requests on, sent to the ROS stack's keypad node (tatbot_session.keypad).

    tatbot_keys_relay.py --dest <ros node rig address>:7644

Stdlib only, no ROS. It reads every keyboard's evdev (`/dev/input/by-id/*-event-kbd`, rescanned so a
keyboard plugged in later is read) WITHOUT grabbing it, so typing reaches the desktop as before, and sends
`TBK1 <key>\\n` in one UDP datagram per press of a drawing key. The drawing keys are F13-F18, which no
typing sends. Fn never reaches the host, so the keyboard's own keymap (VIA) puts them on its Fn layer:

    Fn+W (F13) up     Fn+S (F14) down   Fn+X (F15) reset
    Fn+E (F16) dip    Fn+Q (F17) fewer_dips              Fn+Enter (F18) enter

A held key sends once. No other key is sent or logged.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import os
import select
import socket
import struct
import sys
import time

EVENT = struct.Struct("llHHi")   # struct input_event on a 64-bit kernel: timeval, type, code, value
KEYBOARDS = "/dev/input/by-id/*-event-kbd"
RESCAN_S = 2.0
# KEY_F13 .. KEY_F18 (linux/input-event-codes.h)
KEYS = {183: "up", 184: "down", 185: "reset", 186: "dip", 187: "fewer_dips", 188: "enter"}


def presses(data: bytes) -> list[str]:
    """The drawing key of each key press (EV_KEY, value 1; a repeat is 2) in a read of input events."""
    whole = data[: len(data) - len(data) % EVENT.size]
    return [KEYS[code] for _s, _us, kind, code, value in EVENT.iter_unpack(whole)
            if (kind, value) == (1, 1) and code in KEYS]


def frame(key: str) -> bytes:
    return f"TBK1 {key}\n".encode()


def parse_dest(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"--dest wants HOST:PORT, got {text!r}")
    return host, int(port)


def open_new(pattern: str, fds: dict[str, int], said: dict[str, str]) -> None:
    """Open each keyboard matching `pattern` not yet read; one that cannot be opened is said once, not every rescan."""
    for path in sorted(set(glob.glob(pattern)) - set(fds)):
        try:
            fds[path] = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            print(f"reading {path}", flush=True)
            said.pop(path, None)
        except OSError as exc:
            if said.get(path) != exc.strerror:
                print(f"{path}: {exc.strerror}", file=sys.stderr, flush=True)
            said[path] = exc.strerror


def forward(path: str, fds: dict[str, int], sock: socket.socket, dest: tuple[str, int]) -> None:
    """Send the drawing keys in one read of a keyboard; drop a keyboard that was unplugged."""
    try:
        data = os.read(fds[path], EVENT.size * 64)
    except OSError as exc:
        print(f"{path}: {exc.strerror}; dropped", file=sys.stderr, flush=True)
        os.close(fds.pop(path))
        return
    for key in presses(data):
        try:
            sock.sendto(frame(key), dest)
            print(f"sent {key}", flush=True)
        except OSError as exc:   # the rig link down: the press is lost, as a numpad's unplugged would be
            print(f"{key} not sent: {exc.strerror}", file=sys.stderr, flush=True)


def run(dest: tuple[str, int], pattern: str = KEYBOARDS, stop=lambda: False) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fds: dict[str, int] = {}
    said: dict[str, str] = {}
    scanned = 0.0
    while not stop():
        if time.monotonic() - scanned > RESCAN_S:
            scanned = time.monotonic()
            open_new(pattern, fds, said)
        ready = select.select(list(fds.values()), [], [], 0.5)[0] if fds else time.sleep(0.5) or []
        for path in [path for path, fd in fds.items() if fd in ready]:
            forward(path, fds, sock, dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dest", type=parse_dest, required=True, help="HOST:PORT of the stack's keypad node")
    args = p.parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        run(args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
