#!/usr/bin/env python3
"""The palette Pi's relays, each its own instance: the e-stop to every host that reads it (EST1), the
station probe likewise (PRB1), and the tattoo machine's power switch (MCH1 in, MCS1 out).

    tatbot_estop_relay.py --gpio GPIO17 --dest <ros node rig address>:7640 [--dest ...] [--bind ADDR]
    tatbot_estop_relay.py --device /dev/tatbot-estop --dest <ros node rig address>:7640 [--dest ...] [--bind ADDR]
    tatbot_estop_relay.py --probe GPIO27 --dest <arm host rig address>:7641 [--dest ...] [--bind ADDR]

Stdlib only, no ROS. The e-stop sources send `EST1 <seq> <state>\\n` (state 1 released, 0 pressed),
one frame per UDP datagram, the same frame to every `--dest` (the ros node's driver and, where the arm
node's own e-stop is this button, its monitor). Each reader judges its own heartbeat, so a datagram
lost on the way to one is silence, a stop, at that reader only:

- `--gpio NAME`: the button's NC contact between that header line and GND (the palette: GPIO17,
  header pin 11, and GND, pin 9). The relay holds the line with its pull-up and sends a frame at
  100 Hz from the raw level, as firmware/estop_pico does: LOW (contact closed) is released; HIGH
  (pressed, or a broken wire) is a stop. It never debounces; the driver does.
- `--device PATH`: a Pico running firmware/estop_pico. The relay is the only process that opens
  its serial port and forwards every complete line unchanged (a line over 128 bytes is dropped),
  never interpreting, filtering, debouncing or synthesizing frames. The Pico's own sequence number
  is what the driver checks, so no header is added.

Silence on the wire is what stops the arm, so a dead relay, Pi, Pico or link is a stop. A line or
device that fails goes silent and is reopened every 0.5 s. `--bind` pins the source address the
driver accepts (`estop_relay_addr`) when the Pi has more than one interface.

`--probe NAME` is the station's normally-closed touch probe on that header line (the palette:
GPIO27, header pin 13, through a 10 kOhm pull-up and a Schottky diode). It runs as a separate
instance so that nothing about the probe can touch the e-stop. It sends
`PRB1 <seq> <state> <edge_ns> <now_ns>\\n` to every `--dest`:
- `state` is 1 while the line is HIGH (touched, or a broken wire) and 0 at rest;
- `edge_ns` is the kernel's CLOCK_MONOTONIC stamp of the latest edge, taken in the GPIO interrupt
  (0 before the first);
- `now_ns` is the relay's CLOCK_MONOTONIC when it sends, which lets the driver map relay time onto
  its own.
A frame goes out the moment the kernel reports an edge, and one every 20 ms with the line's
level. The relay never debounces: kernel debounce would move the stamp one debounce period late,
and the probe's edges are clean.

    tatbot_estop_relay.py --machine GPIO22 --from <ros node rig address> [--listen PORT] [--estop-port PORT]

`--machine NAME` is the tattoo machine's power switch on that header line (HIGH powers it), its own
instance again. The session sends `MCH1 <seq> <on>\\n` to --listen every 20 ms, off as well as on; the
relay drives the line HIGH only while the newest command from --from asks for it and is at most
--timeout old, and the newest `EST1` frame on 127.0.0.1:--estop-port (the e-stop relay's, which gets
that --dest) reads released and is at most 0.15 s old. Silence from either, a pressed e-stop and a
relay that exits are all LOW, and once the e-stop reads anything but released the line stays LOW until
the machine is asked off: releasing the e-stop never restarts it. It answers each command with
`MCS1 <seq> <on> <powered> <estop>\\n` to its sender: the command's own seq and on, the line as driven,
and the e-stop as read (1 released, 0 pressed, 2 silent).
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import glob
import itertools
import math
import os
import select
import socket
import struct
import sys
import time
import tty

MAX_FRAME = 128     # bytes, newline included; the driver rejects longer datagrams too
REOPEN_S = 0.5      # cpp/teleop estop_monitor REOPEN_PERIOD_MS
READ_TIMEOUT_S = 0.5
FRAME_PERIOD_NS = 10_000_000   # 100 Hz, firmware/estop_pico/code.py

# The GPIO character device, uAPI v2 (linux/gpio.h); sizes and offsets checked against the header
# on the palette Pi.
GPIO_NAME_SIZE = 32
GPIO_LINE_FLAG_INPUT = 1 << 2
GPIO_LINE_FLAG_OUTPUT = 1 << 3
GPIO_LINE_FLAG_EDGE_RISING = 1 << 4
GPIO_LINE_FLAG_EDGE_FALLING = 1 << 5
GPIO_LINE_FLAG_BIAS_PULL_UP = 1 << 8
CHIP_INFO_SIZE, CHIP_INFO_LINES = 68, 64          # struct gpiochip_info, .lines
LINE_INFO_SIZE, LINE_INFO_OFFSET = 256, 64        # struct gpio_v2_line_info, .offset
LINE_REQUEST_SIZE = 592                           # struct gpio_v2_line_request:
REQUEST_CONSUMER, REQUEST_FLAGS, REQUEST_NUM_LINES, REQUEST_FD = 256, 288, 560, 588
LINE_EVENT_SIZE = 48                              # struct gpio_v2_line_event: u64 timestamp_ns, u32 id, ...
EVENT_RISING_EDGE = 1                             # gpio_v2_line_event.id; 2 is the falling edge
CONSUMER = b"tatbot_estop_relay"
PROBE_PERIOD_NS = 20_000_000   # the probe's level at 50 Hz between edges
ESTOP_STALE_S = 0.15           # an older EST1 is a stop, as in the driver (stack.yaml estop.udp_timeout_s)
MACHINE_TICK_S = 0.005


def _iowr(nr: int, size: int, read_only: bool = False) -> int:
    return ((2 if read_only else 3) << 30) | (size << 16) | (0xB4 << 8) | nr


GPIO_GET_CHIPINFO_IOCTL = _iowr(0x01, CHIP_INFO_SIZE, read_only=True)
GPIO_V2_GET_LINEINFO_IOCTL = _iowr(0x05, LINE_INFO_SIZE)
GPIO_V2_GET_LINE_IOCTL = _iowr(0x07, LINE_REQUEST_SIZE)
GPIO_V2_LINE_GET_VALUES_IOCTL = _iowr(0x0E, 16)
GPIO_V2_LINE_SET_VALUES_IOCTL = _iowr(0x0F, 16)


def split_lines(buffer: bytes, chunk: bytes) -> tuple[list[bytes], bytes]:
    """Complete lines (each ending in b"\\n", at most MAX_FRAME bytes) and the unfinished rest."""
    parts = (buffer + chunk).split(b"\n")
    rest = parts.pop()
    lines = [part + b"\n" for part in parts if len(part) + 1 <= MAX_FRAME]
    if len(rest) > 4 * MAX_FRAME:   # no newline in sight: garbage, drop it
        rest = b""
    return lines, rest


def open_device(path: str) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    if os.isatty(fd):
        tty.setraw(fd)
    return fd


def find_line(name: str) -> tuple[str, int]:
    """The (chip, offset) of the GPIO line with this name, such as GPIO17 on the Pi's header."""
    for chip in sorted(glob.glob("/dev/gpiochip*")):
        fd = os.open(chip, os.O_RDWR | os.O_CLOEXEC)
        try:
            info = bytearray(CHIP_INFO_SIZE)
            fcntl.ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, info)
            for offset in range(struct.unpack_from("I", info, CHIP_INFO_LINES)[0]):
                line = bytearray(LINE_INFO_SIZE)
                struct.pack_into("I", line, LINE_INFO_OFFSET, offset)
                fcntl.ioctl(fd, GPIO_V2_GET_LINEINFO_IOCTL, line)
                if line[:GPIO_NAME_SIZE].split(b"\0", 1)[0] == name.encode():
                    return chip, offset
        finally:
            os.close(fd)
    raise OSError(errno.ENODEV, f"no GPIO line named {name}")


def request_line(name: str, flags: int) -> int:
    """Hold the named line with these gpio_v2 flags; returns the line's file descriptor."""
    chip, offset = find_line(name)
    request = bytearray(LINE_REQUEST_SIZE)
    struct.pack_into("I", request, 0, offset)
    request[REQUEST_CONSUMER:REQUEST_CONSUMER + len(CONSUMER)] = CONSUMER
    struct.pack_into("Q", request, REQUEST_FLAGS, flags)
    struct.pack_into("I", request, REQUEST_NUM_LINES, 1)
    fd = os.open(chip, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.ioctl(fd, GPIO_V2_GET_LINE_IOCTL, request)
    finally:
        os.close(fd)
    return struct.unpack_from("i", request, REQUEST_FD)[0]


def line_high(line_fd: int) -> bool:
    values = bytearray(struct.pack("QQ", 0, 1))   # bits, mask
    fcntl.ioctl(line_fd, GPIO_V2_LINE_GET_VALUES_IOCTL, values)
    return bool(struct.unpack_from("Q", values)[0] & 1)


def open_gpio(name: str):
    """Hold the named line as an input with its pull-up. Returns (read, close); read() is True
    while the line is HIGH: the NC contact is open."""
    line_fd = request_line(name, GPIO_LINE_FLAG_INPUT | GPIO_LINE_FLAG_BIAS_PULL_UP)
    return (lambda: line_high(line_fd)), (lambda: os.close(line_fd))


def open_probe(name: str):
    """Hold the probe's line as an input with both edges stamped by the kernel. Returns
    ((fd, level), close): select on fd, read gpio_v2_line_events from it; level() reads it now."""
    line_fd = request_line(name, GPIO_LINE_FLAG_INPUT | GPIO_LINE_FLAG_BIAS_PULL_UP
                           | GPIO_LINE_FLAG_EDGE_RISING | GPIO_LINE_FLAG_EDGE_FALLING)
    return (line_fd, lambda: line_high(line_fd)), (lambda: os.close(line_fd))


def parse_events(data: bytes) -> list[tuple[int, bool]]:
    """(kernel stamp in ns, HIGH after the edge) for each whole gpio_v2_line_event in data."""
    return [(stamp, kind == EVENT_RISING_EDGE)
            for stamp, kind in (struct.unpack_from("QI", data, at)
                                for at in range(0, len(data) - LINE_EVENT_SIZE + 1, LINE_EVENT_SIZE))]


def parse_dest(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"--dest wants HOST:PORT, got {text!r}")
    return host, int(port)


def pump(fd: int, sock: socket.socket, dests: list[tuple[str, int]], stop) -> None:
    """Forward lines to every destination until the device fails or stop() is true."""
    buffer = b""
    while not stop():
        ready, _, _ = select.select([fd], [], [], READ_TIMEOUT_S)
        if not ready:
            continue
        chunk = os.read(fd, 256)
        if not chunk:
            return   # EOF: the device went away
        lines, buffer = split_lines(buffer, chunk)
        for line in lines:
            for dest in dests:
                with contextlib.suppress(OSError):   # an unreachable reader is silence there: a stop
                    sock.sendto(line, dest)


def heartbeat(read, sock: socket.socket, dests: list[tuple[str, int]], stop, seq) -> None:
    """A frame from read() to every destination every 10 ms until read() fails or stop() is true; seq
    numbers them, one sequence for all destinations."""
    deadline = time.monotonic_ns()
    while not stop():
        frame = b"EST1 %d %d\n" % (next(seq), 0 if read() else 1)
        for dest in dests:
            with contextlib.suppress(OSError):
                sock.sendto(frame, dest)
        deadline += FRAME_PERIOD_NS
        delay = deadline - time.monotonic_ns()
        if delay > 0:
            time.sleep(delay / 1e9)
        else:
            deadline = time.monotonic_ns()   # late: carry on from now, never burst


def probe_frames(line, sock: socket.socket, dests: list[tuple[str, int]], stop, seq) -> None:
    """PRB1 frames: one per kernel edge as it arrives, and the line's level every 20 ms, until the
    line fails or stop() is true."""
    fd, level = line
    edge_ns = 0   # the latest edge's kernel stamp, carried by every frame after it

    def send(state: int) -> None:
        frame = b"PRB1 %d %d %d %d\n" % (next(seq), state, edge_ns, time.monotonic_ns())
        for dest in dests:
            with contextlib.suppress(OSError):
                sock.sendto(frame, dest)

    send(int(level()))
    due = time.monotonic_ns() + PROBE_PERIOD_NS
    while not stop():
        ready, _, _ = select.select([fd], [], [], max(0, due - time.monotonic_ns()) / 1e9)
        if ready:
            data = os.read(fd, LINE_EVENT_SIZE * 16)
            if not data:
                raise OSError(errno.EIO, "the probe line closed")
            for stamp, high in parse_events(data):
                edge_ns = stamp
                send(int(high))
            continue
        send(int(level()))
        due = time.monotonic_ns() + PROBE_PERIOD_NS


def serve(label: str, opener, send, dests: list[tuple[str, int]], bind: str | None, stop) -> None:
    """Open the source, send from it until it fails, reopen every REOPEN_S; silence in between."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if bind:
        sock.bind((bind, 0))
    reported = None
    try:
        while not stop():
            try:
                handle, close = opener()
            except OSError as error:
                if reported != error.errno:
                    print(f"tatbot_estop_relay: {label}: {error.strerror}; retrying", file=sys.stderr, flush=True)
                    reported = error.errno
                time.sleep(REOPEN_S)
                continue
            print(f"tatbot_estop_relay: {label} -> {', '.join(f'{h}:{p}' for h, p in dests)}", file=sys.stderr,
                  flush=True)
            reported = None
            try:
                send(handle, sock)
            except OSError as error:
                print(f"tatbot_estop_relay: {label}: {error.strerror}; reopening", file=sys.stderr, flush=True)
            finally:
                close()
            time.sleep(REOPEN_S)
    finally:
        sock.close()


def run(device: str, dests: list[tuple[str, int]], bind: str | None = None, stop=lambda: False) -> None:
    """Forward a Pico's serial lines to every destination."""
    def opener():
        fd = open_device(device)
        return fd, lambda: os.close(fd)
    serve(device, opener, lambda fd, sock: pump(fd, sock, dests, stop), dests, bind, stop)


def run_gpio(name: str, dests: list[tuple[str, int]], bind: str | None = None, stop=lambda: False,
             open_line=open_gpio) -> None:
    """Send the NC contact on a GPIO line as EST1 frames at 100 Hz to every destination."""
    seq = itertools.count()   # advances across reopens
    serve(name, lambda: open_line(name), lambda read, sock: heartbeat(read, sock, dests, stop, seq), dests, bind,
          stop)


def run_probe(name: str, dests: list[tuple[str, int]], bind: str | None = None, stop=lambda: False,
              open_line=open_probe) -> None:
    """Send the station probe on a GPIO line as PRB1 frames to every destination."""
    seq = itertools.count()   # advances across reopens
    serve(name, lambda: open_line(name), lambda line, sock: probe_frames(line, sock, dests, stop, seq),
          dests, bind, stop)


def open_output(name: str):
    """Hold the named line as an output, LOW. Returns (write, close); close drives it LOW first."""
    line_fd = request_line(name, GPIO_LINE_FLAG_OUTPUT)

    def write(high: bool) -> None:
        fcntl.ioctl(line_fd, GPIO_V2_LINE_SET_VALUES_IOCTL, bytearray(struct.pack("QQ", int(high), 1)))

    def close() -> None:
        with contextlib.suppress(OSError):
            write(False)
        os.close(line_fd)
    return write, close


ESTOP_PRESSED, ESTOP_RELEASED, ESTOP_SILENT = 0, 1, 2   # MCS1's e-stop field


class MachineGate:
    """Whether the machine may run: the newest MCH1 asks for it and is at most timeout_s old; the newest
    EST1 reads released and is at most ESTOP_STALE_S old; and the machine has been asked off since the
    e-stop last read otherwise, so releasing the e-stop never restarts it by itself. A frame older in
    sequence than the last one taken is dropped (reordered), unless the last is stale (a restart)."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self.asked, self.released = (0, False, -math.inf), (0, False, -math.inf)   # (seq, value, when)
        self.armed = False

    @staticmethod
    def _take(last, frame: bytes, tag: bytes, now: float, stale_s: float):
        parts = frame.split()
        if len(parts) < 3 or parts[0] != tag or not (parts[1].isdigit() and parts[2] in (b"0", b"1")):
            return None
        seq = int(parts[1])
        if seq <= last[0] and now - last[2] <= stale_s:
            return None
        return seq, parts[2] == b"1", now

    def command(self, frame: bytes, now: float) -> int | None:
        """Take an MCH1 frame; its seq, or None when it is not taken."""
        taken = self._take(self.asked, frame, b"MCH1", now, self.timeout_s)
        if taken:
            self.asked = taken
            self.armed |= not taken[1] and self.estop(now) == ESTOP_RELEASED
        return taken[0] if taken else None

    def take_estop(self, frame: bytes, now: float) -> None:
        self.released = self._take(self.released, frame, b"EST1", now, ESTOP_STALE_S) or self.released

    def estop(self, now: float) -> int:
        if now - self.released[2] > ESTOP_STALE_S:
            return ESTOP_SILENT
        return ESTOP_RELEASED if self.released[1] else ESTOP_PRESSED

    def powered(self, now: float) -> bool:
        """Whether the line is HIGH now; an e-stop that is not released disarms it until the machine is asked off."""
        self.armed &= self.estop(now) == ESTOP_RELEASED
        return self.armed and self.asked[1] and now - self.asked[2] <= self.timeout_s


def machine_receive(ready, est, gate: MachineGate, command_from: str, ignored: set, now: float) -> list:
    """Take the frames waiting on the ready sockets; returns (seq, sender) for each command taken."""
    answers = []
    for sock in ready:
        data, sender = sock.recvfrom(MAX_FRAME)
        if sock is est:
            gate.take_estop(data, now)
        elif sender[0] == command_from:
            seq = gate.command(data, now)
            answers += [] if seq is None else [(seq, sender)]
        elif sender[0] not in ignored:
            ignored.add(sender[0])
            print(f"tatbot_estop_relay: machine: ignoring commands from {sender[0]}", file=sys.stderr, flush=True)
    return answers


def run_machine(name: str, listen: int, command_from: str, estop_port: int, timeout_s: float,
                stop=lambda: False, open_line=open_output) -> None:
    """Drive the machine's line from MachineGate every MACHINE_TICK_S and answer every command taken."""
    cmd, est = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2))
    cmd.bind(("0.0.0.0", listen))
    est.bind(("127.0.0.1", estop_port))   # loopback: only the Pi's own e-stop relay reaches it
    gate, ignored = MachineGate(timeout_s), set()
    try:
        while not stop():
            try:
                write, close = open_line(name)
            except OSError as error:
                print(f"tatbot_estop_relay: machine {name}: {error.strerror}; retrying", file=sys.stderr, flush=True)
                time.sleep(REOPEN_S)
                continue
            on = False
            try:
                write(False)
                while not stop():
                    ready, _, _ = select.select([cmd, est], [], [], MACHINE_TICK_S)
                    now = time.monotonic()
                    answers = machine_receive(ready, est, gate, command_from, ignored, now)
                    if gate.powered(now) != on:
                        on = not on
                        write(on)
                    for seq, sender in answers:
                        with contextlib.suppress(OSError):
                            cmd.sendto(b"MCS1 %d %d %d %d\n" % (seq, gate.asked[1], on, gate.estop(now)), sender)
            except OSError as error:
                print(f"tatbot_estop_relay: machine {name}: {error.strerror}; reopening", file=sys.stderr, flush=True)
            finally:
                close()
            time.sleep(REOPEN_S)
    finally:
        cmd.close()
        est.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_estop_relay", description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--device", default="/dev/tatbot-estop", help="a Pico's serial port")
    source.add_argument("--gpio", metavar="NAME", help="the GPIO line wired to the NC contact, e.g. GPIO17")
    source.add_argument("--probe", metavar="NAME", help="the GPIO line wired to the station probe, e.g. GPIO27")
    source.add_argument("--machine", metavar="NAME", help="the GPIO line that powers the tattoo machine")
    parser.add_argument("--dest", type=parse_dest, action="append", default=[],
                        help="HOST:PORT, repeatable: every e-stop reader's UDP port (the ros node's driver, the arm "
                             "node's monitor); with --probe, every arm host's probe port")
    parser.add_argument("--bind", default=None, help="local source address to send from")
    parser.add_argument("--from", dest="command_from", help="--machine: the only address taken commands from")
    parser.add_argument("--listen", type=int, default=7642, help="--machine: the command port")
    parser.add_argument("--estop-port", type=int, default=7643, help="--machine: the loopback EST1 port")
    parser.add_argument("--timeout", type=float, default=0.2, help="--machine: a command older than this is off")
    args = parser.parse_args(argv)
    if len(set(args.dest)) != len(args.dest):
        parser.error("a --dest is given twice")
    if bool(args.machine) != bool(args.command_from) or bool(args.machine) == bool(args.dest):
        parser.error("--machine takes --from and no --dest; every other source needs a --dest")
    with contextlib.suppress(KeyboardInterrupt):
        if args.machine:
            run_machine(args.machine, args.listen, args.command_from, args.estop_port, args.timeout)
        elif args.probe:
            run_probe(args.probe, args.dest, args.bind)
        elif args.gpio:
            run_gpio(args.gpio, args.dest, args.bind)
        else:
            run(args.device, args.dest, args.bind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
