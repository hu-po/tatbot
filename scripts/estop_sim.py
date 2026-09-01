#!/usr/bin/env python3
"""Simulate the tatbot e-stop box on a PTY, for desk-testing without hardware.

Speaks the same protocol as firmware/estop_pico/code.py: "EST1 <seq> <state>"
frames at 100 Hz, state 1 = released/OK, 0 = pressed. Prints the PTY path to
point the consumer at, e.g.:

    ./scripts/estop_sim.py
    # in another terminal:
    ./cpp/teleop/build/wxai_teleop --estop /dev/pts/N

Keys (single keypress, no Enter):
    p   toggle pressed/released      (latching button press / twist-release)
    k   toggle heartbeat on/off      (simulates unplug / dead firmware)
    q   quit
"""

import contextlib
import os
import pty
import select
import sys
import termios
import time
import tty

FRAME_PERIOD_S = 0.010  # 100 Hz, same as the firmware


def main() -> None:
    master, slave = pty.openpty()
    print(f"e-stop simulator on: {os.ttyname(slave)}")
    print("keys: [p] press/release  [k] heartbeat on/off  [q] quit")

    pressed = False
    heartbeat = True
    seq = 0
    stdin_fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)
    try:
        next_frame = time.monotonic()
        while True:
            timeout = max(0.0, next_frame - time.monotonic())
            readable, _, _ = select.select([stdin_fd, master], [], [], timeout)
            if stdin_fd in readable:
                key = os.read(stdin_fd, 1).decode(errors="replace").lower()
                if key == "p":
                    pressed = not pressed
                    print(f"\r[sim] button {'PRESSED' if pressed else 'released'}   ")
                elif key == "k":
                    heartbeat = not heartbeat
                    print(f"\r[sim] heartbeat {'ON' if heartbeat else 'OFF (silent)'}   ")
                elif key == "q":
                    return
            if master in readable:
                os.read(master, 4096)  # consumer echo/noise; discard
            if time.monotonic() >= next_frame:
                if heartbeat:
                    frame = f"EST1 {seq} {0 if pressed else 1}\n".encode()
                    with contextlib.suppress(OSError):  # no reader attached yet
                        os.write(master, frame)
                    seq += 1
                next_frame += FRAME_PERIOD_S
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_termios)
        os.close(master)
        os.close(slave)


if __name__ == "__main__":
    main()
