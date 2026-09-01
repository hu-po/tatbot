#!/usr/bin/env python3
"""Bench-check the Tatbot Pico e-stop without connecting to either arm."""

from __future__ import annotations

import argparse
import os
import re
import select
import sys
import termios
import time
from dataclasses import dataclass

FRAME_RE = re.compile(rb"EST1 ([0-9]+) ([01])$")


@dataclass(frozen=True)
class Sample:
    elapsed_s: float
    sequences: tuple[int, ...]
    states: tuple[int, ...]
    invalid_frames: int

    @property
    def rate_hz(self) -> float:
        return len(self.sequences) / self.elapsed_s


def sample_device(device: str, duration_s: float) -> Sample:
    fd = os.open(device, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        if os.isatty(fd):
            attrs = termios.tcgetattr(fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            # Let any USB-endpoint backlog drain, then discard it so the rate
            # measurement reflects live frames rather than bytes queued while
            # no process had the tty open.
            time.sleep(0.1)
            termios.tcflush(fd, termios.TCIFLUSH)

        start = time.monotonic()
        deadline = start + duration_s
        buffer = b""
        sequences: list[int] = []
        states: list[int] = []
        invalid_frames = 0
        synchronized = False

        while time.monotonic() < deadline:
            timeout = max(0.0, min(0.05, deadline - time.monotonic()))
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                continue
            try:
                chunk = os.read(fd, 512)
            except BlockingIOError:
                continue
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                if not synchronized:
                    # Opening a streaming tty can begin halfway through a
                    # frame. The first newline establishes a clean boundary.
                    synchronized = True
                    continue
                match = FRAME_RE.fullmatch(line.rstrip(b"\r"))
                if match is None:
                    invalid_frames += 1
                    continue
                sequences.append(int(match.group(1)))
                states.append(int(match.group(2)))

        return Sample(
            elapsed_s=time.monotonic() - start,
            sequences=tuple(sequences),
            states=tuple(states),
            invalid_frames=invalid_frames,
        )
    finally:
        os.close(fd)


def validate_sample(sample: Sample, expect: str, min_rate_hz: float) -> list[str]:
    errors: list[str] = []
    if not sample.sequences:
        return ["no valid heartbeat frames received"]
    if any(
        current <= previous
        for previous, current in zip(sample.sequences, sample.sequences[1:], strict=False)
    ):
        errors.append("heartbeat sequence is not strictly increasing")
    if sample.invalid_frames:
        errors.append(f"received {sample.invalid_frames} malformed frame(s)")
    if sample.rate_hz < min_rate_hz:
        errors.append(f"heartbeat rate {sample.rate_hz:.1f} Hz is below {min_rate_hz:.1f} Hz")

    observed = set(sample.states)
    required = {
        "released": {1},
        "stopped": {0},
        "cycle": {0, 1},
    }.get(expect)
    if required is not None and observed != required:
        errors.append(f"expected states {sorted(required)}, observed {sorted(observed)}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("TATBOT_ESTOP_DEVICE") or "/dev/tatbot-estop")
    parser.add_argument("--duration", type=float, default=1.0, help="sample duration in seconds")
    parser.add_argument(
        "--expect",
        choices=("any", "released", "stopped", "cycle"),
        default="any",
        help="required observed state: released=1, stopped=0, cycle=both",
    )
    parser.add_argument("--min-rate-hz", type=float, default=80.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.min_rate_hz < 0:
        parser.error("--min-rate-hz must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        sample = sample_device(args.device, args.duration)
    except OSError as exc:
        print(f"FAIL device={args.device}: {exc}", file=sys.stderr)
        return 2

    observed = sorted(set(sample.states))
    errors = validate_sample(sample, args.expect, args.min_rate_hz)
    status = "PASS" if not errors else "FAIL"
    first = sample.sequences[0] if sample.sequences else "none"
    last = sample.sequences[-1] if sample.sequences else "none"
    print(
        f"{status} device={args.device} frames={len(sample.sequences)} "
        f"rate_hz={sample.rate_hz:.1f} seq={first}..{last} states={observed} "
        f"invalid={sample.invalid_frames}"
    )
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
