#!/usr/bin/env python3
"""Analyze a wxai_teleop flight-recorder log (.wxtl) for timing hiccups and
tracking quality. Pure stdlib — no numpy required.

Usage: analyze_log.py <log.wxtl> [--top N]

Format (all little-endian doubles unless noted; see wxai_teleop.cpp):
  header: 8s magic "WXTLOG1\\0", u64 num_joints, f64 period_s, f64 tau_s,
          f64 goal_time_s, f64 ff_gain, u64 abs_gripper, i64 wall_start_ns
  record: t_sched, t_wake, t_leader_read, t_follower_read, t_cmd,
          leader_pos[n], leader_vel[n],
          follower_pos[n], follower_vel[n], follower_eff[n], target[n]
"""

import argparse
import struct
import sys
from array import array
from datetime import datetime

HEADER_FMT = "<8sQdddd Qq".replace(" ", "")
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def fmt_ms(seconds):
    return f"{seconds * 1e3:.3f} ms"


def stats_line(name, vals):
    s = sorted(vals)
    return (
        f"  {name:<22} mean {fmt_ms(sum(s) / len(s)):>10}   p50 {fmt_ms(percentile(s, 50)):>10}   "
        f"p99 {fmt_ms(percentile(s, 99)):>10}   max {fmt_ms(s[-1]):>10}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=10, help="worst hiccups to list")
    args = ap.parse_args()

    with open(args.log, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
        magic, n, period, tau, goal_time, ff_gain, abs_gripper, wall_ns = struct.unpack(
            HEADER_FMT, raw_header
        )
        if magic != b"WXTLOG1\x00":
            sys.exit(f"not a wxai_teleop log (magic {magic!r})")
        n = int(n)
        data = array("d")
        data.frombytes(f.read())

    rec_len = 5 + 6 * n
    ticks = len(data) // rec_len
    if ticks == 0:
        sys.exit("log contains no records")
    data = data[: ticks * rec_len]

    def col(i):
        return data[i::rec_len]

    t_sched = col(0)
    t_wake = col(1)
    t_leader = col(2)
    t_follower = col(3)
    t_cmd = col(4)

    duration = t_cmd[-1] - t_wake[0]
    print(f"log:        {args.log}")
    print(f"started:    {datetime.fromtimestamp(wall_ns / 1e9)}")
    print(
        f"config:     {n} joints, period {fmt_ms(period)} ({1 / period:.0f} Hz), "
        f"tau {fmt_ms(tau)}, goal_time {fmt_ms(goal_time)}, "
        f"ff_gain {ff_gain}, abs_gripper {bool(abs_gripper)}"
    )
    print(
        f"duration:   {duration:.2f} s, {ticks} ticks, "
        f"achieved {ticks / duration:.1f} Hz (target {1 / period:.0f} Hz)"
    )

    print("\n-- loop timing --")
    periods = [t_wake[i + 1] - t_wake[i] for i in range(ticks - 1)]
    lateness = [t_wake[i] - t_sched[i] for i in range(ticks)]
    leader_read = [t_leader[i] - t_wake[i] for i in range(ticks)]
    follower_read = [t_follower[i] - t_leader[i] for i in range(ticks)]
    command = [t_cmd[i] - t_follower[i] for i in range(ticks)]
    busy = [t_cmd[i] - t_wake[i] for i in range(ticks)]
    print(stats_line("tick-to-tick period", periods))
    print(stats_line("wake lateness", lateness))
    print(stats_line("leader state read", leader_read))
    print(stats_line("follower state read", follower_read))
    print(stats_line("command send", command))
    print(stats_line("total busy time", busy))
    overruns = sum(1 for b in busy if b > period)
    print(f"  overruns (busy > period): {overruns} ({overruns / ticks * 100:.2f}%)")

    print(f"\n-- worst {args.top} hiccups (tick-to-tick period) --")
    worst = sorted(range(ticks - 1), key=lambda i: periods[i], reverse=True)[: args.top]
    for i in sorted(worst):
        print(
            f"  t={t_wake[i]:9.3f} s  period {fmt_ms(periods[i]):>11}  "
            f"(leader read {fmt_ms(leader_read[i])}, follower read {fmt_ms(follower_read[i])}, "
            f"command {fmt_ms(command[i])})"
        )

    print("\n-- tracking (target vs follower position, per joint) --")
    lag_ticks = max(1, round((tau + goal_time) / period))
    print(f"  (follower compared against the target from {lag_ticks} ticks earlier "
          f"to discount commanded lag)")
    for j in range(n):
        tgt = col(5 + 5 * n + j)
        fpos = col(5 + 2 * n + j)
        errs = [abs(fpos[i + lag_ticks] - tgt[i]) for i in range(ticks - lag_ticks)]
        errs.sort()
        unit = "m" if j == n - 1 else "rad"
        moved = max(tgt) - min(tgt)
        print(
            f"  joint {j}: err p50 {percentile(errs, 50):.5f} p99 {percentile(errs, 99):.5f} "
            f"max {errs[-1]:.5f} {unit}   (target range {moved:.4f} {unit})"
        )

    print("\n-- follower external effort (contact/grip forces, per joint) --")
    for j in range(n):
        eff = sorted(abs(v) for v in col(5 + 4 * n + j))
        unit = "N" if j == n - 1 else "Nm"
        line = (
            f"  joint {j}: p50 {percentile(eff, 50):.2f} p99 {percentile(eff, 99):.2f} "
            f"max {eff[-1]:.2f} {unit}"
        )
        if j == n - 1:
            grip = list(col(5 + 4 * n + j))
            for threshold in (50.0, 100.0):
                above = sum(1 for v in grip if abs(v) > threshold) * period
                line += f"   >{threshold:.0f}N for {above:.1f}s"
        print(line)

    print("\n-- leader velocity (smoothness of your hand-guiding, per joint) --")
    for j in range(n):
        vel = [abs(v) for v in col(5 + n + j)]
        vel.sort()
        unit = "m/s" if j == n - 1 else "rad/s"
        print(
            f"  joint {j}: p50 {percentile(vel, 50):.4f} p99 {percentile(vel, 99):.4f} "
            f"max {vel[-1]:.4f} {unit}"
        )


if __name__ == "__main__":
    main()
