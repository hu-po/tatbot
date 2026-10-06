#!/usr/bin/env python3
"""The bench M0 numbers from a driver flight ring (<run>/<arm>-flight.bin, ros/README.md "Flight log").

    python3 flight_summary.py RUN_DIR [--arm right] [--no-tip]

RUN_DIR is a ros-stack run (`tatbot ros logs last --workflow ros-stack`). Prints one JSON line:
  records, seconds         what the ring holds
  period_max_ms            the longest gap between records (400 Hz = 2.5 ms); jitter_ms = that - 2.5
  rt_period_max_ms         the driver's own read-to-read maximum
  latches                  each latch: reason, e-stop state and frame age at the latch, the fastest
                           joint in the 50 ms before, and how long until every joint read < 5 mrad/s
  carriage_effort_n        the carriage's external effort while the arm rests (running, unlatched,
                           every joint < 0.01 rad/s): median, p5, p95 (the contact cap is 20 N above it)
  tip_lag_mm               signed lag of the measured tip behind the command along the commanded tcp +z
                           (toward the paper) while running unlatched: p50, p99, max (the guard trips
                           at 0.8 mm held 0.15 s). Needs tatbot_motion and $TATBOT_REPO; --no-tip skips it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

J = ("q", "qd", "effort", "cmd_q", "sent_q", "sent_qd")
DTYPE = np.dtype([("t", "<f8"), *((name, "<f4", 7) for name in J), ("estop_age_s", "<f4"),
                  ("rt_period_max_ms", "<f4"), ("estop_ok", "u1"), ("latched", "u1"), ("latch_reason", "u1"),
                  ("phase", "u1")])
RUNNING = 1


def read(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header, _, body = raw.partition(b"\n")
    size = int(header.split()[-1])
    if size != DTYPE.itemsize:
        raise ValueError(f"{path}: {size}-byte records, this reader knows {DTYPE.itemsize}")
    return np.frombuffer(body[: len(body) // size * size], dtype=DTYPE)


def summarize(r: np.ndarray, kin=None) -> dict:
    t = r["t"] - r["t"][0] if len(r) else r["t"]
    gaps = np.diff(t) * 1000 if len(t) > 1 else np.zeros(1)
    out = {"records": int(len(r)), "seconds": round(float(t[-1]) if len(t) else 0.0, 1),
           "period_max_ms": round(float(gaps.max()), 3), "jitter_ms": round(float(gaps.max()) - 2.5, 3),
           "rt_period_max_ms": round(float(r["rt_period_max_ms"].max()) if len(r) else 0.0, 3), "latches": []}
    speed = np.abs(r["qd"][:, :6]).max(axis=1) if len(r) else np.zeros(0)
    for i in np.flatnonzero(np.diff(r["latched"].astype(int)) == 1) + 1:
        before = speed[(t >= t[i] - 0.05) & (t < t[i])]
        still = np.flatnonzero((t >= t[i]) & (speed < 0.005))
        out["latches"].append({"t": round(float(t[i]), 3), "reason": int(r["latch_reason"][i]),
                               "estop_ok": int(r["estop_ok"][i]), "estop_age_s": round(float(r["estop_age_s"][i]), 3),
                               "qd_max_before": round(float(before.max()) if len(before) else 0.0, 4),
                               "still_after_ms": round(float(t[still[0]] - t[i]) * 1000, 1) if len(still) else None})
    running = (r["phase"] == RUNNING) & (r["latched"] == 0)
    rest = running & (np.abs(r["qd"]).max(axis=1) < 0.01) if len(r) else running
    if rest.any():
        e = r["effort"][rest, 6]
        out["carriage_effort_n"] = {k: round(float(np.percentile(e, p)), 2) for k, p in (("median", 50), ("p5", 5), ("p95", 95))}
    if kin is not None and running.any():
        idx = np.flatnonzero(running)[:: max(1, int(running.sum()) // 20000)]
        lag = []
        for i in idx:
            cmd, meas = kin.fk(r["cmd_q"][i].astype(float)), kin.fk(r["q"][i].astype(float))
            lag.append(float((cmd[:3, 3] - meas[:3, 3]) @ cmd[:3, 2]))
        lag = np.asarray(lag) * 1000
        out["tip_lag_mm"] = {"p50": round(float(np.median(lag)), 3), "p99": round(float(np.percentile(lag, 99)), 3),
                             "max": round(float(lag.max()), 3), "samples": int(len(lag))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flight_summary.py", description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--no-tip", action="store_true")
    args = parser.parse_args(argv)
    ring = Path(args.run_dir).expanduser() / f"{args.arm}-flight.bin"
    if not ring.is_file():
        print(f"no flight ring at {ring}", file=sys.stderr)
        return 2
    kin = None
    if not args.no_tip:
        from tatbot_description import robot_description
        from tatbot_motion import Kinematics

        kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    print(json.dumps({"ring": str(ring), **summarize(read(ring), kin)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
