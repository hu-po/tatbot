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
  tracking                 while the commanded tip moves at drawing speed (1-20 mm/s, held 0.3 s): the
                           measured tip minus the sent one in mm along the path, across it and along the
                           tool axis (p50, p95, p99), per-joint |error| in encoder counts (p95), how much
                           larger it is in the 0.3 s after a joint reverses, and the seconds measured.
                           --since/--until (Unix s, version 2 rings) narrow it to one draw or wave.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

J = ("q", "qd", "effort", "cmd_q", "sent_q", "sent_qd")
_TAIL = [("estop_age_s", "<f4"), ("rt_period_max_ms", "<f4"), ("estop_ok", "u1"), ("latched", "u1"),
         ("latch_reason", "u1"), ("phase", "u1")]
DTYPES = {
    1: np.dtype([("t", "<f8"), *((name, "<f4", 7) for name in J), *_TAIL]),
    # 2 adds the controller's compensation effort, rotor temperatures and output header (id, its µs clock)
    2: np.dtype([("t", "<f8"), *((name, "<f4", 7) for name in (*J, "compensation", "rotor_temp")),
                 ("controller_id", "<u4"), ("controller_us", "<u8"), *_TAIL]),
}
DTYPE = DTYPES[2]
RUNNING = 1
ENCODER_RAD = 2 * np.pi / 16471   # one count of an arm joint's reported position (2026-10-13 draw)


def header(path: Path) -> dict:
    """The ring's header line: version, arm, record bytes and, from version 2, realtime0 (Unix s at t == 0), or
    clock=sim on a sim arm, whose t is the sim server's clock."""
    with open(path, "rb") as f:
        words = f.readline().decode().split()
    out = {"version": int(words[1]), "arm": words[2], "record_bytes": int(words[-1])}
    out.update((k, v if k == "clock" else float(v)) for k, _, v in (w.partition("=") for w in words[3:-1]) if v)
    return out


def read(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    line, _, body = raw.partition(b"\n")
    words = line.split()
    dtype = DTYPES.get(int(words[1]))
    size = int(words[-1])
    if dtype is None or size != dtype.itemsize:
        raise ValueError(f"{path}: version {words[1].decode()} with {size}-byte records, which this reader does not know")
    return np.frombuffer(body[: len(body) // size * size], dtype=dtype)


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
        out["tracking"] = tracking(r[running], kin)
    return out


def _pct(x, ps=(50, 95, 99)) -> dict:
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps}


def tracking(r: np.ndarray, kin, step: int = 4, min_mm_s: float = 1.0, max_mm_s: float = 20.0) -> dict:
    """Measured minus sent tip while the command moves at drawing speed (every `step`th record), split along the
    commanded path, across it and along the tool axis; per-joint errors in encoder counts, and near reversals."""
    r = r[np.isfinite(r["sent_q"][:, 0])][::step]
    if len(r) < 50:
        return {"seconds": 0.0}
    t = r["t"]
    pc, pm, z = np.empty((len(r), 3)), np.empty((len(r), 3)), np.empty((len(r), 3))
    for k in range(len(r)):
        cmd = kin.fk(r["sent_q"][k].astype(float))
        pc[k], z[k] = cmd[:3, 3], cmd[:3, 2]
        pm[k] = kin.fk(r["q"][k].astype(float))[:3, 3]
    dt = np.gradient(t)
    v = np.gradient(pc, axis=0) / dt[:, None]
    speed = np.linalg.norm(v, axis=1) * 1000
    contiguous = np.r_[True, np.diff(t) < 0.03]
    window = max(1, int(round(0.3 / (0.0025 * step))))
    steady = np.convolve((speed > min_mm_s) & (speed < max_mm_s) & contiguous, np.ones(window) / window, "same") > 0.99
    if not steady.any():
        return {"seconds": 0.0}
    e = (pm - pc) * 1000
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    along, depth = (e * u).sum(1), (e * z).sum(1)
    cross = np.linalg.norm(e - along[:, None] * u - depth[:, None] * z, axis=1)
    counts = np.abs(r["q"][:, :6] - r["sent_q"][:, :6]).astype(float) / ENCODER_RAD
    reversal = []
    for j in range(6):
        s = np.sign(r["sent_qd"][:, j])
        near = np.zeros(len(r), bool)
        for i in np.flatnonzero((s[1:] != s[:-1]) & (s[1:] != 0)) + 1:
            near |= (t >= t[i]) & (t < t[i] + 0.3)
        a, b = counts[steady & near, j], counts[steady & ~near, j]
        reversal.append(round(float(np.percentile(a, 95) / max(np.percentile(b, 95), 1e-9)), 2)
                        if len(a) > 20 and len(b) > 20 else None)
    return {"seconds": round(float(steady.sum() * np.median(np.diff(t))), 1),
            "speed_mm_s": _pct(speed[steady], (50, 95)),
            "along_mm": _pct(along[steady]), "cross_mm": _pct(cross[steady]), "depth_mm": _pct(depth[steady]),
            "joint_counts_p95": [round(float(np.percentile(counts[steady, j], 95)), 2) for j in range(6)],
            "reversal_ratio_p95": reversal}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flight_summary.py", description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--no-tip", action="store_true")
    parser.add_argument("--since", type=float, help="Unix s: only records from then")
    parser.add_argument("--until", type=float, help="Unix s: only records until then")
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
    r = read(ring)
    realtime0 = header(ring).get("realtime0")
    if args.since or args.until:
        if realtime0 is None:
            print("--since/--until need a version 2 ring (its header carries realtime0)", file=sys.stderr)
            return 2
        wall = r["t"] + realtime0
        r = r[(wall >= (args.since or -np.inf)) & (wall <= (args.until or np.inf))]
    print(json.dumps({"ring": str(ring), "realtime0": realtime0, **summarize(r, kin)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
