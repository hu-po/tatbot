#!/usr/bin/env python3
"""Reader for wxai_teleop flight logs (.wxtl), and where the arm held still.

Format (from cpp/teleop/wxai_teleop.cpp, mirrored in rust/visiond/src/teleop.rs):
a 64-byte little-endian header then fixed records of `5 + 6*num_joints` f64 per
tick. A truncated trailing record is dropped rather than treated as an error.

The still intervals are the point of this module. Sampling the arm only while
it is stationary takes cross-host timing out of the error budget — within a
pause any timestamp is as good as any other — and gives blur-free tag corners
at the same time.

This is also the single home of "the arm held still" and "the pen was pressing
on something": the log carries the follower's *external* (gravity-compensated)
efforts at 400 Hz, so a touch-off is a still interval whose effort sits above
the log's own free-space baseline. Consumers: fuse_session.py, il_touchoff.py.

  python3 teleop_log.py <log.wxtl> [--tolerance-rad 0.003] [--min-still 0.5]
"""

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"WXTLOG1\0"
HEADER_LEN = 64


class TeleopLog:
    def __init__(self, path):
        raw = Path(path).expanduser().read_bytes()
        if len(raw) < HEADER_LEN or raw[:8] != MAGIC:
            raise ValueError(f"{path} is not a wxai_teleop flight log")
        self.num_joints = struct.unpack_from("<Q", raw, 8)[0]
        if not 0 < self.num_joints <= 32:
            raise ValueError(f"implausible num_joints {self.num_joints}")
        self.period_s = struct.unpack_from("<d", raw, 16)[0]
        self.wall_start_ns = struct.unpack_from("<Q", raw, 56)[0]

        values = 5 + 6 * self.num_joints
        payload = raw[HEADER_LEN:]
        count = len(payload) // (values * 8)
        table = np.frombuffer(payload[:count * values * 8], dtype="<f8").reshape(count, values)
        joints = self.num_joints
        self.t_wake = table[:, 1]
        self.leader_pos = table[:, 5:5 + joints]
        self.follower_pos = table[:, 5 + 2 * joints:5 + 3 * joints]
        self.follower_vel = table[:, 5 + 3 * joints:5 + 4 * joints]
        # External (gravity-compensated) efforts: quiet in free space, elevated
        # on contact — the channel a touch-off reads. Last joint is the gripper
        # (N, not Nm) and is excluded from the arm contact signal so grip force
        # never masquerades as a touch.
        self.follower_eff = table[:, 5 + 4 * joints:5 + 5 * joints]
        arm_cols = self.follower_eff[:, :-1] if joints > 1 else self.follower_eff
        self.arm_eff = np.abs(arm_cols).max(axis=1)
        # Absolute wall time per tick: this is what lets camera observations,
        # audio and joint angles meet on one timeline.
        self.unix_seconds = self.wall_start_ns / 1e9 + self.t_wake

    def __len__(self):
        return len(self.t_wake)

    def duration_s(self):
        return float(self.t_wake[-1] - self.t_wake[0]) if len(self) else 0.0

    def still_intervals(self, tolerance_rad=0.003, min_duration=0.5, arm="follower"):
        """Windows where no joint wanders more than `tolerance_rad`.

        Deliberately position-based, not velocity-based: the logged velocity
        has a noise floor around 0.007 rad/s and never reads zero, so a speed
        threshold finds nothing. What matters anyway is whether the arm stayed
        put, which position deviation states directly. 0.003 rad is under a
        millimetre of wrist travel at this arm's reach.
        """
        positions = self.follower_pos if arm == "follower" else self.leader_pos
        intervals = []
        start = 0
        for index in range(1, len(positions)):
            deviation = np.abs(positions[start:index + 1] - positions[start]).max()
            if deviation > tolerance_rad:
                if index - 1 > start:
                    intervals.append((start, index - 1))
                start = index
        if len(positions) - 1 > start:
            intervals.append((start, len(positions) - 1))

        speed = np.abs(self.follower_vel).max(axis=1)
        out = []
        for first, last in intervals:
            duration = float(self.unix_seconds[last] - self.unix_seconds[first])
            if duration < min_duration:
                continue
            eff = self.arm_eff[first:last + 1]
            out.append({
                "start_unix": float(self.unix_seconds[first]),
                "end_unix": float(self.unix_seconds[last]),
                "duration_s": duration,
                "ticks": last - first + 1,
                # Median over hundreds of ticks: encoder noise averages away.
                "follower_pos": np.median(self.follower_pos[first:last + 1], axis=0).tolist(),
                "max_speed_rad_s": float(speed[first:last + 1].max()),
                "arm_eff_med_nm": float(np.median(eff)),
                "arm_eff_p95_nm": float(np.percentile(eff, 95)),
            })
        return out

    def window_sample(self, start_unix, end_unix, span_s=0.4, tolerance_rad=0.010):
        """The quietest joint sample inside a wall-clock window.

        For GUIDED touches: the timeline says when the operator pressed, so
        the sample is the lowest-motion `span_s` sub-window in that window —
        with a tolerance looser than free-space stillness, because pressing a
        pen through teleop trembles (the 2026-08-21 session left two of eight
        touches with no 0.003 rad still interval at all). Effort rides along
        as a diagnostic only: at pen forces it sits inside the pose-dependent
        gravity-comp residual (0.9-7 Nm on the same log) and cannot gate.
        Returns None when even the loose tolerance never holds.
        """
        mask = (self.unix_seconds >= start_unix) & (self.unix_seconds <= end_unix)
        indices = np.nonzero(mask)[0]
        span = max(2, int(round(span_s / max(self.period_s, 1e-4))))
        if len(indices) < span:
            return None
        first, last = indices[0], indices[-1]
        best_start, best_spread = None, tolerance_rad
        for window_start in range(first, last - span + 2):
            block = self.follower_pos[window_start:window_start + span]
            spread = float((block.max(axis=0) - block.min(axis=0)).max())
            if spread < best_spread:
                best_start, best_spread = window_start, spread
        if best_start is None:
            return None
        block = slice(best_start, best_start + span)
        return {
            "joints": np.median(self.follower_pos[block], axis=0).tolist(),
            "start_unix": float(self.unix_seconds[best_start]),
            "end_unix": float(self.unix_seconds[best_start + span - 1]),
            "duration_s": float(span * self.period_s),
            "spread_rad": best_spread,
            "arm_eff_med_nm": float(np.median(self.arm_eff[block])),
        }

    def classify_contacts(self, intervals, min_rise_nm=1.0, mad_k=6.0):
        """Mark which still intervals are touches, against the log's own baseline.

        The threshold is estimated from this log, not a fixed constant: the
        free-space effort level depends on payload and pose, and contact is a
        small fraction of any session, so the whole-log median is a robust
        free-space baseline. `min_rise_nm` keeps a near-zero MAD (a very quiet
        log) from turning noise into touches. Each interval gains a "contact"
        bool; returns the threshold details for the audit trail.
        """
        baseline = float(np.median(self.arm_eff))
        mad = float(np.median(np.abs(self.arm_eff - baseline)))
        threshold = baseline + max(min_rise_nm, mad_k * 1.4826 * mad)
        for interval in intervals:
            interval["contact"] = bool(interval["arm_eff_med_nm"] >= threshold)
        return {"baseline_nm": baseline, "mad_nm": mad, "threshold_nm": threshold,
                "min_rise_nm": min_rise_nm, "mad_k": mad_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tolerance-rad", type=float, default=0.003)
    ap.add_argument("--min-still", type=float, default=0.5)
    args = ap.parse_args()
    log = TeleopLog(args.log)
    print(f"{len(log)} ticks, {log.num_joints} joints, {log.duration_s():.1f} s, "
          f"period {log.period_s * 1000:.2f} ms")
    import datetime
    started = datetime.datetime.fromtimestamp(log.wall_start_ns / 1e9)
    print(f"wall start: {started:%Y-%m-%d %H:%M:%S}")
    intervals = log.still_intervals(args.tolerance_rad, args.min_still)
    touch = log.classify_contacts(intervals)
    print(f"\n{len(intervals)} still intervals "
          f"(within {args.tolerance_rad} rad for >{args.min_still}s), "
          f"{sum(i['contact'] for i in intervals)} in contact "
          f"(effort >= {touch['threshold_nm']:.2f} Nm, "
          f"baseline {touch['baseline_nm']:.2f}):")
    for i, interval in enumerate(intervals[:12]):
        offset = interval["start_unix"] - log.unix_seconds[0]
        mark = "TOUCH" if interval["contact"] else "     "
        print(f"  {i + 1:3d}: t+{offset:6.1f}s  {interval['duration_s']:5.2f}s  "
              f"{interval['ticks']:5d} ticks  peak {interval['max_speed_rad_s']:.4f} rad/s"
              f"  eff {interval['arm_eff_med_nm']:5.2f} Nm  {mark}")
    if len(intervals) > 12:
        print(f"  ... {len(intervals) - 12} more")
    if intervals:
        total = sum(i["duration_s"] for i in intervals)
        print(f"\nstill for {total:.1f}s of {log.duration_s():.1f}s "
              f"({100 * total / max(log.duration_s(), 1e-9):.0f}%)")


if __name__ == "__main__":
    main()
