#!/usr/bin/env python3
"""Forward-kinematic read of a rollout flight log — a PRIMITIVE, not a pipeline.

Prints tip-height distribution, sustained lift events, xy footprint, and (for
high starts) a descent summary. Adapt freely: the tip offset and tip link are
CLI args precisely so a swapped end-effector or a re-run touch-off changes one
flag, not the code. This is a starting point — copy and edit it for whatever
the current session actually needs to measure.

    flight_fk.py <run_dir_or_csv> [--tip-offset X,Y,Z_m] [--link right/ee_gripper_link]
                                  [--settle 8] [--lift-mm 6]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(run):
    p = Path(run)
    if p.is_dir():
        c = sorted(p.glob("flight-*.csv")) or sorted(p.glob("*.csv"))
        if not c:
            sys.exit(f"no flight csv in {p}")
        p = c[-1]
    return list(csv.DictReader(open(p)))

def resolve_tip_offset(a):
    """--tip-offset if given, else the touch-off's measured offset.

    Reading it beats a copied constant: the offset is per-tool and per-mount,
    so a hardcoded default silently measures the wrong tool's tip the day a
    pen is swapped.
    """
    if a.tip_offset:
        return [float(x) for x in a.tip_offset.split(",")]
    repo = Path(a.urdf).resolve().parent.parent
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    import tool_spec
    workspace = tool_spec.read_workspace(repo)
    offset = tool_spec.tip_offset_m(workspace, a.prefix)
    if offset is None:
        sys.exit(f"no pen_tip_offset in {repo}/config/workspace.yaml — run a "
                 "touch-off, or pass --tip-offset X,Y,Z")
    tool = (workspace.get(a.prefix) or {}).get("tool_id")
    print(f"tip offset {[round(v * 1000, 2) for v in offset]} mm from workspace.yaml"
          f" (tool {tool or 'unnamed'})")
    return list(offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--urdf", default=str(REPO / "urdf/tatbot.urdf"))
    ap.add_argument("--kin", default=str(REPO / "scripts/vision"))
    ap.add_argument("--link", default="right/ee_gripper_link")
    ap.add_argument("--prefix", default="right")
    ap.add_argument("--tip-offset", default=None,
                    help="pen tip in EE frame, metres; defaults to the measured "
                         "offset in config/workspace.yaml (whatever tool the "
                         "last touch-off calibrated)")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--lift-mm", type=float, default=6.0)
    a = ap.parse_args()
    sys.path.insert(0, a.kin)
    from urdf_kinematics import UrdfChain
    chain = UrdfChain(a.urdf)
    names = chain.arm_joint_names(a.prefix)
    off = np.array(list(resolve_tip_offset(a)) + [1.0])
    rows = load(a.run)
    t0 = float(rows[0]["t_mono"])
    track = [(float(r["t_mono"]) - t0,
              (chain.link_pose(a.link, dict(zip(names, [float(r[f"pos_joint_{i}"]) for i in range(6)],
                                                strict=True))) @ off)[:3])
             for r in rows]
    tf = np.array([p for _, p in track])
    ts = np.array([tt for tt, _ in track])
    win = ts >= a.settle
    z = tf[win, 2] * 1000
    tw = ts[win]
    floor = np.percentile(z, 2)
    lifted = z > floor + a.lift_mm
    ev = tot = 0
    start = None
    for i in range(len(tw)):
        if lifted[i] and start is None:
            start = tw[i]
        elif not lifted[i] and start is not None:
            if tw[i]-start > 0.3:
                ev += 1
                tot += tw[i]-start
            start = None
    xy = tf[win, :2]*1000
    print(f"tip z mm  floor(p2) {floor:.1f}  med {np.median(z):.1f}  p95 {np.percentile(z,95):.1f}")
    print(f"lifts >{a.lift_mm:.0f}mm >0.3s: {ev}  airborne {tot:.1f}s of {tw[-1]-tw[0]:.0f}s")
    print(f"xy footprint {np.ptp(xy[:,0]):.0f} x {np.ptp(xy[:,1]):.0f} mm")
    zf = tf[:,2]*1000
    if zf[0] > floor + 23:  # started high -> descent read
        band = floor + 3
        below = zf < band
        if below.any():
            i = int(np.argmax(below))
            seg = zf[:i+1]
            after = zf[(ts>=ts[i]) & (ts<=ts[i]+1.5)]
            print(f"descent {zf[0]:.0f}mm -> band {ts[i]:.1f}s  "
                  f"monotone {(np.diff(seg)<=0.5).mean():.2f}  "
                  f"overshoot {floor-after.min():+.1f}mm")
        else:
            print(f"descent started {zf[0]:.0f}mm up, never reached band")

if __name__ == "__main__":
    main()
