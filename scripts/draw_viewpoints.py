#!/usr/bin/env python3
"""Score wrist-camera viewpoints against a mapped surface: from a teleop rehearsal log, and for the orbit generator.

    draw_viewpoints.py <teleop.wxtl> <surface.npz> [--every-s 1.0] [--standoffs 60,80,120] [--tilt 15]

For every sampled follower pose in the flight log: the tip's gap above the
surface, the tool-axis lean, and per D405 whether the contact patch (the anchor
plus an 8 mm ring) is inside the frustum, at what distance and incidence. Then
the same score for the orbit `draw_stage.py orbit` would generate from the
log's first pose at each candidate standoff, so the operator's instinct and the
generator can be compared in one table. Geometry only: no joint plan is run, so
a rehearsal that starts near a joint limit still scores (docs/draw.md).

The surface's chart is used analytically (a plane or a cylinder), so poses far
outside the mapped patch still get a gap and a uv.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts" / "lib"), str(REPO / "scripts"), str(REPO / "scripts" / "vision")]

import draw_kinematics as dk  # noqa: E402
import draw_path as dp  # noqa: E402
import draw_stage  # noqa: E402
import draw_surface as ds  # noqa: E402

HEADER_FMT = "<8sQddddQq"
DEPTH_RANGE_M = (0.07, 0.5)
PATCH_RING_M = 0.008


def read_flight_log(path: Path):
    raw = Path(path).read_bytes()
    magic, n, period, *_ = struct.unpack(HEADER_FMT, raw[:64])
    if magic != b"WXTLOG1\x00":
        raise SystemExit(f"{path}: not a wxai_teleop flight log")
    n = int(n)
    data = np.frombuffer(raw[64:], dtype="<f8")
    rec = 5 + 6 * n
    ticks = len(data) // rec
    data = data[: ticks * rec].reshape(ticks, rec)
    return data[:, 1], data[:, 5 + 2 * n: 5 + 3 * n], float(period)


def patch_points(surface):
    ring = np.array([[0.0, 0.0]] + [[PATCH_RING_M * np.cos(a), PATCH_RING_M * np.sin(a)]
                                    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    pts, _, _, normals = surface.frame(ring + surface.anchor_uv)
    return pts, normals


def camera_view(root_from_camera: np.ndarray, pts: np.ndarray, normals: np.ndarray, intrinsics=dk.D405_INTRINSICS):
    """(fraction of patch points in the frustum and depth range, mean distance m, mean incidence deg)."""
    return dk.camera_view(root_from_camera, pts, normals, intrinsics, DEPTH_RANGE_M)


def gap_and_lean(surface, tip_root: np.ndarray, axis_root: np.ndarray):
    """Tip gap above the chart (analytic) and the tool axis lean off the chart normal there."""
    chart = surface.chart
    if chart.kind == "cylinder":
        a, n = chart.rot[:, 0], chart.rot[:, 2]
        d = tip_root - (chart.center - chart.radius * n)
        radial = d - (d @ a) * a
        gap = float(np.linalg.norm(radial) - chart.radius)
        normal = radial / max(np.linalg.norm(radial), 1e-12)
    else:
        normal = chart.rot[:, 2]
        gap = float((tip_root - chart.center) @ normal)
    lean = float(np.degrees(np.arccos(np.clip(-normal @ axis_root, -1.0, 1.0))))
    return gap, lean, chart.invert(tip_root[None])[0]


def score_pose(surface, chain, axis6, pts, normals, joints, carriage_m):
    p, rotation, _ = dk.fk_ballpoint(joints, carriage_m)
    tip = dk.root_from_base(p)
    gap, lean, uv = gap_and_lean(surface, tip, rotation @ axis6)
    values = dk.joint_map(joints, carriage_m)
    cams = {role: camera_view(chain.link_pose(link, values), pts, normals)
            for role, link in draw_stage.CAMERA_LINKS.items()}
    return {"gap_mm": gap * 1e3, "lean_deg": lean, "uv_mm": (uv * 1e3).tolist(), "cams": cams}


def score_cartesian(surface, pts, normals, p_base, rotation, axis6, carriage_m):
    tip = dk.root_from_base(p_base)
    gap, lean, uv = gap_and_lean(surface, tip, rotation @ axis6)
    cams = {role: dk.camera_view(pose, pts, normals) for role, pose in dk.rig_cameras(tip, rotation, carriage_m).items()}
    return {"gap_mm": gap * 1e3, "lean_deg": lean, "uv_mm": (uv * 1e3).tolist(), "cams": cams}


def fmt_cam(c):
    frac, dist, inc = c
    return f"{frac * 100:3.0f}% {dist * 1e3 if dist == dist else float('nan'):4.0f} mm {inc if inc == inc else float('nan'):3.0f}deg"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("surface")
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--standoffs", default="60,80,120", help="orbit standoffs (mm) to score, comma separated")
    ap.add_argument("--tilt", type=float, default=None, help="orbit tilt deg (default: draw.json default)")
    ap.add_argument("--json", help="write the scores here")
    args = ap.parse_args(argv)

    t, follower, period = read_flight_log(Path(args.log))
    surface = ds.HeightFieldSurface.from_npz(args.surface)
    chain = dk.urdf_chain()
    axis6 = dk.tool_axis_in_link6()
    pts, normals = patch_points(surface)
    step = max(1, int(round(args.every_s / period)))
    print(f"{args.log}: {len(t)} ticks, {t[-1]:.1f} s; surface: {surface.chart.kind} chart"
          f"{'' if surface.chart.kind != 'cylinder' else f' r={surface.chart.radius * 1e3:.1f} mm'}; "
          f"patch = anchor + {PATCH_RING_M * 1e3:.0f} mm ring")
    print(f"{'t s':>6} {'gap mm':>6} {'lean':>4} {'tip uv mm':>11} | upper: in-view, depth, incidence | lower")
    rows = []
    both = 0
    for i in range(0, len(t), step):
        s = score_pose(surface, chain, axis6, pts, normals, follower[i, :6], follower[i, 6])
        u, low = s["cams"]["wrist_upper"], s["cams"]["wrist_lower"]
        ok = u[0] == 1.0 and low[0] == 1.0
        both += ok
        print(f"{t[i]:6.1f} {s['gap_mm']:6.0f} {s['lean_deg']:4.0f} {s['uv_mm'][0]:5.0f},{s['uv_mm'][1]:5.0f} | "
              f"{fmt_cam(u)} | {fmt_cam(low)}{'  <- both' if ok else ''}")
        rows.append({"t": float(t[i]), **s, "joints": follower[i].tolist()})
    print(f"{both} of {len(rows)} poses have the whole patch in both cameras")

    j0, c0 = follower[0, :6], dk.CARRIAGE_IK_BIAS_M
    p0, r0, _ = dk.fk_ballpoint(j0, c0)
    trigger = {"schema": "tatbot.draw-pose/1", "frame": dp.SAMPLES_FRAME, "period_s": period, "joints": j0.tolist(),
               "carriage_m": c0, "tip": p0.tolist(), "rotation": r0.tolist(), "tool": "lutin-ballpoint-dot", "t_wall": 0.0}
    orbits = []
    for mode, standoff in [("camera", None)] + [("tip", float(v)) for v in args.standoffs.split(",")]:
        cfg = json.loads(json.dumps(draw_stage.DEFAULT_CONFIG))
        cfg["orbit"]["mode"] = mode
        if standoff is not None:
            cfg["orbit"]["standoff_mm"] = standoff
        if args.tilt is not None:
            cfg["orbit"]["tilt_deg"] = args.tilt
        try:
            samples, rep = dp.orbit_samples(cfg, trigger, period, axis6)
        except dp.DrawRefusal as refusal:
            print(f"\norbit mode {mode}: refused: {refusal.code} ({refusal.detail})")
            continue
        label = (f"camera mode, cameras {rep['camera_distance_mm']:.0f} mm off the patch, {rep['off_axis_deg']:.0f} deg off axis"
                 if mode == "camera" else f"tip mode, standoff {standoff:.0f} mm")
        print(f"\norbit from the log's first pose: {label}, tilt {cfg['orbit']['tilt_deg']:.0f} deg, "
              f"{int(samples.capture.max())} viewpoints, {samples.duration_s:.0f} s")
        views = []
        for row in np.nonzero(samples.capture > 0)[0]:
            s = score_cartesian(surface, pts, normals, samples.p[row], samples.R[row], axis6, c0)
            print(f"  viewpoint {int(samples.capture[row])}: tip {s['gap_mm']:4.0f} mm above, lean {s['lean_deg']:3.0f} | "
                  f"{fmt_cam(s['cams']['wrist_upper'])} | {fmt_cam(s['cams']['wrist_lower'])}")
            views.append(s)
        orbits.append({"mode": mode, "standoff_mm": standoff, "tilt_deg": cfg["orbit"]["tilt_deg"], "viewpoints": views})
    if args.json:
        Path(args.json).write_text(json.dumps({"log": rows, "orbits": orbits}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
