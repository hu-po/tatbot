"""The wrist D405 as a gauge of the pen's height over the paper (ros/README.md 4.3).

The pen and the wrist camera both ride link 6, so the pen's tip is one fixed point in the camera's frame, and a
depth frame of the paper beside the pen gives the paper's plane there. The tip's height over that plane needs
neither the arm's kinematics nor the camera's mounting: on 2026-10-03 sixteen contacts from three touch runs read
within 0.25 mm (1 sigma) of each other, while the arm's own tip heights for them spread 4 mm with the tool's turn.

The tip is fitted from touch snapshots (touches/<touch>-<trip>.npz and .png, which the executor saves at every
trip): the pen's points pooled over every frame and taken at their end along the paper's normal, then the offset
that puts the lowest cluster of trips, the contacts, at zero (`ros2 run tatbot_session gauge fit RUN... --write`).
It holds while the tool and the camera stay mounted as they are. Once fitted, the pen is found by its place
on the fitted axis rather than its colour, so any ink's cartridge serves (on 2026-10-03's frames of the teal
sky-blue cartridge the two ways' heights agreed within 0.12 mm, 1 sigma). Its end along the axis does not show
a moved pen: without the colour, the depth's mixed pixels between the tip and the paper run on along the axis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB = Path("~/tatbot-ros/calib").expanduser()
PEN_HUE = (78, 100)        # the ballpoint's teal body in OpenCV's 0-180 hue
PEN_NEAR_M = 0.17          # the pen rides 100-150 mm from the camera; anything teal farther is the room
PAPER_BOX_PX = 150         # the paper fitted: within this of the pen in the image, and behind its middle
CONTACT_BAND_M = 0.001     # a fit's contacts: the trips within this of the lowest decile
AXIS_M = 0.003             # a fitted pen's points: within this of its axis (the cone's narrow end) ...
CONE_M = (-0.015, 0.0025)  # ... and this far along it from the fitted cone end (a hover keeps the paper ~6 mm past it)
BODY_M = 0.008             # no paper within this of the axis short of the cone end (the cartridge and machine)
SURFACE_S_MAX = 150        # the surface drawn on is bright and not strongly coloured: white paper and pink silicone
                           # (S ~75) alike; the print's dots and the arm's shadow drop out by value


def calibration_path(arm: str) -> Path:
    return CALIB / f"wrist-gauge-{arm}.json"


@dataclass
class Frame:
    """One depth frame, in the colour camera's frame: the pen's points and the paper's plane (c, n toward the
    camera), with the plane's rms."""
    pen: np.ndarray
    c: np.ndarray
    n: np.ndarray
    rms: float


def _points(depth_raw, meta: dict, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The depth frame's points in the colour camera's frame and the colour pixel (u, v) each falls on."""
    depth = np.asarray(depth_raw, np.float32) * float(meta["depth_units_m"])
    di = meta["intrinsics"]
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    ok = (depth > 0.04) & (depth < 0.40)
    z = depth[ok]
    pts = np.stack([(u[ok] - di["ppx"]) / di["fx"] * z, (v[ok] - di["ppy"]) / di["fy"] * z, z], axis=1)
    ex = meta["color_from_depth"]
    pc = pts @ np.array(ex["rotation"], float).reshape(3, 3, order="F").T + np.array(ex["translation_m"], float)
    k = np.array(meta["color_intrinsics"]["k"], float)
    uc = np.round(k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2]).astype(int)
    vc = np.round(k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]).astype(int)
    inside = (uc >= 0) & (uc < shape[1]) & (vc >= 0) & (vc < shape[0])
    return pc[inside], uc[inside], vc[inside]


def on_paper(depth_raw, meta: dict, frame: Frame, shape, within_m: float = 0.003, close_px: int = 9,
             trim_px: int = 7) -> np.ndarray:
    """The colour pixels whose depth puts them on this frame's paper (within within_m of its plane), the gaps
    between the depth's samples closed and the edges trimmed: the pen, the machine and the arm, whatever their
    colour, and the depth's holes about them are not paper."""
    import cv2

    pc, uc, vc = _points(depth_raw, meta, shape)
    on = np.abs((pc - frame.c) @ frame.n) < within_m
    out = np.zeros(shape[:2], np.uint8)
    out[vc[on], uc[on]] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return cv2.erode(out, np.ones((trim_px, trim_px), np.uint8)).astype(bool)


def measure(depth_raw, color_bgr, meta: dict, cal: dict | None = None) -> Frame | None:
    """The pen and the paper in one frame (WristCamera depth, unaligned, and its colour); None when either is
    missing. The pen is teal (the sky-blue cartridge the fit was made with), or with a fitted `cal` the points on
    its axis by the cone end, whatever their colour; the paper (any surface drawn on) is bright and not strongly
    coloured (the arm's shadow darkens it, the print's dots drop out by value) and off the pen."""
    import cv2

    pc, uc, vc = _points(depth_raw, meta, color_bgr.shape)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    teal = ((hsv[..., 0] >= PEN_HUE[0]) & (hsv[..., 0] <= PEN_HUE[1]) & (hsv[..., 1] > 90)
            & (hsv[..., 2] > 70)).astype(np.uint8)
    seen = hsv[vc, uc].astype(int)
    paper = (seen[:, 1] < SURFACE_S_MAX) & (seen[:, 2] > 45) & ~teal[vc, uc].astype(bool)
    if cal is not None and "cone_end_cam_m" in cal:
        s, off = _along(pc, cal)
        pen = (off < AXIS_M) & (s > CONE_M[0]) & (s < CONE_M[1]) & (pc[:, 2] < PEN_NEAR_M)
        paper &= ~((off < BODY_M) & (s < 0.001))
    else:
        core = cv2.erode(teal, np.ones((7, 7), np.uint8))   # off the body's edges, where depths blend
        pen = core[vc, uc].astype(bool) & (pc[:, 2] < PEN_NEAR_M)
    if pen.sum() < 100:
        return None
    pp = pc[pen]
    box = ((uc > uc[pen].min() - PAPER_BOX_PX) & (uc < uc[pen].max() + PAPER_BOX_PX)
           & (vc > vc[pen].min() - PAPER_BOX_PX) & (vc < vc[pen].max() + PAPER_BOX_PX))
    q = pc[paper & box & (pc[:, 2] > np.median(pp[:, 2]))]
    if len(q) < 300:
        return None
    keep = np.ones(len(q), bool)
    for _ in range(4):   # a plane, refitted to the points within 3 MADs (0.8 mm at least) of it
        c = q[keep].mean(axis=0)
        n = np.linalg.svd(q[keep] - c, full_matrices=False)[2][-1]
        r = (q - c) @ n
        mid = np.median(r[keep])
        keep = np.abs(r - mid) < max(0.0008, 3.0 * np.median(np.abs(r[keep] - mid)))
    if n @ c > 0:
        n = -n
    return Frame(pp, c, n, float(np.std(r[keep])))


def load_snapshot(npz: Path, cal: dict | None = None) -> Frame | None:
    import cv2

    data = np.load(npz)
    return measure(data["depth_raw"], cv2.imread(str(npz.with_suffix(".png"))), json.loads(str(data["meta"])), cal)


def _along(points, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each point's distance along the fitted axis past the fitted cone end (toward the paper), and from the axis."""
    end, a = np.asarray(cal["cone_end_cam_m"], float), np.asarray(cal["axis_cam"], float)
    d = np.asarray(points, float) - end
    s = d @ a
    return s, np.linalg.norm(d - np.outer(s, a), axis=1)


def height(frame: Frame, cal: dict) -> float:
    """The pen's tip over the paper (m): the fitted tip's distance along the paper's normal, less the offset."""
    return float((np.asarray(cal["tip_cam_m"], float) - frame.c) @ frame.n) - float(cal["offset_m"])


def _axis(points) -> tuple[np.ndarray, np.ndarray]:
    """A line through the pen's points (a point on it, its unit direction), refitted to those within 3 mm of it:
    by the narrow tip the depth reads the paper behind it, and those points lie well off the pen."""
    keep = np.ones(len(points), bool)
    for _ in range(6):
        m = points[keep].mean(axis=0)
        d = np.linalg.svd(points[keep] - m, full_matrices=False)[2][0]
        off = np.linalg.norm((points - m) - np.outer((points - m) @ d, d), axis=1)
        keep = off < max(0.003, 2.5 * float(np.median(off[keep])))
    return points[keep].mean(axis=0), d


def fit(frames: list[Frame]) -> tuple[dict, np.ndarray]:
    """The pen's contact point in the camera frame, and every frame's height over its paper. The pen's pooled
    points give its axis (_axis) and the cone's end on it; the contact lies on the axis past that end by the
    length that puts the contacts at zero, the contacts being the frames within CONTACT_BAND_M of the lowest
    decile (the floor a trip cannot go under). The contact is a true point on the pen (the close-up anchors the
    camera's pose on it): the lowest pooled points the first fit took were paper read through the tip, ~15 mm
    deeper along its line of sight, which kept heights right and put the camera ~10 mm off sideways."""
    pool = np.concatenate([f.pen for f in frames])
    n0 = np.mean([f.n for f in frames], axis=0)
    n0 /= np.linalg.norm(n0)
    m, a = _axis(pool)
    if a @ n0 > 0:   # toward the paper, away from the camera's side of it
        a = -a
    s = (pool - m) @ a
    near = np.linalg.norm((pool - m) - np.outer(s, a), axis=1) < 0.003
    end = m + float(np.percentile(s[near], 99.5)) * a
    # each frame's distance from the cone's end to its paper along the axis; the contacts' is the tip's length
    reach = np.array([float((f.c - end) @ f.n) / float(a @ f.n) for f in frames])
    contacts = reach[reach <= np.percentile(reach, 10) + CONTACT_BAND_M]
    length = float(np.median(contacts))
    contact = end + length * a
    heights = np.array([float((contact - f.c) @ f.n) for f in frames])
    cal = {"tip_cam_m": contact.tolist(), "offset_m": 0.0, "cone_end_cam_m": end.tolist(), "axis_cam": a.tolist(),
           "tip_length_m": length, "normal_cam": n0.tolist(), "contacts": int(len(contacts)),
           "contact_sd_m": float(np.std(heights[reach <= np.percentile(reach, 10) + CONTACT_BAND_M])),
           "frames": len(frames)}
    return cal, heights


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_session gauge", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fit", help="fit the pen's tip in the wrist camera's frame from touch runs' snapshots")
    f.add_argument("runs", nargs="+", help="ros-touch run directories whose touches/ hold trip snapshots")
    f.add_argument("--arm", default="right")
    f.add_argument("--tool", default="", help="the fitted tool, recorded with the fit")
    f.add_argument("--seed", default="", help="a previous fit (a wrist-gauge json) whose axis finds the pen whatever "
                   "its colour: another cartridge in the same mount; without it the pen must be the teal cartridge")
    f.add_argument("--write", action="store_true", help=f"adopt: write {CALIB}/wrist-gauge-<arm>.json")
    args = parser.parse_args(argv)
    names, frames = [], []
    seed = json.loads(Path(args.seed).expanduser().read_text()) if args.seed else None
    for run in map(Path, args.runs):
        for npz in sorted(run.glob("touches/*.npz"), key=lambda p: [int(x) for x in p.stem.split("-")]):
            frame = load_snapshot(npz, seed)
            if frame is not None:
                names.append(f"{run.name} {npz.stem}")
                frames.append(frame)
    if len(frames) < 4:
        print(json.dumps({"ok": False, "message": f"{len(frames)} usable snapshots; a fit needs 4"}))
        return 1
    cal, heights = fit(frames)
    for name, frame, h in zip(names, frames, heights, strict=True):
        print(f"  {name}: {h * 1e3:+6.2f} mm over the paper (plane rms {frame.rms * 1e3:.2f} mm)")
    cal.update(arm=args.arm, tool=args.tool, runs=[Path(r).name for r in args.runs],
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps({"ok": True, **cal}))
    if args.write:
        CALIB.mkdir(parents=True, exist_ok=True)
        calibration_path(args.arm).write_text(json.dumps(cal, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
