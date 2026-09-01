#!/usr/bin/env python3
"""Independently verify a board-session calibration (runs on the camera node).

The solve minimizes reprojection per camera; that says nothing about whether
the cameras agree with EACH OTHER in the world. This does: for every shot seen
by two or more cameras, each camera independently estimates the board's pose
and it is mapped into the world frame. If the extrinsics are right, all
cameras land on the same board pose, so the spread across cameras is a direct,
physical measure of cross-camera accuracy — in millimetres, not pixels.

  python3 verify_calibration.py <session_dir>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory  # noqa: E402
from fiducials.board import rigid_board_instance_ids  # noqa: E402
from fuse_session import filter_shot_dirs  # noqa: E402

_INVENTORY = load_inventory()
# This verifier is run on the same isolated board-phase stills as the solver.
# IDs 3/6/7/8 are reused by the wrist/palette elsewhere, but within these
# windows they are the physical board instances and must participate in the
# independent cross-camera check.
BOARD_IDS = list(_INVENTORY.target("board").ids)
BOARD_ROOT = _INVENTORY.target("board").calibration_root_id
MAX_BOARD_INSTANCE_ERROR_M = 0.025


def board_instance_ids(tags, layout, intrinsics, dist):
    return rigid_board_instance_ids(
        tags,
        layout,
        intrinsics,
        dist,
        board_ids=BOARD_IDS,
        root_id=BOARD_ROOT,
        max_error_m=MAX_BOARD_INSTANCE_ERROR_M,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--calibration", default=None,
                    help="bundle to test against this session's board evidence")
    args = ap.parse_args()
    session = Path(args.session_dir).expanduser()
    calibration = (Path(args.calibration).expanduser() if args.calibration
                   else session / "calibration.json")
    bundle = json.loads(calibration.read_text())
    layout = {int(i): np.asarray(c, np.float64)
              for i, c in json.loads((session / "board_layout.json").read_text()).items()}

    cameras = {}
    for name, cal in bundle["cameras"].items():
        k = cal["intrinsics"]
        world_from_cam = np.eye(4)
        world_from_cam[:3, :3] = np.asarray(cal["world_from_camera"]["rotation"]).reshape(3, 3)
        world_from_cam[:3, 3] = cal["world_from_camera"]["translation_m"]
        cameras[name] = {
            "K": np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]),
            "d": np.asarray(cal["distortion"]["coefficients"], np.float64),
            "world_from_cam": world_from_cam,
        }

    per_shot, all_err, per_camera = [], [], {}
    shot_dirs = sorted(session.glob("shot_*/"))
    windows_path = session / "board_windows.json"
    if windows_path.is_file():
        windows = json.loads(windows_path.read_text())
        shot_dirs = filter_shot_dirs(shot_dirs, None, None, windows=windows)
        print(f"using {len(shot_dirs)} guided board-still shots")
    for shot_dir in shot_dirs:
        if "anchor" in shot_dir.name:
            continue
        detections = json.loads((shot_dir / "detections.json").read_text())
        centers = {}
        for name, info in detections.items():
            if name not in cameras:
                continue
            tags = {int(i): np.asarray(c, np.float64)
                    for i, c in info.get("corners", {}).items()}
            seen = board_instance_ids(
                tags, layout, cameras[name]["K"], cameras[name]["d"]
            )
            if len(seen) < 2:
                continue
            obj = np.concatenate([layout[i] for i in seen])
            img = np.concatenate([tags[i] for i in seen])
            ok, rvec, tvec = cv2.solvePnP(obj, img, cameras[name]["K"], cameras[name]["d"],
                                          flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            cam_from_board = np.eye(4)
            cam_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
            cam_from_board[:3, 3] = tvec.reshape(3)
            centers[name] = (cameras[name]["world_from_cam"] @ cam_from_board)[:3, 3]
        if len(centers) >= 2:
            names = list(centers)
            points = np.stack([centers[n] for n in names])
            spread = np.linalg.norm(points - points.mean(axis=0), axis=1) * 1000
            per_shot.append((shot_dir.name, len(centers), float(spread.max())))
            all_err.extend(spread.tolist())
            for name, value in zip(names, spread, strict=True):
                per_camera.setdefault(name, []).append(float(value))

    print(f"{len(per_shot)} multi-camera shots\n")
    print("shot                          cams   max disagreement")
    for name, n, worst in per_shot:
        print(f"  {name:<28} {n}    {worst:6.1f} mm")
    if per_camera:
        print("\nper-camera deviation from the multi-camera consensus:")
        for name in sorted(per_camera):
            values = np.array(per_camera[name])
            print(f"  {name}: median {np.median(values):5.1f} mm   "
                  f"p95 {np.percentile(values, 95):5.1f} mm   n={len(values)}")
    if all_err:
        errs = np.array(all_err)
        print(f"\ncross-camera board-position disagreement over {len(errs)} camera-views:")
        print(f"  median {np.median(errs):.1f} mm   p95 {np.percentile(errs, 95):.1f} mm   "
              f"max {errs.max():.1f} mm")


if __name__ == "__main__":
    main()
