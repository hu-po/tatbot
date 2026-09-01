#!/usr/bin/env python3
"""Estimate provisional PoE camera extrinsics from the palette tag (16h5 id 8).

Samples multiple frames per camera from the RTSP main streams, detects the
palette tag, and solves IPPE_SQUARE PnP on the per-corner median across
samples. The world frame is the palette tag: origin at the tag center, X/Y in
the tag plane, Z out of the tag face. Output is a DRAFT visiond
CalibrationBundle (bundle_id empty) plus a stats report; finalize with:

  tatbot-visiond finalize-calibration <draft.json>

Intrinsics are NOMINAL (2.8 mm / 1/2.7" pinhole, zero distortion) and each
camera is tagged provisional in bundle metadata. Cameras that never see the
tag (camera1) are reported and omitted.

  python3 estimate_extrinsics.py --outdir ~/tatbot-logs/vision/extrinsics \
      [--cameras camera2,camera3,camera4,camera5] [--samples 15] [--tag-id 8]
      [--tag-size-m 0.041]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from tag_scan import detect, nominal_intrinsics, rtsp_url  # noqa: E402

_PALETTE = load_inventory().target("palette")


def sample_corners(camera: str, tag_id: int, samples: int, timeout_s: float):
    cap = cv2.VideoCapture(rtsp_url(camera, "main"), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None, []
    got, deadline = [], time.monotonic() + timeout_s
    resolution = None
    while len(got) < samples and time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        resolution = (frame.shape[1], frame.shape[0])
        for det_id, corners in detect(frame):
            if det_id == tag_id:
                got.append(corners)
    cap.release()
    return resolution, got


def solve(camera: str, resolution, corner_sets, tag_size_m: float):
    corners = np.median(np.stack(corner_sets), axis=0)
    spread = float(np.std(np.stack(corner_sets), axis=0).mean())
    w, h = resolution
    intrinsics = nominal_intrinsics(w, h)
    obj = tag_model_corners(tag_size_m)
    # IPPE on a planar square has a well-known two-fold ambiguity (mirrored
    # about the tag plane). Physical prior: every camera looks DOWN at the
    # palette, so its position must be on the +Z side of the tag. Take both
    # solutions, keep the physically valid one, tie-break on reprojection.
    count, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, corners.astype(np.float64), intrinsics, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if count == 0:
        return None
    candidates = []
    for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errs).flatten(), strict=False):
        rot, _ = cv2.Rodrigues(rvec)
        above = (-rot.T @ tvec.reshape(3))[2] > 0.0
        candidates.append((float(err), above, rvec, tvec))
    valid = [c for c in candidates if c[1]] or candidates
    err_best, _, rvec, tvec = min(valid, key=lambda c: c[0])
    errors = sorted(c[0] for c in candidates)
    ambiguity = errors[0] / errors[1] if len(errors) > 1 and errors[1] > 0 else 0.0
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics, None)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners) ** 2, axis=1))))
    rot_cam_from_world, _ = cv2.Rodrigues(rvec)          # tag(world) -> camera
    t_cam_from_world = tvec.reshape(3)
    rot_world_from_cam = rot_cam_from_world.T                           # camera -> world
    t_world_from_cam = -rot_cam_from_world.T @ t_cam_from_world
    return {
        "pose_ambiguity_ratio": ambiguity,
        "camera": camera,
        "resolution": [w, h],
        "samples": len(corner_sets),
        "corner_spread_px": spread,
        "reproj_rmse_px": rmse,
        "range_m": float(np.linalg.norm(t_cam_from_world)),
        "intrinsics": intrinsics.tolist(),
        "world_from_camera": {
            "rotation": rot_world_from_cam.flatten().tolist(),
            "translation_m": t_world_from_cam.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="camera1,camera2,camera3,camera4,camera5")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--tag-id", type=int, default=_PALETTE.ids[0])
    ap.add_argument("--tag-size-m", type=float, default=_PALETTE.edge_m)
    ap.add_argument("--fps", type=int, default=20, help="configured main-stream fps")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(os.path.expanduser(args.outdir)) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    results, missing = [], []
    for cam in args.cameras.split(","):
        cam = cam.strip()
        resolution, corner_sets = sample_corners(cam, args.tag_id, args.samples, args.timeout_s)
        if not corner_sets:
            missing.append(cam)
            print(f"{cam}: tag {args.tag_id} not seen", flush=True)
            continue
        r = solve(cam, resolution, corner_sets, args.tag_size_m)
        if r is None:
            missing.append(cam)
            continue
        results.append(r)
        print(f"{cam}: {r['samples']} samples, spread {r['corner_spread_px']:.2f}px, "
              f"rmse {r['reproj_rmse_px']:.2f}px, range {r['range_m']:.3f}m", flush=True)

    if not results:
        sys.exit("no camera saw the tag; no bundle written")

    cameras = {}
    for r in results:
        w, h = r["resolution"]
        intrinsics = np.array(r["intrinsics"])
        cameras[r["camera"]] = {
            "sensor_name": r["camera"],
            "profile": {"stream": "main", "width": w, "height": h,
                        "fps_num": args.fps, "fps_den": 1, "format": "h264"},
            "intrinsics": {"width": w, "height": h, "fx": intrinsics[0, 0], "fy": intrinsics[1, 1],
                           "cx": intrinsics[0, 2], "cy": intrinsics[1, 2]},
            "distortion": {"model": "brown_conrady", "coefficients": [0.0] * 5},
            "world_from_camera": r["world_from_camera"],
            "depth_to_color": None,
            "metadata": {
                "provisional": "nominal_intrinsics_single_tag_pnp",
                "tag_id": str(args.tag_id),
                "tag_size_m": str(args.tag_size_m),
                "samples": str(r["samples"]),
                "corner_spread_px": f"{r['corner_spread_px']:.3f}",
                "reproj_rmse_px": f"{r['reproj_rmse_px']:.3f}",
                "captured_at": ts,
            },
        }
    bundle = {
        "schema_version": 1,
        "bundle_id": "",
        "world_frame": f"palette_tag{args.tag_id}",
        "cameras": cameras,
    }
    draft = outdir / "calibration_draft.json"
    with open(draft, "w") as f:
        json.dump(bundle, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump({"timestamp": ts, "results": results, "missing": missing}, f, indent=2)
    print(f"draft bundle: {draft}")
    print("finalize with: tatbot-visiond finalize-calibration", draft)


if __name__ == "__main__":
    main()
