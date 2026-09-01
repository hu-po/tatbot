#!/usr/bin/env python3
"""Scan tatbot cameras for AprilTag 16h5 markers (OpenCV aruco).

Grabs fresh frames from the Amcrest RTSP streams (credentials from
~/.config/tatbot/cameras.env), detects DICT_APRILTAG_16H5, and writes per
camera: the raw frame (PNG), an annotated JPEG, and a JSON report with corner
coordinates and (optionally) IPPE_SQUARE pose estimates under one or more tag
size hypotheses using provisional intrinsics.

Intrinsics are NOMINAL (2.8 mm lens on a 1/2.7" sensor) until a real
calibration exists; every pose is flagged provisional. 16h5 is a small
dictionary with a known false-positive rate, so detections are restricted to
the current printed inventory and corner-refined.

Usage (on any node with LAN access to the cameras):
  python3 tag_scan.py --outdir ~/tatbot-logs/vision/tag-scan \
      [--cameras camera1,camera2,...] [--stream main|sub] [--frames 5]
  python3 tag_scan.py --image path.png --outdir ...   # detect on a saved file
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Amcrest RTSP over UDP drops packets under load; force TCP before cv2 opens
# FFmpeg so sampled frames are never half-decoded.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fiducials import (  # noqa: E402
    load_inventory,
    tag_model_corners,
)
from fiducials.detector import DetectorConfig, FiducialDetector  # noqa: E402

INVENTORY = load_inventory()
WRIST_TARGET = INVENTORY.target("wrist")
BOARD_TARGET = INVENTORY.target("board")
PALETTE_TARGET = INVENTORY.target("palette")
PRINTED_IDS = set(INVENTORY.printed_ids)
BOARD_IDS = set(BOARD_TARGET.ids)
PALETTE_IDS = set(PALETTE_TARGET.ids)
WRIST_IDS = set(WRIST_TARGET.ids)
# These IDs occur only on the board and provide geometric context when a
# reused board/EE/palette ID appears more than once in one image.
BOARD_UNIQUE_IDS = BOARD_IDS - WRIST_IDS - PALETTE_IDS
SIZE_HYPOTHESES_M = {
    f"{target.name}_{target.edge_m * 1000:g}mm": target.edge_m
    for target in (WRIST_TARGET, BOARD_TARGET, PALETTE_TARGET)
}
BOARD_TAG_SIZE_M = BOARD_TARGET.edge_m
PALETTE_TAG_SIZE_M = PALETTE_TARGET.edge_m
KNOWN_SIZES_M = {
    tag_id: sizes[0]
    for tag_id in PRINTED_IDS
    if len(sizes := INVENTORY.size_hypotheses(tag_id)) == 1
}
_CALIBRATION_PROFILE = INVENTORY.detector_profiles["calibration"]
_DETECTOR = FiducialDetector(
    PRINTED_IDS,
    DetectorConfig.from_profile(_CALIBRATION_PROFILE),
    family=INVENTORY.family,
)


def _camera_address(n: int) -> str:
    """Address of PoE camera `n` from the visiond sensor registry."""
    import tomllib
    reg = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "rust/visiond/config/vision.toml").read_text())
    for cam in reg.get("cameras", {}).get("poe", []):
        if cam.get("name") == f"camera{n}":
            return str(cam["address"])
    raise SystemExit(f"sensor registry has no camera{n}")


def rtsp_url(camera: str, stream: str) -> str:
    n = camera.removeprefix("camera")
    pw = os.environ.get(f"TATBOT_CAMERA_PASSWORD_CAMERA{n}")
    if not pw:
        sys.exit(f"missing TATBOT_CAMERA_PASSWORD_CAMERA{n} (source cameras.env)")
    subtype = 0 if stream == "main" else 1
    return (f"rtsp://admin:{pw}@{_camera_address(n)}:554"
            f"/cam/realmonitor?channel=1&subtype={subtype}")


def nominal_intrinsics(w: int, h: int) -> np.ndarray:
    # 2.8 mm lens, 1/2.7" sensor (~5.37 mm active width) -> f ~= w * 2.8/5.37.
    f = w * 2.8 / 5.37
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def grab_frame(camera: str, stream: str, warmup: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(rtsp_url(camera, stream), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    frame = None
    # Drain a few frames so we return a fresh, fully-received image.
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def detect(frame: np.ndarray):
    return [(item.tag_id, item.corners_px) for item in _DETECTOR.detect("scan", frame, 0)]


def detection_candidates(detections):
    """Group every decoded square without collapsing physical copies by id."""
    by_id = {}
    for tag_id, corners in detections:
        by_id.setdefault(tag_id, []).append(corners)
    return by_id


def resolve_candidate_groups(by_id):
    """Resolve repeated physical IDs using board-only tags as context.

    The EE reuses board ids 3/6/7/8 and the palette also uses id 8. Keying by
    id alone would silently swap physical objects. The board copy is nearest
    its unique 4/5/9/10/11 context. A second id-8 candidate is retained as
    legacy palette metadata; other non-board duplicates are dropped.

    This is only a duplicate guard. A lone reused ID cannot be assigned to a
    physical target from pixels, so calibration and tracking still require
    phase/scene separation.
    """
    siblings = [
        c[0].mean(axis=0) for i, c in by_id.items() if i in BOARD_UNIQUE_IDS and len(c) == 1
    ]
    board, palette = {}, None
    for tag_id, candidates in by_id.items():
        if len(candidates) == 1:
            board[tag_id] = candidates[0]
            continue
        if siblings and tag_id in BOARD_IDS:
            centroid = np.mean(siblings, axis=0)
            order = sorted(candidates,
                           key=lambda c: np.linalg.norm(c.mean(axis=0) - centroid))
            board[tag_id] = order[0]
            if tag_id in PALETTE_IDS:
                palette = order[-1]
        else:
            # No siblings to judge against: keep none rather than guess wrong.
            continue
    palette_id = next(iter(PALETTE_IDS))
    if palette_id in board and siblings and palette is None:
        centroid = np.mean(siblings, axis=0)
        # A lone palette-id candidate far from the board context is not board.
        if np.linalg.norm(board[palette_id].mean(axis=0) - centroid) > 400:
            palette = board.pop(palette_id)
    return board, palette


def resolve_duplicates(detections):
    """Compatibility wrapper for callers that only need one resolved view."""
    return resolve_candidate_groups(detection_candidates(detections))


def pose_report(corners: np.ndarray, size_m: float, intrinsics: np.ndarray):
    obj = tag_model_corners(size_m)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), intrinsics, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics, None)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners) ** 2, axis=1))))
    return {
        "rvec": rvec.flatten().tolist(),
        "tvec_m": tvec.flatten().tolist(),
        "range_m": float(np.linalg.norm(tvec)),
        "reproj_rmse_px": err,
    }


def scan_camera(camera: str, stream: str, frames: int, outdir: Path) -> dict:
    frame = grab_frame(camera, stream, frames)
    result = {"camera": camera, "stream": stream, "ok": frame is not None,
              "detections": [], "intrinsics": "nominal_2.8mm_1/2.7in"}
    if frame is None:
        return result
    h, w = frame.shape[:2]
    result["resolution"] = [w, h]
    intrinsics = nominal_intrinsics(w, h)
    result["intrinsics_nominal"] = intrinsics.tolist()
    cv2.imwrite(str(outdir / f"{camera}_{stream}.png"), frame)
    annotated = frame.copy()
    for tag_id, corners in detect(frame):
        det = {"id": tag_id, "corners_px": corners.tolist(),
               "side_px": float(np.mean([np.linalg.norm(corners[k] - corners[(k + 1) % 4])
                                         for k in range(4)])),
               "known_size_m": KNOWN_SIZES_M.get(tag_id),
               "size_hypotheses_m": list(INVENTORY.size_hypotheses(tag_id)),
               "poses_provisional": {}}
        for name, size in SIZE_HYPOTHESES_M.items():
            p = pose_report(corners, size, intrinsics)
            if p:
                det["poses_provisional"][name] = p
        result["detections"].append(det)
        cv2.aruco.drawDetectedMarkers(annotated, [corners.reshape(1, 4, 2).astype(np.float32)],
                                      np.array([[tag_id]]))
        c = corners.mean(axis=0).astype(int)
        cv2.putText(annotated, f"id{tag_id}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 4)
    cv2.imwrite(str(outdir / f"{camera}_{stream}_annotated.jpg"), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="camera1,camera2,camera3,camera4,camera5")
    ap.add_argument("--stream", default="main", choices=["main", "sub"])
    ap.add_argument("--frames", type=int, default=5, help="frames to drain before using one")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image", help="detect on an existing image instead of RTSP")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(os.path.expanduser(args.outdir)) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    reports = []
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"cannot read {args.image}")
        name = Path(args.image).stem
        h, w = frame.shape[:2]
        intrinsics = nominal_intrinsics(w, h)
        r = {"camera": name, "ok": True, "resolution": [w, h], "detections": []}
        annotated = frame.copy()
        for tag_id, corners in detect(frame):
            det = {"id": tag_id, "corners_px": corners.tolist(),
                   "known_size_m": KNOWN_SIZES_M.get(tag_id),
                   "size_hypotheses_m": list(INVENTORY.size_hypotheses(tag_id)),
                   "poses_provisional": {}}
            for hname, size in SIZE_HYPOTHESES_M.items():
                p = pose_report(corners, size, intrinsics)
                if p:
                    det["poses_provisional"][hname] = p
            r["detections"].append(det)
            cv2.aruco.drawDetectedMarkers(annotated, [corners.reshape(1, 4, 2).astype(np.float32)],
                                          np.array([[tag_id]]))
        cv2.imwrite(str(outdir / f"{name}_annotated.jpg"), annotated)
        reports.append(r)
    else:
        for cam in args.cameras.split(","):
            r = scan_camera(cam.strip(), args.stream, args.frames, outdir)
            ids = [d["id"] for d in r["detections"]]
            print(f"{cam}: {'OK' if r['ok'] else 'NO FRAME'} tags={ids}", flush=True)
            reports.append(r)

    with open(outdir / "report.json", "w") as f:
        json.dump({"timestamp": ts, "reports": reports}, f, indent=2)
    print(outdir)


if __name__ == "__main__":
    main()
