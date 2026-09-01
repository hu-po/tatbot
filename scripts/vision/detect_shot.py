#!/usr/bin/env python3
"""Reduce one calibration-shot burst to per-camera stills + detections.

Runs on the camera node. Input: a capture-poe-all --decoded evidence dir. For each camera
the LAST complete decoded frame is kept (the burst is of a static scene, so
any sharp frame will do; the last one is past pipeline warm-up), tags are
detected, and the frame is written as PNG next to a detections JSON. The
heavyweight burst directory is then deleted.

  python3 detect_shot.py <burst_dir> <shot_out_dir>

Prints one JSON line: {camera: {"ids": [...], "center_xy": [...]}} for the
session driver to display.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_scan import BOARD_IDS, detect, resolve_duplicates  # noqa: E402


def last_frame(camera_dir: Path):
    with open(camera_dir / "frames.jsonl") as f:
        entries = [json.loads(line) for line in f]
    for entry in reversed(entries):
        meta = entry.get("metadata", entry)
        profile = meta["profile"]
        w, h = profile["width"], profile["height"]
        payload = camera_dir / f"{meta['sequence']:012d}.bgr8"
        if not payload.is_file():
            continue
        data = np.fromfile(payload, dtype=np.uint8)
        if data.size == w * h * 3:
            return data.reshape(h, w, 3)
    return None


def main():
    burst = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for camera_dir in sorted(burst.glob("camera*/")):
        camera = camera_dir.name.rstrip("/")
        frame = last_frame(camera_dir)
        if frame is None:
            report[camera] = {"ids": [], "error": "no complete frame"}
            continue
        # Board, EE, and palette reuse IDs; keyed by id alone one physical
        # target could silently overwrite another.
        resolved, _palette = resolve_duplicates(detect(frame))
        detections = list(resolved.items())
        cv2.imwrite(str(out / f"{camera}.png"), frame)
        board = [(i, c) for i, c in detections if i in BOARD_IDS]
        report[camera] = {
            "ids": sorted(i for i, _ in detections),
            "corners": {str(i): c.tolist() for i, c in detections},
            "center_xy": [
                float(np.mean([c[0] for _, cs in board for c in cs])),
                float(np.mean([c[1] for _, cs in board for c in cs])),
            ]
            if board
            else None,
            "resolution": [frame.shape[1], frame.shape[0]],
        }
    with open(out / "detections.json", "w") as f:
        json.dump(report, f)
    shutil.rmtree(burst)
    print(json.dumps({cam: {"ids": r["ids"], "center_xy": r.get("center_xy")}
                      for cam, r in report.items()}))


if __name__ == "__main__":
    main()
