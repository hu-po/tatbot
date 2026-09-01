#!/usr/bin/env python3
"""Render a print-ready sheet of wrist tags the pipeline can actually decode.

    uv run --no-project --with numpy --with opencv-python-headless \
        python scripts/vision/generate_wrist_tags.py

History: the ORIGINAL (pre-2026-08-21) wrist tags decoded in no OpenCV
dictionary — wrong family, no quiet zone — which is why this generator
exists. The current EE uses calibrated 16h5 ids 3/6/7/8.
Every pipeline detector is DICT_APRILTAG_16H5, so any replacement must be too.

The sheet is generated at exactly 300 DPI: the black square prints at 56 mm
(same as the board — verify with calipers, print at 100%/no scaling) inside a
14 mm white quiet zone (>1.5 modules; the detector segments quads by the
white-around-black transition, so mounting flush on a black bracket without
this margin kills detection).

The default sheet reproduces the configured wrist set for replacement only.
Never bring a second copy into the five-camera scene: ids 3/6/7/8 are also on
the calibration board, and id 8 is also on the palette. Remove the old target
before using a replacement and rerun wrist calibration after remounting it.
"""

from pathlib import Path

import cv2
import numpy as np
from fiducials import load_inventory

DPI = 300
MM = DPI / 25.4
INVENTORY = load_inventory()
WRIST = INVENTORY.target("wrist")
TAG_MM = WRIST.edge_m * 1000
MARGIN_MM = 14
IDS = WRIST.ids
OUT = Path(__file__).resolve().parents[2] / "docs" / "wrist-tags-16h5.png"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(str(i) for i in IDS),
                    help="comma-separated 16h5 ids (must be UNUSED in the scene)")
    args = ap.parse_args()
    ids = tuple(int(v) for v in args.ids.split(","))
    allowed = set(WRIST.ids) | set(INVENTORY.spare_ids)
    unexpected = sorted(set(ids) - allowed)
    if unexpected:
        raise SystemExit(
            f"ids {unexpected} are not configured wrist or spare ids; update "
            "config/fiducials.json before printing"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
    tag_px = round(TAG_MM * MM)          # 661 px = 56.0 mm at 300 DPI
    margin_px = round(MARGIN_MM * MM)
    cell = tag_px + 2 * margin_px
    label_px = round(10 * MM)
    sheet = np.full((len(ids) * (cell + label_px) + label_px, cell + 2 * label_px),
                    255, np.uint8)
    for row, tag_id in enumerate(ids):
        # 6 modules across (16h5 = 4x4 data + 1-module black border ring);
        # render at an exact multiple then resize NEAREST to the physical size.
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 6 * 120)
        marker = cv2.resize(marker, (tag_px, tag_px),
                            interpolation=cv2.INTER_NEAREST)
        y = label_px + row * (cell + label_px) + margin_px
        x = label_px + margin_px
        sheet[y:y + tag_px, x:x + tag_px] = marker
        cv2.putText(sheet, f"16h5 id {tag_id} - black square {TAG_MM:.1f} mm - do not scale",
                    (label_px, y + tag_px + margin_px + round(4 * MM)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)
    print(f"wrote {OUT} — print at 300 DPI / 100% scale, caliper-check 56 mm,")
    print("keep the full white margin when cutting (>= 14 mm on every side)")


if __name__ == "__main__":
    main()
