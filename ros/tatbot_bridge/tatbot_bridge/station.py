"""The shared roof-tag observation producer for calibration and drawing; capture only, no arm motion."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tatbot_motion import station

from tatbot_bridge import capture as register

DARK_GRAY = 30.0


class UnmeasuredError(RuntimeError):
    """The station cannot be measured now."""


def observe(repo: Path, arm: str, shots: int, run_dir: Path, registration_path):
    """The fix from `shots` D555 captures of the roof tag, through the arm's adopted registration."""
    register._lib(repo)
    from fiducials import load_inventory
    from fiducials.detector import DetectorConfig, FiducialDetector


    reg_path = str(registration_path) if registration_path is not None else None
    if not reg_path or not Path(reg_path).expanduser().is_file():
        raise UnmeasuredError(f"no adopted registration for the {arm} arm ({reg_path}): "
                              f"tatbot ros register --arm {arm}")
    raw = Path(reg_path).expanduser().read_bytes()
    registration = json.loads(raw)
    if (registration.get('schema') != 'tatbot.arm-registration/1' or registration.get('arm') != arm
            or not registration.get('calibration_id') or type(shots) is not int or shots < 2):
        raise UnmeasuredError('station observation requires this arm\'s identified registration and at least two shots')
    run_dir.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory(repo / "config" / "fiducials.json")
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges",
                                                                        cell_margin=0.30), target="palette")
    palette_urdf = repo / "urdf" / "palette.urdf"
    view = _TagView(inventory.target("palette"), station.palette_from_tag(palette_urdf, require_confirmed=True), registration)
    camera = register.Camera(repo)
    rows = []
    try:
        for i in range(shots):
            shot = camera.capture(time.time_ns())
            if shot["metadata"].get("calibration_id") != registration.get("calibration_id"):
                raise UnmeasuredError(f"the D555's frames are in camera bundle "
                                      f"{shot['metadata'].get('calibration_id')!r} but the {arm} arm is registered in "
                                      f"{registration.get('calibration_id')!r}: tatbot ros register --arm {arm}")
            if i == 0:
                cv2.imwrite(str(run_dir / "overhead.jpg"), shot["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            seen = detector.detect(register.COLOR, shot["image"], shot["stamp_ns"])
            if not any(d.tag_id in view.target.ids for d in seen):
                seen = detector.detect(register.COLOR, _stretched(shot["image"]), shot["stamp_ns"])
            gray = round(float(cv2.cvtColor(shot["image"], cv2.COLOR_BGR2GRAY).mean()), 1)
            rows.append({"shot": i, "gray": gray, **view.solve(shot, seen)})
    finally:
        camera.close()
        with (run_dir / "shots.jsonl").open("w") as out:
            out.writelines(json.dumps(row) + "\n" for row in rows)
    # a shot placed by depth and one by its corners alone differ by the light's error (station.tag_centre_by_depth)
    by_depth = [row for row in rows if row.get("placed_by") == "depth"]
    poses = [row["base_from_palette"] for row in (by_depth or rows) if "base_from_palette" in row]
    if len(poses) < max(2, shots // 2):
        why = sorted({row["why"] for row in rows if "why" in row})
        gray = float(np.median([row["gray"] for row in rows if "gray" in row] or [255.0]))
        dark = (f"; the frames average gray {gray:.0f} of 255, and under {DARK_GRAY:.0f} the tag has not been found: "
                "the room may need a light" if gray < DARK_GRAY else "")
        raise UnmeasuredError(f"the D555 measured the roof tag in {len(poses)} of {shots} shots "
                              f"({'; '.join(why)}){dark}")
    first = dt.datetime.fromtimestamp(rows[0]["stamp_ns"] / 1e9, dt.timezone.utc)
    # the camera's optical centre in the arm base: the post it stands on is the wrist's to keep clear of (reach)
    camera = np.linalg.inv(np.asarray(registration["world_from_arm_base"], float))[:3, 3]
    detail = {"camera": register.COLOR, "calibration_id": registration["calibration_id"], "registration": reg_path,
              "rms_px": max(row.get("rms_px", 0.0) for row in rows), "overhead_camera_m": camera.round(4).tolist(),
              "registration_sha256": hashlib.sha256(raw).hexdigest(),
              "placed_by_depth": sum(row.get("placed_by") == "depth" for row in rows)}
    found = station.from_poses(arm, poses, first.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                               palette_urdf, "overhead_tag", detail)
    (run_dir / 'station.json').write_text(json.dumps(found.as_dict(), indent=1)+'\n')
    return found


def _stretched(image):
    """Stretch grayscale between its 1st and 99.5th percentiles and return BGR for dim-tag detection."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(gray, [1.0, 99.5])
    out = np.clip((gray - lo) * (255.0 / max(float(hi - lo), 1.0)), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


class _TagView:
    """Solve one roof-tag capture into a level palette pose through the adopted optical-frame registration."""

    def __init__(self, target, palette_tag, registration: dict):
        from fiducials import tag_model_corners
        from tatbot_motion.dip import _pose

        self.target, self.palette_tag = target, palette_tag
        self.model = tag_model_corners(target.edge_m)
        self.camera_from_base = _pose(registration["world_from_arm_base"])

    def solve(self, shot: dict, detections) -> dict:


        row = {"stamp_ns": shot["stamp_ns"]}
        seen = [d for d in detections if d.tag_id in self.target.ids]
        if not seen:
            return {**row, "why": "the tag unseen (is the arm over the palette?)"}
        k, dist = register.k_dist(register.intrinsics_of(shot["metadata"]))
        try:
            pose, rms = station.level_pose(seen[0].corners_px, k, dist, self.camera_from_base, self.model,
                                           self.palette_tag)
        except ValueError as error:
            return {**row, "corners": seen[0].corners_px.tolist(), "why": str(error)}
        row.update(corners=seen[0].corners_px.tolist(), rms_px=round(rms, 4), placed_by="corners")
        depth = shot.get("depth_m")
        tag = None if depth is None else station.tag_centre_by_depth(seen[0].corners_px, depth, k, dist)
        if tag is not None:   # the range by depth, not by the tag's apparent size (station.tag_centre_by_depth)
            placed = station.placed_at(pose, self.palette_tag, (np.linalg.inv(self.camera_from_base)
                                                                @ np.append(tag, 1.0))[:3])
            row.update(placed_by="depth", depth_moved_mm=np.round((placed - pose)[:3, 3] * 1000.0, 2).tolist())
            pose = placed
        return {**row, "base_from_palette": pose.tolist()}
