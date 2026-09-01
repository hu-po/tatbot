#!/usr/bin/env python3
"""Replay flight CSV telemetry through the measured-motion watchdog.

This is hardware-free: it opens only recorded CSV files and never imports a
robot, camera, or controller driver.  It is intended for incident regression
and threshold audits, not as a substitute for an operator-observed acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

MOTION_SAFETY_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "lerobot_robot_tatbot"
    / "src"
    / "lerobot_robot_tatbot"
    / "motion_safety.py"
)
SPEC = importlib.util.spec_from_file_location("tatbot_motion_safety", MOTION_SAFETY_SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load measured-motion watchdog from {MOTION_SAFETY_SOURCE}")
MOTION_SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTION_SAFETY)
MotionSafetyError = MOTION_SAFETY.MotionSafetyError
MotionSafetyWatchdog = MOTION_SAFETY.MotionSafetyWatchdog


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(path: Path) -> dict:
    guard = MotionSafetyWatchdog(
        velocity_limit=2.5,
        acceleration_limit=80.0,
        reversal_window_s=1.0,
        reversal_min_velocity=0.2,
        reversal_limit=4,
        clamp_grace_s=2.5,
        clamp_window_s=1.0,
        clamp_fraction=0.8,
        clamp_min_samples=20,
        overforce_limit=9.0,
        overforce_window_s=0.5,
        overforce_fraction=0.5,
        overforce_min_samples=8,
    )
    samples = 0
    first_t = None
    last_t = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            now = float(row["t_mono"])
            last_t = now
            if first_t is None:
                first_t = now
                guard.reset(now)
            velocity = [float(row[f"vel_joint_{joint}"]) for joint in range(6)]
            effort = [float(row[f"eff_joint_{joint}"]) for joint in range(6)]
            clamped = any(
                abs(float(row[f"raw_joint_{joint}"]) - float(row[f"pos_joint_{joint}"]))
                > 0.5
                for joint in range(6)
            )
            samples += 1
            try:
                guard.update(
                    now=now,
                    velocities=velocity,
                    external_efforts=effort,
                    clamped=clamped,
                )
            except MotionSafetyError as error:
                return {
                    "path": str(path),
                    "sha256": sha256(path),
                    "samples_read": samples,
                    "elapsed_s": now - first_t,
                    "verdict": "abort",
                    "code": error.code,
                    "message": str(error),
                    "metrics": error.metrics,
                }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "samples_read": samples,
        "elapsed_s": 0.0 if first_t is None else last_t - first_t,
        "verdict": "no_abort",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flight_csv", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "kind": "hardware-free measured-motion safety replay",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [replay(path) for path in args.flight_csv],
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
