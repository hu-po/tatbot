"""`ros2 run tatbot_bridge page_watch [--seconds 30] [--arm right] [--json]`: the stencil page as the
bridge would see it, without ROS: subscribe (read only) to tatbot/tracking/target/<pattern_id> on the
tatbot bus for a while and print each sample's page pose in world and in <arm>/base_link through the
arm's adopted registration. Bench runbook step 6 ("the stencil pose appears in right/base_link").
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np

from tatbot_bridge import page, stack


def finite(value, digits=3):
    return round(float(value), digits) if np.isfinite(value) else None


def describe(sample: dict, world_from_arm_base, now_ns: int) -> dict:
    row = {"source": sample["source"], "seq": sample["seq"],
           "age_ms": round((now_ns - sample["stamp_ns"]) / 1e6, 1) if sample["stamp_ns"] else None,
           "sigma_mm": finite(sample["translation_sigma_m"] * 1000), "sigma_deg": finite(np.degrees(sample["rotation_sigma_rad"])),
           "print_id": sample["print_id"], "identity_verified": sample["identity_verified"],
           "calibration_id": sample["calibration_id"]}
    if sample["world_from_target"] is not None:
        wfp = page.world_from_page(sample["world_from_target"])
        row["world_xyz_mm"] = np.round(wfp[:3, 3] * 1000, 2).tolist()
        row["world_z_axis"] = np.round(wfp[:3, 2], 4).tolist()
        if world_from_arm_base is not None:
            bfp = page.base_from_page(world_from_arm_base, sample["world_from_target"])
            row["base_xyz_mm"] = np.round(bfp[:3, 3] * 1000, 2).tolist()
            row["base_rpy_deg"] = np.round(page.rpy_deg(bfp[:3, :3]), 2).tolist()
            row["base_z_axis"] = np.round(bfp[:3, 2], 4).tolist()
            row["base_from_page"] = np.round(bfp, 6).tolist()
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="page_watch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--arm", default="right")
    ap.add_argument("--config", default="", help="stack.yaml (default: the installed one)")
    ap.add_argument("--pattern-id", default="", help="default: stack.yaml page.pattern_id")
    ap.add_argument("--registration", default="", help="arm-registration-<arm>-current.json (default: stack.yaml registration.<arm>)")
    ap.add_argument("--endpoint", default="", help="default: the bus router's lan endpoint (config/nodes.json)")
    ap.add_argument("--json", action="store_true", help="one JSON line per sample, then a summary line")
    a = ap.parse_args(argv)
    cfg = stack.load(a.config)
    pattern_id = a.pattern_id or cfg["page"]["pattern_id"]
    if a.registration:
        cfg.setdefault("registration", {})[a.arm] = a.registration
    wfb = stack.world_from_arm_base(cfg, a.arm)
    reg_path = stack.registration_path(cfg, a.arm)
    reg_calibration = json.loads(reg_path.read_text()).get("calibration_id") if wfb is not None else None
    endpoint = a.endpoint or stack.bus_endpoint()
    print(f"# {page.topic(pattern_id)} on {endpoint} for {a.seconds:g} s (read only); registration "
          f"{reg_path if wfb is not None else 'absent'}", file=sys.stderr)
    rows, lock = [], threading.Lock()

    def on_sample(zsample):
        sample = page.parse(zsample.payload.to_bytes(), pattern_id)
        if sample is not None:
            row = describe(sample, wfb, time.time_ns())
            with lock:
                rows.append(row)
            print(json.dumps(row) if a.json else
                  f"{row['source']:8s} seq {row['seq']} age {row['age_ms']} ms sigma {row['sigma_mm']} mm "
                  f"world {row.get('world_xyz_mm')} base {row.get('base_xyz_mm')} rpy {row.get('base_rpy_deg')}", flush=True)

    bus = stack.open_bus(endpoint)
    sub = bus.declare_subscriber(page.topic(pattern_id), on_sample)
    try:
        time.sleep(a.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        sub.undeclare()
        bus.close()
    with lock:
        measured = [r for r in rows if r["source"] == "measured"]
        summary = {"pattern_id": pattern_id, "endpoint": endpoint, "seconds": a.seconds, "samples": len(rows),
                   "measured": len(measured), "lost": len(rows) - len(measured),
                   "registration": str(reg_path) if wfb is not None else None,
                   "registration_calibration_id": reg_calibration,
                   "same_camera_bundle": (rows[-1]["calibration_id"] == reg_calibration) if rows and reg_calibration else None,
                   "last": measured[-1] if measured else (rows[-1] if rows else None)}
    print(json.dumps({"summary": summary}) if a.json else "summary " + json.dumps(summary, indent=1))
    return 0 if measured else 1


if __name__ == "__main__":
    sys.exit(main())
