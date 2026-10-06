"""Offline duration components. Drawing uses the planner's law; no IK or robot I/O."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

from tatbot_motion import load_motion, timelaw
from tatbot_motion.plan import drawing_time_law


def drawing_seconds(op, speed, motion):
    points = np.asarray(op["points_m"], float)
    if op.get("closed") and not np.array_equal(points[0], points[-1]):
        points = np.concatenate([points, points[:1]])
    dt = 1 / motion["control_rate_hz"]
    if timelaw.polyline_length(points) <= 1e-6:
        return max(1, round(2 * motion["draw"]["min_ease_s"] / dt)) * dt
    t, _, _, _ = drawing_time_law(points, min(speed, motion["tip_speed"]["pen_down_max_m_s"]), motion)
    return len(t) * dt  # the solver dispatches one sample per fixed control tick


def _workflow_kind(op):
    if op["op"] == "tool_change" or (op["op"] == "pause" and op.get("reason", "").startswith("swap pen:")):
        return "pen_change"
    return op["op"]


def estimate_duration(ops, speed_m_s, travel_m, motion, *, resources=None):
    """Known subtotal plus explicit unknown costs; total_s is null when workflow costs are absent.

    Optional duration_estimate.<kind>_s values are measured additional workflow costs,
    including dip travel/dwell/return or pen-swap waiting, excluding the stroke lift/descent/settle
    already counted here. No defaults invent these measurements.
    Approach/lift/travel remain rough kinematic estimates; setup, IK and interruptions are excluded.
    """
    motion = motion or load_motion()
    strokes = [op for op in ops if op["op"] == "stroke"]
    touches = sum(not op.get("continues", False) for op in strokes)
    approach = motion["approach"]
    standoff, final = approach["standoff_m"], approach["final_m"]
    table = {row['id']: row for row in resources} if resources is not None else None
    settle = sum(float(motion['pen'][table[op['resource_id']]['pen_mode'] if table is not None else
                                    motion['pen']['mode']]['settle_s'])
                 for op in strokes if not op.get('continues', False))
    components = {
        "drawing": sum(drawing_seconds(op, speed_m_s, motion) for op in strokes),
        "descent": touches * ((standoff - final) / approach["descend_m_s"] + final / approach["final_m_s"]),
        "settle": settle,
        "lift": touches * standoff / approach["lift_m_s"],
        "travel": travel_m / (0.5 * motion["tip_speed"]["pen_up_max_m_s"]),
    }
    unknown = {}
    counts = Counter(_workflow_kind(op) for op in ops if op["op"] != "stroke")
    measured = motion.get("duration_estimate", {})
    for kind, count in sorted(counts.items()):
        value = measured.get(f"{kind}_s")
        if value is None:
            unknown[kind] = count
            continue
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"duration_estimate.{kind}_s must be finite and nonnegative")
        components[kind] = count * float(value)
    total = sum(components.values())
    return {"model": "tatbot-duration/2", "modeled_s": round(total, 4),
            "total_s": None if unknown else round(total, 4),
            "components_s": {k: round(v, 4) for k, v in components.items()},
            "unknown_operations": unknown, "workflow_counts": dict(counts),
            "motion_sha256": hashlib.sha256(json.dumps(motion, sort_keys=True).encode()).hexdigest(),
            "scope": "Cartesian drawing law; approximate descent/lift/travel; excludes setup, IK retiming and interruptions"}
