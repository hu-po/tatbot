"""Native Cartesian time laws, CLIK and drawing choreography; no ROS imports.
Frames are <arm>/base_link <- X in metres, with six revolute joints and the carriage in metres.
The loaded TCP's +z points along the tool; drawing holds tcp z = -page z (ros/README.md).
"""
from __future__ import annotations

import math
from pathlib import Path

from tatbot_motion.clik import Kinematics, PlanError
from tatbot_motion.plan import (
    HOVER,
    PHASE_DESCEND,
    PHASE_DRAW,
    PHASE_LIFT,
    PHASE_PAUSED,
    PHASE_SETTLE,
    PHASE_TOUCH,
    PHASE_TRAVEL,
    PRESS,
    RIDE,
    PenDown,
    Trajectory,
    dispatch_drift,
    guard_arm_time_s,
    pen_down,
    plan_hover,
    plan_lift,
    plan_op,
    plan_touch,
    plan_travel,
    probe_travel_m,
    to_knots,
)
from tatbot_motion.trim import Trim, compose

MOTION_FORMAT = "tatbot-motion"
MOTION_VERSION = 1

__all__ = [
    "Kinematics", "PlanError", "Trajectory", "load_motion", "motion_path", "plan_op", "plan_travel", "plan_hover", "plan_lift", "plan_touch",
    "probe_travel_m", "to_knots", "dispatch_drift", "guard_arm_time_s", "PenDown", "pen_down", "PRESS", "RIDE", "HOVER",
    "PHASE_TRAVEL", "PHASE_DESCEND", "PHASE_SETTLE", "PHASE_DRAW", "PHASE_LIFT", "PHASE_TOUCH", "PHASE_PAUSED",
    "Trim", "compose",
]


def motion_path() -> Path:
    """The motion.yaml load_motion reads by default: the source tree's, else the installed share's."""
    path = Path(__file__).resolve().parents[1] / "config" / "motion.yaml"
    if not path.is_file():
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("tatbot_motion")) / "config" / "motion.yaml"
    return path


def load_motion(path: str | Path | None = None) -> dict:
    """motion.yaml (format tatbot-motion, version 1): the source tree's config/motion.yaml by default."""
    import yaml

    path = Path(path) if path else motion_path()
    motion = yaml.safe_load(path.read_text())
    if (motion.get("format"), motion.get("version")) != (MOTION_FORMAT, MOTION_VERSION):
        raise ValueError(f"{path}: not {MOTION_FORMAT} version {MOTION_VERSION}")
    _check_pen(motion["pen"], path)
    return motion


def _check_pen(pen: dict, path) -> None:
    """motion.yaml `pen`: a mode naming one of its blocks, each block's numbers in range."""
    if pen.get("mode") not in (PRESS, RIDE, HOVER):
        raise ValueError(f"{path}: pen.mode must be {PRESS}, {RIDE} or {HOVER}, got {pen.get('mode')!r}")
    lift, fraction, hover = float(pen[PRESS]["lift_m"]), float(pen[RIDE]["fraction"]), float(pen[HOVER]["height_m"])
    if not (math.isfinite(lift) and abs(lift) <= 0.003):
        raise ValueError(f"{path}: pen.press.lift_m must be within +-0.003, got {lift}")
    if not 0.0 < fraction < 2.0:
        # Over 1 is in use: the running ball reaches further than one stroke below the touched page
        # (0.5 and 0.95 pressed it in, operator 2026-10-02), so the touch does not find the top of its stroke.
        raise ValueError(f"{path}: pen.ride.fraction must be in (0, 2), got {fraction}")
    if not 0.0 < hover <= 0.010:
        # under the stroke's 10 mm standoff, over any tip's reach past its touched end
        raise ValueError(f"{path}: pen.hover.height_m must be in (0, 0.010], got {hover}")
    if not all(float(pen[mode]["settle_s"]) >= 0.0 for mode in (PRESS, RIDE, HOVER)):
        raise ValueError(f"{path}: pen.press, pen.ride and pen.hover settle_s must be >= 0")
    trim = pen["trim"]
    if not 0.0 < float(trim["step_m"]) <= float(trim["limit_m"]) or not all(
            float(trim[key]) > 0.0 for key in ("speed_m_s", "min_s", "lead_s")):
        raise ValueError(f"{path}: pen.trim needs 0 < step_m <= limit_m and positive speed_m_s, min_s and lead_s")
