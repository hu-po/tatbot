"""Adopting a probe tip candidate (tatbot_calib.program, or the joint-6 sweep's, tatbot_calib.sweep): the fitted
cartridge's tip, for the record its ROS owner keeps (cartridge_fit).

`tatbot ros calib apply --run ID` calls this on the ROS owner when the operator asks for it. It sets the arm's
pen_tip_offset_x/y to the tip the run fitted across the axis, and its _z to the fitted length when the run measured it
(a sweep's tilts: `fit.length_measured`), carries mechanical_contact_* along with the tip, and names the run in the
touch-off record. The fitted tip (`fit.tip_m`) is the tip in the tool mount itself, not a change to the tip the run
was commanded with (`tip0_m`): a run adopted twice, or after a newer one, gives the same tip. A candidate measured
with another tool than the one fitted now is refused, because its tip belongs to that
tool. A cartridge's first adoption (its tip still nominal: `tool_fit.status: nominal`) writes the touch-off record in
place of that mark, its contact at the nominal tip and its reference and profile the tool datasheet's `calibration`,
before the run moves them. Pure: no ROS import, no file.
"""
from __future__ import annotations

import math

from cartridge_fit import TIP_KEYS

# The touch-off record's note: what the run measured (setup's measured_lines reads it back).
LENGTH_NOTE = "the tip and its length"
AXIS_NOTE = "the tip across its axis only; its length is the installed one"
TOUCHOFF_KEYS = ("utc", "session", "method", "n_plate", "n_pad", "cond", "residual_mm", "holdout_mm", "tip_loo_max_mm",
                 "spread_deg", "note")


def _first_touchoff(tip: dict, calibration: dict) -> dict:
    """A nominal tip with the measured tip's record in place of its `tool_fit` mark."""
    out = {f"pen_tip_offset_{axis}": tip[f"pen_tip_offset_{axis}"] for axis in "xyz"}
    out.update({f"mechanical_contact_{axis}": tip[f"pen_tip_offset_{axis}"] for axis in "xyz"})
    out.update(mechanical_contact_reference=calibration.get("contact_reference"),
               mechanical_contact_profile=calibration.get("accuracy_profile"),
               mechanical_contact_status="fit_passed", mechanical_contact_session=None,
               touchoff=dict.fromkeys(TOUCHOFF_KEYS))
    return out


def _mm(value: float) -> float | None:
    return round(value * 1000, 3) if math.isfinite(value) else None


def apply(candidate: dict, section: dict, calibration: dict | None = None, *, across_only: bool = False
          ) -> tuple[dict, list[str]]:
    """(the arm's tip keys with the candidate's tip adopted, []), or ({}, why not). `section`: the arm's workspace
    section as its owner reads it (tool_spec.read_workspace); `calibration`: the tool datasheet's, for a cartridge's
    first adoption. `across_only` keeps the installed length although a sweep measured one: a fresh cartridge's tube
    reached 2.45 mm farther while its needles, which the machine drives to the same bottom, reached that much less
    past it, and the length alone would have ridden every stroke that shallow (2026-11-10)."""
    if candidate.get("tool") != section.get("tool_id"):
        return {}, [f"the candidate measured {candidate.get('tool')}, but {section.get('tool_id')} is fitted"]
    tip = {key: section[key] for key in TIP_KEYS if key in section}
    if (tip.get("tool_fit") or {}).get("status") == "nominal" and "mechanical_contact_session" not in tip:
        tip = _first_touchoff(tip, calibration or {})
    measured = candidate["fit"]["tip_m"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in measured):
        return {}, ["the candidate's tip is not finite"]
    length = bool(candidate["fit"].get("length_measured")) and not across_only
    for k, axis in enumerate("xyz"[:3 if length else 2]):
        offset, contact = f"pen_tip_offset_{axis}", f"mechanical_contact_{axis}"
        for key in (offset, contact):
            if not isinstance(tip.get(key), (int, float)):
                return {}, [f"the fitted tip has no {key}"]
        # the contact reference keeps its place against the tip (a tube's end short of it, a ball's radius)
        tip[contact] = round(float(tip[contact]) - float(tip[offset]) + measured[k], 6)
        tip[offset] = round(float(measured[k]), 6)
    held, moved = candidate.get("held_out_rms_m") or {}, candidate.get("tip_moved_m") or {}
    tip["mechanical_contact_session"] = candidate["run_id"]
    tip["touchoff"] = {**dict.fromkeys(TOUCHOFF_KEYS), **(tip.get("touchoff") or {}),
                       "utc": candidate["utc"], "session": f"ros-calib/{candidate['run_id']}",
                       "method": candidate.get("method", "probe-station side pairs at upright yaws (tatbot ros calib run)"),
                       "n_plate": candidate["contacts"], "residual_mm": _mm(max(candidate["fit"]["rms_m"].values())),
                       "holdout_mm": _mm(max(held.values(), default=math.nan)),
                       "tip_loo_max_mm": _mm(max(moved.values(), default=math.nan)),
                       "note": LENGTH_NOTE if length else AXIS_NOTE}
    return tip, []
