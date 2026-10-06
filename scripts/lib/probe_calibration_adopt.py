"""Adopting a probe tip candidate (tatbot_calib.program, or the joint-6 sweep's, tatbot_calib.sweep): the arm's tip
across its axis into config/workspace.yaml.

`tatbot ros calib apply --run ID` calls this when the operator asks for it. It moves the arm's pen_tip_offset_x/y
and mechanical_contact_x/y by the fitted change across the axis (their lengths stay as they are) and names the run
in the touch-off record; every other line stays as it was. A candidate measured with another tool than the one
fitted now is refused, because its tip belongs to that tool. Pure text: no ROS import.
"""
from __future__ import annotations

import math
import re


def _section(text: str, arm: str) -> tuple[int, int]:
    start = re.search(rf"^{re.escape(arm)}:\s*$", text, re.MULTILINE)
    if start is None:
        raise ValueError(f"workspace.yaml has no {arm}: section")
    following = re.search(r"^\S", text[start.end():], re.MULTILINE)
    return start.start(), start.end() + following.start() if following else len(text)


def _pattern(key: str):
    return re.compile(rf"^(\s+{re.escape(key)}:)[ \t]*([^\n#]*)", re.MULTILINE)


def _get(section: str, key: str) -> str:
    found = _pattern(key).search(section)
    if found is None:
        raise ValueError(f"workspace.yaml's section has no {key}")
    return found.group(2).strip().strip('"')


def _set(section: str, key: str, value: str) -> str:
    _get(section, key)
    return _pattern(key).sub(lambda m: f"{m.group(1)} {value}", section, count=1)


def apply(candidate: dict, text: str) -> tuple[str, list[str]]:
    """(workspace.yaml with the candidate's tip across the axis, []), or (the text unchanged, why not)."""
    start, end = _section(text, candidate["arm"])
    section = text[start:end]
    if candidate.get("tool") != _get(section, "tool_id"):
        return text, [f"the candidate measured {candidate.get('tool')}, but {_get(section, 'tool_id')} is fitted"]
    tip, tip0 = candidate["fit"]["tip_m"], candidate["tip0_m"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*tip, *tip0]):
        return text, ["the candidate's tip is not finite"]
    for axis, k in (("x", 0), ("y", 1)):
        for key in (f"pen_tip_offset_{axis}", f"mechanical_contact_{axis}"):
            section = _set(section, key, f"{float(_get(section, key)) + tip[k] - tip0[k]:.6f}")
    held = candidate.get("held_out_rms_m") or {}
    for key, value in (("mechanical_contact_session", candidate["run_id"]), ("utc", candidate["utc"]),
                       ("session", f"ros-calib/{candidate['run_id']}"),
                       ("method", candidate.get("method", "probe-station side pairs at upright yaws (tatbot ros calib run)")),
                       ("n_plate", str(candidate["contacts"])),
                       ("residual_mm", f"{max(candidate['fit']['rms_m'].values()) * 1000:.3f}"),
                       ("holdout_mm", f"{max(held.values(), default=float('nan')) * 1000:.3f}"),
                       ("tip_loo_max_mm", f"{max((candidate.get('tip_moved_m') or {}).values(), default=float('nan')) * 1000:.3f}"),
                       ("note", '"the tip across its axis only; its length is the installed one"')):
        section = _set(section, key, value)
    return text[:start] + section + text[end:], []
