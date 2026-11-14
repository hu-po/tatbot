"""A wrist gauge's tip derived from the fitted cartridge's tip, in place of measuring it again per camera.

A wrist gauge file (calib/wrist-gauge-<arm>[-<role>].json) holds where the tip the arm commands stands in that camera's
view, `tip_cam_m`. The camera rides the tool mount, so when `tatbot ros calib apply` moves the tip by d in the mount
(a sweep's tip after a cartridge swap), the tip moves by the same d in the camera, turned by the camera's hand-eye:
tip_cam += R(camera <- mount) d. What `calib gauge` measured on top of the hand-eye, the camera's depth along the
tip's ray, is the camera's and stays. The pen's cone end moves with the tip; its axis is kept.

On the 3RL -> 7RL swap of 2026-11-10 the session camera's gauge derived this way stood 0.11 mm along the plane's
normal from the one `calib gauge` then measured to +-0.13 mm, and 0.24 mm across it, which the measurement cannot
see: it moves the tip along its ray and keeps its pixel (test_gauge_derive).

The hand-eye is the file's `mount_from_camera` (`calib wristcam` writes it), and the tip the gauge stands for its
`tip_mount_m`. A gauge adopted before they were written names its hand-eye through its runs' candidates (each `calib
gauge` run's gauge names the run it started from, back to a `calib wristcam` run), and stands for the tip installed
before this adoption when it was measured with the fitted cartridge since that tip was adopted. Otherwise it is
not derived, and the reason is said. Pure: no ROS import, no numpy; ros_cartridge reads and writes the files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RUN_ID = re.compile(r"\d{8}T\d{6}Z-[\w-]+?-[0-9a-f]{4}")
CHAIN_MAX = 12   # runs followed back to a hand-eye: one `calib gauge` run per link


class NotDerivedError(Exception):
    """Why a gauge cannot follow the tip."""


def _candidate(runs: Path, run: str) -> dict:
    try:
        return json.loads((runs / run / "candidate.json").read_text())
    except (OSError, ValueError):
        return {}


def hand_eye(gauge: dict, runs: Path) -> list[list[float]]:
    """mount <- camera's rotation, rows: the file's own, else the `calib wristcam` run its runs lead back to."""
    mount = gauge.get("mount_from_camera")
    run = gauge.get("run")
    for _ in range(CHAIN_MAX):
        if mount or not run:
            break
        candidate = _candidate(runs, run)
        mount = candidate.get("mount_from_camera")
        named = RUN_ID.findall((candidate.get("gauge") or {}).get("source") or "")
        run = named[0] if named and named[0] != run else None
    if not mount:
        raise NotDerivedError("its hand-eye is in neither the file nor its runs' candidates (tatbot ros calib "
                              "wristcam --adopt writes it, then tatbot ros calib gauge)")
    return [list(row[:3]) for row in mount[:3]]


def tip_before(gauge: dict, tool_id: str, installed: list[float], installed_utc: str | None) -> list[float]:
    """The tip in the mount the gauge stands for: its `tip_mount_m`, else the tip installed before this adoption
    when the gauge was measured with the fitted cartridge no earlier than that tip."""
    if gauge.get("tip_mount_m"):
        return [float(v) for v in gauge["tip_mount_m"]]
    if gauge.get("measured_with") != tool_id:
        raise NotDerivedError(f"it was measured with {gauge.get('measured_with') or 'an unrecorded tool'} and does "
                              f"not record the tip it stands for; {tool_id} is fitted")
    if str(gauge.get("utc") or "") < str(installed_utc or ""):
        raise NotDerivedError(f"it was measured {gauge.get('utc')}, before the installed tip ({installed_utc}), and "
                              "does not record the tip it stands for")
    return [float(v) for v in installed]


def in_camera(mount_from_camera: list[list[float]], moved_mount: list[float]) -> list[float]:
    """A move in the tool mount as the camera sees it: the hand-eye's rotation, transposed, times the move."""
    return [sum(mount_from_camera[row][col] * moved_mount[row] for row in range(3)) for col in range(3)]


def derive(gauge: dict, tip_then: list[float], tip_now: list[float], mount_from_camera: list[list[float]], *,
           tool_id: str, run: str, utc: str) -> dict:
    """The gauge with its tip, and its pen's cone end, moved as the tip moved in the mount from `tip_then` to
    `tip_now`; marked `derived` with the run whose tip it follows, the gauge's own measurement kept beside it."""
    moved = in_camera(mount_from_camera, [now - then for now, then in zip(tip_now, tip_then, strict=True)])
    out = dict(gauge)
    for key in ("tip_cam_m", "cone_end_cam_m"):
        if key in gauge:
            out[key] = [value + step for value, step in zip(gauge[key], moved, strict=True)]
    step = {"utc": utc, "run": run, "from_tool": gauge.get("measured_with"),
            "tip_moved_mount_mm": [round((now - then) * 1000, 3) for now, then in zip(tip_now, tip_then, strict=True)],
            "tip_moved_cam_mm": [round(value * 1000, 3) for value in moved]}
    out.update(tip_mount_m=[float(v) for v in tip_now], tool=tool_id, measured_with=tool_id, utc=utc,
               derived=[*gauge.get("derived", []), step])
    return out


def along_normal_mm(gauge: dict, derived: dict) -> float | None:
    """How far the derived tip stands from the gauge's along the plane's normal (toward the camera when positive):
    the height the gauge reads changes by this much."""
    normal = gauge.get("normal_cam")
    if not normal:
        return None
    return round(sum((new - old) * n for new, old, n in zip(derived["tip_cam_m"], gauge["tip_cam_m"], normal,
                                                           strict=True)) * 1000, 3) + 0.0
