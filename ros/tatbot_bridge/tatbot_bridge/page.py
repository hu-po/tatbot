"""The page frame from stencild's target frame (ros/README.md section 3), and the sample it comes in.

stencild's `world_from_target` (tatbot.target-pose/1 on tatbot/tracking/target/<pattern_id>) is the
print's centre with x along reference u (right) and y along reference v (down the image), so z points
into the paper (scripts/vision/stencil_pose.py object points: ((uv - 0.5) * page_mm, 0)). The page
frame keeps x, flips y to the top of the print (-v) and z out of the paper: a half turn about x.

Pure numpy, no ROS: the node, the page watch and the tests share it.
"""
from __future__ import annotations

import json
import math

import numpy as np
from tatbot_description.transforms import matrix_quat as quaternion  # noqa: F401 -- existing page API
from tatbot_description.transforms import rpy_matrix as from_xyz_rpy  # noqa: F401

SCHEMA = "tatbot.target-pose/1"
TARGET_FROM_PAGE = np.diag([1.0, -1.0, -1.0, 1.0])


def topic(pattern_id: str) -> str:
    return f"tatbot/tracking/target/{pattern_id}"


def world_from_page(world_from_target) -> np.ndarray:
    return np.asarray(world_from_target, float) @ TARGET_FROM_PAGE


def base_from_page(world_from_arm_base, world_from_target) -> np.ndarray:
    """The page in an arm's base: inv(world_from_arm_base) @ world_from_page. This is the crossing the
    frozen session used (target_pose.to_arm_base), with the page frame's axes."""
    return np.linalg.inv(np.asarray(world_from_arm_base, float)) @ world_from_page(world_from_target)


def rigid(value) -> np.ndarray:
    """A finite rigid 4x4, or ValueError."""
    m = np.asarray(value, float)
    if (m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("pose is not a finite rigid 4x4 matrix")
    return m


def parse(data: bytes | str, pattern_id: str | None = None) -> dict | None:
    """One stencild sample as a flat dict, or None when it is not this print's target-pose.

    A `measured` sample carries `world_from_target`; a `lost` one carries None.
    """
    try:
        envelope = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or (pattern_id and payload.get("target_id") != pattern_id):
        return None
    support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
    source = payload.get("source")
    pose = None
    if source == "measured":
        try:
            pose = rigid(payload.get("world_from_target"))
        except (ValueError, TypeError):
            return None
    elif source != "lost":
        return None

    def number(key):
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.nan

    return {
        "pattern_id": payload.get("target_id", ""),
        "source": source,
        "world_from_target": pose,
        "translation_sigma_m": number("translation_sigma_m"),
        "rotation_sigma_rad": number("rotation_sigma_rad"),
        "stamp_ns": int((envelope.get("stamp") or {}).get("wall_ns") or 0),
        "seq": envelope.get("seq"),
        "calibration_id": payload.get("calibration_id") or "",
        "print_id": str(support.get("physical_instance_id") or ""),
        "identity_verified": support.get("physical_instance_identity_verified") is True,
        "support": support,
    }


def copies(sample: dict) -> list[dict]:
    """A measured sample's sheets of the print, largest first, each a sample of its own: stencild's pose, and the others
    its support's `copies` carried (a malformed one is left out). No support here carries `copies` on."""
    support = {key: value for key, value in sample["support"].items() if key != "copies"}
    sheets = [{**sample, "support": support}]
    for sheet in sample["support"].get("copies") or []:
        try:
            pose = rigid(sheet["world_from_target"])
            sheets.append({**sample, "world_from_target": pose, "support": {**support, **sheet["support"]},
                           "translation_sigma_m": float(sheet["translation_sigma_m"]),
                           "rotation_sigma_rad": float(sheet["rotation_sigma_rad"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sheets


def centre(sample: dict) -> np.ndarray:
    return np.asarray(sample["world_from_target"], float)[:3, 3]


def others_m(chosen: dict, sheets: list[dict]) -> list[float]:
    """How far each sheet but `chosen` stands from it, m."""
    return [round(float(np.linalg.norm(centre(s) - centre(chosen))), 3) for s in sheets if s is not chosen]


class PageChooser:
    """Which sheet of a print is the page. The other mounts' sheets, spares and the next rig's page read the same
    markers as the page itself, so what tells them apart is where they stand:

    - The sheet within `same_m` of where the page last stood is the page, however few of its markers the arm leaves
      and however many another shows: the largest group alone flipped between two sheets every few seconds
      (2026-11-04). Readings that close carry it along as it slides.
    - With none there it stands lost: the arm hides it, or it moved too far at once to follow. Another sheet never
      stands in for it, however long that lasts. Only on a rig with one sheet is the one the overhead reads the page
      wherever it stands, the page moved: while it has never read two at once and no sheet was chosen, since this
      began or in the record it began from, since a table that needs a choice has others.
    - `choose_at` names the page anew: the sheet nearest a point is the page from the turn it is read there on, and
      until then there is none. With no place yet (no touches, no choice) and several sheets in view, none is the page
      until one is chosen.

    Mounts carry no identity of their own, so they may move, come and go in any number. Pure, and in the arm's own
    world: every stack chooses for itself. `place`: where the page stood when this began (page_place); `chosen`: a
    choice put it there."""

    def __init__(self, cfg: dict, place=None, chosen: bool = False):
        self.same_m = float(cfg["same_m"])
        self.place = None if place is None else np.asarray(place, float)
        self.wanted: tuple[np.ndarray, np.ndarray] | None = None   # (xy in an arm's base, base_from_world), unread
        self.alone = not chosen   # one sheet on the table, as far as anything has shown

    def choose_at(self, xy_m, base_from_world) -> None:
        """The page is the sheet nearest xy_m (m, in the base of base_from_world) from the next turn it is read on."""
        self.wanted, self.alone = (np.asarray(xy_m, float), np.asarray(base_from_world, float)), False

    def choose(self, sheets: list[dict]) -> dict | None:
        """The page among one turn's `sheets` (copies), or None: it stands lost."""
        self.alone = self.alone and len(sheets) < 2
        if self.wanted is not None:
            chosen = self._nearest(sheets, self._from_wanted)
        else:
            chosen = (None if self.place is None else self._nearest(sheets, self._from_place)) or (
                sheets[0] if self.alone and sheets else None)
        if chosen is not None:
            self.place, self.wanted = centre(chosen), None
        return chosen

    def why_none(self) -> str:
        """Why the last turn had no page, for a log."""
        if self.wanted is not None:
            x, y = self.wanted[0] * 1e3
            return f"none within {self.same_m * 1e3:.0f} mm of the chosen ({x:+.0f}, {y:+.0f}) mm"
        if self.place is not None:
            return (f"none within {self.same_m * 1e3:.0f} mm of where the page stood (the arm hides it, or it moved "
                    "too far at once to follow)")
        return "none chosen"

    def _nearest(self, sheets: list[dict], off) -> dict | None:
        nearest = min(sheets, key=off, default=None)
        return nearest if nearest is not None and off(nearest) <= self.same_m else None

    def _from_place(self, sheet: dict) -> float:
        return float(np.linalg.norm(centre(sheet) - self.place))

    def _from_wanted(self, sheet: dict) -> float:
        xy, base_from_world = self.wanted
        return float(np.linalg.norm((base_from_world[:3, :3] @ centre(sheet) + base_from_world[:3, 3])[:2] - xy))


def parse_at(text: str) -> tuple[str, np.ndarray]:
    """`<arm>,<x_mm>,<y_mm>` (the bridge's page_at): the arm, and the point in its base in m."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"page_at: expected <arm>,<x_mm>,<y_mm>, got {text!r}")
    xy = np.array([float(parts[1]), float(parts[2])]) / 1e3
    if not np.isfinite(xy).all():
        raise ValueError(f"page_at: x and y must be finite, got {text!r}")
    return parts[0], xy


def page_place(path, world_from_arm_base) -> np.ndarray | None:
    """Where a record put the page, in the world: the session's last accepted touches (page-touched-<arm>.json) or the
    bridge's last choice (page-chosen-<arm>.json), each holding the page in the arm's base as `camera`. The place to
    look first after a restart; None without the record or the arm's registration."""
    if world_from_arm_base is None:
        return None
    try:
        record = json.loads(path.read_text())
        return (np.asarray(world_from_arm_base, float) @ rigid(record["camera"]))[:3, 3]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def covariance(translation_sigma_m: float, rotation_sigma_rad: float) -> list[float]:
    """Row-major 6x6 (x y z rx ry rz): the sigmas squared on the diagonal; -1 on the first entry when
    unknown (the ROS convention for an unknown covariance)."""
    cov = [0.0] * 36
    if not (math.isfinite(translation_sigma_m) and math.isfinite(rotation_sigma_rad)):
        cov[0] = -1.0
        return cov
    for i in range(3):
        cov[7 * i] = translation_sigma_m ** 2
        cov[7 * (i + 3)] = rotation_sigma_rad ** 2
    return cov


def rpy_deg(rotation) -> tuple[float, float, float]:
    """Roll, pitch, yaw in degrees (for logs)."""
    r = np.asarray(rotation, float)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return (math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(pitch), math.degrees(math.atan2(r[1, 0], r[0, 0])))
