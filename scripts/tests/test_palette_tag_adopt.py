"""palette_tag_adopt: a calib wristcam run's roof-tag seat moves the palette design's seat and urdf/palette.urdf's
tag joint together, to where the run measured the sticker: applied twice, or after another run, it gives the same seat."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import ink_spec
import numpy as np
import palette_tag_adopt
import pytest

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / ink_spec.palette_geometry(REPO)["cad_design"]
URDF = REPO / "urdf" / "palette.urdf"


def _origin(urdf_text: str) -> tuple[np.ndarray, np.ndarray]:
    m = re.search(r'<child link="palette_tag"/>\s*<origin xyz="([^"]*)" rpy="([^"]*)"', urdf_text)
    return np.fromstring(m.group(1), sep=" "), np.fromstring(m.group(2), sep=" ")


def measured(run: str, turn_deg: float, shift_mm: list[float], urdf_text: str) -> dict:
    """A calib wristcam candidate that measured the sticker that far off the seat `urdf_text` carries."""
    xyz, rpy = _origin(urdf_text)
    return {"tag_seat": {"turn_deg": turn_deg, "shift_mm": shift_mm, "rise_mm": 0.3,
                         "against": {"xyz_m": xyz.tolist(), "yaw_deg": math.degrees(rpy[2])}},
            "gauge": {"arm": "xarm", "run": run}}


def test_a_measured_seat_moves_the_design_and_the_urdf_together():
    """The ID-0 sticker turned 1.29 deg and 0.9 mm aside (2026-11-09): the design's seat moves by both, its height
    kept, and the URDF's joint stands where build.py would emit it from that seat; nothing else in either changes."""
    design_text, urdf_text = DESIGN.read_text(), URDF.read_text()
    candidate = measured("20261109T201731Z-node-d23f", 1.29, [0.64, -0.62], urdf_text)
    edited_design, edited_urdf, lines = palette_tag_adopt.apply(candidate, design_text, urdf_text)
    before, after = json.loads(design_text), json.loads(edited_design)
    tag0, tag = before["runtime"]["installed_tag"], after["runtime"]["installed_tag"]
    assert tag["xyz_mm"] == pytest.approx(np.add(tag0["xyz_mm"], [0.64, -0.62, 0.0]).tolist())
    assert tag["yaw_deg"] == pytest.approx(tag0["yaw_deg"] + 1.29) and "d23f" in tag["measured"]
    after["runtime"]["installed_tag"] = tag0
    assert after == before
    xyz, rpy = _origin(edited_urdf)
    probe = before["probe"]
    root = np.array([*probe["xy"], probe["ball_z"] - before["runtime"]["ball_above_root_mm"]])
    assert xyz == pytest.approx((np.asarray(tag["xyz_mm"]) - root) / 1000.0, abs=1e-9)
    assert rpy == pytest.approx([0.0, 0.0, math.radians(tag["yaw_deg"])], abs=1e-9)
    assert re.sub(r'<origin xyz="[^"]*" rpy="[^"]*"', "", edited_urdf) == re.sub(r'<origin xyz="[^"]*" rpy="[^"]*"', "", urdf_text)
    assert "turned +1.29 deg" in lines[0]


def test_a_seat_applied_twice_or_after_another_run_is_the_seat_that_run_measured():
    """The seat used to be added to the design's each time: a second apply turned the sticker twice as far, and a run
    measured before another's adoption landed on the sum of both."""
    design_text, urdf_text = DESIGN.read_text(), URDF.read_text()
    older = measured("20261109T201731Z-node-d23f", 1.29, [0.64, -0.62], urdf_text)
    once = palette_tag_adopt.apply(older, design_text, urdf_text)
    again = palette_tag_adopt.apply(older, *once[:2])
    assert again[:2] == once[:2] and "already adopted" in again[2][0]
    newer = palette_tag_adopt.apply(measured("20261110T000000Z-node-0001", -0.4, [0.2, 0.1], urdf_text),
                                    design_text, urdf_text)
    assert newer[:2] != once[:2]
    assert palette_tag_adopt.apply(older, *newer[:2])[:2] == once[:2]


def test_a_run_that_does_not_record_the_seat_it_measured_against_is_adopted_only_where_it_already_is():
    """Runs before this record hold the turn and shift alone. The one the design names is its seat already; another
    was measured against a seat nobody can name now, so a new run is asked for."""
    design_text, urdf_text = DESIGN.read_text(), URDF.read_text()
    named = json.loads(design_text)["runtime"]["installed_tag"]["measured"].rsplit("run ", 1)[1].rstrip(")")
    old = {"tag_seat": {"turn_deg": 1.48, "shift_mm": [0.82, -0.80], "rise_mm": 0.3}, "gauge": {"run": named}}
    same = palette_tag_adopt.apply(old, design_text, urdf_text)
    assert same[:2] == (design_text, urdf_text) and "already adopted" in same[2][0]
    old["gauge"]["run"] = "20261110T165413Z-node-292a"
    with pytest.raises(ValueError, match="calib wristcam"):
        palette_tag_adopt.apply(old, design_text, urdf_text)


def test_a_run_without_a_seat_adopts_nothing():
    with pytest.raises(ValueError, match="no tag seat"):
        palette_tag_adopt.apply({"tip_cam_m": [0.0, 0.0, 0.15]}, DESIGN.read_text(), URDF.read_text())
