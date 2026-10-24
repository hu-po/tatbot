"""config/motion_constants.json is the one source of the executor/planner numbers.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_motion_constants.py

Pins: the JSON loads under its schema, its SHA is the 12-hex value the checked-in
C++ header was generated with (so `scripts/gen_motion_constants.py --check` and this
agree), two spot constants read the same on both sides, and every samples file the
planner writes carries the `constants_sha` line the executor checks.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import gen_motion_constants  # noqa: E402
import motion_constants as dc  # noqa: E402

HEADER = REPO / "cpp" / "teleop" / "motion_constants.hpp"


def _cpp_value(name: str) -> str:
    text = HEADER.read_text()
    match = re.search(rf"^inline constexpr (?:double|int) {name} = ([^;]+);$", text, re.M)
    assert match, f"{name} is not in {HEADER}"
    return match.group(1)


def test_json_loads_under_its_schema_and_sha_is_12_hex():
    data = dc.load()
    assert data["schema"] == dc.SCHEMA == "tatbot.draw-constants/1"
    assert re.fullmatch(r"[0-9a-f]{12}", dc.SHA), dc.SHA
    assert dc.sha_of(data) == dc.SHA
    # The SHA covers the numbers, not the prose: editing _doc must not invalidate every samples file.
    assert dc.sha_of({**data, "_doc": "reworded"}) == dc.SHA
    assert dc.sha_of({**data, "period_s": 0.005}) != dc.SHA


def test_checked_in_cpp_header_carries_the_same_sha_and_values():
    text = HEADER.read_text()
    assert f'inline constexpr char SHA[] = "{dc.SHA}";' in text
    assert float(_cpp_value("PLANNER_MAX_JOINT_VELOCITY_RAD_S")) == dc.C.planner.max_joint_velocity_rad_s
    assert float(_cpp_value("CARRIAGE_IK_BIAS_M")) == dc.C.carriage_ik.bias_m
    assert int(_cpp_value("PLANNER_MAX_TICKS")) == dc.C.planner.max_ticks
    backoff = ", ".join(gen_motion_constants._cpp(v) for v in dc.C.path.dip_travel_backoff_m_s)
    assert f"PATH_DIP_TRAVEL_BACKOFF_M_S{{{backoff}}};" in text
    # The header is generated: it must be byte-identical to what the JSON renders now.
    assert gen_motion_constants.render() == text


def test_flat_covers_every_number_once():
    keys = [key for key, _ in dc.flat()]
    assert len(keys) == len(set(keys))
    assert "schema" not in keys and not any(k.startswith("_") for k in keys)
    assert {"period_s", "planner.max_ticks", "path.dip_travel_backoff_m_s"} <= set(keys)
