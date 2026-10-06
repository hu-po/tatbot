"""config/motion_constants.json is the one source of the executor/planner numbers.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_motion_constants.py

Pins: the JSON loads under its schema, its SHA is the 12-hex value the checked-in
C++ header was generated with (so `scripts/gen_motion_constants.py --check` and this
agree), two spot constants read the same on both sides, the Python aliases the
planner keeps for its old names resolve to the JSON, and every samples file the
planner writes carries the `constants_sha` line the executor checks.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

import arm_kinematics as dk  # noqa: E402
import gen_motion_constants  # noqa: E402
import motion_constants as dc  # noqa: E402
import pen_path as dp  # noqa: E402

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


def test_python_aliases_resolve_to_the_json():
    c = dc.C
    assert c.planner.max_ticks == dk.PLAN_MAX_TICKS
    assert (c.carriage_ik.min_m, c.carriage_ik.max_m) == (dk.CARRIAGE_IK_MIN_M, dk.CARRIAGE_IK_MAX_M)


def test_samples_files_carry_the_constants_sha(tmp_path):
    n = 4
    samples = dp.Samples(
        period_s=dc.C.period_s, t=np.arange(n) * dc.C.period_s, p=np.zeros((n, 3)), v=np.zeros((n, 3)),
        R=np.tile(np.eye(3), (n, 1, 1)), pen=np.zeros(n, np.int64), capture=np.zeros(n, np.int64))
    for kind in ("orbit", "path"):
        path = tmp_path / f"{kind}.csv"
        dp.write_samples_csv(path, samples, kind, dk.BALLPOINT_TIP_IN_LINK6)
        lines = path.read_text().splitlines()
        assert lines[0] == f"schema,{dp.SAMPLES_SCHEMA}"
        assert lines[1] == f"constants_sha,{dc.SHA}"
        _, header = dp.read_samples_csv(path)
        assert header["constants_sha"] == dc.SHA
    # A caller may restate the line only if it agrees.
    dp.write_samples_csv(tmp_path / "ok.csv", samples, "path", dk.BALLPOINT_TIP_IN_LINK6, {"constants_sha": dc.SHA})
    try:
        dp.write_samples_csv(tmp_path / "bad.csv", samples, "path", dk.BALLPOINT_TIP_IN_LINK6,
                             {"constants_sha": "deadbeefcafe"})
    except ValueError as exc:
        assert "constants_sha" in str(exc)
    else:
        raise AssertionError("a conflicting constants_sha in extra must be refused")
