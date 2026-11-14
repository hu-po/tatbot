"""Adopting a probe tip: the fitted cartridge's tip stands where the run measured it across its axis, its length
stays, the touch-off record names the run, and nothing outside the tip changes. A run applied twice, or after a newer
one, gives that run's tip and never a sum. Another tool's candidate is refused."""
from __future__ import annotations

from pathlib import Path

import cartridge_fit
import probe_calibration_adopt as adopt
import pytest
import tool_spec

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = tool_spec.parse_simple_yaml((REPO / "config" / "workspace.yaml").read_text())
SECTION = WORKSPACE["right"]
FITTED = SECTION["tool_id"]


def candidate(**over):
    base = {"arm": "right", "tool": FITTED, "run_id": "20261003T010000Z-node-abcd",
            "utc": "2026-10-03T01:30:00Z", "contacts": 32, "tip0_m": [0.00002, -0.002674, 0.074495],
            "held_out_rms_m": {"S1": 0.0003, "S4 -60/+0": 0.0004}, "tip_moved_m": {"S1": 0.0002},
            "fit": {"tip_m": [0.00042, 0.000826, 0.074495], "rms_m": {"side": 0.0002}}}
    base.update(over)
    return base


def adopted(run: dict, section: dict, **how) -> dict:
    """The section as its owner reads it once `run` is adopted."""
    tip, why = adopt.apply(run, section, **how)
    assert why == []
    return {**section, **tip}


def test_the_tip_stands_where_the_run_measured_it_across_its_axis_and_only_tip_keys_come_back():
    tip, why = adopt.apply(candidate(), SECTION)
    assert why == []
    assert set(tip) <= set(cartridge_fit.TIP_KEYS)
    assert (tip["pen_tip_offset_x"], tip["pen_tip_offset_y"]) == (0.00042, 0.000826)
    assert tip["pen_tip_offset_z"] == SECTION["pen_tip_offset_z"]
    for axis in "xyz":   # the contact reference keeps its place against the tip (the 3RL's: 0.25 mm short of it)
        assert (tip[f"mechanical_contact_{axis}"] - tip[f"pen_tip_offset_{axis}"] == pytest.approx(
            SECTION[f"mechanical_contact_{axis}"] - SECTION[f"pen_tip_offset_{axis}"], abs=1e-9)), axis
    assert tip["mechanical_contact_session"] == "20261003T010000Z-node-abcd"
    assert tip["touchoff"]["session"] == "ros-calib/20261003T010000Z-node-abcd"
    assert tip["touchoff"]["residual_mm"] == 0.2 and tip["touchoff"]["holdout_mm"] == 0.4
    assert tip["touchoff"]["note"] == adopt.AXIS_NOTE


def test_a_measured_length_moves_the_tip_along_it_unless_across_only():
    swept = candidate(fit={"tip_m": [0.00042, 0.000826, 0.0765], "rms_m": {"side": 0.0002}, "length_measured": True})
    tip, _ = adopt.apply(swept, SECTION)
    assert tip["pen_tip_offset_z"] == 0.0765
    assert tip["touchoff"]["note"] == adopt.LENGTH_NOTE
    tip, _ = adopt.apply(swept, SECTION, across_only=True)
    assert tip["pen_tip_offset_z"] == SECTION["pen_tip_offset_z"] and tip["touchoff"]["note"] == adopt.AXIS_NOTE


def test_a_run_applied_twice_changes_nothing_the_second_time():
    """`calib apply` used to add the run's change (fit.tip_m - tip0_m) to the installed tip, so a second apply of one
    sweep left the tip that change farther again: (2.817, -2.736, 73.382) became (4.634, -2.272, 75.834) mm."""
    swept = candidate(tip0_m=[0.001002, -0.003199, 0.070932],
                      fit={"tip_m": [0.002817, -0.002736, 0.073382], "rms_m": {"side": 0.0002}, "length_measured": True})
    for how in ({}, {"across_only": True}):
        once = adopted(swept, SECTION, **how)
        assert adopted(swept, once, **how) == once, how
    assert [once[f"pen_tip_offset_{axis}"] for axis in "xy"] == [0.002817, -0.002736]


def test_an_older_run_applied_after_a_newer_one_gives_the_older_runs_tip():
    """A run's tip is the tip in the tool mount, whatever tip the arm carried when it was measured or carries now."""
    older = candidate(run_id="20261110T161721Z-node-4625", tip0_m=[0.001002, -0.003199, 0.070932],
                      fit={"tip_m": [0.002817, -0.002736, 0.073382], "rms_m": {"side": 0.0002}, "length_measured": True})
    newer = candidate(run_id="20261111T193729Z-node-0a58", utc="2026-11-11T19:45:47Z",
                      tip0_m=[0.002817, -0.002736, 0.073382],
                      fit={"tip_m": [0.002641, -0.002892, 0.072574], "rms_m": {"side": 0.0004}, "length_measured": True})
    alone = adopted(older, SECTION)
    assert adopted(older, adopted(newer, alone)) == alone
    assert adopted(older, adopted(newer, SECTION)) == alone


def test_another_tools_candidate_and_a_nonfinite_tip_are_refused():
    assert adopt.apply(candidate(tool="picosecond-laser-pen"), SECTION) == ({}, [
        f"the candidate measured picosecond-laser-pen, but {FITTED} is fitted"])
    bad = candidate()
    bad["fit"]["tip_m"][0] = float("nan")
    assert adopt.apply(bad, SECTION)[1] == ["the candidate's tip is not finite"]


def test_a_missing_holdout_is_recorded_as_none_never_nan():
    """parse_simple_yaml refuses a non-finite number, and JSON has none: an absent figure is null."""
    tip, why = adopt.apply(candidate(held_out_rms_m={}, tip_moved_m={}), SECTION)
    assert why == [] and tip["touchoff"]["holdout_mm"] is None and tip["touchoff"]["tip_loo_max_mm"] is None


def test_a_cartridges_first_adoption_writes_its_touchoff_record_in_place_of_its_nominal_mark():
    """A cartridge fitted at its datasheet's nominal tip (cartridge_fit.fit) gains the record a measured one has, the
    contact at the nominal tip before the run moves it, and tool_spec then reads a measured tip."""
    calibration = tool_spec.load_tool(FITTED, REPO).raw["calibration"]
    never_measured = {**cartridge_fit.seeded(WORKSPACE["xarm"], "xarm"), "tips": {}}
    record = cartridge_fit.fit(never_measured, FITTED, (0.0, 0.0, 0.073), instance=None, utc="2026-11-10T00:00:00Z")
    nominal = cartridge_fit.fitted_section(WORKSPACE["xarm"], record)
    assert tool_spec.tip_offset_m({"xarm": nominal}, "xarm") is None
    tip, why = adopt.apply(candidate(arm="xarm", tip0_m=[0.0, 0.0, 0.073]), nominal, calibration)
    assert why == []
    assert "tool_fit" not in tip
    assert tip["mechanical_contact_reference"] == calibration["contact_reference"]
    assert tip["mechanical_contact_profile"] == calibration["accuracy_profile"]
    assert tip["mechanical_contact_x"] == pytest.approx(0.00042) and tip["pen_tip_offset_z"] == 0.073
    adopted = cartridge_fit.fitted_section(nominal, cartridge_fit.adopt(record, tip, run_id="r", utc="u"))
    assert tool_spec.tip_offset_m({"xarm": adopted}, "xarm") == pytest.approx((0.00042, 0.000826, 0.073))
