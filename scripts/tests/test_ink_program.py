"""InkProgram intent must be an exact, session-independent view of ink_spec."""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import ink_program  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic_catalog(tmp_path, monkeypatch):
    """Exercise ink planning without reading or changing the deployment inventory."""
    source = REPO
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy2(source / "config/arms.json", config / "arms.json")
    shutil.copytree(source / "config/tools", config / "tools")
    shutil.copy2(source / "config/examples/inks.yaml", config / "inks.yaml")
    palette = (source / "config/examples/palette.yaml").read_text()
    palette += "\nslots:\n  inkcap_medium_1:\n    arm: right\n    size: large\n  inkcap_medium_2:\n    arm: right\n    size: large\n"
    (config / "palette.yaml").write_text(palette)
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)


def _policy(tool_id: str = "lutin-ballpoint-dot"):
    tool = tool_spec.load_tool(tool_id, REPO)
    policy = ink_spec.policy_for(tool)
    if tool_id == "lutin-ballpoint-dot":
        policy = replace(ink_spec.policy_for(tool_spec.load_tool("lutin-3rl-bugpin", REPO)), mode="rehearsal")
    return tool, policy


def _curve(length_m: float) -> dict:
    return {
        "coordinates": [
            {"topology_sha256": "1" * 64, "face_index": 1, "barycentric": [1, 0, 0]},
            {"topology_sha256": "1" * 64, "face_index": 2, "barycentric": [0, 1, 0]},
        ],
        "rest_surface_arc_length_m": length_m,
        "width_m": 0.0008,
        "deposition": 0.7,
        "direction": "forward",
        "source_primitive_sha256": "2" * 64,
        "compiler_sha256": "3" * 64,
    }


def _program(needs: list[tuple[float, str]], policy, initial_fraction: float = 0.0) -> dict:
    charge = ink_spec.Charge(
        initial_fraction * policy.charge_capacity_ul,
        policy.charge_capacity_ul,
        needs[0][1],
    )
    events = []
    for index, (length_m, ink_id) in enumerate(needs):
        seconds = length_m / 0.004
        cost = policy.stroke_ul(length_m * 1000.0, seconds)
        reason = None
        if index == 0 and charge.ul <= 0:
            reason = "session_start"
        elif charge.ink_id != ink_id:
            reason = "color_change"
        elif cost > charge.ul:
            reason = "low_charge"
        if reason:
            if reason == "color_change":
                charge.ul = 0.0
            charge.credit(policy.uptake_ul, ink_id)
            events.append({
                "kind": "dip",
                "ink_id": ink_id,
                "target_load": [charge.frac, min(1.0, charge.frac + 0.05)],
                "trigger_reason": reason,
                "dependency": f"before-stroke-{index}",
                "expected_load_after": charge.frac,
            })
        events.append({
            "kind": "stroke",
            "curve": _curve(length_m),
            "start_choice": "start",
            "allowed_tool_class": "ballpoint",
            "speed_m_s": 0.004,
            "orientation_tolerance_rad": 0.12,
            "contact_envelope_m": [-0.0002, 0.0002],
            "research_depth_band_m": None,
            "ordering_rationale": "test order",
        })
        charge.debit(cost)
    return {
        "schema": "tatbot.ink-program/1",
        "content_sha256": "4" * 64,
        "tattoo_program_sha256": "5" * 64,
        "surface_placement_sha256": "6" * 64,
        "compiler_sha256": "3" * 64,
        "initial_ink_state": {"ink_id": needs[0][1], "load_fraction": initial_fraction},
        "predicted_ink_state": {"ink_id": charge.ink_id, "load_fraction": charge.frac},
        "events": events,
        "total_material_path_length_m": sum(length for length, _ in needs),
        "uncertainty": {"length_sigma_m": 0, "load_sigma": 0},
        "provenance": {},
    }


def test_mapping_table_has_one_owner_for_every_session_sensitive_field():
    sources = {source: (destination, owner) for source, destination, owner in ink_program.FIELD_MAPPING}
    assert sources["DipPlan.slot_id"] == ("palette.resolved_caps.slot", "ExecutionProgram")
    assert sources["SlotLoad.fill_ul"] == ("palette load snapshot", "ExecutionProgram")
    assert sources["StrokeNeed.contact_mm"] == (
        "stroke.curve.rest_surface_arc_length_m",
        "InkProgram",
    )


def test_dry_rehearsal_resolves_and_produces_a_derivable_ledger():
    tool, policy = _policy()
    palette = ink_spec.load_palette(REPO)
    dry = ink_spec.supply_load("dry", palette, repo=REPO)
    program = _program([(0.02, "example_black"), (0.02, "example_black")], policy)
    dips = ink_program.resolve_dips(program, policy, palette, dry, tool_id=tool.tool_id)
    assert [(dip.before_stroke, dip.reason) for dip in dips] == [(0, "session_start")]
    ledger = ink_program.expected_ledger_events(program, policy, dips)
    assert [event["kind"] for event in ledger] == ["dip", "stroke", "stroke"]
    assert sum(event.get("ul", 0) for event in ledger) == pytest.approx(
        sum(policy.stroke_ul(stroke.need.contact_mm, stroke.need.contact_s) for stroke in ink_program.program_strokes(program))
    )


def test_real_multi_ink_and_low_charge_use_existing_planner():
    tool, policy = _policy("lutin-3rl-bugpin")
    palette = ink_spec.load_palette(REPO)
    load = {slot: ink_spec.SlotLoad(slot, None) for slot in palette}
    load["inkcap_medium_1"] = ink_spec.SlotLoad("inkcap_medium_1", "example_black", 600)
    load["inkcap_medium_2"] = ink_spec.SlotLoad("inkcap_medium_2", "example_red", 600)
    program = _program(
        [(0.12, "example_black"), (0.12, "example_black"), (0.02, "example_red")],
        policy,
    )
    dips = ink_program.resolve_dips(
        program,
        policy,
        palette,
        load,
        tool_id=tool.tool_id,
        inks=ink_spec.load_inks(REPO),
    )
    assert [dip.reason for dip in dips] == ["session_start", "low_charge", "color_change"]
    assert [dip.slot_id for dip in dips] == ["inkcap_medium_1", "inkcap_medium_1", "inkcap_medium_2"]


def test_no_ink_unavailable_cap_over_capacity_and_tampered_dip_refuse():
    palette = ink_spec.load_palette(REPO)
    dry = ink_spec.supply_load("dry", palette, repo=REPO)
    laser, none = _policy("picosecond-laser-pen")
    with pytest.raises(ink_program.InkProgramRefusal) as caught:
        ink_program.resolve_dips(_program([(0.01, "example_black")], none), none, palette, dry, tool_id=laser.tool_id)
    assert caught.value.code == "ink_supply_unavailable"

    needle, real = _policy("lutin-3rl-bugpin")
    program = _program([(0.01, "example_black")], real)
    with pytest.raises(ink_program.InkProgramRefusal) as caught:
        ink_program.resolve_dips(program, real, palette, dry, tool_id=needle.tool_id)
    assert caught.value.code == "ink_supply_unavailable"

    too_long = _program([(1.0, "example_black")], real)
    with pytest.raises(ink_program.InkProgramRefusal) as caught:
        ink_program.resolve_dips(too_long, real, palette, ink_spec.supply_load("wet", palette, "example_black", REPO))
    assert caught.value.code == "stroke_over_capacity"

    full = ink_spec.supply_load("wet", palette, "example_black", REPO)
    program["events"][0]["trigger_reason"] = "unrecorded_reason"
    with pytest.raises(ink_program.InkProgramRefusal) as caught:
        ink_program.resolve_dips(program, real, palette, full)
    assert caught.value.code == "dip_intent_mismatch"


def test_tool_classes_are_closed_and_explicit():
    assert ink_program.tool_class("lutin-ballpoint-dot") == "ballpoint"
    assert ink_program.tool_class("lutin-3rl-bugpin") == "tattoo-needle"
    with pytest.raises(ink_program.InkProgramRefusal):
        ink_program.tool_class("unknown")
