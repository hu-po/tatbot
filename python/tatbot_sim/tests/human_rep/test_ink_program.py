from __future__ import annotations

import json
from dataclasses import replace

import pytest
from shapely import LineString, Polygon
from tatbot_sim import tools
from tatbot_sim.human_rep.contracts import ContractError, load_contract, validate_contract
from tatbot_sim.human_rep.ink_program import (
    compile_ink_program,
    exact_program_module,
    material_strokes,
)
from tatbot_sim.human_rep.placement import make_surface_placement, upgrade_body_placement
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root

REPO = repo_root()
EXAMPLES = REPO / "config" / "human-representation" / "examples"


@pytest.fixture(scope="module")
def body_context():
    rig = load_body_rig()
    atlas = json.loads((REPO / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json").read_text())
    faces = [index for index, eligible in enumerate(atlas["eligible_faces"]) if eligible]
    return rig, faces


def _placement(program, body_context, *, scale=(0.025, 0.025)):
    rig, faces = body_context
    return make_surface_placement(
        tattoo_program_sha256=program["content_sha256"],
        body_identity_sha256=rig.identity_sha256,
        rest_surface_sha256=rig.surface_sha256,
        topology_sha256=rig.topology_sha256,
        semantic_site="forearm",
        laterality="right",
        anchor=(8729, [1 / 3, 1 / 3, 1 / 3]),
        physical_scale_m=scale,
        rotation_rad=0.0,
        mirrored=False,
        supported_faces=faces,
        margin_m=0.003,
        review={"status": "pending", "reviewer": "maintainer", "evidence_sha256": "0" * 64},
        provenance={
            "producer": "tatbot-p4-test",
            "version": "1",
            "created_utc": "2026-09-04T14:00:00Z",
            "source_sha256": program["content_sha256"],
        },
    )


# The ballpoint's datasheet says ink.mode cartridge since 3ca4ade7 (an internal
# supply the palette never replenishes; the operator ruled out fake dry-dip
# replenishment). Dip parity, capacity splitting and colour changes are
# rehearsal/real-tool semantics, so this module compiles the ballpoint's
# geometry under its former rehearsal ink block as a fixture, and asserts the
# cartridge truth separately.
REHEARSAL_INK = {"mode": "rehearsal", "charge_capacity_ul": 2.0, "uptake_ul": 1.5,
                 "deposit_ul_per_mm": 0.004, "bleed_ul_per_s": 0.01, "dip_depth_m": 0.003,
                 "dip_dwell_s": 0.4, "min_fill_frac": 0.0}
REHEARSAL = "rehearsal-ballpoint"


def _load_tool(tool_id):
    if tool_id != REHEARSAL:
        return tools.registry().load_tool(tool_id, REPO)
    ball = tools.registry().load_tool("lutin-ballpoint-dot", REPO)
    return replace(ball, raw={**ball.raw, "ink": REHEARSAL_INK})


def _compile(name, body_context, *, tool_id=REHEARSAL, scale=(0.025, 0.025), **extra):
    program = load_contract(EXAMPLES / name / "program.json")
    placement = _placement(program, body_context, scale=scale)
    tool = _load_tool(tool_id)
    return compile_ink_program(
        program,
        placement,
        tool=tool,
        provenance={
            "producer": "tatbot-p4-test",
            "version": "1",
            "created_utc": "2026-09-04T14:00:00Z",
            "source_sha256": placement["content_sha256"],
        },
        **extra,
    )


def test_the_fitted_ballpoint_compiles_without_a_timed_supply_or_dips(body_context):
    result = _compile("linework", body_context, tool_id="lutin-ballpoint-dot")
    assert not [e for e in result["events"] if e["kind"] == "dip"]
    assert result["predicted_ink_state"]["load_fraction"] == 1.0


def test_body_v2_uses_identical_geometry_and_material_events(body_context):
    artwork = load_contract(EXAMPLES / "linework/program.json")
    old = _placement(artwork, body_context)
    new = upgrade_body_placement(old)
    tool = _load_tool(REHEARSAL)
    original = compile_ink_program(artwork, old, tool=tool, provenance=old["provenance"])
    migrated = compile_ink_program(artwork, new, tool=tool, provenance=old["provenance"])
    assert migrated["schema"] == "tatbot.ink-program/2"
    assert migrated["surface_placement_sha256"] == new["content_sha256"]
    assert migrated["events"] == original["events"]
    assert migrated["predicted_ink_state"] == original["predicted_ink_state"]


@pytest.mark.parametrize("name", ["linework", "blackwork", "stipple", "color"])
def test_every_artwork_class_materializes_to_soma_ink_program(name, body_context):
    result = _compile(name, body_context)
    strokes = [event for event in result["events"] if event["kind"] == "stroke"]
    assert strokes
    assert result["total_material_path_length_m"] == pytest.approx(
        sum(event["curve"]["rest_surface_arc_length_m"] for event in strokes)
    )
    assert all(
        coordinate["topology_sha256"] == body_context[0].topology_sha256
        for event in strokes
        for coordinate in event["curve"]["coordinates"]
    )


def test_fill_hatching_respects_negative_space_and_stipple_is_explicit(body_context):
    blackwork = load_contract(EXAMPLES / "blackwork" / "program.json")
    placement = _placement(blackwork, body_context)
    strokes = material_strokes(blackwork, placement)
    assert len(strokes) > 10
    # Test the complete footprint, including inset hole contours. A contour's
    # mean point lies inside its hole even though the stroke never enters it.
    half_diagonal = .01 * .025 / .06
    mask = Polygon([(0, -half_diagonal), (half_diagonal, 0),
                    (0, half_diagonal), (-half_diagonal, 0)])
    for stroke in strokes:
        footprint = LineString(stroke.points_m).buffer(stroke.width_m / 2, quad_segs=32)
        assert footprint.intersection(mask).area < 1e-16

    stipple = load_contract(EXAMPLES / "stipple" / "program.json")
    dots = material_strokes(stipple, _placement(stipple, body_context))
    assert len(dots) == 6
    assert all(len(stroke.points_m) == 13 for stroke in dots)


def test_multi_ink_has_explicit_color_change_and_exact_dip_parity(body_context):
    result = _compile("color", body_context)
    dips = [event for event in result["events"] if event["kind"] == "dip"]
    transitions = [event for event in dips if event["trigger_reason"] != "low_charge"]
    assert [event["trigger_reason"] for event in transitions] == ["session_start", "color_change"]
    assert [event["ink_id"] for event in transitions] == ["bright_red", "true_blue"]

    exact = exact_program_module()
    palette = exact.ink_spec.load_palette(REPO)
    load = {slot: exact.ink_spec.SlotLoad(slot, None) for slot in palette}
    load["inkcap_medium_1"] = exact.ink_spec.SlotLoad("inkcap_medium_1", "bright_red", 600)
    load["inkcap_medium_2"] = exact.ink_spec.SlotLoad("inkcap_medium_2", "true_blue", 600)
    tool = tools.registry().load_tool("lutin-3rl-bugpin", REPO)
    policy = exact.ink_spec.policy_for(tool)
    needle_result = _compile("color", body_context, tool_id="lutin-3rl-bugpin")
    resolved = exact.resolve_dips(
        needle_result,
        policy,
        palette,
        load,
        tool_id=tool.tool_id,
        inks=exact.ink_spec.load_inks(REPO),
    )
    slots = {"bright_red": "inkcap_medium_1", "true_blue": "inkcap_medium_2"}
    assert [item.slot_id for item in resolved] == [slots[e["ink_id"]] for e in needle_result["events"] if e["kind"] == "dip"]


def test_capacity_splitting_is_explicit_and_no_ink_tool_refuses(body_context):
    program = load_contract(EXAMPLES / "linework" / "program.json")
    placement = _placement(program, body_context)
    tool = _load_tool(REHEARSAL)
    policy = replace(tools.ink_registry().policy_for(tool), uptake_ul=0.05)
    result = compile_ink_program(
        program,
        placement,
        tool=tool,
        policy=policy,
        provenance={
            "producer": "tatbot-p4-test",
            "version": "1",
            "created_utc": "2026-09-04T14:00:00Z",
            "source_sha256": placement["content_sha256"],
        },
    )
    rationales = [event["ordering_rationale"] for event in result["events"] if event["kind"] == "stroke"]
    assert any("capacity split" in rationale for rationale in rationales)

    with pytest.raises(ContractError) as caught:
        _compile("linework", body_context, tool_id="picosecond-laser-pen")
    assert caught.value.code == "ink_supply_unavailable"


def test_fill_style_is_carried_only_when_not_the_default(body_context):
    """A program compiled with the hatch planner names it, following the
    legacy operating_budget_s precedent; a concentric program's bytes are unchanged
    and the field validates as an enum in the Python reader."""
    program = load_contract(EXAMPLES / "blackwork" / "program.json")
    placement = _placement(program, body_context)
    tool = _load_tool(REHEARSAL)
    provenance = {"producer": "tatbot-p4-test", "version": "1", "created_utc": "2026-09-04T14:00:00Z",
                  "source_sha256": placement["content_sha256"]}
    concentric = compile_ink_program(program, placement, tool=tool, provenance=provenance)
    hatch = compile_ink_program(program, placement, tool=tool, provenance=provenance, fill_style="hatch")
    assert "fill_style" not in concentric and hatch["fill_style"] == "hatch"
    assert hatch == compile_ink_program(program, placement, tool=tool, provenance=provenance, fill_style="hatch")
    assert hatch["content_sha256"] != concentric["content_sha256"]
    rationales = [e["ordering_rationale"] for e in hatch["events"] if e["kind"] == "stroke"]
    assert any("depth 3" in r for r in rationales) and all(r.startswith("source layer 0,") for r in rationales)
    broken = dict(hatch, fill_style="stipple")
    with pytest.raises(ContractError) as caught:
        validate_contract(broken, expected_schema=hatch["schema"])
    assert caught.value.code == "wrong_enum"
    with pytest.raises(ContractError) as caught:
        compile_ink_program(program, placement, tool=tool, provenance=provenance, fill_style="stipple")
    assert caught.value.code == "wrong_enum"


def test_compilation_is_byte_deterministic(body_context):
    assert _compile("linework", body_context) == _compile("linework", body_context)
