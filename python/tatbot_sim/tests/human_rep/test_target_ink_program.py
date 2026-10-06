from copy import deepcopy

import pytest
from tatbot_sim import tools
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, load_contract, validate_contract
from tatbot_sim.human_rep.ink_program import compile_ink_program, exact_program_module
from tatbot_sim.human_rep.placement import make_target_placement
from tatbot_sim.repo import repo_root

ROOT = repo_root()
EXAMPLES = ROOT / "config/human-representation/examples"


def compile_chart(kind, *, artwork="linework", loaded=1.0):
    art = load_contract(EXAMPLES / artwork / "program.json")
    source = load_contract(EXAMPLES / f"{kind}-placement.json")
    placed = make_target_placement(
        target=source["target"], tattoo_program_sha256=art["content_sha256"],
        physical_scale_m=[0.015, 0.015], rotation_rad=0, mirrored=False,
        review=source["review"], provenance=source["provenance"],
    )
    tool = tools.registry().load_tool("lutin-ballpoint-dot", ROOT)
    # Dip/colour tests select a volumetric policy explicitly; the fitted pen is a cartridge.
    policy = exact_program_module().ink_spec.policy_for(tools.registry().load_tool("lutin-3rl-bugpin", ROOT))
    result = compile_ink_program(art, placed, tool=tool, policy=policy, initial_load_fraction=loaded,
                                 provenance=source["provenance"])
    return placed, result, tool


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
@pytest.mark.parametrize("artwork", ["linework", "blackwork", "stipple", "color"])
def test_all_artwork_classes_use_same_target_and_supply_compiler(kind, artwork):
    placed, result, _ = compile_chart(kind, artwork=artwork)
    assert result["schema"] == "tatbot.ink-program/2"
    strokes = [event for event in result["events"] if event["kind"] == "stroke"]
    assert strokes
    for stroke in strokes:
        assert all(point["target_sha256"] == placed["content_sha256"] for point in stroke["curve"]["coordinates"])
    assert result["total_material_path_length_m"] == pytest.approx(sum(
        e["curve"]["rest_surface_arc_length_m"] for e in strokes))
    if artwork == "color":
        assert any(e["kind"] == "dip" and e["trigger_reason"] == "color_change" for e in result["events"])


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_loaded_pen_needs_no_palette_but_exhausted_supply_is_not_ignored(kind):
    _, loaded, tool = compile_chart(kind)
    exact = exact_program_module()
    policy = exact.ink_spec.policy_for(tools.registry().load_tool("lutin-3rl-bugpin", ROOT))
    assert not [e for e in loaded["events"] if e["kind"] == "dip"]
    assert exact.resolve_dips(loaded, policy, {}, {}) == []
    _, empty, _ = compile_chart(kind, loaded=0)
    assert any(e["kind"] == "dip" for e in empty["events"])
    with pytest.raises(exact.InkProgramRefusal, match="ink_supply_unavailable"):
        exact.resolve_dips(empty, policy, {}, {})


def test_chart_binding_and_material_length_cannot_be_rehashed_away():
    _, original, _ = compile_chart("plane")
    for mutation in ("length", "target", "mixed", "old_schema"):
        result = deepcopy(original)
        curve = next(e["curve"] for e in result["events"] if e["kind"] == "stroke")
        if mutation == "length":
            curve["rest_surface_arc_length_m"] += 0.001
            result["total_material_path_length_m"] += 0.001
        elif mutation == "target":
            for point in curve["coordinates"]:
                point["target_sha256"] = "a" * 64
        elif mutation == "mixed":
            curve["coordinates"][-1] = {"topology_sha256": "b" * 64, "face_index": 1, "barycentric": [1, 0, 0]}
        else:
            result["schema"] = "tatbot.ink-program/1"
        result["content_sha256"] = canonical_digest(result)
        with pytest.raises(ContractError):
            validate_contract(result)


@pytest.mark.parametrize('kind', ['plane', 'cylinder'])
def test_cartridge_has_no_timed_cutoff_or_dips(kind):
    placed, _, tool = compile_chart(kind)
    art = load_contract(EXAMPLES / 'linework' / 'program.json')
    policy = exact_program_module().ink_spec.InkPolicy(mode='cartridge')
    kwargs = {"tool": tool, "policy": policy, "initial_load_fraction": 1,
              "provenance": placed['provenance']}
    result = compile_ink_program(art, placed, **kwargs)
    assert not any(event['kind'] == 'dip' for event in result['events'])
    assert result['predicted_ink_state']['load_fraction'] == 1.0
    legacy = compile_ink_program(art, placed, operating_budget_s=.001, **kwargs)
    assert legacy['predicted_ink_state']['load_fraction'] == 1.0
    assert legacy['operating_budget_s'] == .001
