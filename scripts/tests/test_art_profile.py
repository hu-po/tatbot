"""Offline time profiles explain the saved compiler model and charge only real workflow changes."""
import hashlib
import json
from copy import deepcopy

import pytest
from tatbot_art import cli, profile
from tatbot_art.render import minutes
from tatbot_art.timing import workflow


def _program():
    """Three resource blocks: a refit of one ink, then a new ink on the same cartridge."""
    activation = {"method": "rinse", "slot": "water", "ink_id": "water", "dwell_s": 30.,
                  "above_ink_m": 0., "first_dip_dwell_s": 3.}
    dip = {"mm_per_dip": 60., "dwell_s": 1.5, "above_ink_m": 0.,
           "hover_m": .012, "speed_m_s": .01, "wall_margin_m": .001}
    resources = [{"id": rid, "ink_id": ink, "tool": {"id": "liner", "line_width_m": .0005, "ink": "dip"},
                  "pen_mode": "press", "slot": "colour", "dip": dip, "activation": activation}
                 for rid, ink in (("red-a", "red"), ("red-b", "red"), ("black", "black"))]
    ops = []
    for i, resource in enumerate(resources):
        rid = resource["id"]
        ops += [{"op": "tool_change", "id": f"t{i}", "resource_id": rid, "initial": i == 0,
                 "activation": activation},
                {"op": "dip", "id": f"d{i}", "resource_id": rid, "slot": "colour"},
                {"op": "stroke", "id": f"s{i}", "resource_id": rid,
                 "points_m": [[-.005, 0.], [.005, 0.]], "generation_width_m": .0005}]
    components = {"drawing": 30., "descent": 12., "settle": 3., "lift": 6., "travel": 9.}
    return {"format": "tatbot-program", "version": 2, "arm": "right", "draw_speed_m_s": .0035,
            "page": {"kind": "stencil", "size_m": [.04, .06], "clear_m": [.04, .06]},
            "preparation": {"adapter": "dbv3-paths-to-ros/2"}, "resources": resources, "ops": ops,
            "stats": {"time_estimate": {"model": "tatbot-duration/2", "modeled_s": 60., "total_s": None,
                      "components_s": components, "unknown_operations": {"dip": 3, "pen_change": 3},
                      "workflow_counts": {"dip": 3, "pen_change": 3}, "motion_sha256": "a" * 64,
                      "scope": "saved fixture model"}}}


def test_initial_and_same_tool_and_ink_selections_do_not_rinse():
    program = _program()
    assert [s["kind"] for s in workflow(program)] == ["selection", "dip", "selection", "dip", "rinse", "dip"]
    report = profile.build(program)
    t = report["time"]
    assert (t["dips"], t["rinses"], t["swaps"], t["selections"]) == (3, 1, 0, 2)
    assert t["estimated_s"] == 60 + 3 * 48 + 40
    assert sum(r["seconds"] for r in report["components"]) == t["estimated_s"]
    assert sum(r["share"] for r in report["components"]) == pytest.approx(1, abs=2e-6)
    assert report["totals"] == {"strokes": 3, "touchdowns": 3, "contact_mm": 30.}
    assert t == minutes(program)


def test_swap_and_user_gap_change_only_the_missing_workflow_costs():
    program = _program()
    program["resources"][-1]["activation"] = {"method": "manual", "action": "exchange"}
    time = profile.build(program, gaps_s={"dip": 7, "swap": 90})["time"]
    assert (time["rinses"], time["swaps"]) == (0, 1)
    assert time["estimated_s"] == 60 + 3 * 7 + 90
    assert time["basis"]["swap"] == "user-supplied allowance"


def test_recorded_workflow_costs_are_not_charged_twice_or_overridden():
    program = _program()
    estimate = program["stats"]["time_estimate"]
    estimate["components_s"].update(dip=21., pen_change=33.)
    estimate.update(modeled_s=114., total_s=114., unknown_operations={})
    time = profile.build(program, gaps_s={"dip": 1000, "rinse": 1000})["time"]
    assert time["estimated_s"] == 114.
    assert "rinse" not in time["components_s"]
    assert time["basis"]["dip"] == "recorded compiler estimate"


def test_chunk_continuations_and_closed_path_length_are_counted_as_drawn():
    program = _program()
    program["ops"][2]["closed"] = True
    program["ops"][5]["continues"] = True
    report = profile.build(program)
    assert report["totals"] == {"strokes": 3, "touchdowns": 2, "contact_mm": 40.}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_gaps_fail_before_writing_any_report(tmp_path, value):
    source, out = tmp_path / "program.json", tmp_path / "out"
    source.write_text(json.dumps(_program()))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        profile.write(source, out, {"dip": value})
    assert not out.exists()


@pytest.mark.parametrize("change", ["rehearsal", "ops", "components"])
def test_transformed_or_inconsistent_estimates_are_refused(change):
    program = _program()
    if change == "rehearsal":
        program["rehearsal"] = {"of": "b" * 64}
    elif change == "ops":
        del program["ops"][1]
    else:
        program["stats"]["time_estimate"]["components_s"]["drawing"] = 61
    with pytest.raises(ValueError, match="compiled program|recorded timing|components exceed"):
        profile.build(program)


def test_report_keeps_input_bytes_and_escapes_labels(tmp_path, capsys):
    program = _program()
    program["stats"]["time_estimate"]["scope"] = "<script>bad()</script>"
    source, out = tmp_path / "program.json", tmp_path / "out"
    raw = json.dumps(program).encode()
    source.write_bytes(raw)
    assert cli.main(["profile", str(source), "--dip-s", "7", "--out", str(out)]) == 0
    report = json.loads((out / "profile.json").read_text())
    assert source.read_bytes() == raw
    assert report["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    html = (out / "profile.html").read_text()
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html and "<script>" not in html
    assert "http://" not in html and "https://" not in html
    assert report["time"]["estimated_s"] == 121
    assert "Travel between strokes" in capsys.readouterr().out
    before = deepcopy(program)
    profile.build(program)
    assert program == before


def test_profile_never_overwrites_its_source(tmp_path):
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(_program()))
    with pytest.raises(ValueError, match="overwrite"):
        profile.write(source, tmp_path)
    assert json.loads(source.read_text()) == _program()
