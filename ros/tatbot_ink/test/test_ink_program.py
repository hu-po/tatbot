"""DBV3 preparation preserves acquired traversal, physical meaning and executor contracts."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import tatbot_ink
from ink_designs import REPO, artwork, design, element, placement, program, write
from tatbot_contracts.canonical import canonical_digest
from tatbot_ink import CompileError
from tatbot_ink.__main__ import main
from tatbot_motion import load_motion
from tatbot_motion.estimate import drawing_seconds


def line_art(*, width=.0005, closed=False):
    return artwork(program([("pen-1", [element([[.002, .01], [.018, .01]], width=width, closed=closed)])]))


@pytest.mark.parametrize('seconds', [2, 60])
def test_compiled_closed_and_repeated_traversals_keep_source_addresses(tmp_path, seconds):
    from tatbot_motion.timelaw import polyline_length

    loop = [[.002, .002], [.018, .002], [.018, .018]]
    art = artwork(program([('pen-1', [element(loop, closed=True), element(loop, closed=True),
                                     element(list(reversed(loop)), closed=True)])]))
    strokes = [op for op in tatbot_ink.compile(write(tmp_path, art), repo=REPO, max_segment_s=seconds)['ops']
               if op['op'] == 'stroke']
    for op in strokes:
        assert op['src']['arc_m'][1] - op['src']['arc_m'][0] == pytest.approx(polyline_length(op['points_m']))
    assert len({op['src']['path'] for op in strokes}) == 3


def test_native_artwork_preserves_repeated_passes_and_direction(tmp_path):
    paths = [element([[.002, .01], [.018, .01]]), element([[.002, .01], [.018, .01]]),
             element([[.018, .005], [.002, .005]])]
    art = artwork(program([("pen-1", paths)]))
    path = write(tmp_path, art)
    result = tatbot_ink.compile(path, repo=REPO, at_m=[.015, -.028], width_m=.02)
    assert (result["format"], result["version"]) == ("tatbot-program", 2)
    assert result["design"]["at_m"] == [.015, -.028] and result["design"]["width_m"] == .02
    assert result["stats"]["paths"] == result["stats"]["strokes"] == 3
    ops = [op for op in result["ops"] if op["op"] == "stroke"]
    assert [op["src"]["path"] for op in ops] == ["path-0-0", "path-0-1", "path-0-2"]
    np.testing.assert_allclose(ops[0]["points_m"], [[.007, -.028], [.023, -.028]])
    assert ops[0]["points_m"] == ops[1]["points_m"]
    assert ops[2]["points_m"][0][0] > ops[2]["points_m"][-1][0]
    assert all(op["src"]["artwork_sha256"] == art["content_sha256"] and not op["continues"] for op in ops)
    assert result == tatbot_ink.compile(path, repo=REPO, at_m=[.015, -.028], width_m=.02)
    assert json.loads(tatbot_ink.write_program(result, tmp_path/'program.json').read_text()) == result


def test_closed_path_retains_closure_and_arc_when_chunked(tmp_path):
    art = artwork(program([("pen-1", [element([[.002, .002], [.018, .002], [.018, .018]], closed=True)])]))
    path = write(tmp_path, art)
    whole = tatbot_ink.compile(path, repo=REPO)
    assert whole["ops"][1]["closed"] and whole["ops"][1]["points_m"][0] == whole["ops"][1]["points_m"][-1]
    split = tatbot_ink.compile(path, repo=REPO, max_segment_s=2)
    ops = [op for op in split["ops"] if op["op"] == "stroke"]
    assert len(ops) > 2 and [op["continues"] for op in ops] == [False] + [True] * (len(ops)-1)
    assert all(not op["closed"] and op["src"]["closed"] for op in ops)
    assert ops[0]["points_m"][0] == ops[-1]["points_m"][-1]
    assert ops[-1]["src"]["arc_m"][1] == pytest.approx(whole["ops"][1]["src"]["arc_m"][1])
    motion = load_motion()
    for op in ops:
        assert drawing_seconds(op, split["draw_speed_m_s"], motion) <= 2 + 1e-10
        assert op["planned_drawing_s"] == drawing_seconds(op, split["draw_speed_m_s"], motion)
    for a, b in zip(ops, ops[1:], strict=False):
        assert a["points_m"][-1] == b["points_m"][0]
        assert a["src"]["arc_m"][1] == b["src"]["arc_m"][0]


def test_multicolour_requires_exact_bindings_and_keeps_colour_sequence(tmp_path):
    art = artwork(program([(pen, [element([[.002, y], [.018, y]])]) for pen, y in
                           [("black", .002), ("red", .01), ("black", .018)]],
                          inks={"black": [0, 0, 0], "red": [1, 0, 0]}))
    path = write(tmp_path, art)
    with pytest.raises(CompileError, match="explicit"):
        tatbot_ink.compile(path, repo=REPO)
    import shutil

    from resource_fixtures import fixture_repo, manifest, resource
    from tatbot_ink.input import read_input

    repo = fixture_repo(tmp_path/'repo')
    motion = repo/'ros/tatbot_motion/config/motion.yaml'
    motion.parent.mkdir(parents=True)
    shutil.copyfile(REPO/'ros/tatbot_motion/config/motion.yaml', motion)
    inkfile = manifest(tmp_path/'inks.yaml', read_input(art),
                       [resource('B'), resource('R', ink='red')], [('black', 'B'), ('red', 'R')])
    result = tatbot_ink.compile(path, repo=repo, inks_path=inkfile)
    assert [op['op'] for op in result['ops']] == ['tool_change', 'stroke', 'tool_change', 'stroke', 'tool_change', 'stroke']
    assert [op['resource_id'] for op in result['ops'] if op['op'] == 'stroke'] == ['B', 'R', 'B']
    assert result['stats']['dips'] == 0 and result['stats']['tool_changes'] == 3
    assert result['stats']['time_estimate']['unknown_operations'] == {'pen_change': 3}
    assert result["stats"]["time_estimate"]["total_s"] is None


def test_inlumino_pack_shares_geometry_and_preserves_color_exchanges_without_dips(tmp_path):
    import yaml
    from resource_fixtures import manifest, resource
    from tatbot_ink.input import read_input

    stock = yaml.safe_load((REPO/'config/inventory.yaml').read_text())['cartridges']
    colors = [row['ink'] for row in stock.values() if row.get('fits') == 'lutin-ballpoint-dot' and row.get('ink')]
    assert len(colors) == 5
    # Return to the first color: identical geometry must not merge pigments or
    # erase the last exchange. Native pen display colors deliberately all match.
    pens = [f'native-{i}' for i in range(5)]
    art = artwork(program([(pen, [element([[.002, .002+i*.002], [.018, .002+i*.002]])])
                           for i, pen in enumerate(pens+[pens[0]])]))
    path = write(tmp_path, art)
    inkfile = manifest(tmp_path/'inks.yaml', read_input(art),
                       [resource(f'color-{i}', tool='lutin-ballpoint-dot', ink=ink) for i, ink in enumerate(colors)],
                       [(pen, f'color-{i}') for i, pen in enumerate(pens)], substrate='paper_pad')
    result = tatbot_ink.compile(path, repo=REPO, inks_path=inkfile)
    rows = result['resources']
    assert [row['ink_id'] for row in rows] == colors
    assert all(row['tool'] == rows[0]['tool'] for row in rows)
    assert rows[0]['tool']['id'] == 'lutin-ballpoint-dot'
    assert len({row['ink_id'] for row in rows}) == 5
    assert all(row['activation'] == {'method': 'manual', 'action': 'exchange'}
               and row['slot'] is row['dip'] is None for row in rows)
    assert [op['resource_id'] for op in result['ops'] if op['op'] == 'tool_change'] == [f'color-{i}' for i in range(5)]+['color-0']
    assert [op['ink'] for op in result['ops'] if op['op'] == 'stroke'] == colors+[colors[0]]
    assert result['stats']['dips'] == 0 and result['stats']['tool_changes'] == 6


def test_size_change_and_unbound_or_tampered_artwork_refused(tmp_path):
    art = line_art()
    with pytest.raises(CompileError, match="regenerate"):
        tatbot_ink.compile(write(tmp_path, art), repo=REPO, width_m=.026)
    corrupted = copy.deepcopy(art)
    corrupted["program"]["layers"][0]["elements"][0]["points_m"][0][0] += .001
    with pytest.raises(CompileError, match="digest"):
        tatbot_ink.compile(write(tmp_path, corrupted), repo=REPO)
    corrupted = copy.deepcopy(art)
    corrupted["conversion"]["adapter"] = "public-svg"
    corrupted["content_sha256"] = canonical_digest(corrupted)
    with pytest.raises(CompileError, match="DBV3"):
        tatbot_ink.compile(write(tmp_path, corrupted), repo=REPO)
    with pytest.raises(CompileError, match="clear area"):
        tatbot_ink.compile(write(tmp_path, art), repo=REPO, at_m=[.03, .05])


def test_placed_design_is_verified_before_overrides(tmp_path):
    art = line_art()
    doc = design([("p", art["program"], placement((.02, .02), mirrored=True))])
    result = tatbot_ink.compile(write(tmp_path, doc), repo=REPO, at_m=[.01, .01], width_m=.02)
    assert result["ops"][1]["points_m"][0][0] > result["ops"][1]["points_m"][-1][0]
    doc["placements"][0]["placement"]["target"]["anchor_uv_m"] = [.005, .005]
    with pytest.raises(CompileError, match="digest"):
        tatbot_ink.compile(write(tmp_path, doc), repo=REPO, at_m=[0, 0])


def test_generation_width_is_not_a_measurement_or_a_geometry_edit(tmp_path, monkeypatch):
    import tatbot_ink.program as adapter

    path = write(tmp_path, line_art(width=.0003))
    result = tatbot_ink.compile(path, repo=REPO, tool_id="lutin-ballpoint-dot")
    assert result["resources"][0]["tool"]["line_width_status"] == "measured"
    assert result["resources"][0]["tool"]["line_width_m"] == .0005
    assert result["preparation"]["generation_widths_m"] == [.0003]
    assert result["stats"]["notes"]
    motion = copy.deepcopy(load_motion())
    motion['pen']['mode'] = 'ride'
    monkeypatch.setattr(adapter, 'load_motion', lambda *args: copy.deepcopy(motion))
    with pytest.raises(CompileError, match='riding tool stroke_mm'):
        tatbot_ink.compile(path, repo=REPO, arm='left')
    motion['pen']['mode'] = 'press'
    prop = tatbot_ink.compile(path, repo=REPO, arm="left")
    assert prop["resources"][0]["tool"]["line_width_status"] == "unknown" and prop["resources"][0]["tool"]["line_width_m"] is None
    assert prop["ops"][1]["points_m"] == result["ops"][1]["points_m"]
    assert 'generation width' in tatbot_ink.write_preview(prop, tmp_path/'preview.svg').read_text()


def test_cli_prepares_native_artwork(tmp_path, capsys):
    assert main(["compile", str(write(tmp_path, line_art())), "--speed", "5", "--width", "20",
                 "--repo", str(REPO), "-o", str(tmp_path/'out')]) == 0
    assert json.loads(capsys.readouterr().out)["paths"] == 1
    result = json.loads((tmp_path/'out/program.json').read_text())
    assert result["draw_speed_m_s"] == .005
    assert (tmp_path/'out/preview.svg').is_file()


def test_unmeasured_profile_and_dipping_are_explicit(tmp_path):
    from tatbot_ink.tools import load_tool

    tools = tmp_path / "config/tools"
    tools.mkdir(parents=True)
    sheet = tools / "cartridge.yaml"
    sheet.write_text("line: {width_mm: 0.35, status: assumed}\nink: {mode: cartridge}\n")
    profile = load_tool(tmp_path, "right", "cartridge")
    assert profile["line_width_status"] == "assumed" and profile["line_width_m"] == .00035
    sheet.write_text("ink: {mode: cartridge}\n")
    profile = load_tool(tmp_path, "right", "cartridge")
    assert profile["line_width_status"] == "unknown" and profile["line_width_m"] is None
    sheet.write_text("ink: {mode: dip}\n")
    with pytest.raises(CompileError, match="explicit resource naming its cap"):
        load_tool(tmp_path, "right", "cartridge")


def test_both_round_needle_groupings_publish_qualified_dips_and_resource_pen_modes(tmp_path):
    import shutil

    from resource_fixtures import fixture_repo, manifest, resource
    from tatbot_contracts.ros_program import validate_for_execution
    from tatbot_ink.input import read_input

    art = artwork(program([('outline', [element([[.002, .005], [.018, .005]])]),
                           ('shade', [element([[.002, .015], [.018, .015]])])]))
    repo = fixture_repo(tmp_path/'repo')
    motion = repo/'ros/tatbot_motion/config/motion.yaml'
    motion.parent.mkdir(parents=True)
    shutil.copyfile(REPO/'ros/tatbot_motion/config/motion.yaml', motion)
    bindings = manifest(tmp_path/'resources.yaml', read_input(art),
                        [resource('outline', tool='round-3'), resource('shade', tool='round-5', mode='ride')],
                        [('outline', 'outline'), ('shade', 'shade')], substrate='practice')
    result = tatbot_ink.compile(write(tmp_path, art), repo=repo, inks_path=bindings, max_segment_s=2)
    validate_for_execution(result)
    assert result['stats']['tool_changes'] == 2 and result['stats']['dips'] == 4
    assert [row['pen_mode'] for row in result['resources']] == ['press', 'ride']
    assert result['stats']['time_estimate']['total_s'] is None
    assert all(op['slot'] == next(row['slot'] for row in result['resources'] if row['id'] == op['resource_id'])
               for op in result['ops'] if op['op'] == 'dip')
