"""Physical frames, size and footprint bounds survive rigid placement."""
import json
import math

import numpy as np
import pytest
from ink_designs import artwork, design, element, placement, program
from tatbot_description import repo_root
from tatbot_ink import CompileError
from tatbot_ink.input import read_input
from tatbot_ink.place import place_points, placed_strokes, print_page


def test_rotation_mirror_and_page_y_follow_inkmap():
    canvas = (.02, .02)
    p = placement(canvas, anchor=(.01, -.02))
    np.testing.assert_allclose(place_points([[.02, .02]], canvas, p), [[.02, -.01]])
    p = placement(canvas, rotation=math.pi/2, mirrored=True)
    np.testing.assert_allclose(place_points([[.02, .01]], canvas, p), [[0, -.01]], atol=1e-15)
    with pytest.raises(CompileError, match="regenerate"):
        place_points([[.02, .01]], canvas, placement((.04, .04)))


def test_canvas_height_and_line_footprint_must_fit_clear_area():
    prog = program([("pen", [element([[0, .01], [.02, .01]])])])
    doc = read_input(design([("p", prog, placement((.02, .02), anchor=(.035, 0)))]))
    with pytest.raises(CompileError, match="footprint"):
        placed_strokes(doc, .0005)
    tall = artwork(program([("pen", [element([[.002, .05], [.018, .05]])])], canvas=(.02, .14)))
    with pytest.raises(CompileError, match="canvas"):
        placed_strokes(read_input(tall), .0005)


def test_a_generated_print_is_the_page_its_design_must_fit(tmp_path):
    """--stencil: the program's page is the print's, with its clear centre; a design the nominal 128 x 172 mm page
    holds can leave a half-skin print's centre."""
    (tmp_path / "tracking.json").write_text(json.dumps({
        "pattern_id": "stencil-half", "page_mm": [83, 127], "clear_center_uv": [19 / 83, 19 / 127, 64 / 83, 108 / 127]}))
    page = print_page(tmp_path / "tracking.json", repo_root(None))
    assert page == {"kind": "stencil", "size_m": [0.083, 0.127], "clear_m": [0.045, 0.089], "margin_m": 0.005}
    wide = read_input(artwork(program([("pen", [element([[.001, .01], [.045, .01]])])], canvas=(.046, .02))))
    assert placed_strokes(wide, .0005)[1]["clear_m"] == [0.090, 0.134]
    with pytest.raises(CompileError, match="canvas"):
        placed_strokes(wide, .0005, page=page)
    narrow = read_input(artwork(program([("pen", [element([[.001, .01], [.039, .01]])])], canvas=(.04, .02))))
    assert placed_strokes(narrow, .0005, page=page)[1] == page
    # a print whose band is shallower than the 5 mm margin keeps its whole clear centre
    (tmp_path / "tracking.json").write_text(json.dumps({
        "pattern_id": "stencil-thin", "page_mm": [40, 60], "clear_center_uv": [4 / 40, 4 / 60, 36 / 40, 56 / 60]}))
    thin = print_page(tmp_path, repo_root(None))
    assert thin["clear_m"] == [0.032, 0.052] and thin["margin_m"] == 0.004
    square = read_input(artwork(program([("pen", [element([[.001, .001], [.031, .031]])])], canvas=(.032, .032))))
    assert placed_strokes(square, .0005, page=thin)[1] == thin
    (tmp_path / "empty").mkdir()
    with pytest.raises(CompileError, match="not a generated stencil print"):
        print_page(tmp_path / "empty", repo_root(None))


def test_acquired_layers_stack_on_one_origin_and_move_together():
    """One DBV3 acquisition per pen, drawn in order without composing a design by hand (2026-10-13)."""
    from tatbot_ink.input import read_layers
    from tatbot_ink.place import relocate

    first = artwork(program([("ink", [element([(0, 0), (0.01, 0)])])]))
    second = artwork(program([("ink", [element([(0, 0.01), (0.01, 0.01)])])]))
    stacked = read_layers([first, second])
    assert [item["id"] for item in stacked["placements"]] == ["layer1", "layer2"]
    moved = relocate(stacked, at_m=[0.005, -0.01])
    assert {tuple(item["placement"]["target"]["anchor_uv_m"]) for item in moved["placements"]} == {(0.005, -0.01)}
    with pytest.raises(CompileError, match="one canvas size"):
        read_layers([first, artwork(program([("ink", [element([(0, 0), (0.01, 0)])])], canvas=(0.03, 0.03)))])


def _stroke(points, path):
    from tatbot_ink.place import Stroke

    return Stroke(np.asarray(points, float), False, .0003, {"placement": "p", "layer": "l", "path": path})


def test_a_keep_region_keeps_its_paths_as_placed_and_cuts_one_drawn_on_both_sides():
    """A part of a design drawn again over its own (2026-10-21: one side of a forearm portrait): nothing moves; a
    crossing path goes with most of its length, unless both sides hold over KEEP_CUT_M of it."""
    from tatbot_ink.place import keep_region

    right = [[0, -.03], [.03, -.03], [.03, .03], [0, .03]]
    outline = [[-.025, -.02], [.025, -.02], [.025, -.01], [-.025, -.01]]     # 50 mm out, 60 in: cut
    wiggle = [[-.025, .02], [.0005, .02], [-.025, .021], [-.025, .022], [.025, .022]]   # its 0.5 mm in is a dot
    strokes = [_stroke([[.005, 0], [.015, 0]], "in"), _stroke([[-.015, 0], [-.005, 0]], "out"),
               _stroke([[-.002, .005], [.018, .005]], "tail"), _stroke([[-.018, .01], [.002, .01]], "mostly-out"),
               _stroke(outline, "outline"), _stroke(wiggle, "wiggle"), _stroke([[.01, .015]], "dot")]
    kept, record = keep_region(strokes, right)
    assert [s.src["path"] for s in kept] == ["in", "tail", "outline", "wiggle", "dot"]
    assert kept[0] is strokes[0] and kept[1] is strokes[2]
    np.testing.assert_array_equal(kept[2].points_m, [[0, -.02], [.025, -.02], [.025, -.01], [0, -.01]])
    assert kept[2].arc0_m == pytest.approx(.025) and not kept[2].closed
    np.testing.assert_allclose(kept[3].points_m, [[0, .022], [.025, .022]], atol=1e-15)
    assert kept[3].arc0_m == pytest.approx(.0255 + np.hypot(.0255, .001) + .001 + .025)
    assert record["paths"] == {"inside": 2, "outside": 1}
    assert [(c["path"], c["decision"], c["pieces"]) for c in record["crossing"]] == [
        ("tail", "kept", 1), ("mostly-out", "dropped", 0), ("outline", "cut", 1), ("wiggle", "cut", 1)]
    assert record["crossing"][2]["inside_m"] == pytest.approx(.06)
    with pytest.raises(CompileError, match="none of the design"):
        keep_region(strokes[1:2], right)
    with pytest.raises(CompileError, match="polygon"):
        keep_region(strokes, right[:2])


def test_a_kept_region_compiles_on_the_whole_designs_placement(tmp_path):
    import tatbot_ink
    from ink_designs import REPO, write

    zigzag = [[.004, .018], [.016, .018], [.004, .017], [.016, .016], [.004, .015]]   # 24 mm each side of x = 0
    lines = [element([[.004, .01], [.018, .01]]), element([[.002, .004], [.008, .004]]), element(zigzag)]
    path = write(tmp_path, artwork(program([("pen", lines)])))
    whole = tatbot_ink.compile(path, repo=REPO)
    half = tatbot_ink.compile(path, repo=REPO, keep_region_m=[[0, -.01], [.01, -.01], [.01, .01], [0, .01]])
    ops = [op for op in half["ops"] if op["op"] == "stroke"]
    assert [op["src"]["path"] for op in ops] == ["path-0-0", "path-0-2", "path-0-2"]
    assert ops[0]["points_m"] == next(op["points_m"] for op in whole["ops"] if op["op"] == "stroke")
    np.testing.assert_allclose(ops[1]["points_m"], [[0, .008], [.006, .008], [0, .0075]], atol=1e-15)
    assert ops[1]["src"]["arc_m"] == pytest.approx([.006, .012 + np.hypot(.006, .0005)])
    assert [c["decision"] for c in half["preparation"]["keep_region"]["crossing"]] == ["kept", "cut"]
    assert half["stats"]["strokes"] == 3 and not any(op["continues"] for op in ops)
    assert '<polygon id="keep-region"' in tatbot_ink.write_preview(half, tmp_path / "p.svg").read_text()
    assert "keep_region" not in whole["preparation"]
