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
    doc = read_input(design([("p", prog, placement((.02, .02), anchor=(.021, 0)))]))
    with pytest.raises(CompileError, match="footprint"):
        placed_strokes(doc, .0005)
    tall = artwork(program([("pen", [element([[.002, .05], [.018, .05]])])], canvas=(.02, .12)))
    with pytest.raises(CompileError, match="canvas"):
        placed_strokes(read_input(tall), .0005)


def test_a_generated_print_is_the_page_its_design_must_fit(tmp_path):
    """--stencil: the program's page is the print's, its clear centre the largest about the page centre inside the
    border's innermost ink; a design the nominal 100 x 150 mm page holds can leave a half-skin print's centre."""
    (tmp_path / "tracking.json").write_text(json.dumps({
        "pattern_id": "stencil-half", "page_mm": [83, 127], "generator": {"svg_sha256": "abc"},
        "clear_center_uv": [19 / 83, 19 / 127, 64 / 83, 108 / 127]}))
    (tmp_path / "settings.json").write_text(json.dumps({"artwork_svg_sha256": "abc",
                                                        "border_inner_mm": [19.897, 19.643, 63.077, 107.357]}))
    page = print_page(tmp_path / "settings.json", repo_root(None))
    assert page == {"kind": "stencil", "size_m": [0.083, 0.127], "clear_m": [0.043154, 0.087714], "margin_m": 0.005}
    wide = read_input(artwork(program([("pen", [element([[.001, .01], [.045, .01]])])], canvas=(.046, .02))))
    assert placed_strokes(wide, .0005)[1]["clear_m"] == [0.062, 0.112]
    with pytest.raises(CompileError, match="canvas"):
        placed_strokes(wide, .0005, page=page)
    narrow = read_input(artwork(program([("pen", [element([[.001, .01], [.039, .01]])])], canvas=(.04, .02))))
    assert placed_strokes(narrow, .0005, page=page)[1] == page
    (tmp_path / "empty").mkdir()
    with pytest.raises(CompileError, match="not a generated stencil print"):
        print_page(tmp_path / "empty", repo_root(None))
