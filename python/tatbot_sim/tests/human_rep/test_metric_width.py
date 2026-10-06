"""Physical pen widths survive artwork placement; geometry alone is resized."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from tatbot_sim import tools
from tatbot_sim.human_rep.contracts import canonical_digest, load_contract
from tatbot_sim.human_rep.ink_program import (
    compile_ink_program,
    material_strokes,
    scheduled_material_strokes,
)
from tatbot_sim.inkmap.program_target import render_program_target
from tatbot_sim.repo import repo_root

ROOT = repo_root()
EXAMPLES = ROOT / "config/human-representation/examples"


def artwork(*, dots=False, fill=False):
    program = load_contract(EXAMPLES / "linework/program.json")
    line = program["layers"][0]["elements"][0]
    line.update(points_m=[[.01, .03], [.05, .03]], width_m=.0005, deposition=1)
    program["layers"][0]["elements"] = [line]
    if dots:
        program["layers"][0]["elements"].append({**deepcopy(line), "id": "dot", "kind": "dots", "points_m": [[.03, .02]]})
    if fill:
        program["layers"][0]["elements"] = [{**deepcopy(line), "kind": "region", "closed": True, "fill": True,
                                              "points_m": [[.02, .02], [.04, .02], [.04, .04], [.02, .04]]}]
    program["content_sha256"] = canonical_digest(program)
    return program


def placed(program, scale, target="plane"):
    filename = "body-placement-v2" if target == "body" else f"{target}-placement"
    placement = load_contract(EXAMPLES / f"{filename}.json")
    placement.update(tattoo_program_sha256=program["content_sha256"], physical_scale_m=[.06 * scale, .06 * scale],
                     rotation_rad=0, mirrored=False)
    if target != "body":
        placement["target"].update(canvas_m=[.2, .2], anchor_uv_m=[0, 0], margin_m=.001)
        if target == "cylinder":
            placement["target"]["radius_m"] = .06
    placement["content_sha256"] = canonical_digest(placement)
    return placement


@pytest.mark.parametrize("target", ["plane", "cylinder", "body"])
@pytest.mark.parametrize("scale", [.4, 1, 2])
@pytest.mark.parametrize("tool_width", [None, .0007])
def test_resizing_moves_geometry_but_keeps_line_and_dot_widths(target, scale, tool_width):
    program = artwork(dots=True)
    original = deepcopy(program)
    strokes = material_strokes(program, placed(program, scale, target), tool_width_m=tool_width)
    width = .0005 if tool_width is None else tool_width
    assert [stroke.width_m for stroke in strokes] == pytest.approx([width, width])
    np.testing.assert_allclose(strokes[0].points_m, [[-.02 * scale, 0], [.02 * scale, 0]], atol=1e-15)
    dot_centre = [0, -.01 * scale]
    assert np.linalg.norm(strokes[1].points_m - dot_centre, axis=1) == pytest.approx(width / 2)
    assert program == original


def test_known_tool_controls_fill_spacing_and_larger_art_needs_more_ink():
    program = artwork(fill=True)
    plans = [material_strokes(program, placed(program, scale), tool_width_m=.0007) for scale in (.4, 1)]
    lengths = [sum(np.linalg.norm(np.diff(s.points_m, axis=0), axis=1).sum() for s in strokes) for strokes in plans]
    assert lengths[1] > lengths[0] * 4  # Area grows 6.25x; the pen did not grow with it.
    for strokes in plans:
        assert all(s.width_m == .0007 for s in strokes)
    # The public scheduled path must apply the tool before planning a fill,
    # not just use it afterwards when estimating overlap.
    scheduled = scheduled_material_strokes(program, placed(program, .4), footprint_width_m=.0007)
    assert scheduled and all(s.width_m == .0007 for s in scheduled)


@pytest.mark.parametrize("scale", [.4, 1, 2])
@pytest.mark.parametrize("tool_width", [None, .001])
def test_reference_raster_keeps_the_same_metric_line_thickness(scale, tool_width):
    program = artwork()
    image = render_program_target(program, placed(program, scale), tool_width_m=tool_width)
    # The middle column intersects the straight part, not the rounded caps.
    measured = image.coverage[:, image.cols // 2].sum() / image.pixels_per_m
    assert measured == pytest.approx(tool_width or .0005, abs=1 / (image.pixels_per_m * image.supersample))
    assert image.width_m == pytest.approx(.06 * scale)
    assert image.tool_width_m == tool_width


def test_compiled_ink_uses_the_fitted_pen():
    # The ROS stack no longer compiles tattoo programs (it draws frozen DBV3
    # paths), so there is no second centerline implementation to agree with.
    program = artwork(dots=True)
    # Deliberately different acquisition width: the fitted tool wins.
    for element in program["layers"][0]["elements"]:
        element["width_m"] = .0008
    program["content_sha256"] = canonical_digest(program)
    tool = tools.registry().load_tool("lutin-ballpoint-dot", ROOT)
    for scale in (.4, 1):
        placement = placed(program, scale)
        ink = compile_ink_program(program, placement, tool=tool, provenance=placement["provenance"])
        events = [event for event in ink["events"] if event["kind"] == "stroke"]
        assert events and all(event["curve"]["width_m"] == tool.line_width_m for event in events)


@pytest.mark.parametrize("width", [0, -0.1, float("inf"), float("nan")])
def test_invalid_tool_widths_refuse_before_material_or_target_generation(width):
    program = artwork()
    placement = placed(program, 1)
    with pytest.raises(ValueError, match="positive finite line width"):
        material_strokes(program, placement, tool_width_m=width)
    with pytest.raises(ValueError, match="positive finite line width"):
        render_program_target(program, placement, tool_width_m=width)
