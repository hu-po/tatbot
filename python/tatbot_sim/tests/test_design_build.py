"""Headless image/SVG to portable design: the browser's tracer, reader and digests.

The disc below is the same shape `web/inkmap/tests/trace.test.ts` traces, built
from the same formula. Both suites therefore trace one known raster through one
implementation, and the pinned digest here fires if the vendored vtracer wasm,
the threshold, or either caller's pixel handling moves.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import numpy as np
import pytest
from tatbot_contracts.artwork import canvas_m, max_width_m
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest
from tatbot_sim.human_rep.ink_program import InkProgramError
from tatbot_sim.inkmap.design import scan_coverage, validate_design
from tatbot_sim.inkmap.design_build import (
    DEFAULT_WIDTH_MM,
    DesignBuildError,
    artwork_from_svg,
    cylinder_placement,
    design_from_artwork,
    file_source,
    fit_size_mm,
    generated_source,
    plane_placement,
    trace_image,
    write_json,
)

# The traced disc, byte for byte. A vendored-tracer bump is a deliberate change:
# re-pin this next to the TRACE_ALGORITHM tag, never silently.
DISC_SVG_SHA256 = "de6a9722915199344640b35adf0639be1e5b3b9f49a2df78ade700de9ff472f2"


def disc_png(size: int = 128, radius: int = 40, ink: int = 20, paper: int = 235) -> bytes:
    """web/inkmap/tests/trace.test.ts `disc()`, as a PNG."""
    import cv2

    y, x = np.mgrid[0:size, 0:size]
    inside = (x - size / 2) ** 2 + (y - size / 2) ** 2 < radius**2
    gray = np.where(inside, ink, paper).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert ok
    return buffer.tobytes()


@pytest.fixture(scope="module")
def traced() -> tuple[str, dict]:
    return trace_image(disc_png())


@pytest.fixture(scope="module")
def record(traced) -> dict:
    svg, trace = traced
    return artwork_from_svg(svg, name="Disc", size_mm=fit_size_mm(tuple(trace["size_px"]), (60.0, 90.0)),
                            source=file_source(svg, path="disc.png", trace=trace))


def test_the_traced_disc_is_the_browsers_traced_disc(traced):
    svg, trace = traced
    assert svg.count("<path") == 1 and 'fill="#111111"' in svg
    assert 80 <= trace["size_px"][0] <= 100 and 80 <= trace["size_px"][1] <= 100
    assert trace["algorithm"] == "inkmap-vtracer-otsu-v2"
    assert 20 <= trace["threshold"] < 235
    assert abs(trace["coverage"] - math.pi * 40 * 40 / 128**2) < 0.01
    assert hashlib.sha256(svg.encode()).hexdigest() == DISC_SVG_SHA256


def test_only_png_and_jpeg_reach_the_tracer():
    for payload in (b"", b"GIF89a" + b"\0" * 64, b"<svg/>", b"\x89PNG\r\n\x1a\n" + b"\0" * 32):
        with pytest.raises(DesignBuildError):
            trace_image(payload)


def test_fit_size_keeps_the_aspect_ratio_inside_the_box():
    assert fit_size_mm((100, 200), (60, 90)) == (45.0, 90.0)
    assert fit_size_mm((200, 100), (60, 90)) == (60.0, 30.0)
    for bad in ((0, 10), (10, float("nan"))):
        with pytest.raises(DesignBuildError):
            fit_size_mm(bad, (60, 90))
        with pytest.raises(DesignBuildError):
            fit_size_mm((100, 100), bad)


def test_an_svg_from_disk_keeps_its_own_provenance(record):
    assert record["source"]["kind"] == "imported"
    assert record["source"]["generation"] is None
    identifier = json.loads(record["source"]["identifier"])
    assert identifier["path"] == "disc.png"
    assert identifier["svg_sha256"] == record["source_sha256"]
    assert record["conversion"]["adapter"] == "tatbot-svg-paint/1"
    assert max_width_m(record) == 0.0003


def test_a_generated_artwork_records_prompt_model_seed_and_tracer(traced):
    svg, _ = traced
    record = artwork_from_svg(svg, name="Swallow", size_mm=(40.0, 40.0),
                              source=generated_source(prompt="tattoo flash design of a swallow",
                                                      model="Tongyi-MAI/Z-Image-Turbo", seed=42))
    generation = record["source"]["generation"]
    assert record["source"]["kind"] == "generated"
    assert generation["seed"] == 42 and generation["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert generation["tracing"] == "inkmap-vtracer-otsu-v2"


def test_cylinder_design_round_trips_through_the_browser_reader(record):
    design = design_from_artwork(record, name="Disc on a cylinder", kind="cylinder",
                                 radius_m=0.04, canvas_m=(0.08, 0.11))
    assert design["schema"] == "tatbot.inkmap-design/1"
    # The browser reader is the judge, and its digest must be Python's digest.
    assert validate_design(design) == design
    assert canonical_digest(design) == design["content_sha256"]
    placement = design["placements"][0]["placement"]
    assert placement["target"]["kind"] == "cylinder"
    assert placement["target"]["radius_m"] == 0.04
    assert placement["tattoo_program_sha256"] == record["program"]["content_sha256"]


def test_scan_coverage_reads_the_cylinder_design(record):
    design = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    coverage = scan_coverage(design)
    assert coverage["design_sha256"] == design["content_sha256"]
    assert coverage["target"]["kind"] == "cylinder" and coverage["target"]["radius_m"] == 0.04
    assert coverage["material_strokes"] >= 1
    # The disc fits inside its own placement: a footprint bounded by the artwork.
    assert 0 < coverage["radius_m"] <= max(canvas_m(record))


def test_mixed_chart_geometry_is_refused(record):
    plane = plane_placement(record)
    cylinder = cylinder_placement(record, radius_m=0.04, canvas_m=(0.08, 0.11))
    from tatbot_sim.inkmap.design_build import design_from_placements
    design = design_from_placements("Mixed", [("flat", record, plane), ("round", record, cylinder)])
    with pytest.raises(ValueError, match="one target chart geometry"):
        scan_coverage(design)


def test_artwork_larger_than_its_canvas_is_refused(record):
    # Two independent refusals stand behind this: the placement contract checks
    # the rotated canvas, and every compiled stroke then goes through the
    # target's own margin rule. Either is the refusal preparation would raise.
    with pytest.raises((ContractError, InkProgramError), match="outside_domain|footprint|margin"):
        cylinder_placement(record, radius_m=0.035, canvas_m=(0.01, 0.01))


def test_a_margin_that_no_longer_fits_is_refused(record):
    canvas = tuple(1.4 * v for v in canvas_m(record))
    plane_placement(record, canvas_m=canvas, margin_m=0.001)
    with pytest.raises((ContractError, InkProgramError), match="outside_domain|footprint|margin"):
        plane_placement(record, canvas_m=canvas, margin_m=0.05)


def test_a_chart_closing_the_full_circumference_is_refused(record):
    """Three quarters of the way round is a chart (the paper cylinder's band);
    the full circumference closes on itself and is not."""
    radius = 0.035
    circumference = 2 * math.pi * radius
    cylinder_placement(record, radius_m=radius, canvas_m=(0.08, 0.75 * circumference))
    with pytest.raises(DesignBuildError, match="full"):
        cylinder_placement(record, radius_m=radius, canvas_m=(0.08, circumference))


@pytest.mark.parametrize("radius", [0, -0.01, float("inf"), None])
def test_a_cylinder_needs_a_real_radius(record, radius):
    with pytest.raises(DesignBuildError):
        cylinder_placement(record, radius_m=radius, canvas_m=(0.08, 0.11))


def test_unusable_sizes_and_names_are_refused(traced):
    svg, _ = traced
    for size in ((0, 10), (10, -1), (float("nan"), 10)):
        with pytest.raises(DesignBuildError):
            artwork_from_svg(svg, name="X", size_mm=size, source=file_source(svg))
    with pytest.raises(DesignBuildError):
        artwork_from_svg(svg, name="  ", size_mm=(10, 10), source=file_source(svg))
    with pytest.raises(DesignBuildError):
        artwork_from_svg(svg, name="X", size_mm=(10, 10), source=file_source(svg), width_mm=3)


def test_trace_planning_width_comes_from_the_tool_line_when_not_given():
    """`design trace --tool ID` plans at the datasheet's recorded line; a tool
    that records none refuses rather than falling back to 0.3 mm silently."""
    import argparse

    from tatbot_sim.inkmap.design_cli import _planning_width_mm, tool_line_width_mm

    width, line = tool_line_width_mm("lutin-ballpoint-dot")
    assert width == pytest.approx(0.5) and line["status"] in ("measured", "assumed")
    assert _planning_width_mm(argparse.Namespace(width_mm=None, tool="lutin-ballpoint-dot"))[0] == pytest.approx(width)
    assert _planning_width_mm(argparse.Namespace(width_mm=0.35, tool="lutin-ballpoint-dot")) == (0.35, "--width-mm")
    assert _planning_width_mm(argparse.Namespace(width_mm=None, tool=None)) == (DEFAULT_WIDTH_MM, "default")
    with pytest.raises(DesignBuildError, match="records no `line:` width"):
        tool_line_width_mm("picosecond-laser-pen")
    with pytest.raises(DesignBuildError, match="cannot load tool"):
        tool_line_width_mm("no-such-tool")


def test_a_detached_placement_cannot_be_rehashed_into_a_design(record):
    design = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    broken = deepcopy(design)
    broken["placements"][0]["placement"]["tattoo_program_sha256"] = "a" * 64
    broken["placements"][0]["placement"]["content_sha256"] = canonical_digest(broken["placements"][0]["placement"])
    broken["content_sha256"] = canonical_digest(broken)
    with pytest.raises(ContractError):
        validate_design(broken)


def test_designs_are_written_once(record, tmp_path):
    design = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    path = write_json(tmp_path / "nested" / "design.json", design)
    assert json.loads(path.read_text())["content_sha256"] == design["content_sha256"]
    with pytest.raises(DesignBuildError, match="refusing to overwrite"):
        write_json(path, design)


def test_the_browsers_fixture_is_exactly_what_this_builder_writes():
    """web/inkmap/tests/design.test.ts opens this file; regenerate it here.

    Two suites, one fixture: the TS reader proves it parses, and this proves the
    Python builder still writes those bytes. Neither can drift alone.
    """
    from tatbot_sim.repo import repo_root

    record = json.loads((repo_root() / "web/inkmap/public/designs/dbv3-orbit/artwork.json").read_text())
    design = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    path = repo_root() / "web/inkmap/tests/fixtures/python-cylinder-design.json"
    assert path.read_text() == json.dumps(design, indent=2) + "\n", (
        f"regenerate {path} from this builder")


def test_design_strokes_exports_the_material_centerlines(record):
    """`tatbot design strokes`: the same material strokes session preparation compiles, as chart
    millimetres about chart zero, with the footprint scan_coverage reports."""
    from tatbot_sim.inkmap.design import scan_coverage
    from tatbot_sim.inkmap.design_strokes import SCHEMA, design_strokes

    design = design_from_artwork(record, kind="cylinder", radius_m=0.04, canvas_m=(0.08, 0.11))
    exported = design_strokes(design)
    coverage = scan_coverage(design)
    assert exported["schema"] == SCHEMA
    assert exported["design_sha256"] == design["content_sha256"]
    assert exported["target"] == coverage["target"]
    assert exported["stroke_count"] == coverage["material_strokes"] == len(exported["strokes_mm"])
    assert exported["stroke_placement_ids"] == [design["placements"][0]["id"]] * exported["stroke_count"]
    assert abs(exported["radius_mm"] + exported["stroke_width_mm"] / 2 - coverage["radius_m"] * 1000) < 1e-6
    for stroke in exported["strokes_mm"]:
        assert len(stroke) >= 2 and all(len(point) == 2 for point in stroke)
    assert exported["length_mm"] > 0


def test_centerline_strokes_draw_a_line_once_at_the_planning_width():
    """`design trace --strokes centerline`: a stroked SVG path is one open tier-0
    stroke per subpath at the planning width, not a ring the planner draws
    around it; the default mode is unchanged and records no `strokes` field."""
    from tatbot_sim.inkmap.design_strokes import design_strokes

    # A 30 mm horizontal line and a closed 10 mm square, both 2-unit strokes.
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50" fill="none" stroke="#111" stroke-width="2">'
           '<path d="M5 10 L35 10 M10 25 L20 25 L20 35 L10 35 Z"/></svg>')
    source = file_source(svg, path="lines.svg")
    lines = artwork_from_svg(svg, name="Lines", size_mm=(50.0, 50.0), source=source, width_mm=0.5,
                             strokes="centerline")
    outline = artwork_from_svg(svg, name="Lines", size_mm=(50.0, 50.0), source=source, width_mm=0.5)
    assert lines["program"]["content_sha256"] != outline["program"]["content_sha256"]
    elements = [element for layer in lines["program"]["layers"] for element in layer["elements"]]
    assert [(e["kind"], e["fill"], e["closed"], e["width_m"]) for e in elements] == [
        ("path", False, False, 0.0005), ("path", False, True, 0.0005)]
    exported = design_strokes(design_from_artwork(lines))
    assert exported["stroke_count"] == 2 and exported["stroke_tiers"] == [0, 0]
    assert exported["stroke_placement_ids"] == ["placement-1", "placement-1"]
    from tatbot_sim.inkmap.design_build import design_from_placements

    placement = plane_placement(lines)
    twice = design_strokes(design_from_placements("Two printed placements", [
        ("first-print", lines, placement), ("second-print", lines, placement)]))
    assert twice["stroke_placement_ids"] == ["first-print"] * 2 + ["second-print"] * 2
    assert exported["length_mm"] == pytest.approx(30 + 40, abs=0.01)
    ringed = design_strokes(design_from_artwork(outline))
    assert all(element["kind"] == "region" and element["fill"]
               for layer in outline["program"]["layers"] for element in layer["elements"])
    assert ringed["length_mm"] > 2 * exported["length_mm"]
    with pytest.raises(DesignBuildError, match="stroke mode"):
        artwork_from_svg(svg, name="Lines", size_mm=(50.0, 50.0), source=source, strokes="dashed")
