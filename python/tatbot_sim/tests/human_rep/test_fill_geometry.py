"""Independent finite-footprint qualification, not centerline-only checks."""
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely import LineString, Polygon, affinity, box, normalize, union_all
from tatbot_sim.human_rep.contracts import ContractError
from tatbot_sim.human_rep.fill_geometry import deposited_coverage, fill_paths, fill_plan, paint_union
from tatbot_sim.human_rep.ink_program import material_strokes
from tatbot_sim.human_rep.placement import make_surface_placement
from tatbot_sim.inkmap.artwork import make_artwork_record
from tatbot_sim.repo import repo_root


def test_triangulation_is_not_ink_and_fill_is_deterministic():
    triangles = [np.array([[0, 0], [.02, 0], [.02, .02]]),
                 np.array([[0, 0], [.02, .02], [0, .02]])]
    target = paint_union(triangles, [])
    untriangulated = paint_union([np.array(box(0, 0, .02, .02).exterior.coords)], [])
    paths = fill_paths(target, .0005)
    expected = fill_paths(untriangulated, .0005)
    assert len(paths) == len(expected)
    for actual, other in zip(paths, expected, strict=True):
        np.testing.assert_array_equal(actual, other)
        # Long triangulation diagonals must never be inked. Short links
        # between inset contours are valid and preserve the finite footprint.
        delta = np.diff(actual, axis=0)
        diagonal = ~(np.isclose(delta[:, 0], 0) | np.isclose(delta[:, 1], 0))
        assert np.all(np.linalg.norm(delta[diagonal], axis=1) < .0005)
    coverage = deposited_coverage(paths, .0005)
    assert coverage.difference(target).area < 1e-16
    assert coverage.intersection(target).area / target.area > .999


def test_hatch_rows_lie_inside_the_centre_domain_and_are_deterministic():
    """The triangulation assertion above proves tessellation edges are never
    inked for the concentric style; the hatch style is proved by the stronger,
    angle-independent property that every segment of every path lies inside
    the centre domain (where the tool's centre may travel), so no row and no
    serpentine link crosses paint the tool cannot be in."""
    triangles = [np.array([[0, 0], [.02, 0], [.02, .02]]),
                 np.array([[0, 0], [.02, .02], [0, .02]])]
    target = paint_union(triangles, [])
    untriangulated = paint_union([np.array(box(0, 0, .02, .02).exterior.coords)], [])
    paths = fill_plan(target, .0005, fill_style="hatch")
    expected = fill_plan(untriangulated, .0005, fill_style="hatch")
    assert len(paths) == len(expected)
    for actual, other in zip(paths, expected, strict=True):
        np.testing.assert_array_equal(actual.points, other.points)
        assert actual.depth == other.depth
    centre = normalize(target.buffer(-.00025 / np.cos(np.pi / 128) - 1e-7, quad_segs=32))
    tolerance = centre.buffer(1e-9)
    assert all(tolerance.covers(LineString(path.points)) for path in paths)
    # Two boundary rings (linked into one stroke here, as the ring linker
    # does), then hatch rows reported at depth 3 so the schedule tiers them last.
    depths = [p.depth for p in paths]
    assert depths[0] == 0 and set(depths) <= {0, 1, 3} and depths.count(3) >= 1
    assert depths == sorted(depths)
    coverage = deposited_coverage([p.points for p in paths], .0005)
    assert coverage.difference(target).area < 1e-16
    assert coverage.intersection(target).area / target.area > .999
    # A feather-shaped region is hatched along its long axis: one serpentine.
    feather = Polygon([(0, 0), (.02, .0005), (.03, .003), (.02, .0055), (0, .006)])
    rows = [p for p in fill_plan(feather, .0003, fill_style="hatch") if p.depth == 3]
    assert rows
    for row in rows:
        segments = np.diff(row.points, axis=0)
        along = segments @ np.array([1.0, 0.0])
        # Every hatch segment is either along the long axis or a short link.
        assert np.all((np.abs(along) > 0.9 * np.linalg.norm(segments, axis=1)) | (np.linalg.norm(segments, axis=1) <= .0006))


@pytest.mark.parametrize("fill_style", ["concentric", "hatch"])
def test_entire_footprint_respects_holes_not_only_midpoints(fill_style):
    hole = box(.008, .008, .012, .012)
    target = box(0, 0, .02, .02).difference(hole)
    paths = fill_paths(target, .0005, fill_style=fill_style)
    coverage = deposited_coverage(paths, .0005)
    assert coverage.intersection(hole).area < 1e-16
    assert coverage.difference(target).area < 1e-16
    assert coverage.intersection(target).area / target.area > .998


@pytest.mark.parametrize("fill_style", ["concentric", "hatch"])
def test_small_disconnected_feature_cannot_disappear_in_large_fill(fill_style):
    # A 0.1 mm dot beside a 20 mm square, planned with a 0.5 mm tool. The dot is
    # five times finer than the tool, so tracing its middle would lay 23x its
    # area in ink — over NARROW_OVERDRAW_LIMIT — and the whole drawing is
    # refused. What matters is that the dot cannot be quietly dropped and the
    # square admitted; which of the two geometric rules catches it may change.
    target = box(0, 0, .02, .02).union(box(.03, .03, .0301, .0301))
    with pytest.raises(ContractError, match="component coverage|narrower than the planning width"):
        fill_paths(target, .0005, fill_style=fill_style)
    # The square alone, without the dot, plans normally at the same width.
    assert fill_paths(box(0, 0, .02, .02), .0005, fill_style=fill_style)
    with pytest.raises(ContractError, match="narrower"):
        fill_paths(box(0, 0, .0001, .0001), .0005, fill_style=fill_style)
    with pytest.raises(ContractError, match="fill_style"):
        fill_paths(box(0, 0, .02, .02), .0005, fill_style="stipple")
    with pytest.raises(ContractError, match="invalid"):
        paint_union([np.array([[0, 0], [1, 1], [1, 0], [0, 1]])], [])
    with pytest.raises(ContractError, match="20000 rows"):
        fill_paths(box(0, 0, 1, 1), 1e-6)


@pytest.mark.parametrize("name,canvas", [
    ("linework", (.08, .05)), ("blackwork", (.06, .06)),
    ("negative-space", (.06, .06)), ("stipple", (.06, .04)),
    ("color-layers", (.08, .05)),
])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("fill_style", ["concentric", "hatch"])
def test_all_owned_svg_classes_plan_from_paint_union(name, canvas, transformed, fill_style):
    svg = (repo_root() / f"web/inkmap/tests/fixtures/artwork/{name}.svg").read_text()
    record = make_artwork_record(
        name=name, original_svg=svg,
        source={"kind": "fixture", "identifier": name, "license": "CC0-1.0",
                "attribution": "Tatbot synthetic fixture", "generation": None},
        conversion={"adapter": "tatbot-svg-paint/1", "canvas_m": list(canvas),
                    "semantic_intent": "Owned fill planning fixture", "width_m": .0003,
                    "deposition": 1, "chord_error_m": .000025},
    )
    program = record["program"]
    scale = (1.2, .8) if transformed else (1, 1)
    angle = .37 if transformed else 0
    width = .0003 * np.sqrt(scale[0] * scale[1])
    placed = make_surface_placement(
        tattoo_program_sha256=program["content_sha256"], body_identity_sha256="1" * 64,
        rest_surface_sha256="2" * 64, topology_sha256="3" * 64,
        semantic_site="forearm", laterality="left", anchor=(0, [1, 0, 0]),
        physical_scale_m=(canvas[0] * scale[0], canvas[1] * scale[1]),
        rotation_rad=angle, mirrored=transformed,
        supported_faces=[0], margin_m=0,
        review={"status": "pending", "reviewer": "test", "evidence_sha256": "0" * 64},
        provenance={"producer": "test", "version": "1", "created_utc": "2026-09-05T00:00:00Z",
                    "source_sha256": record["content_sha256"]},
    )
    strokes = material_strokes(program, placed, fill_style=fill_style)
    assert strokes
    assert all("union of" in stroke.ordering_rationale for stroke in strokes)
    metrics = []
    # Independently compare the finite footprint against each original paint
    # layer; occlusion/composited color and body mapping are separate gates.
    for layer_index, layer in enumerate(program["layers"]):
        paths = [s.points_m for s in strokes if s.ordering_rationale.startswith(f"source layer {layer_index},")]
        coverage = deposited_coverage(paths, width)
        polygons = [Polygon(np.array(e["points_m"]) - np.array(canvas) / 2) for e in layer["elements"]]
        target = union_all(polygons)
        target = affinity.scale(target, xfact=scale[0] * (-1 if transformed else 1), yfact=scale[1], origin=(0, 0))
        target = affinity.rotate(target, angle, origin=(0, 0), use_radians=True)
        assert coverage.intersection(target).area / coverage.union(target).area >= .98
        assert coverage.difference(target).area < 1e-14
        metrics.append({"layer_id": layer["id"], "stroke_count": len(paths),
                        "target_area_m2": target.area,
                        "ideal_coverage_iou": coverage.intersection(target).area / coverage.union(target).area,
                        "spill_area_m2": coverage.difference(target).area})
    repeated = material_strokes(deepcopy(program), deepcopy(placed), fill_style=fill_style)
    assert [s.source_primitive_sha256 for s in repeated] == [s.source_primitive_sha256 for s in strokes]
    for actual, other in zip(strokes, repeated, strict=True):
        np.testing.assert_array_equal(actual.points_m, other.points_m)
    if output := os.environ.get("INKMAP_FILL_EVIDENCE"):
        destination = Path(output)
        if not destination.is_absolute() or destination.resolve().is_relative_to(repo_root()):
            raise ValueError("INKMAP_FILL_EVIDENCE must be an absolute directory outside the checkout")
        destination.mkdir(parents=True, exist_ok=True)
        report = {"result": "pass", "fixture": name, "transformed": transformed, "fill_style": fill_style,
                  "source_sha256": record["source_sha256"], "program_sha256": program["content_sha256"],
                  "surface_placement": placed, "shapely": shapely.__version__, "geos": shapely.geos_version_string,
                  "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip(),
                  "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root())),
                  "metrics": metrics, "scope": "planar ideal footprint; not simulator or physical deposition"}
        (destination / f"{name}-{fill_style}-{'transformed' if transformed else 'nominal'}.json").write_text(json.dumps(report, indent=2) + "\n")
