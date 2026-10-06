"""The stroke schedule: tiers from the planner's own data, dedup at the tool's
footprint, nearest-neighbour order within a tier. Geometry only."""
from __future__ import annotations

import numpy as np
import pytest
from shapely import Polygon, box
from tatbot_sim.human_rep.contracts import load_contract
from tatbot_sim.human_rep.fill_geometry import PAINT_COVERAGE_FLOOR, deposited_coverage, fill_plan
from tatbot_sim.human_rep.ink_program import MaterialStroke, material_strokes, stroke_tier
from tatbot_sim.human_rep.stroke_schedule import (
    MAX_COVERAGE_LOSS_FRACTION,
    MIN_NEW_AREA_FRACTION,
    pen_up_travel_m,
    schedule,
)

from .test_ink_program import EXAMPLES, _placement, body_context  # noqa: F401


def _strokes(target, width_m: float, *, layer: int = 0, ink: str = "black") -> list[MaterialStroke]:
    return [MaterialStroke(points_m=path.points, ink_id=ink, width_m=width_m, deposition=1.0,
                           source_primitive_sha256="a" * 64, ordering_rationale=f"source layer {layer}, union of 1",
                           layer_index=layer, inset_depth=path.depth, component_area_m2=path.island_area_m2,
                           component=path.component, tier=stroke_tier(path.depth))
            for path in fill_plan(target, width_m)]


def _explicit(points, *, layer: int = 0, ink: str = "black", width_m: float = 0.0003) -> MaterialStroke:
    return MaterialStroke(points_m=np.asarray(points, float), ink_id=ink, width_m=width_m, deposition=1.0,
                          source_primitive_sha256="b" * 64, ordering_rationale=f"source layer {layer}, element 0",
                          layer_index=layer)


def _length(points) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


# A 3 mm wide square band: its exterior and interior rings at every inset are
# further apart than the 2 x width link, so each ring is its own stroke (a
# solid square would link into one spiral at depth 0).
BAND = box(0, 0, .01, .01).difference(box(.003, .003, .007, .007))


def test_fill_plan_tags_depth_island_and_component():
    thin = box(.02, 0, .0204, .01)  # 0.4 mm wide: one centre ring at 0.3 mm
    paths = fill_plan(BAND.union(thin), .0003)
    thin_paths = [p for p in paths if p.component.equals(thin)]
    thick_paths = [p for p in paths if p.component.equals(BAND)]
    assert thin_paths and thick_paths and len(thin_paths) + len(thick_paths) == len(paths)
    assert [p.depth for p in thin_paths] == [0], "a thin feature has only its boundary ring"
    assert [p.depth for p in thick_paths[:4]] == [0, 0, 1, 1] and max(p.depth for p in thick_paths) >= 3
    assert all(p.island_area_m2 <= p.component.area + 1e-12 for p in paths)
    islands = [p.island_area_m2 for p in thick_paths]
    assert all(b <= a * (1 + 1e-3) for a, b in zip(islands[:-1], islands[1:], strict=True)), "islands shrink with depth"
    # Every path of one component shares the component object, so the
    # schedule can group by identity.
    assert len({id(p.component) for p in thick_paths}) == 1
    assert [stroke_tier(d) for d in (None, 0, 1, 2, 3, 8)] == [0, 0, 1, 1, 2, 2]


def test_tiers_draw_outlines_first_and_explicit_elements_are_never_dropped():
    strokes = _strokes(BAND, .0003)
    dot = _explicit([[.005, .005]] + [[.005 + 1e-4 * np.cos(a), .005 + 1e-4 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 13)][1:])
    report: dict = {}
    scheduled = schedule([*strokes, dot], footprint_width_m=.0006, report=report)
    assert [s.tier for s in scheduled] == sorted(s.tier for s in scheduled)
    assert [s.inset_depth for s in scheduled[:2]] == [0, 0]
    assert any(s.source_primitive_sha256 == "b" * 64 for s in scheduled), "the explicit dot survives at a 0.6 mm footprint"
    assert scheduled[-1].source_primitive_sha256 != "b" * 64, "tier 0 first: the dot is not drawn last"
    assert all(item["inset_depth"] is not None for item in report["dropped"])
    assert [s.schedule_index for s in scheduled] == list(range(len(scheduled)))
    assert report["output_strokes"] == len(scheduled) and report["input_strokes"] == len(strokes) + 1
    assert report["dedup_dropped_strokes"] == len(report["dropped"]) > 0


def test_dedup_drops_only_redundant_rings_and_holds_the_coverage_floor():
    strokes = _strokes(BAND, .0003)
    depths = [s.inset_depth for s in strokes]
    assert max(depths) >= 3 and len(strokes) > 20

    def kept_at(footprint):
        report: dict = {}
        kept = schedule(strokes, footprint_width_m=footprint, report=report)
        coverage = deposited_coverage([s.points_m for s in kept], footprint)
        full = deposited_coverage([s.points_m for s in strokes], footprint)
        fraction = coverage.intersection(BAND).area / BAND.area
        assert fraction >= PAINT_COVERAGE_FLOOR
        # A drop is a deliberate loss and may not spend the qualification
        # budget: the band keeps all but MAX_COVERAGE_LOSS_FRACTION of what
        # every ring would give it.
        assert full.intersection(BAND).area / BAND.area - fraction <= MAX_COVERAGE_LOSS_FRACTION + 1e-9
        for item in report["dropped"]:
            assert item["new_area_fraction"] < MIN_NEW_AREA_FRACTION
            assert item["footprint_width_m"] == footprint
        return kept, report

    wide, wide_report = kept_at(.0006)
    narrow, narrow_report = kept_at(.0003)
    assert sum(s.inset_depth == 0 for s in wide) == 2, "both outline rings are kept at a 0.6 mm footprint"
    # A ring just inside the outline still adds a fifth of its footprint at
    # 0.6 mm; dropping it would cost the band several percent, so it stays.
    # What goes is the converging centre, where rings add next to nothing.
    assert all(item["inset_depth"] >= 3 for item in wide_report["dropped"])
    assert min(item["new_area_fraction"] for item in wide_report["dropped"]) < 0.05
    assert len(wide) < len(narrow) < len(strokes)
    # At the planning width each clean ring adds 45% new area; only the
    # innermost rings, converging on the band's centre, are redundant.
    assert all(item["inset_depth"] >= 3 for item in narrow_report["dropped"])
    assert wide_report["dedup_dropped_m"] > narrow_report["dedup_dropped_m"]
    # A disconnected dot is its own component with one ring: never redundant.
    dot = box(.03, .03, .0306, .0306)
    both = _strokes(BAND.union(dot), .0003)
    kept = schedule(both, footprint_width_m=.0006)
    assert any(s.component.equals(dot) for s in kept)


def test_dedup_respects_component_floor_at_a_fine_footprint():
    # A footprint finer than the planning width cannot reach the floor without
    # every ring, so nothing may be dropped even though rings overlap.
    strokes = _strokes(BAND, .0005)
    report: dict = {}
    kept = schedule(strokes, footprint_width_m=.0002, report=report)
    coverage = deposited_coverage([s.points_m for s in kept], .0002)
    full = deposited_coverage([s.points_m for s in strokes], .0002)
    assert coverage.area == pytest.approx(full.area)
    assert report["dropped"] == []


def test_order_within_tier_rotates_rings_reverses_open_strokes_and_is_deterministic():
    # Two closed rings drawn far apart in generation order, then an open
    # stroke whose far end is nearer.
    ring_a = np.asarray(box(0, 0, .002, .002).exterior.coords)
    ring_b = np.asarray(box(.010, .010, .012, .012).exterior.coords)
    ring_c = np.asarray(box(.003, 0, .005, .002).exterior.coords)
    line = np.asarray([[.030, .030], [.0125, .0125]])
    strokes = [_explicit(ring_a), _explicit(ring_b), _explicit(ring_c), _explicit(line)]
    report: dict = {}
    out = schedule(strokes, report=report)
    assert [np.array_equal(s.points_m, ring_a) for s in out][0], "the first stroke keeps its start"
    assert np.allclose(out[1].points_m[:-1].min(axis=0), ring_c[:-1].min(axis=0)), "nearest ring next"
    assert np.array_equal(out[1].points_m[0], out[1].points_m[-1]) and "rotated" in out[1].ordering_rationale
    assert _length(out[1].points_m) == pytest.approx(_length(ring_c))
    assert np.array_equal(out[3].points_m, line[::-1]) and out[3].ordering_rationale.endswith("; reversed")
    assert report["pen_up_travel_m"]["scheduled"] < report["pen_up_travel_m"]["generation_order"]
    assert pen_up_travel_m(out) == pytest.approx(report["pen_up_travel_m"]["scheduled"])
    again = schedule(strokes)
    for a, b in zip(out, again, strict=True):
        np.testing.assert_array_equal(a.points_m, b.points_m)
        assert a.ordering_rationale == b.ordering_rationale


def test_groups_keep_layer_and_ink_order():
    first = _strokes(BAND, .0003, layer=0, ink="red")
    second = _strokes(box(.02, 0, .03, .01).difference(box(.023, .003, .027, .007)), .0003, layer=1, ink="blue")
    out = schedule([*first, *second])
    inks = [s.ink_id for s in out]
    assert inks == ["red"] * inks.count("red") + ["blue"] * inks.count("blue")
    # Tier order holds within each group, not across them.
    reds = [s.tier for s in out if s.ink_id == "red"]
    blues = [s.tier for s in out if s.ink_id == "blue"]
    assert reds == sorted(reds) and blues == sorted(blues)


def test_stipple_dots_survive_the_schedule_and_rationale_keeps_its_prefix(body_context):  # noqa: F811
    stipple = load_contract(EXAMPLES / "stipple" / "program.json")
    placement = _placement(stipple, body_context)
    raw = material_strokes(stipple, placement)
    out = schedule(raw, footprint_width_m=.002)
    assert len(out) == len(raw) == 6
    for stroke in out:
        assert stroke.ordering_rationale.startswith("source layer 0,")
        assert "; tier 0" in stroke.ordering_rationale
    blackwork = load_contract(EXAMPLES / "blackwork" / "program.json")
    fills = material_strokes(blackwork, _placement(blackwork, body_context))
    assert all("union of" in s.ordering_rationale and f"; tier {s.tier} depth {s.inset_depth}" in s.ordering_rationale
               for s in fills)
    assert isinstance(fills[0].component, Polygon)
