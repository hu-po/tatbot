from copy import deepcopy

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest
from tatbot_sim.human_rep.placement import make_target_placement
from tatbot_sim.inkmap.collection import artwork_record, collection_entries, planar_placement
from tatbot_sim.inkmap.design import make_design, scan_coverage, validate_design


@pytest.fixture(scope="module")
def artwork():
    return artwork_record(next(e for e in collection_entries() if e["id"] == "dbv3-sprout"))


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_browser_and_python_agree_on_portable_target_design(artwork, kind):
    placed = planar_placement(artwork)
    if kind == "cylinder":
        placed = make_target_placement(
            **{k: v for k, v in placed.items() if k not in {"schema", "content_sha256", "target"}},
            target={**placed["target"], "kind": kind, "radius_m": .03},
        )
    design = make_design(name="Fish", artworks={"fish": artwork}, placements=[
        {"id": "first", "artwork_id": "fish", "placement": placed},
        {"id": "second", "artwork_id": "fish", "placement": placed},
    ])
    assert validate_design(design) == design
    assert design["placements"][0]["placement"]["target"]["kind"] == kind
    broken = deepcopy(design)
    broken["placements"][0]["placement"]["tattoo_program_sha256"] = "a" * 64
    broken["placements"][0]["placement"]["content_sha256"] = canonical_digest(broken["placements"][0]["placement"])
    broken["content_sha256"] = canonical_digest(broken)
    with pytest.raises(ContractError, match="different artwork"):
        validate_design(broken)


@pytest.mark.slow
def test_scan_coverage_tracks_rotation_mirroring_and_anchor(artwork):
    def coverage(angle, mirrored, anchor):
        placed = planar_placement(artwork, rotation_rad=angle, mirrored=mirrored)
        placed["target"].update(canvas_m=[.3, .3], anchor_uv_m=anchor)
        placed["content_sha256"] = canonical_digest(placed)
        return scan_coverage(make_design(name="Fish", artworks={"fish": artwork}, placements=[
            {"id": "fish", "artwork_id": "fish", "placement": placed}]))

    base = coverage(0, False, [0, 0])
    moved = coverage(0, False, [.02, -.01])
    assert np.asarray(moved["bounds_uv_m"]) == pytest.approx(np.asarray(base["bounds_uv_m"]) + [.02, -.01])
    mirrored = np.asarray(coverage(0, True, [0, 0])["bounds_uv_m"])
    bounds = np.asarray(base["bounds_uv_m"])
    assert mirrored[:, 0] == pytest.approx(-bounds[::-1, 0])
    rotated = np.asarray(coverage(np.pi / 2, False, [0, 0])["bounds_uv_m"])
    # Portable contracts round the rotation; allow sub-micrometre material differences.
    assert rotated[:, 0] == pytest.approx(-bounds[::-1, 1], abs=1e-7)
    assert rotated[:, 1] == pytest.approx(bounds[:, 0], abs=1e-7)


def test_acquired_scan_retains_all_source_paths_and_refuses_a_small_canvas():
    art = artwork_record(collection_entries()[0])
    placed = planar_placement(art)
    design = make_design(name=art['name'], artworks={'art': art}, placements=[
        {'id': 'art', 'artwork_id': 'art', 'placement': placed}])
    coverage = scan_coverage(design)
    assert coverage['material_strokes'] == sum(len(layer['elements']) for layer in art['program']['layers'])
    report = coverage['schedule'][0]
    assert report['dedup_dropped_strokes'] == 0
    placed = design['placements'][0]['placement']
    placed['target']['canvas_m'] = [.01, .01]
    placed['content_sha256'] = canonical_digest(placed)
    design['content_sha256'] = canonical_digest(design)
    with pytest.raises((ContractError, ValueError)):
        scan_coverage(design)
