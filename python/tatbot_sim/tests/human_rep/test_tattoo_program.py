from __future__ import annotations

import hashlib

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import ContractError, canonical_bytes, load_contract
from tatbot_sim.human_rep.examples import write_examples
from tatbot_sim.human_rep.tattoo_program import (
    program_geometry,
    tattoo_program_bytes,
    tattoo_program_from_materialization,
    tattoo_program_from_svg,
)
from tatbot_sim.inkmap.svg_strokes import compile_svg_strokes
from tatbot_sim.repo import repo_root

EXAMPLES = repo_root() / "config" / "human-representation" / "examples"


def provenance(source: str = "1" * 64) -> dict:
    return {
        "producer": "tatbot-p2-test",
        "version": "1",
        "created_utc": "2026-09-04T13:00:00Z",
        "source_sha256": source,
        "seed": 7,
    }


def test_checked_program_roundtrips_as_canonical_bytes():
    value = load_contract(EXAMPLES / "tattoo-program.json")
    assert tattoo_program_bytes(value) == canonical_bytes(value)
    assert program_geometry(value)


def test_every_supported_svg_primitive_and_transform_preserves_visible_stroke_coverage():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 50" fill="none" stroke="black" stroke-width="1">'
        '<g transform="translate(1 1)">'
        '<path d="M1 2 L8 2 H12 V7 C14 8 15 9 16 10 S18 12 20 10 Q22 8 24 10 T28 10 A3 2 20 0 1 34 12 Z"/>'
        '<line x1="2" y1="20" x2="15" y2="22"/>'
        '<polyline points="18,20 22,24 27,20"/>'
        '<polygon points="30,20 35,25 40,20"/>'
        '<circle cx="48" cy="22" r="4"/>'
        '<ellipse cx="60" cy="22" rx="5" ry="3"/>'
        '<rect x="2" y="32" width="12" height="8"/>'
        '<rect x="20" y="32" width="14" height="9" rx="2"/>'
        '</g></svg>'
    )
    expected = compile_svg_strokes(svg, [80, 50], chord_error_m=0.0001).strokes
    program = tattoo_program_from_svg(
        svg,
        canvas_m=(0.08, 0.05),
        semantic_intent="synthetic primitive matrix",
        provenance=provenance(hashlib.sha256(svg.encode()).hexdigest()),
    )
    # Import now retains painted area, including cap/join/width geometry. Check
    # independently compiled centerline interiors lie in that area, rather than
    # asserting the old lossy representation (one bare boundary per primitive).
    regions = program_geometry(program)
    def inside(point, polygon):
        x, y = point
        found = False
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
                found = not found
        return found
    for centerline in expected:
        for point in (centerline[:-1] + centerline[1:]) / 2:
            canvas = point + [0.04, 0.025]
            assert any(inside(canvas, region) for region in regions), canvas


def test_unsupported_svg_refuses_with_a_typed_element_path():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><image href="x.png"/></svg>'
    with pytest.raises(ContractError) as caught:
        tattoo_program_from_svg(
            svg,
            canvas_m=(0.01, 0.01),
            semantic_intent="unsupported fixture",
            provenance=provenance(hashlib.sha256(svg.encode()).hexdigest()),
        )
    assert caught.value.code == "tattoo_program_unsupported"
    assert caught.value.path == "$.svg.image[1]"
    assert "image" in caught.value.detail


def test_current_materialization_adapter_binds_exact_svg(tmp_path):
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="1" x2="9" y2="9" stroke="black"/></svg>'
    digest = hashlib.sha256(svg.encode()).hexdigest()
    (tmp_path / "design.svg").write_text(svg)
    record = {
        "schema": "tatbot.inkgen-materialization/1",
        "id": "fixture",
        "name": "fixture",
        "sha256": digest,
        "size_mm": [20, 20],
        "source": {"kind": "generated", "model": "fixture", "seed": 11, "prompt": "line"},
        "svg": "design.svg",
        "png": "design.png",
    }
    program = tattoo_program_from_materialization(
        record,
        directory=tmp_path,
        semantic_intent="line",
        created_utc="2026-09-04T13:00:00Z",
    )
    assert program["preview_sha256"] == digest
    assert program["provenance"]["seed"] == 11
    record["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="does not bind"):
        tattoo_program_from_materialization(
            record,
            directory=tmp_path,
            semantic_intent="line",
            created_utc="2026-09-04T13:00:00Z",
        )


def test_checked_program_class_matrix_is_exactly_regenerable():
    write_examples(EXAMPLES, check=True)
