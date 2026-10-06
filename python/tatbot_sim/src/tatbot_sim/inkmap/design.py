"""Portable design handoff, validated by the same reader as the Inkmap editor.

Artwork/placement identity is checked here. Measured registration, supply and
session execution are separate preparation inputs, never design authority.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tatbot_sim.human_rep.artwork_client import artwork_request
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, validate_contract


def validate_design(value: dict[str, Any]) -> dict[str, Any]:
    design = artwork_request({"operation": "validate_design", "design": value})
    for document in [design, *design["artworks"].values()]:
        if canonical_digest(document) != document["content_sha256"]:
            raise ContractError("wrong_hash", "$.content_sha256", "Python and browser design digests differ")
    for artwork in design["artworks"].values():
        validate_contract(artwork["program"], expected_schema="tatbot.tattoo-program/1")
    for item in design["placements"]:
        validate_contract(item["placement"], expected_schema="tatbot.surface-placement/2")
    return design


def make_design(*, name: str, artworks: dict[str, Any], placements: list[dict[str, Any]]) -> dict[str, Any]:
    document = {"schema": "tatbot.inkmap-design/1", "content_sha256": "", "name": name,
                "artworks": artworks, "placements": placements}
    document["content_sha256"] = canonical_digest(document)
    return validate_design(document)


def scan_coverage(value: dict[str, Any], *, fill_style: str = "concentric") -> dict[str, Any]:
    """Material footprint about chart zero; geometry requirements, never registration.

    Arc-length chart distance conservatively bounds Euclidean distance on a
    cylinder. Scan planning must separately establish the chart's measured pose.
    """
    from tatbot_sim.human_rep.ink_program import scheduled_material_strokes, target_stroke_points

    design = validate_design(value)
    geometry = None
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    radius = 0.0
    count = 0
    schedules: list[dict[str, Any]] = []
    for item in design["placements"]:
        placement = item["placement"]
        target = placement["target"]
        if target["kind"] not in {"plane", "cylinder"}:
            raise ValueError("body scan coverage needs a measured body-to-surface address mapping")
        current = {key: target[key] for key in ("kind", "canvas_m")}
        if target["kind"] == "cylinder":
            current["radius_m"] = target["radius_m"]
        if geometry is not None and current != geometry:
            raise ValueError("scan placements must share one target chart geometry")
        geometry = current
        report: dict[str, Any] = {}
        scheduled = scheduled_material_strokes(design["artworks"][item["artwork_id"]]["program"], placement,
                                               report=report, fill_style=fill_style)
        schedules.append({"placement_id": item["id"], **report})
        for stroke in scheduled:
            points = target_stroke_points(stroke, target)
            if not len(points):
                continue
            lower = np.minimum(lower, points.min(axis=0) - stroke.width_m / 2)
            upper = np.maximum(upper, points.max(axis=0) + stroke.width_m / 2)
            radius = max(radius, float(np.linalg.norm(points, axis=1).max()) + stroke.width_m / 2)
            count += 1
    if not count:
        raise ValueError("scan design has no material footprint")
    return {"design_sha256": design["content_sha256"], "target": geometry,
            "bounds_uv_m": [lower.tolist(), upper.tolist()], "radius_m": radius,
            "material_strokes": count, "schedule": schedules}
