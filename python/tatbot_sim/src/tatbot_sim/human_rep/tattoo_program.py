"""Body-independent TattooProgram construction and exact SVG adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tatbot_sim.human_rep.artwork_client import artwork_request
from tatbot_sim.human_rep.contracts import (
    ContractError,
    canonical_bytes,
    validate_contract,
)

SCHEMA = "tatbot.tattoo-program/1"
ADAPTER_VERSION = "tatbot-svg-paint/1"


class TattooProgramError(ContractError):
    """Stable refusal raised at the artwork/materialization boundary."""


def tattoo_program_bytes(value: dict[str, Any]) -> bytes:
    return canonical_bytes(validate_contract(value, expected_schema=SCHEMA))


def tattoo_program_from_svg(
    svg: str,
    *,
    canvas_m: tuple[float, float],
    semantic_intent: str,
    provenance: dict[str, Any],
    ink_id: str = "black",
    color_srgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    width_m: float = 0.0008,
    deposition: float = 0.7,
    chord_error_m: float = 0.0001,
) -> dict[str, Any]:
    """Preserve SVG paint through the same adapter used by the browser.

    Metric regions preserve widths, caps, joins, fills, holes, colors and layer
    order. ``width_m`` controls subsequent fill planning, not SVG stroke width.
    Recoloring is an explicit artwork edit, never an implicit import default.
    """
    if not isinstance(svg, str) or "<svg" not in svg:
        raise TattooProgramError("tattoo_program_invalid", "$.svg", "expected materialized SVG text")
    if ink_id != "black" or color_srgb != (0.0, 0.0, 0.0):
        raise TattooProgramError("tattoo_program_unsupported", "$.svg", "edit SVG paint explicitly before import")
    document = artwork_request({"svg": svg, "options": {
        "canvas_m": list(canvas_m), "semantic_intent": semantic_intent,
        "provenance": provenance, "width_m": width_m,
        "deposition": deposition, "chord_error_m": chord_error_m,
    }})
    return validate_contract(document, expected_schema=SCHEMA)


def tattoo_program_from_materialization(
    record: dict[str, Any],
    *,
    directory: str | Path,
    semantic_intent: str,
    created_utc: str,
) -> dict[str, Any]:
    """Adapt one immutable current Inkgen materialization without networking."""

    if record.get("schema") != "tatbot.inkgen-materialization/1":
        raise TattooProgramError("tattoo_program_invalid", "$.schema", "unsupported generated-design record")
    allowed = {"schema", "id", "name", "sha256", "size_mm", "source", "svg", "png"}
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise TattooProgramError("tattoo_program_invalid", "$", f"unknown fields: {', '.join(unknown)}")
    svg_name = record.get("svg")
    if not isinstance(svg_name, str) or Path(svg_name).name != svg_name:
        raise TattooProgramError("tattoo_program_invalid", "$.svg", "must be a local basename")
    svg_path = Path(directory) / svg_name
    try:
        svg = svg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TattooProgramError("tattoo_program_invalid", "$.svg", str(exc)) from exc
    digest = hashlib.sha256(svg.encode()).hexdigest()
    if record.get("sha256") != digest:
        raise TattooProgramError("tattoo_program_invalid", "$.sha256", "does not bind the SVG bytes")
    size = record.get("size_mm")
    if not isinstance(size, list) or len(size) != 2:
        raise TattooProgramError("tattoo_program_invalid", "$.size_mm", "expected [width,height]")
    source = record.get("source")
    if not isinstance(source, dict):
        raise TattooProgramError("tattoo_program_invalid", "$.source", "expected provenance object")
    checkpoint = source.get("checkpoint_sha256")
    provenance: dict[str, Any] = {
        "producer": ADAPTER_VERSION,
        "version": "1",
        "created_utc": created_utc,
        "source_sha256": digest,
    }
    if isinstance(source.get("seed"), int) and source["seed"] >= 0:
        provenance["seed"] = source["seed"]
    if isinstance(checkpoint, str):
        provenance["checkpoint_sha256"] = checkpoint
    if isinstance(source.get("prompt"), str) and source["prompt"]:
        provenance["prompt"] = source["prompt"]
    return tattoo_program_from_svg(
        svg,
        canvas_m=(float(size[0]) / 1000, float(size[1]) / 1000),
        semantic_intent=semantic_intent,
        provenance=provenance,
    )


def tattoo_program_to_svg(value: dict[str, Any]) -> str:
    """Produce the shared deterministic SVG review image, with transparent masks."""
    return str(artwork_request({"program": validate_contract(value, expected_schema=SCHEMA)})["svg"])


def program_geometry(value: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Return element paths or region boundaries in canvas metres (not fill strokes)."""

    program = validate_contract(value, expected_schema=SCHEMA)
    output: list[np.ndarray] = []
    for layer in program["layers"]:
        for element in layer["elements"]:
            if element["kind"] == "cubic_bezier":
                controls = np.asarray(element["control_points_m"], dtype=np.float64)
                t = np.linspace(0, 1, 257)[:, None]
                points = (
                    (1 - t) ** 3 * controls[0]
                    + 3 * (1 - t) ** 2 * t * controls[1]
                    + 3 * (1 - t) * t**2 * controls[2]
                    + t**3 * controls[3]
                )
            else:
                points = np.asarray(element["points_m"], dtype=np.float64)
            output.append(points)
    return tuple(output)
