"""Read the shared frozen path artwork and the plane subset of Inkmap Design/1."""
from __future__ import annotations

import copy
import math
import re
from datetime import datetime

from tatbot_contracts.artwork import SCHEMA, canvas_m, require_acquired_artwork
from tatbot_contracts.canonical import canonical_digest

from tatbot_ink.errors import CompileError

DESIGN_SCHEMA = "tatbot.inkmap-design/1"
ADAPTER = "dbv3-batik-paths/1"


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise CompileError(f"expected fields: {expected}")


def _text(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 1000:
        raise CompileError("expected bounded nonempty identifier")


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise CompileError("expected finite physical value")


def _pair(value, *, positive=False):
    if not isinstance(value, list) or len(value) != 2:
        raise CompileError("expected two metric coordinates")
    for component in value:
        _number(component, positive=positive)


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise CompileError("expected lowercase SHA-256")


def _metadata(placement):
    review, provenance = placement["review"], placement["provenance"]
    _keys(review, "status reviewer evidence_sha256")
    if review["status"] not in ("pending", "accepted", "rejected"):
        raise CompileError("unknown placement review status")
    _text(review["reviewer"])
    _sha(review["evidence_sha256"])
    _keys(provenance, "producer version created_utc source_sha256")
    for key in ("producer", "version", "created_utc"):
        _text(provenance[key])
    _sha(provenance["source_sha256"])
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", provenance["created_utc"]):
        raise CompileError("placement provenance requires UTC time")
    datetime.strptime(provenance["created_utc"], "%Y-%m-%dT%H:%M:%SZ")


def _digest(value):
    if value["content_sha256"] != canonical_digest(value):
        raise CompileError("shared document digest mismatch")


def _plane(placement, artwork):
    _keys(placement, "schema content_sha256 tattoo_program_sha256 physical_scale_m rotation_rad mirrored warp review provenance target")
    if placement["schema"] != "tatbot.surface-placement/2":
        raise CompileError("expected SurfacePlacement/2")
    _digest(placement)
    if placement["tattoo_program_sha256"] != artwork["program"]["content_sha256"]:
        raise CompileError("placement references a different path program")
    _pair(placement["physical_scale_m"], positive=True)
    _number(placement["rotation_rad"])
    if type(placement["mirrored"]) is not bool or placement["warp"] is not None:
        raise CompileError("plane placement requires a boolean mirror and no warp")
    target = placement["target"]
    if not isinstance(target, dict) or target.get("kind") != "plane":
        raise CompileError("only plane placements are drawable")
    _keys(target, "kind canvas_m anchor_uv_m margin_m")
    _pair(target["canvas_m"], positive=True)
    _pair(target["anchor_uv_m"])
    _number(target["margin_m"])
    if not 0 <= target["margin_m"] < min(target["canvas_m"]) / 2:
        raise CompileError("plane margin consumes the canvas")
    # Review/provenance are retained metadata, never runtime motion authority.
    _metadata(placement)


def _artwork(record):
    try:
        require_acquired_artwork(record)
    except (ValueError, KeyError, TypeError) as error:
        raise CompileError(str(error)) from error
    for layer in record["program"]["layers"]:
        if any(path["deposition"] != 1 for path in layer["elements"]):
            raise CompileError("variable deposition is not implemented by the ROS executor")


def read_input(document):
    """Return validated records and ordered placements; raw artwork uses a centred stencil."""
    from tatbot_ink.place import STENCIL_MARGIN_M, STENCIL_SIZE_M

    if document.get("schema") == SCHEMA:
        _artwork(document)
        return {"name": document["name"], "artworks": {"artwork": document}, "placements": [
            {"id": "placement", "artwork_id": "artwork", "placement": {
                "physical_scale_m": canvas_m(document), "rotation_rad": 0, "mirrored": False,
                "target": {"kind": "plane", "canvas_m": list(STENCIL_SIZE_M),
                           "anchor_uv_m": [0, 0], "margin_m": STENCIL_MARGIN_M}}}]}
    _keys(document, "schema content_sha256 name artworks placements")
    if document["schema"] != DESIGN_SCHEMA:
        raise CompileError(f"expected {SCHEMA} or {DESIGN_SCHEMA}")
    _digest(document)
    _text(document["name"])
    artworks, items = document["artworks"], document["placements"]
    if not isinstance(artworks, dict) or not 1 <= len(artworks) <= 100:
        raise CompileError("expected 1..100 artwork records")
    for key, value in artworks.items():
        _text(key)
        _artwork(value)
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        raise CompileError("expected 1..100 placements")
    seen = set()
    for item in items:
        _keys(item, "id artwork_id placement")
        _text(item["id"])
        if item["id"] in seen or item["artwork_id"] not in artworks:
            raise CompileError("duplicate placement or unknown artwork")
        seen.add(item["id"])
        _plane(item["placement"], artworks[item["artwork_id"]])
    return copy.deepcopy(document)
