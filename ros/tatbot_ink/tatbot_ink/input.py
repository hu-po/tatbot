"""Read the shared frozen path artwork: one acquisition, or several drawn as layers."""
from __future__ import annotations

from tatbot_contracts.artwork import SCHEMA, canvas_m, require_acquired_artwork

from tatbot_ink.errors import CompileError


def _artwork(record):
    try:
        require_acquired_artwork(record)
    except (ValueError, KeyError, TypeError) as error:
        raise CompileError(str(error)) from error


def _centred(document):
    """The artwork at its acquired size, unturned, its canvas centre on the page centre (place.relocate moves it)."""
    return {"physical_scale_m": canvas_m(document), "rotation_rad": 0, "mirrored": False,
            "target": {"anchor_uv_m": [0, 0]}}


def read_layers(documents):
    """Acquired artworks drawn as layers in the order given, each centred on the page like a single artwork:
    one DBV3 acquisition per pen (a design's colour layers, generated at one canvas size)."""
    if not 1 <= len(documents) <= 100:
        raise CompileError("expected 1..100 artwork layers")
    for document in documents:
        if document.get("schema") != SCHEMA:
            raise CompileError(f"a layer must be {SCHEMA}")
        _artwork(document)
    if len({tuple(canvas_m(document)) for document in documents}) != 1:
        raise CompileError("layers drawn together must share one canvas size: regenerate them at one size")
    return {"name": " + ".join(document["name"] for document in documents),
            "artworks": {f"layer{i}": document for i, document in enumerate(documents, 1)},
            "placements": [{"id": f"layer{i}", "artwork_id": f"layer{i}", "placement": _centred(document)}
                           for i, document in enumerate(documents, 1)]}


def read_input(document):
    """Return the validated artwork as one placement, centred on the page."""
    if document.get("schema") != SCHEMA:
        raise CompileError(f"expected {SCHEMA}: acquired DBV3 artwork.json, one file or several as layers")
    _artwork(document)
    return {"name": document["name"], "artworks": {"artwork": document}, "placements": [
        {"id": "placement", "artwork_id": "artwork", "placement": _centred(document)}]}
