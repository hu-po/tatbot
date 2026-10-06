"""Synthetic frozen native-path documents for preparation contracts, not acquisition evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tatbot_contracts.artwork import freeze_artwork
from tatbot_contracts.canonical import canonical_digest
from tatbot_contracts.paths import freeze_program

REPO = Path(__file__).resolve().parents[3]
SOURCE = 'a' * 64
ADAPTER = 'dbv3-batik-paths/1'


def element(points, *, closed=False, width=.0005):
    return {"id": "path", "kind": "path", "closed": closed, "fill": False, "width_m": width, "deposition": 1,
            "points_m": [list(map(float, p)) for p in points]}


def program(layers, *, canvas=(.02, .02), inks=None):
    inks = inks or {ink: [0., 0., 0.] for ink, _ in layers}
    geometry = {"canvas_m": dict(zip(("width", "height"), canvas, strict=True)),
                "inks": [{"id": k, "color_srgb": v} for k, v in inks.items()], "layers": [], "negative_space_masks": []}
    for i, (ink, paths) in enumerate(layers):
        paths = copy.deepcopy(paths)
        for j, path in enumerate(paths):
            path["id"] = f"path-{i}-{j}"
        geometry["layers"].append({"id": f"layer-{i}", "ink_id": ink, "elements": paths})
    return freeze_program(geometry, source_sha256=SOURCE, name="fixture", adapter=ADAPTER)[0]


def artwork(prog):
    return freeze_artwork(prog, name="fixture", source={"kind": "fixture", "identifier": "synthetic-paths",
                          "license": None, "attribution": None, "generation": None},
                          conversion={"adapter": ADAPTER, "recipe_sha256": 'b' * 64, "chord_error_m": .000005})


def placement(size, *, anchor=(0., 0.), rotation=0., mirrored=False, canvas=(.1, .15), margin=.005):
    return {"schema": "tatbot.surface-placement/2", "physical_scale_m": list(size), "rotation_rad": rotation,
            "mirrored": mirrored, "warp": None, "review": {"status": "pending", "reviewer": "fixture",
            "evidence_sha256": SOURCE}, "provenance": {"producer": "fixture", "version": "1",
            "created_utc": "1970-01-01T00:00:00Z", "source_sha256": SOURCE},
            "target": {"kind": "plane", "canvas_m": list(canvas), "anchor_uv_m": list(anchor), "margin_m": margin}}


def design(items, *, name="test design"):
    artworks, placements = {}, []
    for i, (pid, prog, placed) in enumerate(items):
        artworks[f"art-{i}"] = artwork(prog)
        placed = copy.deepcopy(placed)
        placed["tattoo_program_sha256"] = prog["content_sha256"]
        placed["content_sha256"] = canonical_digest(placed)
        placements.append({"id": pid, "artwork_id": f"art-{i}", "placement": placed})
    doc = {"schema": "tatbot.inkmap-design/1", "name": name, "artworks": artworks, "placements": placements}
    doc["content_sha256"] = canonical_digest(doc)
    return doc


def write(tmp_path, document, name="design.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path
