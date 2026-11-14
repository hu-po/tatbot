"""Synthetic frozen native-path documents for preparation contracts, not acquisition evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tatbot_contracts.artwork import freeze_artwork
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


def placement(size, *, anchor=(0., 0.), rotation=0., mirrored=False):
    """The compiler's internal rigid placement of a `size` canvas (tatbot_ink.input._centred, moved and turned)."""
    return {"physical_scale_m": list(size), "rotation_rad": rotation, "mirrored": mirrored,
            "target": {"anchor_uv_m": list(anchor)}}


def write(tmp_path, document, name="artwork.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path
