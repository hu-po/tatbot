"""The same frozen artwork collection used by Inkmap and offline simulation."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
from tatbot_contracts.artwork import canvas_m as artwork_canvas_m
from tatbot_contracts.artwork import max_width_m, validate_path_artwork

from tatbot_sim.human_rep.ink_program import scheduled_material_strokes
from tatbot_sim.human_rep.placement import make_target_placement
from tatbot_sim.repo import repo_root

COLLECTION_PATH = repo_root() / "web/inkmap/public/designs/manifest.json"
SPLITS = ("train", "validation", "test")
EPOCH = "2026-09-06T00:00:00Z"


def collection_entries(split: str = "all") -> tuple[dict, ...]:
    if split not in (*SPLITS, "all"):
        raise ValueError(f"unknown artwork split {split!r}")
    manifest = json.loads(COLLECTION_PATH.read_text())
    if manifest.get("schema") != "tatbot.artwork-collection/1":
        raise ValueError("unsupported artwork collection")
    entries, ids, families, hashes = [], set(), {}, {}
    for entry in manifest["designs"]:
        if entry["id"] in ids:
            raise ValueError("duplicate collection design ID")
        ids.add(entry["id"])
        if entry.get("usage") != "artwork":
            continue
        path = repo_root() / "web/inkmap/public" / entry["path"]
        if path.suffix != ".json" or path.parent.parent.resolve() != COLLECTION_PATH.parent.resolve():
            raise ValueError("collection artwork must be inside its acquisition directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"artwork source digest differs: {entry['id']}")
        if entry["split"] not in SPLITS or not entry["family"]:
            raise ValueError("artwork needs a named family and split")
        for key, table in ((entry["family"], families), (entry["sha256"], hashes)):
            if table.setdefault(key, entry["split"]) != entry["split"]:
                raise ValueError("artwork family/source leaks across evaluation splits")
        low, high = entry["size_range_mm"]
        size = entry["default_size_mm"]
        if not (0 < low <= max(size) <= high <= 100 and 0 < entry["planning_width_mm"] <= 2):
            raise ValueError("invalid artwork size/width domain")
        if split in ("all", entry["split"]):
            record = validate_path_artwork(json.loads(raw))
            _acquired(record)
            entries.append({**entry, "artwork": record})
    if not entries:
        raise ValueError("artwork collection/split is empty")
    return tuple(entries)


def _acquired(record):
    if (record["conversion"]["adapter"] != "dbv3-batik-paths/1" or record["conversion"]["recipe_sha256"] is None
            or record["program"]["provenance"]["producer"] != "dbv3-batik-paths/1"):
        raise ValueError("Generate with DrawingBot V3 and import artwork.json; legacy artwork is retired")


def collection_artifacts(split: str = "all"):
    from tatbot_contracts.paths import render_paths

    from tatbot_sim.inkmap.designs import DesignArtifact
    return tuple(DesignArtifact(
        id=e["id"], name=e["name"], svg=render_paths({key: e["artwork"]["program"][key] for key in ("canvas_m", "inks", "layers", "negative_space_masks")}), size_mm=tuple(e["default_size_mm"]),
        source={**e["source"], "collection": "dbv3-acquired-v1", "family": e["family"],
                "split": e["split"], "size_range_mm": e["size_range_mm"]}, artwork=e["artwork"],
    ) for e in collection_entries(split))


def artwork_record(entry: dict, size_mm: tuple[float, float] | None = None, *, width_mm: float | None = None):
    record = validate_path_artwork(entry["artwork"])
    _acquired(record)
    acquired_size = np.asarray(artwork_canvas_m(record)) * 1000
    if size_mm is not None and not np.allclose(size_mm, acquired_size, rtol=0, atol=1e-6):
        raise ValueError("physical resizing requires DBV3 regeneration at the requested size")
    if width_mm is not None and not np.isclose(width_mm / 1000, max_width_m(record), rtol=0, atol=1e-9):
        raise ValueError("pen width changes require DBV3 regeneration")
    return deepcopy(record)


def planar_placement(record: dict, *, mirrored: bool = False, rotation_rad: float = 0):
    """Nominal plane enclosing the rotated artwork, without physical registration."""
    program = record["program"]
    size = np.asarray(artwork_canvas_m(record))
    c, s = abs(np.cos(rotation_rad)), abs(np.sin(rotation_rad))
    # Enclose the transformed canvas and a full planning-width border. This is
    # an offline material domain, not a guessed paper-pad size or robot pose.
    border = 2 * max_width_m(record)
    canvas = (np.array([[c, s], [s, c]]) @ size + border).tolist()
    return make_target_placement(
        tattoo_program_sha256=program["content_sha256"],
        target={"kind": "plane", "canvas_m": canvas, "anchor_uv_m": [0, 0], "margin_m": 0},
        physical_scale_m=size.tolist(), rotation_rad=rotation_rad,
        mirrored=mirrored,
        review={"status": "pending", "reviewer": "offline planar artwork compiler", "evidence_sha256": "0" * 64},
        provenance={"producer": "tatbot-artwork-planar", "version": "1", "created_utc": EPOCH,
                    "source_sha256": record["content_sha256"]},
    )


def planar_strokes(record: dict, *, mirrored: bool = False, rotation_rad: float = 0,
                   fill_style: str = "concentric"):
    """The scheduled material strokes of one planar placement: what a session draws."""
    return scheduled_material_strokes(record["program"],
                                      planar_placement(record, mirrored=mirrored, rotation_rad=rotation_rad),
                                      fill_style=fill_style)
