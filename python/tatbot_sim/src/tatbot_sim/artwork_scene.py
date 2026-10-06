"""Place whole shared artworks on flat/height-field simulation substrates."""
from __future__ import annotations

import numpy as np

from tatbot_sim.inkmap.collection import artwork_record, collection_entries, planar_strokes
from tatbot_sim.strokes import Stroke


def sample_artwork_scene(rng, sheet, budget_s: float, *, verb="draw", reachable=None,
                         split="train", width_mm=.3, calibration=False, config=None):
    from tatbot_sim.inkmap.designs import spiral_artifact
    from tatbot_sim.inkmap.svg_strokes import compile_svg_strokes
    from tatbot_sim.language import REACH, _cost_s, pacing

    entries = list(collection_entries(split)) if not calibration else [None]
    rng.shuffle(entries)
    rejected = []
    for entry in entries:
        sizes = sorted({s for s in (30, 35, 40, *entry["size_range_mm"], max(entry["default_size_mm"]))
                        if entry["size_range_mm"][0] <= s <= entry["size_range_mm"][1]}) if entry is not None else [30]
        rng.shuffle(sizes)
        for size in sizes:
            dimensions = np.asarray(entry["default_size_mm"] if entry is not None else [size, size], dtype=float)
            dimensions *= size / max(dimensions)
            angle = float(rng.uniform(-np.pi, np.pi))
            mirrored = bool(rng.integers(2))
            if entry is None:
                design = spiral_artifact()
                points = list(compile_svg_strokes(design.svg, design.size_mm).strokes)
                identity = {"design_id": design.id, "source_sha256": design.sha256,
                            "family": "calibration", "split": "calibration", "name": "calibration spiral"}
            else:
                record = artwork_record(entry, tuple(dimensions), width_mm=width_mm)
                points = [s.points_m for s in planar_strokes(record, mirrored=mirrored, rotation_rad=angle)]
                identity = {"design_id": entry["id"], "source_sha256": entry["sha256"],
                            "artwork_record_sha256": record["content_sha256"],
                            "family": entry["family"], "split": entry["split"], "name": entry["name"]}
            cost = pacing(config)[2] + _cost_s(points, config)
            if cost > budget_s:
                rejected.append({**identity, "size_mm": size, "reason": "duration", "estimated_s": cost, "budget_s": budget_s})
                continue
            extent = float(np.abs(np.concatenate(points)).max())
            bound = REACH - extent - width_mm / 2000
            if bound <= 0:
                continue
            for _ in range(40):
                offset = rng.uniform(-bound, bound, 2)
                placed = [p + offset for p in points]
                if reachable is not None and not reachable.ok(np.concatenate(placed)):
                    continue
                return [Stroke(p) for p in placed], {
                    "schema": "tatbot.artwork-scene/1", **identity,
                    "prompt": f"{verb} the {identity['name'].lower()} tattoo design",
                    "size_mm": dimensions.tolist(), "planning_width_mm": width_mm,
                    "rotation_rad": angle if entry is not None else 0,
                    "mirrored": mirrored if entry is not None else False,
                    "offset_m": offset.tolist(), "stroke_count": len(points),
                    "path_length_m": sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in points),
                    "est_cost_s": cost, "rejected_candidates": rejected,
                }
    raise RuntimeError("no whole artwork fits the requested horizon and reachable area; increase horizon")
