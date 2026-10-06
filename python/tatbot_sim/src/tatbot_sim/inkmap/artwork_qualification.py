"""Reproducible artwork/paint/trajectory/deposition qualification without arms."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import Polygon, union_all

from tatbot_sim.config import ARTWORK_HORIZON_STEPS
from tatbot_sim.human_rep.contracts import FILL_STYLES
from tatbot_sim.human_rep.fill_geometry import deposited_coverage
from tatbot_sim.inkfield import InkField
from tatbot_sim.inkmap.artwork import artwork_preview
from tatbot_sim.inkmap.collection import (
    COLLECTION_PATH,
    artwork_record,
    collection_entries,
    planar_placement,
    planar_strokes,
)
from tatbot_sim.inkmap.parity_evidence import mask_metrics
from tatbot_sim.inkmap.program_target import render_program_target
from tatbot_sim.repo import repo_root, source_state
from tatbot_sim.strokes import ShapeConfig, Stroke, build_ee_trajectory
from tatbot_sim.surface import PlanarSurface

PIXELS_PER_M = 48_000


def qualify(output: Path, *, horizon: int = ARTWORK_HORIZON_STEPS, seed: int = 42, design_ids: tuple[str, ...] = (),
            fill_style: str = "concentric") -> dict:
    entries = collection_entries()
    if horizon <= 0 or set(design_ids) - {entry["id"] for entry in entries} or fill_style not in FILL_STYLES:
        raise ValueError(f"qualification requires a positive horizon, known artwork IDs and a fill style in {FILL_STYLES}")
    if output.resolve().is_relative_to(repo_root().resolve()):
        raise ValueError("qualification output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    rows, jobs = [], []
    for entry in entries:
        if design_ids and entry["id"] not in design_ids:
            continue
        for size in entry["size_range_mm"]:
            dimensions = np.asarray(entry["default_size_mm"], dtype=float)
            dimensions *= size / max(dimensions)
            row = {"id": entry["id"], "family": entry["family"], "split": entry["split"],
                   "source_sha256": entry["sha256"], "size_mm": size,
                   "canvas_mm": dimensions.tolist(),
                   "planning_width_mm": entry["planning_width_mm"], "fill_style": fill_style}
            directory = output / f"{entry['id']}-{size}mm"
            directory.mkdir()
            row["directory"] = directory.name
            try:
                record = artwork_record(entry, tuple(dimensions))
                placed = planar_placement(record)
                strokes = planar_strokes(record, fill_style=fill_style)
                target = render_program_target(record["program"], placed, pixels_per_m=PIXELS_PER_M, supersample=4)
                width = entry["planning_width_mm"] / 1000
                paths = [s.points_m for s in strokes]
                coverage = deposited_coverage(paths, width)
                # Independent original paint-region union, in the chart frame.
                offset = dimensions / 2000
                paint = union_all([Polygon(np.asarray(element["points_m"]) - offset)
                                   for layer in record["program"]["layers"] for element in layer["elements"]])
                row["ideal_footprint_iou"] = coverage.intersection(paint).area / coverage.union(paint).area
                row["stroke_count"] = len(strokes)
                row["pen_lifts"] = max(0, len(strokes) - 1)
                row["path_length_mm"] = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in paths) * 1000
                # Fixed measured-band nominal speed; do not time-compress a design.
                trajectory = build_ee_trajectory([Stroke(p) for p in paths], np.random.default_rng(seed),
                    replace(ShapeConfig(), draw_speed_range=(.007, .007)))
                row["duration_s"] = len(trajectory.positions) / 30
                row["fits_horizon"] = len(trajectory.positions) <= horizon
                row["speed_mm_s"] = 7
                surface = PlanarSurface(torch.zeros((1, 3)), torch.eye(3)[None], dimensions[0] / 1000, dimensions[1] / 1000,
                                        target.cols, target.rows)
                field = InkField(1, surface, torch.tensor([width / 2]), torch.tensor([width / 2]), torch.zeros(1, 3))
                for position, active in zip(trajectory.positions, trajectory.pen_down, strict=True):
                    field.deposit_segment(surface, torch.tensor(position[None, :2]), torch.ones(1), torch.tensor([bool(active)]))
                deposited = field.field[0].numpy()
                row["deposition"] = mask_metrics(target.coverage > .5, deposited > .5)
                (directory / "artwork.json").write_text(json.dumps(record, indent=2) + "\n")
                (directory / "original.svg").write_text(entry["svg"])
                (directory / "preview.svg").write_text(artwork_preview(record))
                for name in ("original", "preview"):
                    jobs.append({"svg": str(directory / f"{name}.svg"), "png": str(directory / f"{name}.png"),
                                 "width": target.cols, "height": target.rows})
                for name, mask in (("target", target.coverage), ("deposited", deposited)):
                    cv2.imwrite(str(directory / f"{name}.png"), np.rint(255 * (1 - mask[::-1])).astype(np.uint8))
                np.savez_compressed(directory / "trajectory.npz", positions=trajectory.positions, pen_down=trajectory.pen_down)
                row["status"] = "compiled"
            except (ValueError, RuntimeError) as exc:
                row.update(status="rejected", reason=str(exc))
            rows.append(row)
            print(f"{row['id']} {size} mm: {row['status']}", flush=True)
    request = output / "render-jobs.json"
    request.write_text(json.dumps(jobs))
    subprocess.run(["node", "web/inkmap/tools/artwork-render.mjs", str(request)], cwd=repo_root(), check=True, timeout=180)
    contact_rows = []
    for row in rows:
        if row["status"] != "compiled":
            continue
        directory = output / row["directory"]
        original = cv2.imread(str(directory / "original.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127
        preview = cv2.imread(str(directory / "preview.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127
        target = cv2.imread(str(directory / "target.png"), cv2.IMREAD_GRAYSCALE) < 128
        row["source_preview"] = mask_metrics(original, preview)
        row["preview_target"] = mask_metrics(preview, target)
        row["status"] = "accepted" if (row["ideal_footprint_iou"] >= .98 and row["fits_horizon"]
            and row["source_preview"]["mask_iou"] >= .98 and row["preview_target"]["mask_iou"] >= .98
            and row["deposition"]["mask_iou"] >= .90) else "rejected"
        if row["status"] == "rejected":
            row["reason"] = "qualification metric or episode budget below declared threshold"
        entry = next(entry for entry in entries if entry["id"] == row["id"])
        if row["size_mm"] == entry["size_range_mm"][0]:
            cells = []
            for name in ("original", "preview", "target", "deposited"):
                im = cv2.imread(str(directory / f"{name}.png"), cv2.IMREAD_UNCHANGED)
                if im.ndim == 3:
                    im = 255 - im[..., 3]
                height, width = im.shape
                scale = 220 / max(height, width)
                im = cv2.resize(im, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
                cell = np.full((254, 240), 255, np.uint8)
                height, width = im.shape
                y, x = 30 + (220 - height) // 2, 10 + (220 - width) // 2
                cell[y:y + height, x:x + width] = im
                cv2.putText(cell, f"{row['id']} {name}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .36, 0, 1)
                cells.append(cell)
            contact_rows.append(np.concatenate(cells, axis=1))
    if contact_rows:
        cv2.imwrite(str(output / "comparison.png"), np.concatenate(contact_rows, axis=0))
    report = {"schema": "tatbot.artwork-qualification/1", "source": source_state(),
              "collection": json.loads(COLLECTION_PATH.read_text()), "seed": seed, "horizon": horizon,
              "fill_style": fill_style,
              "scope": "CPU planar expert trajectory and runtime InkField; browser SVG paint; no robot IK or GPU scene",
              "pixels_per_m": PIXELS_PER_M, "thresholds": {"paint_iou": .98, "runtime_deposition_iou": .90},
              "accepted": sum(row["status"] == "accepted" for row in rows), "requested": len(rows), "cases": rows}
    report["complete"] = report["accepted"] == report["requested"]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "rejections.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows if row["status"] != "accepted"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=ARTWORK_HORIZON_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--design-id", action="append", default=[])
    parser.add_argument("--fill-style", choices=FILL_STYLES, default="concentric",
                        help="paint planner to qualify: concentric inset rings (default) or contour-then-hatch")
    args = parser.parse_args()
    report = qualify(args.output_dir.expanduser().resolve(), horizon=args.horizon, seed=args.seed,
                     design_ids=tuple(args.design_id), fill_style=args.fill_style)
    print(f"artwork qualification: {report['accepted']}/{report['requested']}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
