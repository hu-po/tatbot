#!/usr/bin/env python3
"""Compare OpenCV and Rust AprilTag detections on the same evidence sets.

The Rust JSONL is produced by ``tatbot-visiond detect-fiducials``.  This tool
reruns the shared Python detector against the hash-verified evidence and gates
ID agreement plus the direct TL/TR/BR/BL corner contract.  A best-permutation
metric is also reported to make a corner-order regression immediately obvious.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_tracker import evidence_sets  # noqa: E402
from fiducials import load_inventory  # noqa: E402
from fiducials.detector import DetectorConfig, FiducialDetector  # noqa: E402


def _timestamp_ns(metadata: dict) -> int:
    timestamps = metadata["timestamps"]
    for key in ("normalized_unix_ns", "source_ns", "host_unix_ns"):
        value = timestamps.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"{metadata.get('sensor_name')}: no usable timestamp")


def _load_rust(path: Path) -> dict[int, dict]:
    rows = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sequence = int(row["sequence"])
        if sequence in rows:
            raise ValueError(f"{path}:{line_number}: duplicate sequence {sequence}")
        rows[sequence] = row
    if not rows:
        raise ValueError(f"{path}: contains no detection rows")
    return rows


def _pair_same_id(python_items: list[dict], rust_items: list[dict]):
    """Greedily pair duplicate IDs by center; typical IDs have one instance."""
    remaining = list(rust_items)
    pairs = []
    for python_item in python_items:
        if not remaining:
            break
        center = np.asarray(python_item["corners_px"], dtype=float).mean(axis=0)
        index = min(
            range(len(remaining)),
            key=lambda item: np.linalg.norm(
                center - np.asarray(remaining[item]["corners_px"], dtype=float).mean(axis=0)
            ),
        )
        pairs.append((python_item, remaining.pop(index)))
    return pairs


def _corner_metrics(left: dict, right: dict) -> tuple[float, float, tuple[int, ...]]:
    python_corners = np.asarray(left["corners_px"], dtype=float)
    rust_corners = np.asarray(right["corners_px"], dtype=float)
    direct = float(np.sqrt(np.mean(np.square(python_corners - rust_corners))))
    candidates = []
    for permutation in itertools.permutations(range(4)):
        rmse = float(
            np.sqrt(np.mean(np.square(python_corners - rust_corners[list(permutation)])))
        )
        candidates.append((rmse, permutation))
    best, permutation = min(candidates)
    return direct, best, permutation


def compare(args: argparse.Namespace) -> tuple[dict, bool]:
    inventory = load_inventory(args.inventory)
    profile = inventory.detector_profiles[args.profile]
    scale = profile.scale if args.scale is None else args.scale
    target = inventory.target(args.target) if args.target else None
    allowed_ids = frozenset(target.ids) if target else inventory.known_ids
    detector = FiducialDetector(
        allowed_ids,
        DetectorConfig(
            scale=scale,
            adaptive_window_max=profile.adaptive_window_max,
            min_side_px=profile.min_side_px,
            corner_refinement=profile.corner_refinement,
        ),
        inventory.family,
    )
    rust_rows = _load_rust(args.rust_jsonl)
    mismatches = []
    direct_errors = []
    best_errors = []
    permutations: dict[str, int] = {}
    python_latency_ms = []
    rust_latency_ms = []
    compared_sets = 0
    compared_detections = 0
    python_instances = 0
    rust_instances = 0
    matched_instances = 0
    seen_sequences = set()
    missing_rust_rows = 0

    for frame_set in evidence_sets(args.evidence):
        if args.max_sets and compared_sets >= args.max_sets:
            break
        sequence = int(frame_set["sequence"])
        seen_sequences.add(sequence)
        rust_row = rust_rows.get(sequence)
        if rust_row is None:
            mismatches.append({"sequence": sequence, "reason": "missing_rust_row"})
            missing_rust_rows += 1
        else:
            if rust_row.get("inventory_hash") != inventory.inventory_hash:
                raise ValueError(
                    f"Rust row {sequence} inventory hash does not match {inventory.source}"
                )
            evidence_calibration_ids = {
                frame["metadata"].get("calibration_id")
                for frame in frame_set["frames"].values()
                if frame["metadata"].get("calibration_id") is not None
            }
            if evidence_calibration_ids and evidence_calibration_ids != {
                rust_row.get("calibration_id")
            }:
                raise ValueError(
                    f"Rust row {sequence} calibration id does not match evidence: "
                    f"{rust_row.get('calibration_id')!r} vs {sorted(evidence_calibration_ids)}"
                )
        started = time.perf_counter()
        python_by_camera: dict[str, list[dict]] = {}
        for camera, frame in frame_set["frames"].items():
            detections = detector.detect(
                camera,
                frame["image"],
                _timestamp_ns(frame["metadata"]),
            )
            python_by_camera[camera] = [
                {
                    "tag_id": item.tag_id,
                    "corners_px": item.corners_px.tolist(),
                    "side_px": item.side_px,
                }
                for item in detections
            ]
        python_latency_ms.append((time.perf_counter() - started) * 1000.0)
        if rust_row is not None:
            rust_latency_ms.append(float(rust_row.get("detection_latency_ms", 0.0)))
        rust_by_camera = rust_row.get("detections", {}) if rust_row is not None else {}
        for camera in sorted(set(python_by_camera) | set(rust_by_camera)):
            python_items = python_by_camera.get(camera, [])
            rust_items = [
                item for item in rust_by_camera.get(camera, [])
                if int(item["tag_id"]) in allowed_ids
            ]
            python_ids = sorted(int(item["tag_id"]) for item in python_items)
            rust_ids = sorted(int(item["tag_id"]) for item in rust_items)
            python_instances += len(python_ids)
            rust_instances += len(rust_ids)
            if python_ids != rust_ids:
                mismatches.append(
                    {
                        "sequence": sequence,
                        "camera": camera,
                        "reason": "id_multiset",
                        "python_ids": python_ids,
                        "rust_ids": rust_ids,
                    }
                )
            for tag_id in sorted(set(python_ids) & set(rust_ids)):
                left = [item for item in python_items if int(item["tag_id"]) == tag_id]
                right = [item for item in rust_items if int(item["tag_id"]) == tag_id]
                pairs = _pair_same_id(left, right)
                matched_instances += len(pairs)
                for python_item, rust_item in pairs:
                    direct, best, permutation = _corner_metrics(python_item, rust_item)
                    direct_errors.append(direct)
                    best_errors.append(best)
                    key = ",".join(map(str, permutation))
                    permutations[key] = permutations.get(key, 0) + 1
                    compared_detections += 1
        compared_sets += 1

    extra_rust_rows = sorted(set(rust_rows) - seen_sequences) if not args.max_sets else []
    for sequence in extra_rust_rows:
        mismatches.append({"sequence": sequence, "reason": "extra_rust_row"})

    def percentile(values: list[float], value: float) -> float | None:
        return float(np.percentile(values, value)) if values else None

    report = {
        "schema_version": 1,
        "inventory_hash": inventory.inventory_hash,
        "evidence": str(args.evidence.resolve()),
        "rust_jsonl": str(args.rust_jsonl.resolve()),
        "profile": args.profile,
        "target": args.target,
        "scale": scale,
        "compared_sets": compared_sets,
        "missing_rust_rows": missing_rust_rows,
        "extra_rust_rows": extra_rust_rows,
        "compared_detections": compared_detections,
        "python_instances": python_instances,
        "rust_instances": rust_instances,
        "matched_instances": matched_instances,
        "python_only_instances": python_instances - matched_instances,
        "rust_only_instances": rust_instances - matched_instances,
        "instance_disagreement_rate": (
            (python_instances + rust_instances - 2 * matched_instances)
            / max(1, python_instances + rust_instances)
        ),
        "id_mismatch_count": len(mismatches),
        "id_mismatches": mismatches[:100],
        "direct_corner_rmse_px": {
            "median": percentile(direct_errors, 50),
            "p95": percentile(direct_errors, 95),
            "max": max(direct_errors, default=None),
        },
        "best_permutation_corner_rmse_px": {
            "median": percentile(best_errors, 50),
            "p95": percentile(best_errors, 95),
        },
        "best_corner_permutations": permutations,
        "python_latency_ms": {
            "median": percentile(python_latency_ms, 50),
            "p95": percentile(python_latency_ms, 95),
        },
        "rust_latency_ms": {
            "median": percentile(rust_latency_ms, 50),
            "p95": percentile(rust_latency_ms, 95),
        },
        "gates": {
            "minimum_matched_instances": args.min_matched_instances,
            "maximum_instance_disagreement_rate": args.max_id_disagreement_rate,
            "maximum_direct_corner_p95_px": args.max_corner_p95_px,
        },
    }
    corner_p95 = report["direct_corner_rmse_px"]["p95"]
    passed = (
        compared_sets > 0
        and missing_rust_rows == 0
        and not extra_rust_rows
        and matched_instances >= args.min_matched_instances
        and report["instance_disagreement_rate"] <= args.max_id_disagreement_rate
        and corner_p95 is not None
        and corner_p95 <= args.max_corner_p95_px
    )
    report["passed"] = passed
    return report, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("rust_jsonl", type=Path)
    parser.add_argument("--inventory", type=Path, default=Path("config/fiducials.json"))
    parser.add_argument("--profile", choices=("calibration", "live"), default="live")
    parser.add_argument("--target", choices=("wrist", "board", "palette"))
    parser.add_argument("--scale", type=float)
    parser.add_argument("--max-sets", type=int, default=0)
    parser.add_argument("--min-matched-instances", type=int, default=20)
    parser.add_argument("--max-id-disagreement-rate", type=float, default=0.10)
    parser.add_argument("--max-corner-p95-px", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, passed = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
