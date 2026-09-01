#!/usr/bin/env python3
"""Validate and publish one canonical wrist layout plus its generated URDF.

New calibration:

    python scripts/vision/export_wrist_tags.py SESSION/robot_world.json --write

Repository consistency:

    python scripts/vision/export_wrist_tags.py --check

The generated layout stores transforms from the one parent frame declared by
``targets.wrist.parent_frame``. URDF origins and simulator poses are derived
representations, never separately calibrated values. ``--refresh-existing``
migrates a legacy/provisional layout without claiming that its transforms are
calibrated.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAYOUT_PATH = REPO / "config" / "wrist_tags_measured.json"
URDF_PATH = REPO / "urdf" / "tatbot.urdf"
BEGIN_PREFIX = "  <!-- BEGIN GENERATED WRIST FIDUCIALS"
END_MARKER = "  <!-- END GENERATED WRIST FIDUCIALS -->"
LEGACY_BEGIN = "  <!-- Provisional geometry for the new three-fiducial wrist mount"
LEGACY_END = '  <joint name="right/realsense_depth_joint"'


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rpy_extrinsic_xyz(rotation):
    """URDF fixed-axis roll-pitch-yaw with R = Rz(y) Ry(p) Rx(r)."""
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def validate_record(record: dict, inventory, *, require_calibrated: bool) -> None:
    wrist = inventory.target("wrist")
    if record.get("schema_version") != 2:
        raise ValueError(f"wrist layout schema must be 2, got {record.get('schema_version')!r}")
    if record.get("inventory_hash") != inventory.inventory_hash:
        raise ValueError("wrist layout inventory hash is stale")
    if not wrist.parent_frame:
        raise ValueError("canonical wrist target has no parent_frame")
    if record.get("parent_frame") != wrist.parent_frame:
        raise ValueError(
            f"wrist layout parent_frame must be {wrist.parent_frame}"
        )
    if abs(float(record.get("edge_m", 0)) - wrist.edge_m) > 1e-9:
        raise ValueError("wrist layout edge does not match canonical inventory")
    declared = tuple(int(tag_id) for tag_id in record.get("target_ids", ()))
    if declared != wrist.ids:
        raise ValueError(f"wrist layout ids must be {list(wrist.ids)}, got {list(declared)}")
    if require_calibrated and record.get("calibration_status") != "calibrated":
        raise ValueError(f"wrist layout is {record.get('calibration_status')}, not calibrated")
    tag_ids = {int(tag_id) for tag_id in record.get("tags", {})}
    if tag_ids != set(declared):
        raise ValueError(f"wrist layout tag transforms must be exactly {list(declared)}")
    for tag_id, entry in record["tags"].items():
        transform = np.asarray(entry.get("ee_from_tag"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"tag {tag_id} parent_from_tag must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError(f"tag {tag_id} parent_from_tag has an invalid homogeneous row")
        if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError(f"tag {tag_id} parent_from_tag rotation is not orthonormal")


def quality_gate(solved: dict, wrist) -> None:
    solved_ids = {int(tag_id) for tag_id in solved.get("link_from_tag", {})}
    expected = set(wrist.ids)
    if solved_ids != expected:
        raise ValueError(
            f"wrist solve must contain exactly {sorted(expected)}; "
            f"missing={sorted(expected - solved_ids)}, unexpected={sorted(solved_ids - expected)}"
        )
    observations = int(solved.get("observations") or 0)
    minimum = wrist.minimum_calibration_observations or 4
    if observations < minimum:
        raise ValueError(f"wrist solve has {observations} observations; need at least {minimum}")
    minimum_per_id = wrist.minimum_calibration_poses_per_id or 1
    pose_counts = {
        int(tag_id): int(count)
        for tag_id, count in solved.get("pose_observations_by_tag", {}).items()
    }
    under_observed = {
        tag_id: pose_counts.get(tag_id, 0)
        for tag_id in expected
        if pose_counts.get(tag_id, 0) < minimum_per_id
    }
    if under_observed:
        raise ValueError(
            "wrist solve needs at least "
            f"{minimum_per_id} distinct arm poses per id; observed {under_observed}"
        )
    corner_px = solved.get("corner_px_median")
    if (
        corner_px is not None
        and wrist.max_calibration_corner_px is not None
        and float(corner_px) > wrist.max_calibration_corner_px
    ):
        raise ValueError(
            f"wrist solve corner median {corner_px} px exceeds {wrist.max_calibration_corner_px} px"
        )
    residual_mm = solved.get("residual_mm_median")
    if residual_mm is None or (
        wrist.max_calibration_residual_mm is not None
        and float(residual_mm) > wrist.max_calibration_residual_mm
    ):
        raise ValueError(
            f"wrist solve residual {residual_mm!r} mm exceeds "
            f"{wrist.max_calibration_residual_mm} mm"
        )


def record_from_solve(solved: dict, source: Path, inventory) -> dict:
    wrist = inventory.target("wrist")
    quality_gate(solved, wrist)
    link = solved["link"]
    parent = wrist.parent_frame
    if not parent:
        raise ValueError("canonical wrist target has no parent_frame")
    # A conversion between two moving links would require the carriage value
    # at which the source transform was solved. Fail closed and solve directly
    # in the physical mount frame instead.
    if link != parent:
        raise ValueError(
            f"wrist solve link {link!r} must equal configured parent_frame {parent!r}"
        )
    tags = {}
    max_parent_distance_mm = wrist.max_calibration_parent_distance_mm or 150.0
    for tag_id, matrix in sorted(solved["link_from_tag"].items(), key=lambda item: int(item[0])):
        parent_from_tag = np.asarray(matrix, dtype=np.float64)
        distance_mm = float(np.linalg.norm(parent_from_tag[:3, 3])) * 1000
        if distance_mm >= max_parent_distance_mm:
            raise ValueError(
                f"tag {tag_id} is implausibly far from {parent}: {distance_mm:.1f} mm "
                f">= configured {max_parent_distance_mm:.1f} mm"
            )
        # The schema-2 key is retained for Python/Rust wire compatibility;
        # parent_frame defines what the historical `ee` token means.
        tags[str(tag_id)] = {"ee_from_tag": parent_from_tag.tolist()}
    record = {
        "schema_version": 2,
        "calibration_status": "calibrated",
        "generated_utc": utcnow(),
        "inventory_hash": inventory.inventory_hash,
        "target_ids": list(wrist.ids),
        "edge_m": wrist.edge_m,
        "parent_frame": parent,
        "source": str(source.resolve()),
        "source_link": link,
        "source_metrics": {
            key: solved.get(key)
            for key in (
                "observations",
                "pose_observations_by_tag",
                "mode",
                "corner_px_median",
                "residual_mm_median",
                "residual_mm_max",
            )
        },
        "tags": tags,
    }
    validate_record(record, inventory, require_calibrated=True)
    return record


def normalize_existing(record: dict, inventory) -> dict:
    wrist = inventory.target("wrist")
    if (
        record.get("calibration_status") == "calibrated"
        and record.get("parent_frame") != wrist.parent_frame
    ):
        raise ValueError(
            "refusing to relabel calibrated wrist transforms into a different parent frame"
        )
    tags = {
        str(tag_id): {"ee_from_tag": record["tags"][str(tag_id)]["ee_from_tag"]}
        for tag_id in wrist.ids
    }
    normalized = {
        "schema_version": 2,
        "calibration_status": record.get("calibration_status", "pending_recalibration"),
        "generated_utc": record.get("generated_utc") or utcnow(),
        "inventory_hash": inventory.inventory_hash,
        "target_ids": list(wrist.ids),
        "edge_m": wrist.edge_m,
        "parent_frame": wrist.parent_frame,
        "note": record.get("note"),
        "source": record.get("source") or record.get("provisional_source"),
        "tags": tags,
    }
    normalized = {key: value for key, value in normalized.items() if value is not None}
    validate_record(normalized, inventory, require_calibrated=False)
    return normalized


def render_urdf_block(record: dict, layout_sha256: str) -> str:
    status = record["calibration_status"]
    lines = [
        f"{BEGIN_PREFIX} layout_sha256={layout_sha256} -->",
        f"  <!-- status={status}; generated by scripts/vision/export_wrist_tags.py -->",
    ]
    for tag_id, entry in sorted(record["tags"].items(), key=lambda item: int(item[0])):
        transform = np.asarray(entry["ee_from_tag"], dtype=np.float64)
        roll, pitch, yaw = rpy_extrinsic_xyz(transform[:3, :3])
        x, y, z = transform[:3, 3]
        lines.extend(
            [
                f'  <link name="right/wrist_tag{tag_id}"/>',
                f'  <joint name="right/wrist_tag{tag_id}_joint" type="fixed">',
                f'    <origin rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}" '
                f'xyz="{x:.8f} {y:.8f} {z:.8f}"/>',
                f'    <parent link="{record["parent_frame"]}"/>',
                f'    <child link="right/wrist_tag{tag_id}"/>',
                "  </joint>",
            ]
        )
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def replace_urdf_block(text: str, block: str) -> str:
    begin = text.find(BEGIN_PREFIX)
    if begin >= 0:
        end = text.find(END_MARKER, begin)
        if end < 0:
            raise ValueError("URDF generated wrist block has no end marker")
        end += len(END_MARKER)
        if end < len(text) and text[end] == "\n":
            end += 1
        return text[:begin] + block + text[end:]
    begin = text.find(LEGACY_BEGIN)
    end = text.find(LEGACY_END, begin)
    if begin < 0 or end < 0:
        raise ValueError("URDF has neither generated nor recognized legacy wrist block")
    return text[:begin] + block + text[end:]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def serialized(record: dict) -> str:
    return json.dumps(record, indent=2) + "\n"


def check(layout_path: Path, urdf_path: Path, inventory) -> None:
    raw = layout_path.read_bytes()
    record = json.loads(raw)
    validate_record(record, inventory, require_calibrated=False)
    expected = replace_urdf_block(
        urdf_path.read_text(), render_urdf_block(record, hashlib.sha256(raw).hexdigest())
    )
    if expected != urdf_path.read_text():
        raise ValueError("generated wrist URDF block is stale; run --refresh-existing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_world", nargs="?", type=Path)
    parser.add_argument("--inventory", type=Path, default=REPO / "config" / "fiducials.json")
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--urdf", type=Path, default=URDF_PATH)
    parser.add_argument("--write", action="store_true", help="publish a quality-gated calibrated layout")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="canonicalize the existing layout and regenerate its URDF without changing status",
    )
    parser.add_argument("--check", action="store_true", help="verify layout and generated URDF agree")
    args = parser.parse_args()
    inventory = load_inventory(args.inventory)

    if args.check:
        check(args.layout, args.urdf, inventory)
        print(f"ok: {args.layout} and generated wrist URDF agree")
        return 0
    if args.refresh_existing:
        if args.robot_world:
            parser.error("--refresh-existing does not take robot_world")
        record = normalize_existing(json.loads(args.layout.read_text()), inventory)
    elif args.robot_world:
        record = record_from_solve(
            json.loads(args.robot_world.read_text()), args.robot_world, inventory
        )
    else:
        parser.error("provide robot_world, --refresh-existing, or --check")

    layout_text = serialized(record)
    layout_sha256 = hashlib.sha256(layout_text.encode()).hexdigest()
    urdf_text = replace_urdf_block(args.urdf.read_text(), render_urdf_block(record, layout_sha256))
    print(json.dumps(record, indent=2))
    print("\nURDF generated wrist block:\n")
    print(render_urdf_block(record, layout_sha256), end="")
    if not (args.write or args.refresh_existing):
        print("\ndry run — pass --write to update the layout and URDF")
        return 0
    atomic_write(args.layout, layout_text)
    atomic_write(args.urdf, urdf_text)
    check(args.layout, args.urdf, inventory)
    print(f"\nwrote {args.layout} and {args.urdf}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        sys.exit(f"REFUSE: {error}")
