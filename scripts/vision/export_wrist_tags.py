#!/usr/bin/env python3
"""Validate and publish one arm's canonical wrist layout plus its generated URDF block.

New calibration (the target names the physical arm's tag set):

    python scripts/vision/export_wrist_tags.py SESSION/robot_world.json --write --target wrist_left

Repository consistency, every target that declares a layout:

    python scripts/vision/export_wrist_tags.py --check

Each generated layout stores transforms from the one parent frame declared by
``targets.<target>.parent_frame``; the shared URDF carries one generated block
per target, so re-exporting one arm leaves the other arm's block byte for
byte. URDF origins and simulator poses are derived representations, never
separately calibrated values. ``--refresh-existing`` migrates a
legacy/provisional layout without claiming that its transforms are calibrated.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from fiducials import load_inventory  # noqa: E402
from urdf_wrist_blocks import (  # noqa: E402
    DEFAULT_TARGET,
    END_MARKER,
    block_begin,
    replace_urdf_block,
)

REPO = Path(__file__).resolve().parents[2]
URDF_PATH = REPO / "urdf" / "tatbot.urdf"


def target_layout_path(inventory, target: str) -> Path:
    layout = inventory.target(target).layout
    if not layout:
        raise ValueError(f"fiducial target {target!r} declares no layout file")
    return REPO / layout


def wrist_link_prefix(record: dict) -> str:
    parent = record.get("parent_frame") or ""
    prefix = parent.split("/", 1)[0]
    if prefix not in ("left", "right"):
        raise ValueError(f"wrist parent frame {parent!r} is not on a physical arm")
    return prefix


def marker_asset_relpath(family: str, tag_id: int, edge_m: float) -> str:
    edge_mm = edge_m * 1000
    if not float(edge_mm).is_integer():
        raise ValueError(f"viewer assets require an integer-mm black edge, got {edge_mm:g}")
    short_family = family.removeprefix("apriltag_")
    return f"meshes/tags/{short_family}_{tag_id:03d}_{int(edge_mm)}mm/tag.glb"


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rpy_extrinsic_xyz(rotation):
    """URDF fixed-axis roll-pitch-yaw with R = Rz(y) Ry(p) Rx(r)."""
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def _empty_pending_layout(record, require_calibrated):
    return (not require_calibrated and record.get("calibration_status") == "pending_recalibration"
            and not record.get("tags"))


def validate_record(record: dict, inventory, *, require_calibrated: bool, target: str = DEFAULT_TARGET,
                    allow_stale_hash: bool = False) -> None:
    wrist = inventory.target(target)
    if record.get("schema_version") != 2:
        raise ValueError(f"wrist layout schema must be 2, got {record.get('schema_version')!r}")
    if record.get("inventory_hash") != inventory.inventory_hash and not allow_stale_hash:
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
    if tag_ids != set(declared) and not _empty_pending_layout(record, require_calibrated):
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


def record_from_solve(solved: dict, source: Path, inventory, target: str = DEFAULT_TARGET) -> dict:
    wrist = inventory.target(target)
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
    validate_record(record, inventory, require_calibrated=True, target=target)
    return record


def normalize_existing(record: dict, inventory, target: str = DEFAULT_TARGET) -> dict:
    wrist = inventory.target(target)
    if (
        record.get("calibration_status") == "calibrated"
        and record.get("parent_frame") != wrist.parent_frame
    ):
        raise ValueError(
            "refusing to relabel calibrated wrist transforms into a different parent frame"
        )
    if record.get("calibration_status") == "calibrated":
        # A refresh re-stamps the inventory hash; the transforms stay valid as
        # long as the geometry the hash guards (ids, edge, parent) is unchanged,
        # which the remaining checks establish.
        validate_record(record, inventory, require_calibrated=True, target=target, allow_stale_hash=True)
    tags = {
        str(tag_id): {"ee_from_tag": record["tags"][str(tag_id)]["ee_from_tag"]}
        for tag_id in wrist.ids
    } if record.get("tags") else {}
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
        "source_link": record.get("source_link"),
        "source_metrics": record.get("source_metrics"),
        "tags": tags,
    }
    normalized = {key: value for key, value in normalized.items() if value is not None}
    validate_record(normalized, inventory, require_calibrated=False, target=target)
    return normalized


def render_urdf_block(
    record: dict,
    layout_sha256: str,
    target: str = DEFAULT_TARGET,
    family: str = "apriltag_36h11",
) -> str:
    status = record["calibration_status"]
    prefix = wrist_link_prefix(record)
    lines = [
        f"{block_begin(target)} layout_sha256={layout_sha256} -->",
        f"  <!-- status={status}; generated by scripts/vision/export_wrist_tags.py -->",
    ]
    if status != "calibrated":
        lines.append("  <!-- Tag placement withheld until fresh calibration; stored poses are historical/provisional. -->")
        lines.append(END_MARKER)
        return "\n".join(lines) + "\n"
    for tag_id, entry in sorted(record["tags"].items(), key=lambda item: int(item[0])):
        numeric_id = int(tag_id)
        transform = np.asarray(entry["ee_from_tag"], dtype=np.float64)
        roll, pitch, yaw = rpy_extrinsic_xyz(transform[:3, :3])
        x, y, z = transform[:3, 3]
        lines.extend(
            [
                f'  <link name="{prefix}/wrist_tag{tag_id}">',
                '    <visual>',
                '      <origin rpy="0 0 0" xyz="0 0 0.0002"/>',
                f'      <geometry><mesh filename="{marker_asset_relpath(family, numeric_id, record["edge_m"])}"/></geometry>',
                f'      <material name="fiducial_{tag_id}"><color rgba="1 1 1 1"/></material>',
                '    </visual>',
                '  </link>',
                f'  <joint name="{prefix}/wrist_tag{tag_id}_joint" type="fixed">',
                f'    <origin rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}" '
                f'xyz="{x:.8f} {y:.8f} {z:.8f}"/>',
                f'    <parent link="{record["parent_frame"]}"/>',
                f'    <child link="{prefix}/wrist_tag{tag_id}"/>',
                "  </joint>",
            ]
        )
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def serialized(record: dict) -> str:
    return json.dumps(record, indent=2) + "\n"


def check(layout_path: Path, urdf_path: Path, inventory, target: str = DEFAULT_TARGET) -> None:
    raw = layout_path.read_bytes()
    record = json.loads(raw)
    validate_record(record, inventory, require_calibrated=False, target=target)
    spec = inventory.target(target)
    for tag_id in spec.ids:
        # The marker meshes are tracked beside the repository URDF. A solve's
        # candidate copy (<capture>/candidate-config/urdf/tatbot.urdf) names
        # them by the same relative path and is only ever adopted onto that
        # URDF, so the repository's assets satisfy it too.
        relpath = marker_asset_relpath(spec.family, tag_id, spec.edge_m)
        if not any((base / relpath).is_file() for base in (urdf_path.parent, URDF_PATH.parent)):
            raise ValueError(f"missing viewer marker asset {urdf_path.parent / relpath}")
    expected = replace_urdf_block(
        urdf_path.read_text(),
        render_urdf_block(record, hashlib.sha256(raw).hexdigest(), target, spec.family),
        target,
    )
    if expected != urdf_path.read_text():
        raise ValueError(f"generated {target} URDF block is stale; run --refresh-existing --target {target}")


def check_all(urdf_path: Path, inventory) -> list[str]:
    """Every target that declares a layout must agree with the shared URDF."""
    checked = []
    for name, target in inventory.targets.items():
        if target.layout:
            check(REPO / target.layout, urdf_path, inventory, name)
            checked.append(name)
    if not checked:
        raise ValueError("no fiducial target declares a wrist layout")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_world", nargs="?", type=Path)
    parser.add_argument("--inventory", type=Path, default=REPO / "config" / "fiducials.json")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="fiducial target (physical arm's tag set); default: the right arm's `wrist`")
    parser.add_argument("--layout", type=Path, default=None,
                        help="layout file (default: the target's declared layout)")
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
    target = args.target
    if args.layout is None:
        args.layout = target_layout_path(inventory, target)

    if args.check:
        if args.target != DEFAULT_TARGET or args.layout != target_layout_path(inventory, DEFAULT_TARGET):
            check(args.layout, args.urdf, inventory, target)
            print(f"ok: {args.layout} and the generated {target} URDF block agree")
            return 0
        checked = check_all(args.urdf, inventory)
        print(f"ok: generated wrist URDF blocks agree with their layouts ({', '.join(checked)})")
        return 0
    if args.refresh_existing:
        if args.robot_world:
            parser.error("--refresh-existing does not take robot_world")
        record = normalize_existing(json.loads(args.layout.read_text()), inventory, target)
    elif args.robot_world:
        record = record_from_solve(
            json.loads(args.robot_world.read_text()), args.robot_world, inventory, target
        )
    else:
        parser.error("provide robot_world, --refresh-existing, or --check")

    layout_text = serialized(record)
    layout_sha256 = hashlib.sha256(layout_text.encode()).hexdigest()
    family = inventory.target(target).family
    urdf_text = replace_urdf_block(
        args.urdf.read_text(),
        render_urdf_block(record, layout_sha256, target, family),
        target,
    )
    print(json.dumps(record, indent=2))
    print(f"\nURDF generated {target} block:\n")
    print(render_urdf_block(record, layout_sha256, target, family), end="")
    if not (args.write or args.refresh_existing):
        print("\ndry run — pass --write to update the layout and URDF")
        return 0
    atomic_write(args.layout, layout_text)
    atomic_write(args.urdf, urdf_text)
    check(args.layout, args.urdf, inventory, target)
    print(f"\nwrote {args.layout} and {args.urdf}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        sys.exit(f"REFUSE: {error}")
