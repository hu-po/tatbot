"""Materialize an Inkmap placement as a replayable posed tattoo scenario."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.contracts import document_sha256, validate_placement, validate_scenario
from tatbot_sim.inkmap.rig import BODY_ASSET_ROOT, load_body_rig
from tatbot_sim.inkmap.surface_trace import (
    SurfaceAnchor,
    anchor_frame,
    compile_surface_trace,
)
from tatbot_sim.inkmap.svg_strokes import compile_svg_strokes
from tatbot_sim.repo import repo_root

SCENARIO_SCHEMA_VERSION = 1
DEFAULT_TATTOO_TARGET_WORLD_M = np.array([0.29, 0.0, 0.04])
DEFAULT_PATCH_YAW_RAD = np.pi


class ScenarioCompileError(ValueError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _resolve_design(placement_file: dict, placement: dict) -> dict:
    design_id = placement["design_id"]
    embedded = placement_file.get("designs", {}).get(design_id)
    if embedded:
        svg = embedded["svg"]
        return {
            "id": design_id,
            "name": embedded["name"],
            "svg": svg,
            "sha256": _sha256_bytes(svg.encode()),
            "source": deepcopy(embedded.get("source", {"kind": "embedded"})),
        }
    manifest_path = BODY_ASSET_ROOT / "designs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    try:
        design = next(item for item in manifest["designs"] if item["id"] == design_id)
    except StopIteration as exc:
        raise ScenarioCompileError(f"design {design_id!r} is neither embedded nor built in") from exc
    path = BODY_ASSET_ROOT / design["path"]
    svg = path.read_text()
    return {
        "id": design_id,
        "name": design["name"],
        "svg": svg,
        "sha256": _sha256_bytes(svg.encode()),
        "source": {"kind": "builtin", "path": design["path"]},
    }


def _default_world_from_body(
    rig,
    pose_id: str,
    placement: dict,
    target_world_m,
    align_patch_up: bool,
    patch_yaw_rad: float,
) -> np.ndarray:
    posed = rig.posed(pose_id)
    source = placement["anchor"]
    anchor = SurfaceAnchor(source["face"], tuple(source["barycentric"]))
    point, normal, u_axis, v_axis = anchor_frame(
        posed.vertices, anchor.face, anchor.barycentric, float(placement["rotation_rad"]),
    )
    transform = posed.body_from_rest.copy()
    if align_patch_up:
        current = np.stack([u_axis, v_axis, normal], axis=1)
        c, s = np.cos(patch_yaw_rad), np.sin(patch_yaw_rad)
        desired = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotation_delta = desired @ current.T
        transform[:3, :3] = rotation_delta @ transform[:3, :3]
        point = rotation_delta @ point
    transform[:3, 3] = np.asarray(target_world_m, dtype=np.float64) - point
    return transform


def compile_scenario(
    placement_file: dict,
    *,
    placement_id: str | None = None,
    pose_id: str = "supine",
    seed: int = 0,
    world_from_body: np.ndarray | None = None,
    target_world_m=DEFAULT_TATTOO_TARGET_WORLD_M,
    align_patch_up: bool = True,
    patch_yaw_rad: float = DEFAULT_PATCH_YAW_RAD,
    tool_id: str = "lutin-3rl-bugpin",
    support_id: str | None = None,
    created_at: str | None = None,
    git_sha: str | None = None,
    generator: str = "tatbot sim compile",
) -> dict:
    validate_placement(placement_file)
    placements = placement_file["placements"]
    if not placements:
        raise ScenarioCompileError("placement file contains no placements")
    if placement_id is None:
        if len(placements) != 1:
            raise ScenarioCompileError("placement_id is required when the file contains multiple placements")
        placement = placements[0]
    else:
        try:
            placement = next(item for item in placements if item["id"] == placement_id)
        except StopIteration as exc:
            raise ScenarioCompileError(f"unknown placement id {placement_id!r}") from exc
    body = placement_file["body"]
    rig = load_body_rig(body["id"])
    if body["surface_sha256"] != rig.surface_sha256:
        raise ScenarioCompileError("placement surface digest does not match the rig rest surface")
    body_path = BODY_ASSET_ROOT / body["path"]
    if _sha256_file(body_path) != body["asset_sha256"]:
        raise ScenarioCompileError("placement body asset checksum mismatch")

    design = _resolve_design(placement_file, placement)
    metric = compile_svg_strokes(
        design["svg"], placement["size_mm"], mirror=placement["mirror"], rotation_rad=0.0,
    )
    trace = compile_surface_trace(rig, placement, metric.strokes)
    if world_from_body is None:
        world_from_body = _default_world_from_body(
            rig, pose_id, placement, target_world_m, align_patch_up, patch_yaw_rad,
        )
    else:
        world_from_body = np.asarray(world_from_body, dtype=np.float64)
    # Validate the transform and pose now, not on replay.
    rig.posed(pose_id, world_from_body)
    pose = rig.catalog_record["poses"][pose_id]
    urdf = repo_root() / "urdf" / "tatbot.urdf"
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if git_sha is None:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root(), text=True,
        ).strip()
    resolved_placement = deepcopy(placement)
    resolved_placement["source_sha256"] = document_sha256(placement_file)
    scenario = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "units": {"length": "m", "tattoo_size": "mm", "angle": "rad", "up": "+z", "matrix_order": "row-major"},
        "seed": int(seed),
        "body": {
            "id": body["id"], "path": body["path"], "asset_sha256": body["asset_sha256"],
            "surface_sha256": rig.surface_sha256, "rig_id": rig.rig_id,
            "rig_sha256": rig.catalog_record["sidecar_sha256"],
        },
        "pose": {
            "id": pose_id, "catalog_sha256": rig.catalog_sha256, "source": "named",
            "joint_rotations": deepcopy(pose["joint_rotations"]),
            "world_from_body": np.asarray(world_from_body).tolist(),
        },
        "placement": resolved_placement,
        "design": design,
        "trace": trace.as_dict(),
        "robot": {
            "urdf_sha256": _sha256_file(urdf), "tool_id": tool_id,
            "world_from_robot": np.eye(4).tolist(),
        },
        "support": {"id": support_id or pose["support_id"]},
        "provenance": {"created_at": created_at, "git_sha": git_sha, "generator": generator},
    }
    validate_scenario(scenario)
    return scenario
