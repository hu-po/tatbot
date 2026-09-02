"""Numerical access to Inkmap's checked-in humanoid rig and named poses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tatbot_sim.repo import repo_root

CATALOG_PATH = repo_root() / "config" / "inkmap" / "body-poses.json"
BODY_ASSET_ROOT = repo_root() / "web" / "inkmap" / "public"


class BodyRigError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quat_matrix(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-20:
        return np.eye(3)
    s = 2.0 / n
    return np.asarray([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class PosedBody:
    body_id: str
    pose_id: str
    surface_sha256: str
    vertices: np.ndarray
    """Canonical faces, shape (F, 3, 3), in world Z-up metres."""
    body_from_rest: np.ndarray
    part_names: tuple[str, ...]
    part_first_face: np.ndarray
    part_face_count: np.ndarray

    def point(self, face: int, barycentric) -> np.ndarray:
        bary = np.asarray(barycentric, dtype=np.float64)
        if face < 0 or face >= len(self.vertices):
            raise BodyRigError(f"face {face} is out of range")
        if bary.shape != (3,) or np.any(bary < -1e-8) or not np.isclose(bary.sum(), 1.0, atol=1e-6):
            raise BodyRigError("barycentric coordinates must be three normalized nonnegative values")
        return bary @ self.vertices[face]


@dataclass(frozen=True)
class BodyRig:
    body_id: str
    rig_id: str
    surface_sha256: str
    catalog_sha256: str
    catalog_record: dict
    bone_names: tuple[str, ...]
    rest_vertices: np.ndarray
    joint_indices: np.ndarray
    joint_weights: np.ndarray
    pose_ids: tuple[str, ...]
    pose_vertices: np.ndarray
    pose_matrices: np.ndarray
    part_names: tuple[str, ...]
    part_first_face: np.ndarray
    part_face_count: np.ndarray

    def posed(self, pose_id: str, world_from_body: np.ndarray | None = None) -> PosedBody:
        try:
            pose_index = self.pose_ids.index(pose_id)
            pose = self.catalog_record["poses"][pose_id]
        except (ValueError, KeyError) as exc:
            raise BodyRigError(f"{self.body_id}: unknown pose {pose_id!r}") from exc
        vertices = np.asarray(self.pose_vertices[pose_index], dtype=np.float64)
        body_from_rest = np.eye(4)
        body_from_rest[:3, :3] = _quat_matrix(pose["body_rotation_xyzw"])
        if world_from_body is not None:
            body_from_rest = np.asarray(world_from_body, dtype=np.float64)
            if body_from_rest.shape != (4, 4) or not np.allclose(body_from_rest[3], [0, 0, 0, 1]):
                raise BodyRigError("world_from_body must be a homogeneous 4x4 matrix")
        vertices = np.einsum("ij,...j->...i", body_from_rest[:3, :3], vertices)
        vertices += body_from_rest[:3, 3]
        return PosedBody(
            body_id=self.body_id,
            pose_id=pose_id,
            surface_sha256=self.surface_sha256,
            vertices=vertices.astype(np.float32),
            body_from_rest=body_from_rest,
            part_names=self.part_names,
            part_first_face=self.part_first_face.copy(),
            part_face_count=self.part_face_count.copy(),
        )


@lru_cache(maxsize=4)
def load_body_rig(body_id: str) -> BodyRig:
    catalog_bytes = CATALOG_PATH.read_bytes()
    catalog = json.loads(catalog_bytes)
    try:
        record = catalog["bodies"][body_id]
    except KeyError as exc:
        raise BodyRigError(f"unknown rigged body {body_id!r}") from exc
    sidecar = BODY_ASSET_ROOT / record["sidecar_path"]
    if _sha256(sidecar) != record["sidecar_sha256"]:
        raise BodyRigError(f"{body_id}: rig sidecar checksum mismatch")
    with np.load(sidecar, allow_pickle=False) as data:
        values = {name: data[name].copy() for name in data.files}
    surface_sha = str(values["surface_sha256"])
    if surface_sha != record["surface_sha256"]:
        raise BodyRigError(f"{body_id}: sidecar rest-surface digest mismatch")
    weights = values["joint_weights"]
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=2e-7):
        raise BodyRigError(f"{body_id}: joint weights are not finite and normalized")
    pose_ids = tuple(str(value) for value in values["pose_ids"])
    if pose_ids != tuple(catalog["pose_ids"]):
        raise BodyRigError(f"{body_id}: sidecar and catalog pose order differ")
    return BodyRig(
        body_id=body_id,
        rig_id=str(values["rig_id"]),
        surface_sha256=surface_sha,
        catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
        catalog_record=record,
        bone_names=tuple(str(value) for value in values["bone_names"]),
        rest_vertices=values["rest_vertices"],
        joint_indices=values["joint_indices"],
        joint_weights=weights,
        pose_ids=pose_ids,
        pose_vertices=values["pose_vertices"],
        pose_matrices=values["pose_matrices"],
        part_names=tuple(str(value) for value in values["part_names"]),
        part_first_face=values["part_first_face"],
        part_face_count=values["part_face_count"],
    )
