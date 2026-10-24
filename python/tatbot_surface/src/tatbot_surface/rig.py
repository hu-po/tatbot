"""Numerical access to the reference SOMA rig: rest surface plus baked pose bytes.

Generating a rig for another identity needs the body model itself (torch and
SOMA), so that path is not here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from tatbot_contracts.digest import sha256_file

from tatbot_surface.gltf_surface import (
    MODEL_ID,
    MODEL_SPEC_SHA256,
    REST_SURFACE_SHA256,
    TOPOLOGY_SHA256,
    load_canonical_surface,
)
from tatbot_surface.repo import repo_root

CATALOG_PATH = repo_root() / "config" / "inkmap" / "body-poses.json"
CATALOG_DIGEST_PATH = CATALOG_PATH.with_name("body-poses.digest.json")
BODY_ASSET_ROOT = repo_root() / "web" / "inkmap" / "public"


class BodyRigError(ValueError):
    """The sole body or its derived pose cache failed closed."""


def _quat_matrix(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=np.float64)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-20:
        raise BodyRigError("pose_unsupported: zero body quaternion")
    scale = 2.0 / norm
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - w * z), scale * (x * z + w * y)],
            [scale * (x * y + w * z), 1 - scale * (x * x + z * z), scale * (y * z - w * x)],
            [scale * (x * z - w * y), scale * (y * z + w * x), 1 - scale * (x * x + y * y)],
        ]
    )


def _surface_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    indexed = np.empty((18_056, 3), dtype="<f4")
    seen = np.zeros(18_056, dtype=bool)
    expanded = np.asarray(vertices, dtype="<f4").reshape(-1, 3)
    corners = np.asarray(faces).reshape(-1).astype(np.intp)
    indexed[corners] = expanded
    seen[corners] = True
    differs = np.flatnonzero(~np.all(indexed[corners] == expanded, axis=1))
    if differs.size:
        raise BodyRigError(f"body_topology_mismatch: vertex {int(corners[differs[0]])} differs across faces")
    if not seen.all():
        raise BodyRigError("body_topology_mismatch: pose cache omits canonical vertices")
    quantized = np.rint(indexed / 0.00001).astype("<i8")
    header = b"dtype=<i8;shape=18056,3;order=C;quantization_m=0.00001;axes=x,-z,y\n"
    return hashlib.sha256(header + quantized.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PosedBody:
    body_id: str
    pose_id: str
    surface_sha256: str
    vertices: np.ndarray
    """Canonical faces, shape ``(36108, 3, 3)``, in world Z-up metres."""

    body_from_rest: np.ndarray
    part_names: tuple[str, ...]
    part_first_face: np.ndarray
    part_face_count: np.ndarray

    def point(self, face: int, barycentric) -> np.ndarray:
        return self.points(np.asarray(face), np.asarray(barycentric))

    def points(self, faces, barycentric) -> np.ndarray:
        """Resolve canonical face addresses on this exact posed realization."""
        indices = np.asarray(faces)
        bary = np.asarray(barycentric, dtype=np.float64)
        if (not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0)
                or np.any(indices >= len(self.vertices))):
            raise BodyRigError("surface_coordinate_invalid: face is out of range")
        if (bary.shape != (*indices.shape, 3) or not np.isfinite(bary).all()
                or np.any(bary < -1e-8) or not np.allclose(bary.sum(axis=-1), 1.0, atol=1e-6)):
            raise BodyRigError(
                "surface_coordinate_invalid: barycentric coordinates must be normalized"
            )
        return np.einsum("...i,...ij->...j", bary, self.vertices[indices])


@dataclass(frozen=True)
class BodyRig:
    """One immutable identity-bound SOMA surface plus deterministic pose bytes."""

    body_id: str
    rig_id: str
    surface_sha256: str
    topology_sha256: str
    model_spec_sha256: str
    identity_sha256: str
    catalog_sha256: str
    catalog_record: dict
    rest_vertices: np.ndarray
    faces: np.ndarray
    pose_ids: tuple[str, ...]
    pose_vertices: np.ndarray
    part_names: tuple[str, ...]
    part_first_face: np.ndarray
    part_face_count: np.ndarray

    def indexed_vertices(self, pose_id: str | None = None) -> np.ndarray:
        """Preserve upstream vertex IDs for a rest or unrotated articulated pose."""
        if pose_id is not None and pose_id not in self.pose_ids:
            raise BodyRigError(f"pose_unsupported: {pose_id!r}")
        expanded = (self.rest_vertices if pose_id is None else self.pose_vertices[self.pose_ids.index(pose_id)])
        vertices = np.empty((int(self.faces.max()) + 1, 3), dtype=np.float64)
        vertices[self.faces.reshape(-1)] = expanded.reshape(-1, 3)
        return vertices

    def posed(self, pose_id: str, world_from_body: np.ndarray | None = None) -> PosedBody:
        try:
            pose_index = self.pose_ids.index(pose_id)
            pose = self.catalog_record["poses"][pose_id]
        except (ValueError, KeyError) as exc:
            raise BodyRigError(f"pose_unsupported: {pose_id!r}") from exc
        vertices = np.asarray(self.pose_vertices[pose_index], dtype=np.float64)
        body_from_rest = np.eye(4)
        body_from_rest[:3, :3] = _quat_matrix(pose["body_rotation_xyzw"])
        if world_from_body is not None:
            body_from_rest = np.asarray(world_from_body, dtype=np.float64)
            if body_from_rest.shape != (4, 4) or not np.allclose(
                body_from_rest[3], [0, 0, 0, 1]
            ):
                raise BodyRigError("body_units_or_axes_invalid: world_from_body must be 4x4")
        world = np.einsum("ij,...j->...i", body_from_rest[:3, :3], vertices)
        world += body_from_rest[:3, 3]
        return PosedBody(
            body_id=self.body_id,
            pose_id=pose_id,
            surface_sha256=self.surface_sha256,
            vertices=world.astype(np.float32),
            body_from_rest=body_from_rest,
            part_names=self.part_names,
            part_first_face=self.part_first_face.copy(),
            part_face_count=self.part_face_count.copy(),
        )


@lru_cache(maxsize=1)
def load_body_rig() -> BodyRig:
    """Load the canonical browser/execution reference rig with no fallback."""
    catalog_bytes = CATALOG_PATH.read_bytes()
    catalog = json.loads(catalog_bytes)
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    # The browser binds scenarios to the digest published beside the catalog
    # (it cannot hash a JSON import); refuse a digest file that no longer
    # matches the bytes so both sides always agree on one value.
    digest = json.loads(CATALOG_DIGEST_PATH.read_bytes())
    if (
        digest.get("schema") != "tatbot.body-pose-catalog-digest/1"
        or digest.get("sha256") != catalog_sha256
    ):
        raise BodyRigError("body_model_unpinned: pose catalog digest file is stale")
    expected = {
        "schema": "tatbot.body-pose-catalog/2",
        "model_spec_id": MODEL_ID,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
        "rest_surface_sha256": REST_SURFACE_SHA256,
    }
    if any(catalog.get(key) != value for key, value in expected.items()):
        raise BodyRigError("body_model_unpinned: pose catalog contract mismatch")
    rest_record = catalog.get("rest_asset", {})
    pose_record = catalog.get("pose_asset", {})
    rest_path = BODY_ASSET_ROOT / str(rest_record.get("path", ""))
    pose_path = BODY_ASSET_ROOT / str(pose_record.get("path", ""))
    if sha256_file(rest_path) != rest_record.get("sha256"):
        raise BodyRigError("body_asset_hash_mismatch: rest browser asset")
    if sha256_file(pose_path) != pose_record.get("sha256"):
        raise BodyRigError("body_asset_hash_mismatch: pose cache")
    surface = load_canonical_surface(rest_path)
    pose_ids = tuple(catalog.get("pose_ids", ()))
    if not pose_ids or set(pose_ids) != set(catalog.get("poses", {})):
        raise BodyRigError("pose_unsupported: pose catalog order/records differ")
    raw = pose_path.read_bytes()
    expected_shape = (len(pose_ids), 36_108, 3, 3)
    poses = np.frombuffer(raw, dtype="<f4")
    if poses.size != int(np.prod(expected_shape)):
        raise BodyRigError("body_asset_hash_mismatch: pose cache size")
    poses = poses.reshape(expected_shape).copy()
    if not np.isfinite(poses).all():
        raise BodyRigError("body_units_or_axes_invalid: non-finite pose cache")
    for index, pose_id in enumerate(pose_ids):
        record = catalog["poses"][pose_id]
        offset = index * poses[index].nbytes
        if record.get("byte_offset") != offset or record.get("byte_length") != poses[index].nbytes:
            raise BodyRigError(f"body_asset_hash_mismatch: {pose_id} byte range")
        chunk = memoryview(raw)[offset : offset + poses[index].nbytes]
        if hashlib.sha256(chunk).hexdigest() != record.get("chunk_sha256"):
            raise BodyRigError(f"body_asset_hash_mismatch: {pose_id} chunk")
        if _surface_digest(poses[index], surface.faces) != record.get("surface_sha256"):
            raise BodyRigError(f"body_rest_surface_mismatch: {pose_id} surface")
    return BodyRig(
        body_id=MODEL_ID,
        rig_id=MODEL_ID,
        surface_sha256=surface.sha256,
        topology_sha256=surface.topology_sha256,
        model_spec_sha256=catalog["model_spec_sha256"],
        identity_sha256=catalog["identity_sha256"],
        catalog_sha256=catalog_sha256,
        catalog_record=catalog,
        rest_vertices=surface.vertices,
        faces=surface.faces,
        pose_ids=pose_ids,
        pose_vertices=poses,
        part_names=("SOMA",),
        part_first_face=np.asarray([0], dtype=np.int32),
        part_face_count=np.asarray([36_108], dtype=np.int32),
    )


def _axis_quaternion(axis: int, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2
    result = np.zeros(4, dtype=np.float64)
    result[axis] = math.sin(half)
    result[3] = math.cos(half)
    return result


def _multiply_quaternion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def provider_pose_rotations(joint_names: tuple[str, ...], values: dict) -> np.ndarray:
    names = joint_names[1:]
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise BodyRigError(f"pose_unsupported: unknown joints {', '.join(unknown)}")
    result = np.zeros((77, 4), dtype=np.float64)
    result[:, 3] = 1
    lookup = {name: index for index, name in enumerate(names)}
    for name, xyz in sorted(values.items()):
        if len(xyz) != 3 or not np.isfinite(xyz).all():
            raise BodyRigError("pose_unsupported: joint rotations require three finite degrees")
        qx, qy, qz = (_axis_quaternion(axis, float(value)) for axis, value in enumerate(xyz))
        quaternion = _multiply_quaternion(_multiply_quaternion(qz, qy), qx)
        result[lookup[name]] = quaternion / np.linalg.norm(quaternion)
    return result

