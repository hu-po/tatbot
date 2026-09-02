"""Read Inkmap's canonical rest surface directly from a binary glTF asset.

Inkmap anchors index the exact non-indexed geometry produced by `buildSkin`:
named mesh nodes sorted by name, expanded in primitive index order, transformed
through their node hierarchy, then converted from glTF Y-up to Tatbot Z-up.
This small reader mirrors that recipe without depending on Blender or Three.js.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COMPONENTS = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GlbSurfaceError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalPart:
    name: str
    first_face: int
    face_count: int


@dataclass(frozen=True)
class CanonicalSurface:
    vertices: np.ndarray
    """Non-indexed float32 vertices, shape (faces, 3, 3), Z-up metres."""

    parts: tuple[CanonicalPart, ...]

    @property
    def sha256(self) -> str:
        # JavaScript Math.round(x) is floor(x + 0.5), including negative ties.
        # Hash signed 10-micrometre units so harmless cross-runtime float ULPs do not
        # redefine a face/barycentric anchor.
        units_10um = np.floor(np.asarray(self.vertices, dtype=np.float64) * 1e5 + 0.5).astype("<i4")
        return hashlib.sha256(units_10um.tobytes()).hexdigest()


def _load_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise GlbSurfaceError(f"{path}: not a binary glTF file")
    _, version, total = struct.unpack_from("<4sII", raw, 0)
    if version != 2 or total != len(raw):
        raise GlbSurfaceError(f"{path}: unsupported glTF header")
    offset = 12
    chunks: dict[int, bytes] = {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[kind] = raw[offset:offset + length]
        offset += length
    try:
        document = json.loads(chunks[0x4E4F534A].decode("utf-8"))
        binary = chunks[0x004E4942]
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbSurfaceError(f"{path}: missing or invalid JSON/BIN chunk") from exc
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if "sparse" in accessor:
        raise GlbSurfaceError("sparse glTF accessors are not supported")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENTS.get(accessor["componentType"])
    width = _WIDTHS.get(accessor["type"])
    if dtype is None or width is None:
        raise GlbSurfaceError(f"unsupported accessor {accessor['componentType']}/{accessor['type']}")
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    packed = dtype.itemsize * width
    stride = view.get("byteStride", packed)
    if offset + (count - 1) * stride + packed > len(binary):
        raise GlbSurfaceError("accessor extends beyond BIN chunk")
    if stride == packed:
        out = np.frombuffer(binary, dtype=dtype, count=count * width, offset=offset).reshape(count, width)
    else:
        out = np.ndarray((count, width), dtype=dtype, buffer=binary, offset=offset, strides=(stride, dtype.itemsize))
    return out.copy()


def _quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-20:
        return np.eye(4)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    out = np.eye(4)
    out[:3, :3] = [
        [1 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1 - xx - yy],
    ]
    return out


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    translation = np.eye(4)
    translation[:3, 3] = node.get("translation", [0, 0, 0])
    scale = np.eye(4)
    scale[np.arange(3), np.arange(3)] = node.get("scale", [1, 1, 1])
    return translation @ _quat_matrix(node.get("rotation", [0, 0, 0, 1])) @ scale


def _world_matrices(document: dict) -> list[np.ndarray]:
    nodes = document.get("nodes", [])
    parents: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = parent
    cache: dict[int, np.ndarray] = {}

    def world(index: int) -> np.ndarray:
        if index not in cache:
            local = _node_matrix(nodes[index])
            cache[index] = world(parents[index]) @ local if index in parents else local
        return cache[index]

    return [world(i) for i in range(len(nodes))]


def load_canonical_surface(path: str | Path, skin_nodes: tuple[str, ...] = ("Body", "EyeL", "EyeR")) -> CanonicalSurface:
    path = Path(path)
    document, binary = _load_glb(path)
    worlds = _world_matrices(document)
    by_name = {node.get("name"): i for i, node in enumerate(document.get("nodes", []))}
    # Match THREE.Matrix4.makeRotationX(Math.PI / 2) bit-for-bit.  Its cosine
    # is the tiny IEEE value ~6.12e-17, not a hand-written zero; preserving it
    # matters because surface_sha256 is over the resulting float32 bytes.
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    faces: list[np.ndarray] = []
    parts: list[CanonicalPart] = []
    first_face = 0
    for name in sorted(skin_nodes):
        try:
            node_index = by_name[name]
            mesh = document["meshes"][document["nodes"][node_index]["mesh"]]
        except (KeyError, IndexError) as exc:
            raise GlbSurfaceError(f"{path}: skin node {name!r} is missing or has no mesh") from exc
        part_faces: list[np.ndarray] = []
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise GlbSurfaceError(f"{path}: {name} contains a non-triangle primitive")
            positions = _accessor(document, binary, primitive["attributes"]["POSITION"]).astype(np.float64)
            ones = np.ones((len(positions), 1), dtype=np.float64)
            world = (worlds[node_index] @ np.concatenate([positions, ones], axis=1).T).T
            # Spell out Three.Vector3.applyMatrix4's operation order instead
            # of using BLAS: fused/reassociated matrix arithmetic can move a
            # float32 result by one ULP and therefore change the digest.
            positions_z = np.stack([
                world[:, 0],
                c * world[:, 1] - s * world[:, 2],
                s * world[:, 1] + c * world[:, 2],
            ], axis=1)
            if "indices" in primitive:
                indices = _accessor(document, binary, primitive["indices"]).reshape(-1)
            else:
                indices = np.arange(len(positions_z))
            if len(indices) % 3:
                raise GlbSurfaceError(f"{path}: {name} triangle index count is not divisible by three")
            part_faces.append(positions_z[indices].reshape(-1, 3, 3))
        combined = np.concatenate(part_faces).astype("<f4", copy=False)
        faces.append(combined)
        parts.append(CanonicalPart(name=name, first_face=first_face, face_count=len(combined)))
        first_face += len(combined)
    return CanonicalSurface(vertices=np.concatenate(faces), parts=tuple(parts))
