"""Read the one checked-in SOMA mid browser surface without Three.js.

The GLB is a rendering view: vertices are expanded at UV seams, but every
corner carries ``_SOMA_VERTEX``. Reconstructing that attribute proves both
the upstream triangle order and the canonical 18,056-vertex rest surface used
by Python contracts and browser placements.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ID = "mhr-soma-v1"
MODEL_SPEC_SHA256 = "e615b8485c367509833ee68b0405cd1e0ce6015604eaa4c1b2b699f4fc8d5144"
TOPOLOGY_SHA256 = "e0ca7ee25dc0b4c8d841bb2626e364bb88b7af7fae037e30854728842e320a18"
REST_SURFACE_SHA256 = "caa66dff9b3625771c8f4c35bfe59556d30acdc3800880106f98f0ce75c49a95"

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
    """The derived browser asset does not implement its locked surface contract."""


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    dimensions = ",".join(str(item) for item in array.shape)
    header = f"dtype={array.dtype.str};shape={dimensions};order=C\n".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _topology_digest(faces: np.ndarray) -> str:
    return _array_digest(np.asarray(faces, dtype="<i4"))


def _surface_digest(vertices: np.ndarray) -> str:
    quantized = np.rint(np.asarray(vertices) / 0.00001).astype("<i8")
    header = b"dtype=<i8;shape=18056,3;order=C;quantization_m=0.00001;axes=x,-z,y\n"
    return hashlib.sha256(header + quantized.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CanonicalPart:
    name: str
    first_face: int
    face_count: int


@dataclass(frozen=True)
class CanonicalSurface:
    vertices: np.ndarray
    """Face-expanded float32 vertices, shape ``(36108, 3, 3)``."""

    indexed_vertices: np.ndarray
    """Canonical SOMA mid vertices, shape ``(18056, 3)``."""

    faces: np.ndarray
    """Canonical SOMA mid triangle indices, shape ``(36108, 3)``."""

    parts: tuple[CanonicalPart, ...]

    @property
    def sha256(self) -> str:
        return _surface_digest(self.indexed_vertices)

    @property
    def topology_sha256(self) -> str:
        return _topology_digest(self.faces)


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
        if offset + 8 > len(raw):
            raise GlbSurfaceError(f"{path}: truncated GLB chunk header")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + length > len(raw):
            raise GlbSurfaceError(f"{path}: truncated GLB chunk")
        chunks[kind] = raw[offset : offset + length]
        offset += length
    try:
        document = json.loads(chunks[0x4E4F534A].decode("utf-8"))
        binary = chunks[0x004E4942]
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbSurfaceError(f"{path}: missing or invalid JSON/BIN chunk") from exc
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    try:
        accessor = document["accessors"][index]
        view = document["bufferViews"][accessor["bufferView"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise GlbSurfaceError(f"invalid accessor {index}") from exc
    if "sparse" in accessor:
        raise GlbSurfaceError("sparse glTF accessors are not supported")
    dtype = _COMPONENTS.get(accessor.get("componentType"))
    width = _WIDTHS.get(accessor.get("type"))
    if dtype is None or width is None:
        raise GlbSurfaceError(
            f"unsupported accessor {accessor.get('componentType')}/{accessor.get('type')}"
        )
    count = accessor.get("count")
    if not isinstance(count, int) or count <= 0:
        raise GlbSurfaceError("accessor count must be a positive integer")
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    packed = dtype.itemsize * width
    stride = view.get("byteStride", packed)
    if stride < packed or offset + (count - 1) * stride + packed > len(binary):
        raise GlbSurfaceError("accessor extends beyond BIN chunk")
    if stride == packed:
        out = np.frombuffer(binary, dtype=dtype, count=count * width, offset=offset).reshape(
            count, width
        )
    else:
        out = np.ndarray(
            (count, width),
            dtype=dtype,
            buffer=binary,
            offset=offset,
            strides=(stride, dtype.itemsize),
        )
    return out.copy()


def _quat_matrix(value: list[float]) -> np.ndarray:
    x, y, z, w = value
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-20:
        raise GlbSurfaceError("node quaternion is zero")
    scale = 2.0 / norm
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - w * z), scale * (x * z + w * y), 0],
            [scale * (x * y + w * z), 1 - scale * (x * x + z * z), scale * (y * z - w * x), 0],
            [scale * (x * z - w * y), scale * (y * z + w * x), 1 - scale * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


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
            if child in parents:
                raise GlbSurfaceError(f"node {child} has multiple parents")
            parents[child] = parent
    cache: dict[int, np.ndarray] = {}

    def world(index: int, active: frozenset[int] = frozenset()) -> np.ndarray:
        if index in active:
            raise GlbSurfaceError("node hierarchy contains a cycle")
        if index not in cache:
            local = _node_matrix(nodes[index])
            cache[index] = world(parents[index], active | {index}) @ local if index in parents else local
        return cache[index]

    return [world(index) for index in range(len(nodes))]


def load_canonical_surface(path: str | Path) -> CanonicalSurface:
    path = Path(path)
    document, binary = _load_glb(path)
    extras = document.get("extras")
    expected_extras = {
        "schema": "tatbot.soma-browser-surface/1",
        "model_spec_id": MODEL_ID,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "surface_sha256": REST_SURFACE_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
        "coordinate_frame": "tatbot-z-up-front-minus-y-metres",
        "face_order": "SOMA-mid-upstream",
        "uv_set": "st-face-varying",
    }
    if not isinstance(extras, dict) or any(
        extras.get(key) != value for key, value in expected_extras.items()
    ):
        raise GlbSurfaceError(f"{path}: model, frame, or digest metadata is not reviewed")
    nodes = document.get("nodes", [])
    matching = [index for index, node in enumerate(nodes) if node.get("name") == "SOMA"]
    if len(matching) != 1:
        raise GlbSurfaceError(f"{path}: expected exactly one SOMA node")
    node_index = matching[0]
    try:
        primitives = document["meshes"][nodes[node_index]["mesh"]]["primitives"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GlbSurfaceError(f"{path}: SOMA node has no valid mesh") from exc
    if len(primitives) != 1 or primitives[0].get("mode", 4) != 4:
        raise GlbSurfaceError(f"{path}: SOMA must contain one triangle primitive")
    primitive = primitives[0]
    attributes = primitive.get("attributes", {})
    required = {"POSITION", "NORMAL", "TEXCOORD_0", "_SOMA_VERTEX"}
    if set(attributes) != required or "indices" not in primitive:
        raise GlbSurfaceError(f"{path}: SOMA primitive attributes differ from the reviewed set")
    positions = _accessor(document, binary, attributes["POSITION"]).astype(np.float64)
    normals = _accessor(document, binary, attributes["NORMAL"])
    uvs = _accessor(document, binary, attributes["TEXCOORD_0"])
    source = _accessor(document, binary, attributes["_SOMA_VERTEX"]).reshape(-1)
    indices = _accessor(document, binary, primitive["indices"]).reshape(-1)
    if not (
        positions.shape == (108_324, 3)
        and normals.shape == (108_324, 3)
        and uvs.shape == (108_324, 2)
        and source.shape == (108_324,)
        and indices.shape == (108_324,)
        and np.array_equal(indices, np.arange(108_324))
    ):
        raise GlbSurfaceError(f"{path}: accessor shapes or invariant corner indices changed")
    worlds = _world_matrices(document)
    points = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    expanded = (worlds[node_index] @ points.T).T[:, :3].astype("<f4")
    if not np.isfinite(expanded).all() or int(source.min()) != 0 or int(source.max()) != 18_055:
        raise GlbSurfaceError(f"{path}: surface contains invalid values or source indices")
    faces = np.asarray(source, dtype="<i4").reshape(36_108, 3)
    indexed = np.empty((18_056, 3), dtype="<f4")
    seen = np.zeros(18_056, dtype=bool)
    for corner, vertex_index in enumerate(source):
        index = int(vertex_index)
        if seen[index] and not np.array_equal(indexed[index], expanded[corner]):
            raise GlbSurfaceError(f"{path}: source vertex {index} has inconsistent corner positions")
        indexed[index] = expanded[corner]
        seen[index] = True
    if not seen.all():
        raise GlbSurfaceError(f"{path}: not every SOMA mid vertex is referenced")
    result = CanonicalSurface(
        vertices=expanded.reshape(36_108, 3, 3),
        indexed_vertices=indexed,
        faces=faces,
        parts=(CanonicalPart(name="SOMA", first_face=0, face_count=36_108),),
    )
    if result.sha256 != REST_SURFACE_SHA256 or result.topology_sha256 != TOPOLOGY_SHA256:
        raise GlbSurfaceError(f"{path}: reconstructed canonical digest mismatch")
    return result
