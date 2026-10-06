#!/usr/bin/env python3
"""Export the reviewed MHR-through-SOMA reference body and named poses.

The command is intentionally an offline build step.  ``SOMAPosedBody`` first
verifies the immutable cache and exact software lock; only then can this tool
deserialize SOMA assets.  The GLB keeps upstream triangle order, preserves
face-varying UVs, and carries the original SOMA vertex index on every corner so
the browser can reconstruct and hash the canonical indexed surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python" / "tatbot_sim" / "src"))

from tatbot_sim.body_models.mhr_soma import (  # noqa: E402
    SOMAPosedBody,
    canonical_surface_digest,
)
from tatbot_sim.human_rep.contracts import load_contract  # noqa: E402
from tatbot_sim.inkmap.rig import load_body_rig, provider_pose_rotations  # noqa: E402

MODEL_ID = "mhr-soma-v1"
GLB_PATH = Path("bodies/mhr-soma-v1.glb")
POSE_BYTES_PATH = Path("bodies/mhr-soma-v1.poses.bin")
EXCLUSION_PATH = Path("bodies/mhr-soma-v1.exclusions.bin")
DIGEST_SCHEMA = "tatbot.body-pose-catalog-digest/1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _smooth_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face = vertices[faces]
    normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    result = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(result, faces[:, corner], normals)
    lengths = np.linalg.norm(result, axis=1)
    if not np.isfinite(lengths).all() or float(lengths.min()) <= 1e-12:
        raise ValueError("SOMA surface contains a vertex without a finite normal")
    return np.ascontiguousarray(result / lengths[:, None], dtype="<f4")


def _triangle_uvs(asset: np.lib.npyio.NpzFile, faces: np.ndarray) -> np.ndarray:
    polygons = np.asarray(asset["face_vert_indices"], dtype=np.int32).reshape(-1, 4)
    polygon_uvs = np.asarray(asset["uv_indices_st"], dtype=np.int32).reshape(-1, 4)
    uv_values = np.asarray(asset["uv_coord_st"], dtype="<f4")
    if len(faces) != 2 * len(polygons):
        raise ValueError("SOMA polygon and triangle counts do not preserve the reviewed 2:1 map")
    result = np.empty((len(faces), 3, 2), dtype="<f4")
    for polygon_index, polygon in enumerate(polygons):
        lookup = {int(vertex): int(polygon_uvs[polygon_index, corner]) for corner, vertex in enumerate(polygon)}
        pair = faces[2 * polygon_index : 2 * polygon_index + 2]
        if {int(value) for value in pair.flat} != {int(value) for value in polygon}:
            raise ValueError(f"SOMA triangle pair {polygon_index} does not match its source polygon")
        for local_triangle, triangle in enumerate(pair):
            result[2 * polygon_index + local_triangle] = uv_values[
                [lookup[int(vertex)] for vertex in triangle]
            ]
    if not np.isfinite(result).all():
        raise ValueError("SOMA UVs contain a non-finite value")
    return result


class _GlbBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def accessor(
        self,
        value: np.ndarray,
        *,
        component_type: int,
        gltf_type: str,
        target: int | None = None,
        bounds: bool = False,
    ) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        array = np.ascontiguousarray(value)
        offset = len(self.binary)
        self.binary.extend(array.tobytes(order="C"))
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        view_index = len(self.views)
        self.views.append(view)
        count = int(array.shape[0])
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": gltf_type,
        }
        if bounds:
            reshaped = array.reshape(count, -1)
            accessor["min"] = [float(item) for item in reshaped.min(axis=0)]
            accessor["max"] = [float(item) for item in reshaped.max(axis=0)]
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def _glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    model_spec_sha256: str,
    identity_sha256: str,
    surface_sha256: str,
    topology_sha256: str,
) -> bytes:
    positions = np.ascontiguousarray(vertices[faces].reshape(-1, 3), dtype="<f4")
    normals = np.ascontiguousarray(_smooth_normals(vertices, faces)[faces].reshape(-1, 3), dtype="<f4")
    texcoords = np.ascontiguousarray(uvs.reshape(-1, 2), dtype="<f4")
    source_vertex = np.ascontiguousarray(faces.reshape(-1), dtype="<u2")
    indices = np.arange(len(positions), dtype="<u4")
    builder = _GlbBuilder()
    position_accessor = builder.accessor(positions, component_type=5126, gltf_type="VEC3", target=34962, bounds=True)
    normal_accessor = builder.accessor(normals, component_type=5126, gltf_type="VEC3", target=34962)
    uv_accessor = builder.accessor(texcoords, component_type=5126, gltf_type="VEC2", target=34962)
    source_accessor = builder.accessor(source_vertex, component_type=5123, gltf_type="SCALAR", target=34962)
    index_accessor = builder.accessor(indices, component_type=5125, gltf_type="SCALAR", target=34963)
    document = {
        "asset": {"version": "2.0", "generator": "Tatbot deterministic SOMA exporter/1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "SOMA", "mesh": 0}],
        "meshes": [
            {
                "name": "SOMA mid reference",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": uv_accessor,
                            "_SOMA_VERTEX": source_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "Tatbot redistributable procedural skin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.72, 0.48, 0.34, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.86,
                },
                "doubleSided": False,
            }
        ],
        "buffers": [{"byteLength": len(builder.binary)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
        "extras": {
            "schema": "tatbot.soma-browser-surface/1",
            "model_spec_id": MODEL_ID,
            "model_spec_sha256": model_spec_sha256,
            "identity_sha256": identity_sha256,
            "surface_sha256": surface_sha256,
            "topology_sha256": topology_sha256,
            "coordinate_frame": "tatbot-z-up-front-minus-y-metres",
            "face_order": "SOMA-mid-upstream",
            "uv_set": "st-face-varying",
        },
    }
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    bin_chunk = bytes(builder.binary) + b"\0" * ((-len(builder.binary)) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), 0x004E4942)
        + bin_chunk
    )


def _pose_quality(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray, quaternions: np.ndarray) -> dict[str, float]:
    rest_faces = rest[faces]
    posed_faces = posed[faces]
    rest_edges = np.linalg.norm(rest_faces[:, [1, 2, 0]] - rest_faces[:, [0, 1, 2]], axis=2)
    posed_edges = np.linalg.norm(posed_faces[:, [1, 2, 0]] - posed_faces[:, [0, 1, 2]], axis=2)
    rest_area = np.linalg.norm(
        np.cross(rest_faces[:, 1] - rest_faces[:, 0], rest_faces[:, 2] - rest_faces[:, 0]), axis=1
    )
    posed_area = np.linalg.norm(
        np.cross(posed_faces[:, 1] - posed_faces[:, 0], posed_faces[:, 2] - posed_faces[:, 0]), axis=1
    )
    edge_ratio = posed_edges / rest_edges
    area_ratio = posed_area / rest_area
    angles = 2 * np.degrees(np.arccos(np.clip(np.abs(quaternions[:, 3]), 0, 1)))
    return {
        "max_joint_rotation_deg": round(float(angles.max()), 6),
        "edge_length_ratio_p001": round(float(np.quantile(edge_ratio, 0.001)), 6),
        "edge_length_ratio_p99": round(float(np.quantile(edge_ratio, 0.99)), 6),
        "triangle_area_ratio_p01": round(float(np.quantile(area_ratio, 0.01)), 6),
        "triangle_area_ratio_p99": round(float(np.quantile(area_ratio, 0.99)), 6),
    }


def _write_or_check(path: Path, value: bytes, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != value:
            raise ValueError(f"generated artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def pose_binding_updates(catalog: dict, catalog_bytes: bytes) -> dict[Path, bytes]:
    """Validate dependent contracts before writing any generated artifact."""
    contract = REPO / "python/tatbot_sim/src/tatbot_sim/inkmap/contracts.py"
    source = contract.read_text()
    match = re.search(r'^POSE_ASSET_SHA256 = "([a-f0-9]{64})"$', source, re.MULTILINE)
    if match is None:
        raise ValueError("shared pose-asset pin is missing")
    previous, current = match[1], catalog["pose_asset"]["sha256"]
    updates = {contract: source.replace(previous, current).encode()}
    for name in ("tattoo-scenario.schema.json", "tattoo-scenario-v3.schema.json"):
        path = REPO / "config/inkmap" / name
        updates[path] = path.read_text().replace(previous, current).encode()
    path = REPO / "config/inkmap/examples/forearm-scenario-v2.json"
    example = _json(path)
    record = catalog["poses"][example["pose"]["id"]]
    if record["surface_sha256"] != example["pose"]["posed_surface_sha256"]:
        raise ValueError("example pose changed; recompile the scenario before exporting")
    example["body"]["pose_asset_sha256"] = current
    example["pose"]["catalog_sha256"] = _sha256_bytes(catalog_bytes)
    updates[path] = (json.dumps(example, indent=2, sort_keys=True) + "\n").encode()
    manifest = REPO / "web/inkmap/public/showcase/manifest.json"
    updates[manifest] = manifest.read_text().replace(previous, current).encode()
    return updates


def _extension_chunks(body, identity, authoring, rig, selected, reference, runtime):
    chunks, records = [], {}
    for pose_id in authoring["pose_ids"]:
        source = authoring["poses"][pose_id]
        if pose_id not in selected:
            record = rig.catalog_record["poses"][pose_id]
            for field in ("label", "support_id", "body_rotation_xyzw", "constraints"):
                if record[field] != source[field]:
                    raise ValueError(f"existing pose {pose_id} changed; include --pose-id to regenerate it")
            if ("joint_rotations_euler_xyz_deg" in record
                    and record["joint_rotations_euler_xyz_deg"] != source["joint_rotations_euler_xyz_deg"]):
                raise ValueError(f"existing pose {pose_id} joint controls changed; include --pose-id to regenerate it")
            chunk = rig.pose_vertices[rig.pose_ids.index(pose_id)].astype("<f4").tobytes()
        else:
            rotations = provider_pose_rotations(body.joint_names, source["joint_rotations_euler_xyz_deg"])
            surface = body.generate(identity, rotations, apply_correctives=True)
            chunk = surface.face_vertices_m.astype("<f4").tobytes()
            record = {key: source[key] for key in ("label", "support_id", "body_rotation_xyzw", "constraints",
                                                   "joint_rotations_euler_xyz_deg")}
            record |= {"correctives_enabled": True, "provider_runtime": runtime,
                       "surface_sha256": surface.surface_sha256,
                       "quality": _pose_quality(reference, surface.vertices_m, body.faces, rotations)}
        records[pose_id] = {**record, "byte_offset": sum(map(len, chunks)),
                           "byte_length": len(chunk), "chunk_sha256": _sha256_bytes(chunk)}
        chunks.append(chunk)
    return chunks, records


def extend_catalog(args, body, identity, authoring) -> None:
    """Bake selected poses while preserving the immutable reference and prior poses.

    Exact rest regeneration remains required by the full exporter/provider.
    An extension compares its runtime rest against the verified reference at
    every indexed vertex, within two reference quantization units. This admits
    bounded arithmetic differences across CPU architectures, records them,
    and never edits or deforms either generated or reference vertices.
    """
    rig = load_body_rig()
    if args.catalog.resolve() != REPO / "config/inkmap/body-poses.json":
        raise ValueError("catalog extension requires the installed verified catalog")
    if (rig.identity_sha256 != identity["content_sha256"]
            or rig.model_spec_sha256 != body.spec["content_sha256"]):
        raise ValueError("catalog extension identity/model mismatch")
    if list(rig.pose_ids) != authoring["pose_ids"][:len(rig.pose_ids)]:
        raise ValueError("catalog extension must preserve existing pose order")
    selected = args.pose_id or [p for p in authoring["pose_ids"] if p not in rig.pose_ids]
    if not selected or not set(selected) <= set(authoring["pose_ids"]):
        raise ValueError("catalog extension needs authored --pose-id values")
    if set(authoring["pose_ids"]) - set(rig.pose_ids) - set(selected):
        raise ValueError("catalog extension must generate every newly authored pose")
    rotations = provider_pose_rotations(body.joint_names, {})
    actual_rest = body.generate(identity, rotations, apply_correctives=True)
    reference = rig.indexed_vertices()
    error = float(np.linalg.norm(actual_rest.vertices_m - reference, axis=1).max())
    limit = 2 * body.spec["coordinates"]["quantization_m"]
    if not np.isfinite(error) or error > limit:
        raise ValueError(f"runtime rest differs from the verified reference by {error:.9g} m; limit {limit:.9g}")
    import torch
    from pxr import Usd

    runtime = {"architecture": platform.machine(), "soma_version": body.spec["python_package"]["version"],
               "torch_version": torch.__version__, "usd_version": list(Usd.GetVersion()),
               "runtime_rest_surface_sha256": actual_rest.surface_sha256,
               "reference_max_vertex_error_m": error, "reference_error_limit_m": limit}
    chunks, records = _extension_chunks(body, identity, authoring, rig, selected, reference, runtime)
    pose_bytes = b"".join(chunks)
    catalog = {**rig.catalog_record, "pose_ids": authoring["pose_ids"], "poses": records,
               "pose_asset": {**rig.catalog_record["pose_asset"], "sha256": _sha256_bytes(pose_bytes),
                              "size": len(pose_bytes)}}
    value = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode()
    digest = {"schema": DIGEST_SCHEMA, "catalog_path": args.catalog.resolve().relative_to(REPO).as_posix(),
              "sha256": _sha256_bytes(value)}
    bindings = pose_binding_updates(catalog, value)
    _write_or_check(args.output_root.resolve() / POSE_BYTES_PATH, pose_bytes, args.check)
    _write_or_check(args.catalog.resolve(), value, args.check)
    _write_or_check(args.digest.resolve(), (json.dumps(digest, indent=2, sort_keys=True) + "\n").encode(), args.check)
    for path, content in bindings.items():
        _write_or_check(path, content, args.check)
    print(json.dumps({"poses": selected, "pose_asset_sha256": _sha256_bytes(pose_bytes),
                      "reference_comparison": runtime, "check": args.check}, sort_keys=True))


def export(args: argparse.Namespace) -> None:
    spec_path = args.spec.resolve()
    authoring = _json(args.poses.resolve())
    eligibility = _json(args.eligibility.resolve())
    identity = load_contract(args.identity.resolve(), expected_schema="tatbot.body-identity/1")
    body = SOMAPosedBody(spec_path=spec_path, cache_dir=args.cache_dir.resolve(), device="cpu")
    spec = body.spec
    if authoring["model_spec_sha256"] != spec["content_sha256"]:
        raise ValueError("pose authoring model-spec digest mismatch")
    if authoring["identity_sha256"] != identity["content_sha256"]:
        raise ValueError("pose authoring identity digest mismatch")
    if args.extend:
        extend_catalog(args, body, identity, authoring)
        return
    rest = body.rest(identity)

    asset_path = body.cache_dir / "SOMA_neutral.npz"
    with np.load(asset_path, allow_pickle=False) as asset:
        uvs = _triangle_uvs(asset, body.faces)
        segment_names = eligibility["upstream_segments"]["excluded_not_skin"]
        missing_segments = sorted(
            name for name in segment_names if f"segment_{name}" not in asset.files
        )
        if missing_segments:
            raise ValueError(f"eligibility names absent SOMA segments: {', '.join(missing_segments)}")
        excluded_vertices = np.unique(
            np.concatenate([np.asarray(asset[f"segment_{name}"], dtype=np.int32) for name in segment_names])
        )
    excluded_faces = np.any(np.isin(body.faces, excluded_vertices), axis=1).astype("u1")
    exclusion_bytes = excluded_faces.tobytes()
    glb = _glb(
        rest.vertices_m,
        body.faces,
        uvs,
        model_spec_sha256=spec["content_sha256"],
        identity_sha256=identity["content_sha256"],
        surface_sha256=rest.surface_sha256,
        topology_sha256=spec["geometry"]["topology_sha256"],
    )

    pose_chunks: list[bytes] = []
    pose_records: dict[str, Any] = {}
    face_vertex_count = len(body.faces) * 3
    for pose_id in authoring["pose_ids"]:
        source = authoring["poses"][pose_id]
        quaternions = provider_pose_rotations(body.joint_names, source["joint_rotations_euler_xyz_deg"])
        posed = body.generate(identity, quaternions)
        chunk = np.ascontiguousarray(posed.vertices_m[body.faces].reshape(-1, 3), dtype="<f4").tobytes()
        offset = sum(len(value) for value in pose_chunks)
        pose_chunks.append(chunk)
        pose_records[pose_id] = {
            "label": source["label"],
            "support_id": source["support_id"],
            "body_rotation_xyzw": source["body_rotation_xyzw"],
            "constraints": source["constraints"],
            "joint_rotations_euler_xyz_deg": source["joint_rotations_euler_xyz_deg"],
            "correctives_enabled": True,
            "surface_sha256": canonical_surface_digest(posed.vertices_m),
            "byte_offset": offset,
            "byte_length": len(chunk),
            "chunk_sha256": _sha256_bytes(chunk),
            "quality": _pose_quality(rest.vertices_m, posed.vertices_m, body.faces, quaternions),
        }
    pose_bytes = b"".join(pose_chunks)
    catalog = {
        "schema": "tatbot.body-pose-catalog/2",
        "model_spec_id": MODEL_ID,
        "model_spec_sha256": spec["content_sha256"],
        "identity_sha256": identity["content_sha256"],
        "topology_sha256": spec["geometry"]["topology_sha256"],
        "rest_surface_sha256": rest.surface_sha256,
        "rest_asset": {
            "path": GLB_PATH.as_posix(),
            "sha256": _sha256_bytes(glb),
            "size": len(glb),
        },
        "pose_asset": {
            "path": POSE_BYTES_PATH.as_posix(),
            "sha256": _sha256_bytes(pose_bytes),
            "size": len(pose_bytes),
            "format": "little-endian-float32-expanded-face-xyz",
            "face_count": len(body.faces),
            "vertices_per_pose": face_vertex_count,
        },
        "exclusion_asset": {
            "path": EXCLUSION_PATH.as_posix(),
            "sha256": _sha256_bytes(exclusion_bytes),
            "size": len(exclusion_bytes),
            "format": "uint8-per-face-zero-eligible-one-excluded",
            "source_segments": segment_names,
            "excluded_vertices": len(excluded_vertices),
            "excluded_faces": int(excluded_faces.sum()),
        },
        "pose_ids": authoring["pose_ids"],
        "poses": pose_records,
    }
    catalog_bytes = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode()
    bindings = (pose_binding_updates(catalog, catalog_bytes)
                if args.catalog.resolve() == REPO / "config/inkmap/body-poses.json" else {})
    root = args.output_root.resolve()
    _write_or_check(root / GLB_PATH, glb, args.check)
    _write_or_check(root / POSE_BYTES_PATH, pose_bytes, args.check)
    _write_or_check(root / EXCLUSION_PATH, exclusion_bytes, args.check)
    _write_or_check(args.catalog.resolve(), catalog_bytes, args.check)
    # The browser cannot hash the catalog it imports as JSON, so the exact
    # digest of the catalog bytes is published beside it and every scenario
    # binds to that value. Python re-derives it from the bytes and refuses a
    # stale digest file, so the two sides cannot drift apart silently.
    digest = {
        "schema": DIGEST_SCHEMA,
        "catalog_path": args.catalog.resolve().relative_to(REPO).as_posix(),
        "sha256": _sha256_bytes(catalog_bytes),
    }
    digest_bytes = (json.dumps(digest, indent=2, sort_keys=True) + "\n").encode()
    _write_or_check(args.digest.resolve(), digest_bytes, args.check)
    for path, content in bindings.items():
        _write_or_check(path, content, args.check)
    print(
        json.dumps(
            {
                "check": args.check,
                "faces": len(body.faces),
                "glb_sha256": _sha256_bytes(glb),
                "excluded_faces": int(excluded_faces.sum()),
                "poses": len(pose_records),
                "pose_asset_sha256": _sha256_bytes(pose_bytes),
                "rest_surface_sha256": rest.surface_sha256,
                "vertices": len(rest.vertices_m),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=REPO / "config/body-models/mhr-soma-v1.json")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--identity",
        type=Path,
        default=REPO / "config/body-models/mhr-soma-v1/identity.json",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=REPO / "config/body-models/mhr-soma-v1/poses.json",
    )
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=REPO / "config/body-models/mhr-soma-v1/eligibility.json",
    )
    parser.add_argument("--output-root", type=Path, default=REPO / "web/inkmap/public")
    parser.add_argument("--catalog", type=Path, default=REPO / "config/inkmap/body-poses.json")
    parser.add_argument("--digest", type=Path, default=REPO / "config/inkmap/body-poses.digest.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--extend", action="store_true", help="preserve verified reference/prior poses; bake selected poses")
    parser.add_argument("--pose-id", action="append", help="pose to bake/check with --extend; repeatable")
    return parser.parse_args()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tatbot-soma-export-"):
        export(parse_args())
