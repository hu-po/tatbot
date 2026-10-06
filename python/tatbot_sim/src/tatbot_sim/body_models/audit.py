"""CPU-first, reproducible audit of the sole MHR-through-SOMA body path.

The audit writes derived evidence only. It neither downloads assets nor emits
robot commands, and a GPU comparison runs only when an explicit device is
provided by the operator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tatbot_contracts.canonical import write_json
from tatbot_contracts.digest import sha256_file

from tatbot_sim.body_models.mhr_soma import SOMAPosedBody, SOMASurface
from tatbot_sim.human_rep.contracts import canonical_digest, load_contract
from tatbot_sim.inkmap.rig import provider_pose_rotations
from tatbot_sim.repo import git_output, repo_root


def _geometry_metrics(surface: SOMASurface, reference_normals: np.ndarray | None = None) -> dict[str, Any]:
    triangles = surface.vertices_m[surface.faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    metrics: dict[str, Any] = {
        "finite": bool(np.isfinite(surface.vertices_m).all()),
        "vertices": int(len(surface.vertices_m)),
        "triangles": int(len(surface.faces)),
        "degenerate_triangles": int(np.count_nonzero(doubled_area <= 1e-12)),
        "doubled_area_m2": {
            "min": float(doubled_area.min()),
            "p01": float(np.quantile(doubled_area, 0.01)),
            "p50": float(np.quantile(doubled_area, 0.5)),
        },
        "bounds_m": {
            "min": surface.vertices_m.min(axis=0).astype(float).tolist(),
            "max": surface.vertices_m.max(axis=0).astype(float).tolist(),
            "span": np.ptp(surface.vertices_m, axis=0).astype(float).tolist(),
        },
    }
    if reference_normals is not None:
        normal = cross / np.maximum(doubled_area[:, None], 1e-20)
        metrics["normal_reversal_count"] = int(np.count_nonzero(np.einsum("ij,ij->i", normal, reference_normals) < 0))
    return metrics


def _edge_area_ratios(rest: SOMASurface, posed: SOMASurface, face_ids: np.ndarray | None = None) -> dict[str, float]:
    faces = rest.faces if face_ids is None else rest.faces[face_ids]
    a = rest.vertices_m[faces].astype(np.float64)
    b = posed.vertices_m[faces].astype(np.float64)
    edge_a = np.linalg.norm(a[:, [1, 2, 0]] - a[:, [0, 1, 2]], axis=2)
    edge_b = np.linalg.norm(b[:, [1, 2, 0]] - b[:, [0, 1, 2]], axis=2)
    area_a = np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)
    area_b = np.linalg.norm(np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0]), axis=1)
    edge = edge_b / edge_a
    area = area_b / area_a
    return {
        "edge_ratio_min": float(edge.min()),
        "edge_ratio_p001": float(np.quantile(edge, 0.001)),
        "edge_ratio_p50": float(np.quantile(edge, 0.5)),
        "edge_ratio_p99": float(np.quantile(edge, 0.99)),
        "edge_ratio_max": float(edge.max()),
        "area_ratio_min": float(area.min()),
        "area_ratio_p01": float(np.quantile(area, 0.01)),
        "area_ratio_p50": float(np.quantile(area, 0.5)),
        "area_ratio_p99": float(np.quantile(area, 0.99)),
        "area_ratio_max": float(area.max()),
    }


def _segment_triangle(start: np.ndarray, stop: np.ndarray, triangle: np.ndarray, eps: float = 1e-10) -> bool:
    direction = stop - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= eps:
        return False
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u <= eps or u >= 1 - eps:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v <= eps or u + v >= 1 - eps:
        return False
    distance = float(np.dot(edge2, qvec)) * inverse
    return eps < distance < 1 - eps


def _self_intersections(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Count transverse non-neighbor triangle intersections after an R-tree broad phase."""
    from rtree import index as rtree_index

    triangles = vertices[faces].astype(np.float64)
    low = triangles.min(axis=1)
    high = triangles.max(axis=1)
    properties = rtree_index.Property()
    properties.dimension = 3
    entries = (
        (int(face), tuple(np.r_[low[face], high[face]]), None)
        for face in range(len(faces))
    )
    tree = rtree_index.Index(entries, properties=properties)
    tested = 0
    intersections = 0
    examples: list[list[int]] = []
    for left in range(len(faces)):
        bounds = tuple(np.r_[low[left], high[left]])
        for right in tree.intersection(bounds):
            if right <= left or np.intersect1d(faces[left], faces[right], assume_unique=False).size:
                continue
            tested += 1
            a, b = triangles[left], triangles[right]
            hit = any(_segment_triangle(a[i], a[(i + 1) % 3], b) for i in range(3))
            hit = hit or any(_segment_triangle(b[i], b[(i + 1) % 3], a) for i in range(3))
            if hit:
                intersections += 1
                if len(examples) < 20:
                    examples.append([left, int(right)])
    return {
        "method": "rtree-transverse-segment-triangle-v1",
        "broad_phase_pairs_tested": tested,
        "transverse_intersection_pairs": intersections,
        "examples": examples,
        "coplanar_limitation": "coplanar overlaps are not classified by this diagnostic",
    }


def _site_faces(atlas: dict[str, Any], site: str) -> np.ndarray:
    index = atlas["sites"].index(site)
    codes = np.asarray(atlas["faces"], dtype=np.int32)
    return np.flatnonzero((codes >> 2) == index)


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _expanded_indexed_parity(surface: SOMASurface, *, seed: int = 9042026, count: int = 10_000) -> dict[str, Any]:
    """Prove persistent face addresses and the expanded provider agree."""

    rng = np.random.default_rng(seed)
    face_ids = rng.integers(0, len(surface.faces), size=count)
    barycentric = rng.dirichlet(np.ones(3), size=count)
    indexed_triangles = surface.vertices_m[surface.faces[face_ids]].astype(np.float64)
    expanded_triangles = surface.face_vertices_m[face_ids].astype(np.float64)
    indexed_points = np.einsum("ni,nij->nj", barycentric, indexed_triangles)
    expanded_points = np.einsum("ni,nij->nj", barycentric, expanded_triangles)
    errors = np.linalg.norm(indexed_points - expanded_points, axis=1)
    return {
        "seed": seed,
        "samples": count,
        "units": "m",
        "max_error_m": float(errors.max()),
        "rms_error_m": math.sqrt(float(np.mean(errors * errors))),
        "target_max_error_m": 0.00001,
        "status": "pass" if float(errors.max()) <= 0.00001 else "fail",
    }


def _transfer_metrics(body: SOMAPosedBody, identity: dict[str, Any], atlas: dict[str, Any]) -> dict[str, Any]:
    import trimesh

    coefficients = torch.tensor(identity["coefficients"], dtype=torch.float32, device=body.device).reshape(1, 45)
    scales = torch.tensor(identity["scales"], dtype=torch.float32, device=body.device).reshape(1, 68)
    backend = body._layer.identity_model  # audited pinned upstream implementation
    with torch.inference_mode():
        source_cm = backend.get_rest_shape(coefficients, scales)[0]
        transferred_cm = backend.identity_model_to_soma(source_cm[None])[0]
    source = source_cm.detach().cpu().numpy().astype(np.float64)
    transferred = transferred_cm.detach().cpu().numpy().astype(np.float64)
    # These arrays are part of the pinned upstream interpolator instance that
    # performed the transfer. Reusing them avoids reparsing or processing the
    # OBJ and guarantees that this metric compares against the exact source
    # topology seen by SOMA.
    source_faces = np.asarray(backend._to_soma_interp.F_src, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=source, faces=source_faces, process=False)
    _, distances_cm, _ = trimesh.proximity.closest_point(mesh, transferred)
    distances_m = np.asarray(distances_cm) / 100.0
    by_site: dict[str, Any] = {}
    for site in atlas["sites"]:
        face_ids = _site_faces(atlas, site)
        if not len(face_ids):
            by_site[site] = {"vertices": 0, "status": "unsupported"}
            continue
        vertex_ids = np.unique(body.faces[face_ids])
        by_site[site] = {"vertices": int(len(vertex_ids)), **_distribution(distances_m[vertex_ids])}
    return {
        "direction": "transferred_SOMA_vertices_to_native_MHR_surface",
        "units": "m",
        "aggregate": _distribution(distances_m),
        "by_site": by_site,
    }


def _png(path: Path, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB")
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(name: bytes, value: bytes) -> bytes:
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def _render_sheet(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a deterministic three-view silhouette/shading sheet without optional renderers."""
    size = 384
    image = np.full((size, size * 3, 3), 245, dtype=np.uint8)
    views = ((0, 2, 1), (1, 2, 0), (0, 1, 2))
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    for panel, (x_axis, y_axis, depth_axis) in enumerate(views):
        x = centroids[:, x_axis]
        y = centroids[:, y_axis]
        x0, x1 = np.quantile(x, [0.001, 0.999])
        y0, y1 = np.quantile(y, [0.001, 0.999])
        scale = 0.88 * min((size - 1) / max(x1 - x0, 1e-9), (size - 1) / max(y1 - y0, 1e-9))
        px = np.clip(((x - (x0 + x1) / 2) * scale + size / 2).astype(int), 0, size - 1)
        py = np.clip((size / 2 - (y - (y0 + y1) / 2) * scale).astype(int), 0, size - 1)
        order = np.argsort(centroids[:, depth_axis])
        shade = np.clip(100 + 120 * np.abs(normals[:, depth_axis]), 0, 255).astype(np.uint8)
        for index in order:
            value = int(shade[index])
            image[max(0, py[index] - 1):min(size, py[index] + 2), panel * size + max(0, px[index] - 1):panel * size + min(size, px[index] + 2)] = [value, int(value * 0.76), int(value * 0.62)]
    _png(path, image)


def _obj(path: Path, surface: SOMASurface) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("# Derived SOMA audit geometry; metres, Tatbot Z-up\n")
        for x, y, z in surface.vertices_m:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in surface.faces:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _identity_cases(reference: dict[str, Any]) -> list[tuple[str, list[float], list[float]]]:
    cases = [("reference", [0.0] * 45, [0.0] * 68)]
    for axis in range(45):
        coefficients = [0.0] * 45
        coefficients[axis] = 1.0
        cases.append((f"coefficient-{axis:02d}-plus-1sigma", coefficients, [0.0] * 68))
    cases.extend([
        ("mixed-alternating-2sigma", [2.0 if index % 2 == 0 else -2.0 for index in range(45)], [0.0] * 68),
        ("mixed-positive-2sigma", [2.0] * 45, [0.0] * 68),
        ("mixed-negative-2sigma", [-2.0] * 45, [0.0] * 68),
    ])
    for group in range(8):
        scales = [0.0] * 68
        for index in range(group, 68, 8):
            scales[index] = 0.1 if (index // 8) % 2 == 0 else -0.1
        cases.append((f"scale-group-{group}-mixed", [0.0] * 45, scales))
    return cases


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evidence directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for child in ("geometry", "renders"):
        (output / child).mkdir()
    spec = load_contract(args.spec, expected_schema="tatbot.body-model-spec/1")
    reference = load_contract(args.identity, expected_schema="tatbot.body-identity/1")
    authoring = json.loads(args.poses.read_text())
    atlas = json.loads(args.atlas.read_text())
    body = SOMAPosedBody(spec_path=args.spec, cache_dir=args.cache_dir, device="cpu")
    neutral_rotations = np.zeros((77, 4), dtype=np.float64)
    neutral_rotations[:, 3] = 1
    rest = body.rest(reference)
    repeated = body.rest(reference)
    reference_triangles = rest.vertices_m[rest.faces].astype(np.float64)
    reference_normals = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    reference_normals /= np.maximum(np.linalg.norm(reference_normals, axis=1)[:, None], 1e-20)

    identity_records = []
    transfer_records = []
    transfer_names = {"reference", "mixed-alternating-2sigma", "mixed-positive-2sigma", "mixed-negative-2sigma"}
    for case_index, (name, coefficients, scales) in enumerate(_identity_cases(reference)):
        provisional = copy.deepcopy(reference)
        provisional["coefficients"] = coefficients
        provisional["scales"] = scales
        surface = body.generate(provisional, neutral_rotations)
        contract = copy.deepcopy(provisional)
        contract["rest_surface_sha256"] = surface.surface_sha256
        contract["provenance"] = {
            "producer": "tatbot_sim.body_models.audit",
            "version": "1",
            "created_utc": args.created_utc,
            "source_sha256": spec["content_sha256"],
            "seed": case_index,
        }
        contract["content_sha256"] = canonical_digest(contract)
        record = {
            "id": name,
            "contract": contract,
            "geometry": _geometry_metrics(surface, reference_normals if name == "reference" else None),
        }
        identity_records.append(record)
        if name in transfer_names:
            transfer_records.append({"id": name, **_transfer_metrics(body, provisional, atlas)})

    pose_records = []
    pose_sources = {
        **authoring["poses"],
        **{
            name: {
                "label": name,
                "support_id": "audit-only",
                "constraints": ["focused joint stress audit"],
                "body_rotation_xyzw": [0, 0, 0, 1],
                "joint_rotations_euler_xyz_deg": rotations,
            }
            for name, rotations in authoring["audit_poses"].items()
        },
    }
    for name, source in pose_sources.items():
        rotations = provider_pose_rotations(body.joint_names, source["joint_rotations_euler_xyz_deg"])
        posed = body.generate(reference, rotations, apply_correctives=True)
        lbs = body.generate(reference, rotations, apply_correctives=False)
        delta = np.linalg.norm(posed.vertices_m.astype(np.float64) - lbs.vertices_m.astype(np.float64), axis=1)
        per_site: dict[str, Any] = {}
        for site in atlas["sites"]:
            face_ids = _site_faces(atlas, site)
            if not len(face_ids):
                per_site[site] = {"faces": 0, "status": "unsupported"}
                continue
            vertex_ids = np.unique(body.faces[face_ids])
            per_site[site] = {
                "faces": int(len(face_ids)),
                "deformation": _edge_area_ratios(rest, posed, face_ids),
                "correctives_delta_m": _distribution(delta[vertex_ids]),
            }
        record = {
            "id": name,
            "label": source["label"],
            "support_id": source["support_id"],
            "constraints": source["constraints"],
            "surface_sha256": posed.surface_sha256,
            "geometry": _geometry_metrics(posed),
            "deformation": _edge_area_ratios(rest, posed),
            "correctives_delta_m": _distribution(delta),
            "by_site": per_site,
            "self_intersections": _self_intersections(posed.vertices_m, posed.faces),
            "human_visual_review": {"status": "pending", "reviewer": None, "reviewed_utc": None},
        }
        pose_records.append(record)
        _render_sheet(output / "renders" / f"{name}.png", posed.vertices_m, posed.faces)
        if name in authoring["pose_ids"]:
            _obj(output / "geometry" / f"{name}.obj", posed)

    gpu = {
        "status": "pending",
        "reason": "no explicitly assigned GPU device was supplied",
        "device": None,
        "max_error_m": None,
        "rms_error_m": None,
    }
    if args.gpu_device:
        gpu_body = SOMAPosedBody(spec_path=args.spec, cache_dir=args.cache_dir, device=args.gpu_device)
        gpu_surface = gpu_body.rest(reference)
        error = np.linalg.norm(gpu_surface.vertices_m.astype(np.float64) - rest.vertices_m.astype(np.float64), axis=1)
        gpu = {
            "status": "pass" if error.max() <= 0.0001 and math.sqrt(float(np.mean(error * error))) <= 0.00002 else "review",
            "reason": None,
            "device": args.gpu_device,
            "max_error_m": float(error.max()),
            "rms_error_m": math.sqrt(float(np.mean(error * error))),
        }

    address_parity = _expanded_indexed_parity(rest)
    metrics = {
        "reference": _geometry_metrics(rest, reference_normals),
        "repeat_byte_identical": bool(rest.vertices_m.tobytes() == repeated.vertices_m.tobytes()),
        "topology_sha256": spec["geometry"]["topology_sha256"],
        "reference_surface_sha256": rest.surface_sha256,
        "identity_cases": len(identity_records),
        "pose_cases": len(pose_records),
        "transfer": transfer_records,
        "gpu_parity": gpu,
        "expanded_indexed_address_parity": address_parity,
        "must": {
            "finite": bool(np.isfinite(rest.vertices_m).all()),
            "vertex_count": len(rest.vertices_m) == 18_056,
            "triangle_count": len(rest.faces) == 36_108,
            "no_degenerate_rest_triangles": _geometry_metrics(rest)["degenerate_triangles"] == 0,
            "reference_repeat_deterministic": bool(rest.vertices_m.tobytes() == repeated.vertices_m.tobytes()),
            "all_identity_cases_finite": all(item["geometry"]["finite"] for item in identity_records),
            "expanded_indexed_address_parity": address_parity["status"] == "pass",
        },
    }
    write_json(output / "identities.json", {"schema": "tatbot.body-identity-audit/1", "cases": identity_records})
    write_json(output / "poses.json", {"schema": "tatbot.body-pose-audit/1", "cases": pose_records})
    write_json(output / "metrics.json", metrics)
    manifest = {
        "schema": "tatbot.capability-evidence/1",
        "capability": "nominal_body",
        "evidence_type": "software",
        "consumer": "Inkmap body export and simulation rig",
        "unresolved_dependency": "maintainer visual review and assigned-device parity",
        "next_experiment": "compare retained pose assets with the pinned provider",
        "hardware_authority": False,
        "status": "pending_external_review" if gpu["status"] == "pending" else "cpu_and_gpu_complete_review_pending",
        "created_utc": args.created_utc,
        "plan": None,
        "base_git_sha": git_output("rev-parse", "HEAD"),
        "dirty_state": git_output("status", "--short"),
        "remote_comparison": git_output("rev-list", "--left-right", "--count", "origin/main...HEAD"),
        "command": sys.argv,
        "model_spec_sha256": spec["content_sha256"],
        "inputs": {
            "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
            "identity": {"path": str(args.identity), "sha256": sha256_file(args.identity)},
            "poses": {"path": str(args.poses), "sha256": sha256_file(args.poses)},
            "atlas": {"path": str(args.atlas), "sha256": sha256_file(args.atlas)},
            "cache": {"path": str(args.cache_dir), "reviewed": True},
        },
        "runtime": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "outputs": {
            name: {"sha256": sha256_file(output / name), "size": (output / name).stat().st_size}
            for name in ("identities.json", "poses.json", "metrics.json")
        },
        "accepted": {
            "offline_cpu": all(metrics["must"].values()),
            "gpu_parity": gpu["status"] == "pass",
            "human_visual_review": False,
        },
        "known_gaps": [
            *( ["GPU parity needs an explicitly assigned NVIDIA device."] if gpu["status"] == "pending" else [] ),
            "Named poses and identity sheets need maintainer human visual review.",
            "The transverse self-intersection diagnostic does not classify coplanar triangle overlap.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    (output / "audit.md").write_text(
        "# MHR/SOMA body audit\n\n"
        f"Generated {args.created_utc} on `{platform.node()}` from the exact offline cache.\n\n"
        f"- CPU structural and deterministic checks: **{'pass' if manifest['accepted']['offline_cpu'] else 'fail'}**\n"
        f"- Identity cases: {len(identity_records)} (all 45 coefficient axes plus mixed and scale-group cases)\n"
        f"- Named and stress poses: {len(pose_records)}\n"
        f"- GPU parity: **{gpu['status']}**\n"
        "- Human visual review: **pending**\n"
        "- Motion, deployment, collection, and human use: **not authorized**\n",
    )
    manifest["outputs"]["audit.md"] = {"sha256": sha256_file(output / "audit.md"), "size": (output / "audit.md").stat().st_size}
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=root / "config/body-models/mhr-soma-v1.json")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=root / "config/body-models/mhr-soma-v1/identity.json")
    parser.add_argument("--poses", type=Path, default=root / "config/body-models/mhr-soma-v1/poses.json")
    parser.add_argument("--atlas", type=Path, default=root / "web/inkmap/public/bodies/mhr-soma-v1.regions.json")
    parser.add_argument("--created-utc", default=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    parser.add_argument("--gpu-device", help="explicitly assigned CUDA device; omitted means pending")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_audit(args)
    print(json.dumps(manifest, sort_keys=True))
    if not manifest["accepted"]["offline_cpu"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
