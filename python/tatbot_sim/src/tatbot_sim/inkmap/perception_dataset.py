"""Render an audited CPU Inkmap perception pilot from a compiled v3 scenario."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import resource
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import tyro

from tatbot_sim.inkmap.contracts import document_sha256, load_scenario
from tatbot_sim.inkmap.identities import admitted_identity_ids
from tatbot_sim.inkmap.perception import (
    Appearance,
    corrupt_sensor_depth,
    look_at_camera,
    render_perception_labels,
    write_perception_sidecar,
)
from tatbot_sim.inkmap.perception_audit import audit_corpus
from tatbot_sim.inkmap.perception_variation import PerceptionVariation, split_for_groups
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.scenario_scene import materialize_scenario_geometry
from tatbot_sim.repo import source_state


@dataclass
class Args:
    scenario: Path
    output_dir: Path
    seed: int = 0
    views: int = 3
    width: int = 256
    height: int = 256
    focal_px: float = 300.0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _patch_frame(geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch = geometry.surface.patches[0]
    anchor, uv = patch.map_samples(np.zeros((1, 2)))[0]
    triangle = geometry.surface.posed_vertices[0][anchor.face]
    bary = np.asarray(anchor.barycentric)
    center = bary @ triangle
    chart_edges = np.stack([uv[1] - uv[0], uv[2] - uv[0]], axis=1)
    world_edges = np.stack([triangle[1] - triangle[0], triangle[2] - triangle[0]], axis=1)
    derivative = world_edges @ np.linalg.inv(chart_edges)
    tangent_u = derivative[:, 0] / np.linalg.norm(derivative[:, 0])
    tangent_v = derivative[:, 1] - tangent_u * np.dot(tangent_u, derivative[:, 1])
    tangent_v /= np.linalg.norm(tangent_v)
    normal = np.cross(tangent_u, tangent_v)
    normal /= np.linalg.norm(normal)
    expected = geometry.surface.base_normal_np(0)
    if np.dot(normal, expected) < 0:
        normal *= -1
        tangent_v *= -1
    return center, tangent_u, tangent_v, normal


def _occluder(height: int, width: int, fraction: float, depth: float, seed: int) -> np.ndarray:
    result = np.full((height, width), np.nan, dtype=np.float32)
    if fraction <= 0:
        return result
    rng = np.random.default_rng(seed)
    fraction = min(float(fraction), 0.45)
    aspect = float(rng.uniform(0.6, 1.7))
    box_h = max(1, min(height, int(round(math.sqrt(fraction * height * width / aspect)))))
    box_w = max(1, min(width, int(round(box_h * aspect))))
    cy = int(round(height * rng.uniform(0.35, 0.65)))
    cx = int(round(width * rng.uniform(0.35, 0.65)))
    y0, x0 = max(0, cy - box_h // 2), max(0, cx - box_w // 2)
    result[y0 : min(height, y0 + box_h), x0 : min(width, x0 + box_w)] = depth
    return result


def _dependencies() -> dict[str, str]:
    output = {"numpy": np.__version__, "opencv": cv2.__version__}
    for name in ("mani-skill-nightly", "sapien", "shapely"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def render_dataset(args: Args) -> dict:
    if args.views < 1 or args.width < 32 or args.height < 32 or args.focal_px <= 0:
        raise ValueError("views/image dimensions/focal length must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(args.scenario)
    if scenario.get("schema_version") != 3:
        raise ValueError("perception labels require a typed v3 scenario")
    rig = load_body_rig()
    if scenario["body"]["identity_sha256"] not in {
        item["identity_sha256"]
        for item in json.loads(
            (Path(__file__).resolve().parents[5] / "config/inkmap/synthetic-identities.json").read_text()
        )["identities"]
        if item["id"] in admitted_identity_ids()
    }:
        raise ValueError("identity_not_admitted: scenario body has no accepted visual review")
    geometry = materialize_scenario_geometry(scenario)
    if geometry.target is None:
        raise ValueError("typed scenario did not materialize a ProgramTarget")
    source = source_state()
    if not isinstance(source["revision"], str) or len(source["revision"]) != 40:
        raise ValueError("cannot determine full source revision")
    binding = scenario["program_binding"]
    bundle = binding["bundle"]
    placement = next(item for item in bundle["placement_file"]["placements"] if item["id"] == binding["placement_id"])
    artwork = bundle["artworks"][placement["design_id"]]
    surface_placement = next(
        item["placement"] for item in bundle["surface_placements"] if item["id"] == placement["id"]
    )
    provenance = artwork["source"]
    family = _sha256_bytes(
        json.dumps(
            {
                "source_sha256": artwork["source_sha256"],
                "kind": provenance["kind"],
                "identifier": provenance["identifier"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    split = split_for_groups(
        artwork_family_sha256=family,
        identity_sha256=scenario["body"]["identity_sha256"],
        seed=args.seed,
    )
    center, tangent_u, tangent_v, normal = _patch_frame(geometry)
    variation = PerceptionVariation()
    request_entries = []
    started = time.perf_counter()
    for view in range(args.views):
        scene_id = f"{document_sha256(scenario)[:16]}-view-{view:02d}"
        sampled, streams = variation.sample(args.seed, scene_id)
        azimuth = math.radians(sampled["light_azimuth_deg"])
        elevation = math.radians(sampled["light_elevation_deg"])
        light = (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        )
        skin_base = np.asarray([int(bundle["request"]["skin_tone"][i : i + 2], 16) for i in (1, 3, 5)]) / 255
        appearance = Appearance(
            skin_srgb=tuple(np.clip(skin_base * sampled["skin_lightness"], 0, 1)),
            skin_roughness=sampled["skin_roughness"],
            microtexture_strength=sampled["skin_microtexture"],
            light_direction_camera=light,
            light_intensity=sampled["light_intensity"],
            background_srgb=tuple(np.clip(np.asarray([0.08, 0.09, 0.11]) * sampled["background_lightness"], 0, 1)),
            blur_sigma_px=sampled["rgb_blur_sigma_px"],
        )
        lateral = (view - (args.views - 1) / 2) * 0.10
        camera_rng = np.random.default_rng(streams["camera_position_jitter_m"])
        position_jitter = camera_rng.uniform(-1, 1, size=3) * abs(sampled["camera_position_jitter_m"])
        look_rng = np.random.default_rng(streams["camera_look_jitter_m"])
        look_jitter = look_rng.uniform(-1, 1, size=2) * abs(sampled["camera_look_jitter_m"])
        eye = center + normal * 0.30 + tangent_u * lateral + position_jitter
        look = center + tangent_u * look_jitter[0] + tangent_v * look_jitter[1]
        camera = look_at_camera(
            eye,
            look,
            width=args.width,
            height=args.height,
            focal_px=args.focal_px * sampled["camera_focal_scale"],
        )
        center_camera = camera.camera_from_world[:3, :3] @ center + camera.camera_from_world[:3, 3]
        occluder = _occluder(
            args.height,
            args.width,
            sampled["occlusion_fraction"],
            max(0.02, float(center_camera[2]) - 0.025),
            streams["occlusion_fraction"],
        )
        labels = render_perception_labels(
            geometry.surface.posed_vertices[0],
            rig.faces,
            camera,
            surface=geometry.surface,
            target=replace(
                geometry.target,
                coverage=geometry.target.coverage * sampled["tattoo_ink_opacity"],
            ),
            appearance=appearance,
            seed=streams["skin_microtexture"],
            occluder_depth_m=occluder,
        )
        labels = corrupt_sensor_depth(
            labels,
            seed=streams["depth_noise_std_m"],
            noise_std_m=sampled["depth_noise_std_m"],
            dropout_fraction=sampled["depth_dropout_fraction"],
        )
        sampled["tattoo_scale_applied"] = False
        sampled["tattoo_rotation_applied"] = False
        sampled["geometry_note"] = "tattoo scale and rotation are immutable scenario/compiler axes; this fixed-scenario renderer records but does not reinterpret them"
        bindings = {
            "scene_id": scene_id,
            "identity_sha256": scenario["body"]["identity_sha256"],
            "rest_surface_sha256": scenario["body"]["rest_surface_sha256"],
            "posed_surface_sha256": scenario["pose"]["posed_surface_sha256"],
            "design_sha256": artwork["program"]["content_sha256"],
            "placement_sha256": surface_placement["content_sha256"],
            "scenario_sha256": document_sha256(scenario),
            "artwork_family_sha256": family,
            "split": split,
            "source_repository": source["repository"],
            "source_revision": source["revision"],
            "source_dirty": source["dirty"],
            "asset_provenance": {
                "artwork": provenance,
                "body_asset_sha256": scenario["body"]["asset_sha256"],
                "pose_asset_sha256": scenario["body"]["pose_asset_sha256"],
            },
            "dependency_versions": _dependencies(),
        }
        write_perception_sidecar(
            output / "frames" / scene_id,
            labels,
            camera,
            bindings=bindings,
            seed_streams=streams,
            sampled_axes=sampled,
        )
        request_entries.append({"scene_id": scene_id, "status": "accepted"})
    elapsed = time.perf_counter() - started
    (output / "requests.json").write_text(
        json.dumps(
            {"schema": "tatbot.inkmap-perception-requests/1", "requests": request_entries},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return _finalize_report(args, output, elapsed=elapsed)


def _finalize_report(args, output: Path, *, elapsed: float) -> dict:
    """Write the audit and project from it, or say why there is nothing to project."""
    report = audit_corpus(output)
    per_frame = report["bytes_per_accepted_frame"]
    report["performance"] = {
        "elapsed_s": elapsed,
        "frames_per_s": args.views / elapsed,
        "peak_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "projected_540_frame_elapsed_s": elapsed / args.views * 540,
        # A run that accepted nothing has no per-frame size to project from. It
        # still has an audit worth reading, and reporting that is the whole
        # point of the ledger — a TypeError here buried the actual reasons.
        "projected_540_frame_bytes": per_frame * 540 if per_frame is not None else None,
        "note": "linear projection from this CPU reference run; benchmark the assigned production renderer before scaling",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("perception dataset audit failed: " + "; ".join(report["problems"]))
    return report



def main() -> None:
    report = render_dataset(tyro.cli(Args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
