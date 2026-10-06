"""Deterministic CPU reference labels for Inkmap perception scenes.

This renderer is deliberately independent of SAPIEN's display pipeline.  It is
the semantic reference for dense labels and a reproducibility oracle; production
RGB can come from ManiSkill, but face/barycentric and tattoo IDs must agree with
this pass before a frame is admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface, _smooth_normals
from tatbot_sim.inkmap.program_target import ProgramTarget

PERCEPTION_LABEL_VERSION = 1
BACKGROUND_FACE = -1
BACKGROUND_ID = 0


@dataclass(frozen=True)
class PinholeCamera:
    """OpenCV pinhole camera: +x right, +y down, +z forward."""

    width: int
    height: int
    intrinsic: np.ndarray
    camera_from_world: np.ndarray

    def __post_init__(self) -> None:
        k = np.asarray(self.intrinsic, dtype=np.float64)
        extrinsic = np.asarray(self.camera_from_world, dtype=np.float64)
        if self.width < 1 or self.height < 1 or k.shape != (3, 3):
            raise ValueError("invalid pinhole image or intrinsic shape")
        if extrinsic.shape != (4, 4) or not np.allclose(extrinsic[3], [0, 0, 0, 1]):
            raise ValueError("camera_from_world must be a rigid 4x4 transform")
        if not np.isfinite(k).all() or not np.isfinite(extrinsic).all():
            raise ValueError("camera calibration must be finite")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("invalid pinhole intrinsic matrix")

    def record(self) -> dict[str, Any]:
        return {
            "model": "opencv-pinhole/1",
            "width": self.width,
            "height": self.height,
            "intrinsic": np.asarray(self.intrinsic, dtype=float).tolist(),
            "camera_from_world": np.asarray(self.camera_from_world, dtype=float).tolist(),
            "axes": "+x right, +y down, +z forward",
            "pixel_centers": "integer coordinates",
        }


@dataclass(frozen=True)
class Appearance:
    skin_srgb: tuple[float, float, float] = (0.72, 0.48, 0.32)
    skin_roughness: float = 0.75
    microtexture_strength: float = 0.0
    light_direction_camera: tuple[float, float, float] = (0.2, -0.4, -1.0)
    light_intensity: float = 0.8
    ambient: float = 0.35
    background_srgb: tuple[float, float, float] = (0.08, 0.09, 0.11)
    blur_sigma_px: float = 0.0


@dataclass(frozen=True)
class PerceptionLabels:
    rgb_srgb: np.ndarray
    depth_clean_m: np.ndarray
    depth_clean_valid: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray
    normal_camera: np.ndarray
    normal_valid: np.ndarray
    body_visible: np.ndarray
    face_index: np.ndarray
    barycentric: np.ndarray
    barycentric_valid: np.ndarray
    tattoo_coverage: np.ndarray
    tattoo_placement_id: np.ndarray
    tattoo_layer_id: np.ndarray

    def canonical_digest(self) -> str:
        digest = hashlib.sha256(f"tatbot.perception-labels/{PERCEPTION_LABEL_VERSION}\0".encode())
        for name in self.__dataclass_fields__:
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode() + b"\0" + value.dtype.str.encode() + b"\0")
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()


def look_at_camera(
    eye_world: np.ndarray,
    target_world: np.ndarray,
    *,
    width: int,
    height: int,
    focal_px: float,
) -> PinholeCamera:
    """Construct an OpenCV-frame camera with a stable world-up choice."""

    eye = np.asarray(eye_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-20)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= max(np.linalg.norm(right), 1e-20)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    intrinsic = np.asarray(
        [[focal_px, 0, (width - 1) / 2], [0, focal_px, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    return PinholeCamera(width, height, intrinsic, extrinsic)


def _sample_target(target: ProgramTarget, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    x = (float(uv[0]) / target.width_m + 0.5) * target.cols - 0.5
    y = (float(uv[1]) / target.height_m + 0.5) * target.rows - 0.5
    if x < -0.5 or x > target.cols - 0.5 or y < -0.5 or y > target.rows - 0.5:
        return np.float32(0), np.zeros(3, dtype=np.float32), 0, 0
    x = min(max(x, 0.0), target.cols - 1.0)
    y = min(max(y, 0.0), target.rows - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, target.cols - 1), min(y0 + 1, target.rows - 1)
    dx, dy = x - x0, y - y0
    weights = np.asarray([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy])
    positions = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    coverage = np.asarray([target.coverage[p] for p in positions])
    premultiplied = np.asarray([target.color_srgb[p] * target.coverage[p] for p in positions])
    alpha = float(weights @ coverage)
    color = weights @ premultiplied
    if alpha > 1e-8:
        color /= alpha
    nearest_x, nearest_y = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
    return np.float32(alpha), color.astype(np.float32), int(target.placement_id[nearest_y, nearest_x]), int(target.layer_id[nearest_y, nearest_x])


def render_perception_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: PinholeCamera,
    *,
    surface: MeshPatchSurface | None = None,
    target: ProgramTarget | None = None,
    appearance: Appearance | None = None,
    seed: int = 0,
    occluder_depth_m: np.ndarray | None = None,
    occluder_srgb: tuple[float, float, float] = (0.16, 0.18, 0.22),
) -> PerceptionLabels:
    """Rasterize dense labels with perspective-correct surface coordinates.

    ``vertices`` may be indexed ``(V,3)`` or expanded ``(F,3,3)``.  The
    optional occluder contains camera-z depth with NaN/inf meaning absent.
    Occluders contribute valid scene depth but never body/tattoo semantics.
    """

    appearance = appearance or Appearance()
    indexed_faces = np.asarray(faces, dtype=np.int32)
    raw_vertices = np.asarray(vertices, dtype=np.float64)
    triangles_world = raw_vertices[indexed_faces] if raw_vertices.ndim == 2 else raw_vertices
    if triangles_world.shape != (len(indexed_faces), 3, 3):
        raise ValueError("body geometry must resolve to (F,3,3)")
    if (surface is None) != (target is None):
        raise ValueError("surface and target must be supplied together")
    if surface is not None and surface.batch_size != 1:
        raise ValueError("reference label renderer accepts one scene at a time")

    h, w = camera.height, camera.width
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[...] = np.asarray(appearance.background_srgb, dtype=np.float32)
    depth = np.full((h, w), np.inf, dtype=np.float64)
    face_index = np.full((h, w), BACKGROUND_FACE, dtype=np.int32)
    barycentric = np.zeros((h, w, 3), dtype=np.float32)
    normal_camera = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)
    placement_id = np.zeros((h, w), dtype=np.int32)
    layer_id = np.zeros((h, w), dtype=np.int32)

    transform = np.asarray(camera.camera_from_world, dtype=np.float64)
    triangles_camera = np.einsum("ij,fkj->fki", transform[:3, :3], triangles_world) + transform[:3, 3]
    corner_normals_world = _smooth_normals(triangles_world)
    corner_normals_camera = np.einsum("ij,fkj->fki", transform[:3, :3], corner_normals_world)
    k = np.asarray(camera.intrinsic, dtype=np.float64)

    patch_uv: dict[int, np.ndarray] = {}
    if surface is not None:
        patch = surface.patches[0]
        patch_uv = {int(face): patch.triangles_uv[index] for index, face in enumerate(patch.face_indices)}

    skin = np.asarray(appearance.skin_srgb, dtype=np.float32)
    light = np.asarray(appearance.light_direction_camera, dtype=np.float64)
    light /= max(np.linalg.norm(light), 1e-20)
    rng = np.random.default_rng(seed)
    microtexture = rng.standard_normal((h, w)).astype(np.float32)

    for face, triangle in enumerate(triangles_camera):
        z = triangle[:, 2]
        if np.any(z <= 1e-6):
            continue
        projected = np.empty((3, 2), dtype=np.float64)
        projected[:, 0] = k[0, 0] * triangle[:, 0] / z + k[0, 2]
        projected[:, 1] = k[1, 1] * triangle[:, 1] / z + k[1, 2]
        low = np.maximum(np.floor(projected.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(projected.max(axis=0)).astype(int), [w - 1, h - 1])
        if np.any(low > high):
            continue
        denominator = (
            (projected[1, 1] - projected[2, 1]) * (projected[0, 0] - projected[2, 0])
            + (projected[2, 0] - projected[1, 0]) * (projected[0, 1] - projected[2, 1])
        )
        if abs(float(denominator)) < 1e-12:
            continue
        for py in range(low[1], high[1] + 1):
            for px in range(low[0], high[0] + 1):
                b0 = (
                    (projected[1, 1] - projected[2, 1]) * (px - projected[2, 0])
                    + (projected[2, 0] - projected[1, 0]) * (py - projected[2, 1])
                ) / denominator
                b1 = (
                    (projected[2, 1] - projected[0, 1]) * (px - projected[2, 0])
                    + (projected[0, 0] - projected[2, 0]) * (py - projected[2, 1])
                ) / denominator
                screen_bary = np.asarray([b0, b1, 1.0 - b0 - b1])
                if float(screen_bary.min()) < -1e-9:
                    continue
                reciprocal = screen_bary / z
                pixel_depth = 1.0 / float(reciprocal.sum())
                if pixel_depth >= depth[py, px]:
                    continue
                bary = reciprocal * pixel_depth
                normal = bary @ corner_normals_camera[face]
                normal /= max(np.linalg.norm(normal), 1e-20)
                diffuse = max(0.0, float(np.dot(normal, -light)))
                view = -(bary @ triangle)
                view /= max(np.linalg.norm(view), 1e-20)
                half_vector = view - light
                half_vector /= max(np.linalg.norm(half_vector), 1e-20)
                roughness = float(np.clip(appearance.skin_roughness, 0.02, 1.0))
                specular = (1.0 - roughness) * max(0.0, float(np.dot(normal, half_vector))) ** (
                    4.0 + 60.0 * (1.0 - roughness)
                )
                shade = appearance.ambient + appearance.light_intensity * (diffuse + specular)
                shade *= 1.0 + appearance.microtexture_strength * microtexture[py, px]
                color = skin * np.clip(shade, 0.05, 1.5)
                alpha, ink, pid, lid = np.float32(0), np.zeros(3, np.float32), 0, 0
                if target is not None and face in patch_uv:
                    alpha, ink, pid, lid = _sample_target(target, bary @ patch_uv[face])
                    color = color * (1.0 - alpha) + ink * alpha
                depth[py, px] = pixel_depth
                face_index[py, px] = face
                barycentric[py, px] = bary.astype(np.float32)
                normal_camera[py, px] = normal.astype(np.float32)
                coverage[py, px] = alpha
                placement_id[py, px] = pid if alpha > 0 else 0
                layer_id[py, px] = lid if alpha > 0 else 0
                rgb[py, px] = np.clip(color, 0, 1)

    body_visible = face_index != BACKGROUND_FACE
    if occluder_depth_m is not None:
        occ = np.asarray(occluder_depth_m, dtype=np.float64)
        if occ.shape != (h, w):
            raise ValueError("occluder_depth_m shape differs from camera image")
        occluded = np.isfinite(occ) & (occ > 0) & (occ < depth)
        depth[occluded] = occ[occluded]
        rgb[occluded] = np.asarray(occluder_srgb, dtype=np.float32)
        face_index[occluded] = BACKGROUND_FACE
        barycentric[occluded] = 0
        normal_camera[occluded] = 0
        coverage[occluded] = 0
        placement_id[occluded] = BACKGROUND_ID
        layer_id[occluded] = BACKGROUND_ID
        body_visible[occluded] = False

    depth_valid = np.isfinite(depth)
    depth_out = depth.astype(np.float32)
    depth_out[~depth_valid] = np.nan
    normal_valid = body_visible.copy()
    if appearance.blur_sigma_px > 0:
        ksize = max(3, int(math.ceil(appearance.blur_sigma_px * 6)) | 1)
        rgb = cv2.GaussianBlur(rgb, (ksize, ksize), appearance.blur_sigma_px)
    return PerceptionLabels(
        rgb_srgb=np.clip(rgb, 0, 1).astype(np.float32),
        depth_clean_m=depth_out.copy(),
        depth_clean_valid=depth_valid.copy(),
        depth_m=depth_out,
        depth_valid=depth_valid,
        normal_camera=normal_camera,
        normal_valid=normal_valid,
        body_visible=body_visible,
        face_index=face_index,
        barycentric=barycentric,
        barycentric_valid=body_visible.copy(),
        tattoo_coverage=coverage,
        tattoo_placement_id=placement_id,
        tattoo_layer_id=layer_id,
    )


def corrupt_sensor_depth(
    labels: PerceptionLabels,
    *,
    seed: int,
    noise_std_m: float,
    dropout_fraction: float,
) -> PerceptionLabels:
    """Apply sensor corruption without changing clean geometry or semantics."""

    if noise_std_m < 0 or not 0 <= dropout_fraction <= 1:
        raise ValueError("invalid depth corruption parameters")
    rng = np.random.default_rng(seed)
    depth = labels.depth_clean_m.copy()
    valid = labels.depth_clean_valid.copy()
    if noise_std_m:
        depth[valid] += rng.normal(0.0, noise_std_m, size=int(valid.sum())).astype(np.float32)
    if dropout_fraction:
        valid &= rng.random(valid.shape) >= dropout_fraction
    valid &= depth > 0
    depth[~valid] = np.nan
    return PerceptionLabels(
        **{
            **{name: getattr(labels, name) for name in labels.__dataclass_fields__},
            "depth_m": depth,
            "depth_valid": valid,
        }
    )


def write_perception_sidecar(
    root: Path,
    labels: PerceptionLabels,
    camera: PinholeCamera,
    *,
    bindings: dict[str, Any],
    seed_streams: dict[str, int],
    sampled_axes: dict[str, Any],
) -> dict[str, Any]:
    """Write one complete privileged-label frame and its strict manifest."""

    required = {"identity_sha256", "rest_surface_sha256", "posed_surface_sha256", "design_sha256", "placement_sha256", "scenario_sha256", "artwork_family_sha256", "source_revision", "source_repository", "source_dirty", "asset_provenance", "dependency_versions", "split"}
    missing = sorted(required - bindings.keys())
    if missing:
        raise ValueError(f"perception bindings missing: {', '.join(missing)}")
    root.mkdir(parents=True, exist_ok=False)
    array_path = root / "labels.npz"
    np.savez_compressed(array_path, **{name: getattr(labels, name) for name in labels.__dataclass_fields__})
    cv2.imwrite(str(root / "rgb.png"), np.rint(labels.rgb_srgb[..., ::-1] * 255).astype(np.uint8))
    manifest = {
        "schema": "tatbot.inkmap-perception-frame/1",
        "renderer": f"tatbot_sim.inkmap.perception/{PERCEPTION_LABEL_VERSION}",
        "status": "accepted",
        "bindings": bindings,
        "camera": camera.record(),
        "seed_streams": seed_streams,
        "sampled_axes": sampled_axes,
        "labels_sha256": labels.canonical_digest(),
        "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "conventions": {
            "depth_clean": "float32 metric camera-z ground truth; NaN where invalid; separate validity",
            "depth": "float32 metric camera-z sensor sample after declared corruption; NaN where invalid",
            "normal": "float32 outward geometric normal in camera frame; zero where invalid",
            "face_index": "int32 global zero-based SOMA face; -1 for non-body/background/occluder",
            "barycentric": "float32 perspective-correct canonical face weights; zero where invalid",
            "ids": "int32 nearest-sampled; 0 means no tattoo; never interpolated",
            "coverage": "float32 visible effective opacity in [0,1]; bilinear target sampling",
            "occlusion": "occluders keep scene depth valid and clear all body/tattoo semantics",
            "rgb": "float32 unassociated sRGB in NPZ; PNG is an 8-bit review derivative",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
