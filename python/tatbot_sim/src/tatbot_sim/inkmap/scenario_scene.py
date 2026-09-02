"""Materialize and build the kinematic posed-body scene for ManiSkill."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, quat2mat

from tatbot_sim.inkmap.contracts import document_sha256, validate_scenario
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface, mesh_patch_from_scenario
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root

SCENE_GEOMETRY_VERSION = 3
PATCH_VISUAL_OFFSET_M = 2e-4


@dataclass(frozen=True)
class BoxProxy:
    center: np.ndarray
    half_size: np.ndarray
    quaternion_wxyz: np.ndarray | None = None


@dataclass(frozen=True)
class CapsuleProxy:
    start: np.ndarray
    end: np.ndarray
    radius: float


@dataclass(frozen=True)
class ScenarioGeometry:
    root: Path
    body_obj: Path
    patch_obj: Path
    support_boxes: tuple[BoxProxy, ...]
    collision_capsules: tuple[CapsuleProxy, ...]
    surface: MeshPatchSurface


def _write_obj(path: Path, vertices: np.ndarray, triangles_uv: np.ndarray | None = None) -> None:
    material = path.with_suffix(".mtl")
    material.write_text("newmtl skin\nKd 0.72 0.48 0.32\nKa 0.10 0.07 0.05\nNs 8\n")
    lines = [f"mtllib {material.name}", "usemtl skin"]
    flat = vertices.reshape(-1, 3)
    lines.extend(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in flat)
    if triangles_uv is not None:
        uv = triangles_uv.reshape(-1, 2)
        lines.extend(f"vt {point[0]:.9g} {point[1]:.9g}" for point in uv)
        lines.extend(f"f {i}/{i} {i + 1}/{i + 1} {i + 2}/{i + 2}" for i in range(1, len(flat) + 1, 3))
    else:
        lines.extend(f"f {i} {i + 1} {i + 2}" for i in range(1, len(flat) + 1, 3))
    path.write_text("\n".join(lines) + "\n")


def _support_boxes(support_id: str, vertices: np.ndarray, body_id: str) -> tuple[BoxProxy, ...]:
    low, high = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    center = (low + high) / 2
    span = high - low
    if support_id == "tattoo-bed-v1":
        return (BoxProxy(np.array([center[0], center[1], low[2] - 0.04]),
                         np.array([max(0.45, span[0] / 2 + 0.08), max(0.9, span[1] / 2 + 0.08), 0.04])),)
    if support_id.startswith("tattoo-chair-"):
        seat_z = low[2] + span[2] * 0.18
        armrest_z = low[2] + span[2] * (0.365 if body_id == "hbm-female-stylized" else 0.34)
        back_angle = np.deg2rad(-30.0)
        back_q = np.array([np.cos(back_angle / 2), np.sin(back_angle / 2), 0.0, 0.0])
        boxes = [
            BoxProxy(np.array([0.0, center[1], seat_z - 0.035]), np.array([0.28, 0.25, 0.035])),
            BoxProxy(np.array([0.0, center[1] + 0.40, seat_z + 0.32]), np.array([0.26, 0.03, 0.36]), back_q),
            BoxProxy(np.array([0.0, -0.04, low[2] + 0.045]), np.array([0.25, 0.27, 0.035])),
        ]
        if support_id.endswith("left-armrest-v1"):
            boxes.append(BoxProxy(
                np.array([0.45, 0.40, armrest_z]),
                np.array([0.16, 0.19, 0.025]),
            ))
        if support_id.endswith("right-armrest-v1"):
            boxes.append(BoxProxy(
                np.array([-0.45, 0.40, armrest_z]),
                np.array([0.16, 0.19, 0.025]),
            ))
        return tuple(boxes)
    return (BoxProxy(np.array([center[0], center[1], low[2] - 0.025]),
                     np.array([max(0.25, span[0] / 2), max(0.25, span[1] / 2), 0.025])),)


def _scenario_support_boxes(scenario: dict) -> tuple[BoxProxy, ...]:
    """Build support in the authored pose frame, then carry it with the body."""
    rig = load_body_rig(scenario["body"]["id"])
    named_pose = rig.posed(scenario["pose"]["id"])
    local = _support_boxes(scenario["support"]["id"], named_pose.vertices, rig.body_id)
    world_from_body = np.asarray(scenario["pose"]["world_from_body"], dtype=np.float64)
    world_from_named_pose = world_from_body @ np.linalg.inv(named_pose.body_from_rest)
    transformed = []
    for box in local:
        center = world_from_named_pose @ np.append(box.center, 1.0)
        local_rotation = np.eye(3) if box.quaternion_wxyz is None else quat2mat(box.quaternion_wxyz)
        rotation = world_from_named_pose[:3, :3] @ local_rotation
        transformed.append(BoxProxy(center[:3], box.half_size, mat2quat(rotation)))
    return tuple(transformed)


def _capsule_radius(name: str) -> float:
    if name in ("pelvis", "spine_lower", "spine_upper"):
        return 0.095
    if name in ("neck", "head"):
        return 0.075
    if name.startswith(("thigh", "upper_arm")):
        return 0.06
    if name.startswith(("shin", "forearm")):
        return 0.045
    return 0.035


def _collision_capsules(scenario: dict) -> tuple[CapsuleProxy, ...]:
    rig = load_body_rig(scenario["body"]["id"])
    config = json.loads((repo_root() / "config" / "inkmap" / "body-rig.json").read_text())
    joints = config["bodies"][rig.body_id]["joints"]
    specs = {bone["name"]: bone for bone in config["bones"]}
    pose_index = rig.pose_ids.index(scenario["pose"]["id"])
    world = np.asarray(scenario["pose"]["world_from_body"], dtype=np.float64)
    capsules = []
    for bone_index, name in enumerate(rig.bone_names):
        spec = specs[name]
        endpoints = np.asarray([joints[spec["head"]], joints[spec["tail"]]], dtype=np.float64)
        homogeneous = np.concatenate([endpoints, np.ones((2, 1))], axis=1)
        posed = (rig.pose_matrices[pose_index, bone_index] @ homogeneous.T).T
        transformed = (world @ posed.T).T[:, :3]
        if np.linalg.norm(transformed[1] - transformed[0]) > 1e-4:
            capsules.append(CapsuleProxy(transformed[0], transformed[1], _capsule_radius(name)))
    return tuple(capsules)


def materialize_scenario_geometry(scenario: dict, cache_root: Path | None = None) -> ScenarioGeometry:
    validate_scenario(scenario)
    digest = document_sha256(scenario)
    root = (cache_root or Path.home() / ".cache" / "tatbot" / "body-scenarios") / f"v{SCENE_GEOMETRY_VERSION}-{digest}"
    root.mkdir(parents=True, exist_ok=True)
    surface = mesh_patch_from_scenario(scenario)
    posed = surface.posed_vertices[0]
    body_obj = root / "body.obj"
    patch_obj = root / "tattoo-patch.obj"
    if not body_obj.exists():
        _write_obj(body_obj, posed)
    patch = surface.patches[0]
    patch_vertices = posed[patch.face_indices] + PATCH_VISUAL_OFFSET_M * surface.normals[0][patch.face_indices]
    uv = patch.triangles_uv.copy()
    uv[..., 0] = uv[..., 0] / surface.width_m + 0.5
    uv[..., 1] = uv[..., 1] / surface.height_m + 0.5
    if not patch_obj.exists():
        _write_obj(patch_obj, patch_vertices, uv)
    return ScenarioGeometry(
        root=root,
        body_obj=body_obj,
        patch_obj=patch_obj,
        support_boxes=_scenario_support_boxes(scenario),
        collision_capsules=_collision_capsules(scenario),
        surface=surface,
    )


def _capsule_pose(proxy: CapsuleProxy) -> tuple[sapien.Pose, float]:
    delta = proxy.end - proxy.start
    length = float(np.linalg.norm(delta))
    x_axis = delta / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(x_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(helper, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return sapien.Pose(p=((proxy.start + proxy.end) / 2).tolist(), q=mat2quat(rotation).tolist()), length / 2


def build_scenario_actors(scene, scenario: dict, num_envs: int):
    """Add body visual/collision, drawable patch, and support to each subscene."""
    geometry = materialize_scenario_geometry(scenario)
    bodies, patches, supports = [], [], []
    for env_index in range(num_envs):
        body = scene.create_actor_builder()
        body.add_visual_from_file(str(geometry.body_obj))
        for proxy in geometry.collision_capsules:
            pose, half_length = _capsule_pose(proxy)
            body.add_capsule_collision(pose=pose, radius=proxy.radius, half_length=half_length)
        body.set_scene_idxs([env_index])
        bodies.append(body.build_kinematic(name=f"posed_body_{env_index}"))

        patch = scene.create_actor_builder()
        patch_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.85)
        patch_material.set_base_color_texture(sapien.render.RenderTexture2D(
            array=np.full((2, 2, 4), 255, dtype=np.uint8), format="R8G8B8A8Unorm", srgb=True,
        ))
        patch.add_visual_from_file(str(geometry.patch_obj), material=patch_material)
        patch.set_scene_idxs([env_index])
        patches.append(patch.build_kinematic(name=f"tattoo_patch_{env_index}"))

        support = scene.create_actor_builder()
        for box in geometry.support_boxes:
            pose = sapien.Pose(p=box.center.tolist())
            if box.quaternion_wxyz is not None:
                pose = sapien.Pose(p=box.center.tolist(), q=box.quaternion_wxyz.tolist())
            support.add_box_visual(pose=pose, half_size=box.half_size.tolist(), material=[0.18, 0.20, 0.24, 1.0])
            support.add_box_collision(pose=pose, half_size=box.half_size.tolist())
        support.set_scene_idxs([env_index])
        supports.append(support.build_kinematic(name=f"body_support_{env_index}"))
    return bodies, patches, supports, geometry
