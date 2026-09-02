"""Build deterministic humanoid rigs from Inkmap's checked-in HBM GLBs.

The static GLBs remain the canonical placement surfaces. This tool imports one,
fits the shared HBM skeleton, computes automatic heat weights (with an explicit
nearest-bone fallback), solves the named IK poses, and writes:

* ``<body>.rigged.glb`` for Three.js skinning;
* ``<body>.rig.npz`` for Python/ManiSkill named-pose geometry and future LBS;
* ``config/inkmap/body-poses.json`` shared by the web and simulator runtimes.

Run from a repository checkout:

    blender --background --factory-startup --python web/inkmap/tools/rig-hbm.py

No source ``.blend`` is required: the exact checked-in static GLB is the source
of truth, and the rigged export must reproduce its canonical rest-surface hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from math import radians
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

REPO = Path(__file__).resolve().parents[3]
_surface_path = REPO / "python" / "tatbot_sim" / "src" / "tatbot_sim" / "inkmap" / "gltf_surface.py"
_surface_spec = importlib.util.spec_from_file_location("tatbot_gltf_surface", _surface_path)
if _surface_spec is None or _surface_spec.loader is None:
    raise RuntimeError(f"cannot load {_surface_path}")
_surface_module = importlib.util.module_from_spec(_surface_spec)
sys.modules[_surface_spec.name] = _surface_module
_surface_spec.loader.exec_module(_surface_module)
CanonicalSurface = _surface_module.CanonicalSurface
load_canonical_surface = _surface_module.load_canonical_surface

SKIN = (1.0, 1.0, 1.0, 1.0)
EYE = (0.09, 0.10, 0.11, 1.0)
SKIN_NODES = ("Body", "EyeL", "EyeR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--body", action="append", dest="bodies")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.armatures, bpy.data.meshes, bpy.data.materials, bpy.data.curves):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def import_parts(path: Path) -> dict[str, bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.context.scene.objects) - before
    parts: dict[str, bpy.types.Object] = {}
    for name in SKIN_NODES:
        obj = next((o for o in imported if o.name == name), None)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"{path}: missing mesh node {name}")
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        parts[name] = obj
    for obj in imported:
        if obj not in parts.values():
            bpy.data.objects.remove(obj, do_unlink=True)
    return parts


def create_armature(config: dict, joints: dict[str, list[float]]) -> bpy.types.Object:
    data = bpy.data.armatures.new(config["rig_id"])
    armature = bpy.data.objects.new(config["rig_id"], data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for spec in config["bones"]:
        bone = data.edit_bones.new(spec["name"])
        bone.head = joints[spec["head"]]
        bone.tail = joints[spec["tail"]]
        bone.use_deform = bool(spec.get("deform", True))
        if (bone.tail - bone.head).length < 1e-4:
            raise RuntimeError(f"bone {bone.name}: head and tail coincide")
        parent = spec.get("parent")
        if parent:
            bone.parent = data.edit_bones[parent]
        # Stable roll: align local X as closely as possible with body front.
        bone.align_roll(Vector((0, -1, 0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def parent_auto_weights(body: bpy.types.Object, armature: bpy.types.Object) -> int:
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    if "FINISHED" not in result:
        raise RuntimeError("Blender automatic heat weighting did not finish")
    # Parenting is not part of the numerical contract; keep both objects in the
    # same identity frame and retain only the armature modifier.
    world = body.matrix_world.copy()
    body.parent = None
    body.matrix_world = world

    bpy.context.view_layer.objects.active = body
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="BONE_DEFORM", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="BONE_DEFORM", lock_active=False)

    deform = {bone.name for bone in armature.data.bones if bone.use_deform}
    fallback = 0
    segments = {
        bone.name: (armature.matrix_world @ bone.head_local, armature.matrix_world @ bone.tail_local)
        for bone in armature.data.bones if bone.use_deform
    }
    for vertex in body.data.vertices:
        total = sum(g.weight for g in vertex.groups if body.vertex_groups[g.group].name in deform)
        if total > 1e-6:
            continue
        point = body.matrix_world @ vertex.co
        nearest = min(segments, key=lambda name: segment_distance(point, *segments[name]))
        body.vertex_groups[nearest].add([vertex.index], 1.0, "REPLACE")
        fallback += 1
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="BONE_DEFORM", lock_active=False)
    return fallback


def segment_distance(point: Vector, start: Vector, end: Vector) -> float:
    delta = end - start
    t = max(0.0, min(1.0, (point - start).dot(delta) / max(delta.length_squared, 1e-12)))
    return (point - (start + t * delta)).length


def bind_eyes(parts: dict[str, bpy.types.Object], armature: bpy.types.Object) -> None:
    for name in ("EyeL", "EyeR"):
        eye = parts[name]
        group = eye.vertex_groups.new(name="head")
        group.add(list(range(len(eye.data.vertices))), 1.0, "REPLACE")
        modifier = eye.modifiers.new(name="Armature", type="ARMATURE")
        modifier.object = armature


def coordinate_key(point) -> tuple[int, int, int]:
    return tuple(int(np.floor(float(v) * 1e5 + 0.5)) for v in point)


def canonical_vertex_maps(surface: CanonicalSurface, parts: dict[str, bpy.types.Object]) -> dict[str, np.ndarray]:
    mappings: dict[str, np.ndarray] = {}
    for part in surface.parts:
        obj = parts[part.name]
        by_position: dict[tuple[int, int, int], list[int]] = {}
        for vertex in obj.data.vertices:
            by_position.setdefault(coordinate_key(obj.matrix_world @ vertex.co), []).append(vertex.index)
        start, stop = part.first_face, part.first_face + part.face_count
        canonical = surface.vertices[start:stop].reshape(-1, 3)
        mapped = np.empty(len(canonical), dtype=np.int32)
        for i, point in enumerate(canonical):
            candidates = by_position.get(coordinate_key(point))
            if not candidates:
                raise RuntimeError(f"{part.name}: canonical vertex {i} has no imported-mesh match")
            mapped[i] = min(candidates)
        mappings[part.name] = mapped.reshape(part.face_count, 3)
    return mappings


def reset_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.ik_stretch = 0.0
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
    bpy.context.view_layer.update()


def chain_length(bone: bpy.types.PoseBone, count: int) -> float:
    """Return the rest length of an IK chain, starting at its end effector."""
    length = 0.0
    current = bone
    for _ in range(count):
        length += current.bone.length
        if current.parent is None:
            break
        current = current.parent
    return length


def target_offset(request: dict, bone: bpy.types.PoseBone) -> Vector:
    if "target_offset" in request:
        return Vector(request["target_offset"])
    if "target_offset_chain_fraction" in request:
        scale = chain_length(bone, int(request["chain"]))
        return Vector(request["target_offset_chain_fraction"]) * scale
    raise RuntimeError(f"IK request for {bone.name} has no target offset")


def align_bone_with_parent(armature: bpy.types.Object, bone_name: str) -> None:
    """Remove wrist cocking while preserving the solved parent-chain pose."""
    bone = armature.pose.bones[bone_name]
    if bone.parent is None:
        raise RuntimeError(f"cannot align parentless bone {bone_name}")
    current = (bone.tail - bone.head).normalized()
    desired = (bone.parent.tail - bone.parent.head).normalized()
    delta = current.rotation_difference(desired)
    pivot = bone.head.copy()
    bone.matrix = (
        Matrix.Translation(pivot)
        @ delta.to_matrix().to_4x4()
        @ Matrix.Translation(-pivot)
        @ bone.matrix
    )
    bpy.context.view_layer.update()


def point_bone_between(bone: bpy.types.PoseBone, head: Vector, tail: Vector) -> None:
    """Aim a bone with the shortest rest-to-pose rotation and no axial twist."""
    rest = bone.bone
    rest_direction = (rest.tail_local - rest.head_local).normalized()
    posed_direction = (tail - head).normalized()
    rotation = rest_direction.rotation_difference(posed_direction)
    matrix = (rotation.to_matrix() @ rest.matrix_local.to_3x3()).to_4x4()
    matrix.translation = head
    bone.matrix = matrix
    bpy.context.view_layer.update()


def solve_two_bone(armature: bpy.types.Object, request: dict, joints: dict[str, list[float]]) -> None:
    """Solve a hinge chain in its requested bend plane without Blender IK roll."""
    end = armature.pose.bones[request["bone"]]
    upper = end.parent
    if int(request["chain"]) != 2 or upper is None:
        raise RuntimeError(f"analytic two-bone solve requires a two-bone chain at {end.name}")
    root = Vector(joints[request["target_from"]])
    target = root + target_offset(request, end)
    reach = target - root
    distance = reach.length
    upper_length = upper.bone.length
    lower_length = end.bone.length
    if distance >= upper_length + lower_length or distance <= abs(upper_length - lower_length):
        raise RuntimeError(
            f"analytic two-bone target for {end.name} is outside the reachable annulus"
        )
    axis = reach.normalized()
    pole = Vector(joints[request["pole_from"]]) + Vector(request["pole_offset"])
    bend = pole - root
    bend -= axis * bend.dot(axis)
    if bend.length < 1e-6:
        raise RuntimeError(f"analytic two-bone pole for {end.name} lies on the target axis")
    bend.normalize()
    along = (
        upper_length * upper_length
        - lower_length * lower_length
        + distance * distance
    ) / (2 * distance)
    offset = np.sqrt(max(upper_length * upper_length - along * along, 0.0))
    middle = root + axis * along + bend * offset
    point_bone_between(upper, root, middle)
    point_bone_between(end, middle, target)


def solve_pose(armature: bpy.types.Object, pose: dict, joints: dict[str, list[float]]) -> dict[str, list[float]]:
    reset_pose(armature)
    helpers = []
    constrained = []
    for i, request in enumerate(pose.get("ik", [])):
        bone = armature.pose.bones[request["bone"]]
        if request.get("solver") == "two_bone":
            solve_two_bone(armature, request, joints)
            continue
        target = bpy.data.objects.new(f"ik-target-{i}", None)
        bpy.context.scene.collection.objects.link(target)
        target.location = Vector(joints[request["target_from"]]) + target_offset(request, bone)
        constraint = bone.constraints.new("IK")
        constraint.target = target
        constraint.chain_count = int(request["chain"])
        constraint.iterations = 100
        helpers.append(target)
        if "pole_from" in request:
            pole = bpy.data.objects.new(f"ik-pole-{i}", None)
            bpy.context.scene.collection.objects.link(pole)
            pole.location = Vector(joints[request["pole_from"]]) + Vector(request["pole_offset"])
            constraint.pole_target = pole
            constraint.pole_angle = radians(float(request.get("pole_angle_deg", 0)))
            helpers.append(pole)
        constrained.append(bone)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = armature.evaluated_get(depsgraph)
    matrices = {bone.name: evaluated.pose.bones[bone.name].matrix.copy() for bone in armature.pose.bones}
    for bone in constrained:
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
    for helper in helpers:
        bpy.data.objects.remove(helper, do_unlink=True)
    reset_pose(armature)
    for spec in armature.data.bones:
        pose_bone = armature.pose.bones[spec.name]
        pose_bone.matrix_basis = armature.convert_space(
            pose_bone=pose_bone,
            matrix=matrices[spec.name],
            from_space="POSE",
            to_space="LOCAL",
        )
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()
    for bone_name in pose.get("align_with_parent", []):
        align_bone_with_parent(armature, bone_name)

    rotations = {}
    for bone in armature.pose.bones:
        location, quaternion, scale = bone.matrix_basis.decompose()
        if location.length > 1e-5 or (scale - Vector((1, 1, 1))).length > 1e-5:
            raise RuntimeError(
                f"pose {pose['label']}: {bone.name} contains non-rotational bone motion "
                f"location={tuple(round(v, 7) for v in location)} "
                f"scale={tuple(round(v, 7) for v in scale)}"
            )
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = quaternion
        bone.scale = (1, 1, 1)
        rotations[bone.name] = [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)]
    bpy.context.view_layer.update()
    return rotations


def posed_vertices(surface: CanonicalSurface, parts: dict[str, bpy.types.Object], mappings: dict[str, np.ndarray]) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for part in surface.parts:
        obj = parts[part.name].evaluated_get(depsgraph)
        mesh = obj.to_mesh()
        try:
            coordinates = np.asarray([tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices], dtype=np.float32)
            chunks.append(coordinates[mappings[part.name]])
        finally:
            obj.to_mesh_clear()
    return np.concatenate(chunks)


def canonical_weights(surface: CanonicalSurface, parts: dict[str, bpy.types.Object], mappings: dict[str, np.ndarray], bone_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bone_index = {name: i for i, name in enumerate(bone_names)}
    all_indices, all_weights = [], []
    for part in surface.parts:
        obj = parts[part.name]
        index_rows = np.zeros((part.face_count * 3, 4), dtype=np.uint16)
        weight_rows = np.zeros((part.face_count * 3, 4), dtype=np.float32)
        for row, source_index in enumerate(mappings[part.name].reshape(-1)):
            influences = sorted(
                ((bone_index[obj.vertex_groups[g.group].name], g.weight) for g in obj.data.vertices[int(source_index)].groups if obj.vertex_groups[g.group].name in bone_index),
                key=lambda value: value[1], reverse=True,
            )[:4]
            if not influences:
                raise RuntimeError(f"{part.name}: source vertex {source_index} is unweighted")
            total = sum(weight for _, weight in influences)
            for column, (joint, weight) in enumerate(influences):
                index_rows[row, column] = joint
                weight_rows[row, column] = weight / total
        all_indices.append(index_rows)
        all_weights.append(weight_rows)
    return np.concatenate(all_indices), np.concatenate(all_weights)


def pose_matrices(armature: bpy.types.Object, bone_names: list[str]) -> np.ndarray:
    matrices = []
    for name in bone_names:
        rest_inverse = armature.data.bones[name].matrix_local.inverted()
        matrices.append(np.asarray(armature.pose.bones[name].matrix @ rest_inverse, dtype=np.float32))
    return np.stack(matrices)


def posed_joint_positions(config: dict, joints: dict[str, list[float]], armature: bpy.types.Object, body_rotation_xyzw: list[float]) -> dict[str, Vector]:
    """Resolve named skeleton joints in the support/world-oriented pose frame."""
    body_rotation = Quaternion((body_rotation_xyzw[3], *body_rotation_xyzw[:3]))
    positions: dict[str, Vector] = {}
    for spec in config["bones"]:
        bone = armature.pose.bones[spec["name"]]
        rest_inverse = armature.data.bones[spec["name"]].matrix_local.inverted()
        deformation = bone.matrix @ rest_inverse
        for endpoint in ("head", "tail"):
            joint_name = spec[endpoint]
            if joint_name in positions:
                continue
            point = deformation @ Vector(joints[joint_name])
            positions[joint_name] = body_rotation @ point
    return positions


def joint_angle_deg(positions: dict[str, Vector], names: list[str]) -> float:
    first, pivot, last = (positions[name] for name in names)
    before = (first - pivot).normalized()
    after = (last - pivot).normalized()
    return float(np.degrees(np.arccos(np.clip(before.dot(after), -1, 1))))


def validate_anatomy(config: dict, pose: dict, positions: dict[str, Vector]) -> dict[str, float]:
    """Gate semantic joint relationships that topology-only metrics cannot see."""
    metrics: dict[str, float] = {}
    failures = []
    for gate_id in pose.get("anatomy_gates", []):
        gate = config["anatomy_gates"][gate_id]
        for angle in gate.get("angles", []):
            names = angle["joints"]
            value = joint_angle_deg(positions, names)
            key = f"angle_deg:{'-'.join(names)}"
            metrics[key] = value
            if not float(angle["min_deg"]) <= value <= float(angle["max_deg"]):
                failures.append(
                    f"{key}={value:.2f} outside {angle['min_deg']}..{angle['max_deg']}"
                )
        for bend_plane in gate.get("bend_plane", []):
            names = bend_plane["joints"]
            first, pivot, last = (positions[name] for name in names)
            axis = (last - first).normalized()
            bend = pivot - (first + axis * (pivot - first).dot(axis))
            toward = Vector(bend_plane["toward_world"])
            toward -= axis * toward.dot(axis)
            if toward.length < 1e-6:
                raise RuntimeError(f"bend-plane direction is parallel to {'-'.join(names)}")
            toward.normalize()
            signed_offset = float(bend.dot(toward))
            off_axis = float((bend - toward * signed_offset).length)
            offset_key = f"bend_offset_m:{'-'.join(names)}"
            off_axis_key = f"bend_off_axis_m:{'-'.join(names)}"
            metrics[offset_key] = signed_offset
            metrics[off_axis_key] = off_axis
            if signed_offset < float(bend_plane["min_offset_m"]):
                failures.append(
                    f"{offset_key}={signed_offset:.4f} below {bend_plane['min_offset_m']}"
                )
            if off_axis > float(bend_plane["max_off_axis_m"]):
                failures.append(
                    f"{off_axis_key}={off_axis:.4f} above {bend_plane['max_off_axis_m']}"
                )
    if failures:
        raise RuntimeError(f"pose anatomy gate failed ({'; '.join(failures)})")
    return metrics


def validate_lbs(rest: np.ndarray, posed: np.ndarray, indices: np.ndarray, weights: np.ndarray, matrices: np.ndarray) -> float:
    points = rest.reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    predicted = np.zeros((len(points), 3), dtype=np.float32)
    for column in range(4):
        transformed = np.einsum("nij,nj->ni", matrices[indices[:, column]], homogeneous)[:, :3]
        predicted += weights[:, column, None] * transformed
    return float(np.max(np.linalg.norm(predicted - posed.reshape(-1, 3), axis=1)))


def pose_quality(rest: np.ndarray, posed: np.ndarray, rotations: dict[str, list[float]], gates: dict) -> dict[str, float]:
    """Reject numerically extreme deformation that can still pass LBS parity.

    The old raised-arm and seated poses matched Blender perfectly while visibly
    twisting limbs. These topology-relative metrics gate shape quality rather
    than merely proving that two runtimes reproduce the same bad pose.
    """
    rest_edges = np.stack([
        np.linalg.norm(rest[:, 1] - rest[:, 0], axis=1),
        np.linalg.norm(rest[:, 2] - rest[:, 1], axis=1),
        np.linalg.norm(rest[:, 0] - rest[:, 2], axis=1),
    ], axis=1)
    posed_edges = np.stack([
        np.linalg.norm(posed[:, 1] - posed[:, 0], axis=1),
        np.linalg.norm(posed[:, 2] - posed[:, 1], axis=1),
        np.linalg.norm(posed[:, 0] - posed[:, 2], axis=1),
    ], axis=1)
    valid_edges = rest_edges > 1e-6
    edge_ratios = posed_edges[valid_edges] / rest_edges[valid_edges]
    rest_area = np.linalg.norm(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1) / 2
    posed_area = np.linalg.norm(np.cross(posed[:, 1] - posed[:, 0], posed[:, 2] - posed[:, 0]), axis=1) / 2
    valid_areas = rest_area > 1e-10
    area_ratios = posed_area[valid_areas] / rest_area[valid_areas]
    joint_angles = []
    for quaternion in rotations.values():
        q = np.asarray(quaternion, dtype=np.float64)
        joint_angles.append(float(np.degrees(2 * np.arccos(np.clip(abs(q[3]) / np.linalg.norm(q), -1, 1)))))
    metrics = {
        "max_joint_rotation_deg": max(joint_angles, default=0.0),
        "edge_length_ratio_p001": float(np.quantile(edge_ratios, 0.001)),
        "edge_length_ratio_p99": float(np.quantile(edge_ratios, 0.99)),
        "triangle_area_ratio_p01": float(np.quantile(area_ratios, 0.01)),
        "triangle_area_ratio_p99": float(np.quantile(area_ratios, 0.99)),
    }
    failures = []
    if metrics["max_joint_rotation_deg"] > gates["max_joint_rotation_deg"]:
        failures.append("joint rotation")
    if metrics["edge_length_ratio_p001"] < gates["edge_length_ratio_p001_min"]:
        failures.append("edge collapse")
    if metrics["edge_length_ratio_p99"] > gates["edge_length_ratio_p99_max"]:
        failures.append("edge stretch")
    if metrics["triangle_area_ratio_p01"] < gates["triangle_area_ratio_p01_min"]:
        failures.append("triangle collapse")
    if metrics["triangle_area_ratio_p99"] > gates["triangle_area_ratio_p99_max"]:
        failures.append("triangle stretch")
    if failures:
        detail = ", ".join(f"{key}={value:.3f}" for key, value in metrics.items())
        raise RuntimeError(f"pose deformation gate failed ({', '.join(failures)}): {detail}")
    return metrics


def paint(mesh: bpy.types.Mesh, color: tuple[float, float, float, float]) -> None:
    attribute = mesh.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    mesh.color_attributes.active_color = attribute
    for item in attribute.data:
        item.color = color


def export_meshes(surface: CanonicalSurface, indices: np.ndarray, weights: np.ndarray, bone_names: list[str], armature: bpy.types.Object) -> list[bpy.types.Object]:
    exports = []
    corner_offset = 0
    for part in surface.parts:
        vertices = surface.vertices[part.first_face:part.first_face + part.face_count].reshape(-1, 3)
        mesh = bpy.data.meshes.new(f"{part.name}-rigged")
        mesh.from_pydata(vertices.tolist(), [], [(i, i + 1, i + 2) for i in range(0, len(vertices), 3)])
        mesh.update()
        paint(mesh, SKIN if part.name == "Body" else EYE)
        obj = bpy.data.objects.new(part.name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = armature
        for bone_name in bone_names:
            obj.vertex_groups.new(name=bone_name)
        part_indices = indices[corner_offset:corner_offset + len(vertices)]
        part_weights = weights[corner_offset:corner_offset + len(vertices)]
        for vertex in range(len(vertices)):
            for column in range(4):
                weight = float(part_weights[vertex, column])
                if weight > 0:
                    obj.vertex_groups[bone_names[int(part_indices[vertex, column])]].add([vertex], weight, "REPLACE")
        modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
        modifier.object = armature
        exports.append(obj)
        corner_offset += len(vertices)
    return exports


def export_glb(path: Path, armature: bpy.types.Object, meshes: list[bpy.types.Object]) -> None:
    reset_pose(armature)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=False, export_animations=False, export_skins=True,
        export_morph=False, export_materials="NONE", export_colors=True,
        export_normals=True, export_texcoords=False, export_yup=True,
        export_extras=False, export_cameras=False, export_lights=False,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate_surface(vertices: np.ndarray, xyzw: list[float]) -> np.ndarray:
    quaternion = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
    matrix = np.asarray(quaternion.to_matrix(), dtype=np.float32)
    return np.einsum("ij,...j->...i", matrix, vertices)


def save_npz_deterministic(path: Path, **arrays) -> None:
    """Write an ordinary NPZ with fixed ordering, metadata and timestamps."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)


def build_body(repo: Path, config: dict, body_id: str) -> dict:
    source = repo / "web" / "inkmap" / "public" / "bodies" / f"{body_id}.glb"
    rigged = source.with_name(f"{body_id}.rigged.glb")
    sidecar = source.with_name(f"{body_id}.rig.npz")
    surface = load_canonical_surface(source)
    clear_scene()
    parts = import_parts(source)
    joints = config["bodies"][body_id]["joints"]
    armature = create_armature(config, joints)
    fallback = parent_auto_weights(parts["Body"], armature)
    bind_eyes(parts, armature)
    mappings = canonical_vertex_maps(surface, parts)
    bone_names = [bone.name for bone in armature.data.bones if bone.use_deform]
    indices, weights = canonical_weights(surface, parts, mappings, bone_names)
    pose_records = {}
    pose_vertices_out = []
    pose_matrices_out = []
    errors = {}
    gates = config["quality_gates"]
    validation_indices = np.linspace(0, surface.vertices.reshape(-1, 3).shape[0] - 1, 64, dtype=np.int32)
    for pose_id, pose in config["poses"].items():
        rotations = solve_pose(armature, pose, joints)
        vertices = posed_vertices(surface, parts, mappings)
        matrices = pose_matrices(armature, bone_names)
        error = validate_lbs(surface.vertices, vertices, indices, weights, matrices)
        if error > 1e-4:
            raise RuntimeError(f"{body_id}/{pose_id}: Python LBS differs from Blender by {error * 1000:.3f} mm")
        errors[pose_id] = error
        try:
            quality = pose_quality(surface.vertices, vertices, rotations, gates)
        except RuntimeError as exc:
            raise RuntimeError(f"{body_id}/{pose_id}: {exc}") from exc
        try:
            anatomy = validate_anatomy(
                config,
                pose,
                posed_joint_positions(config, joints, armature, pose["body_rotation_xyzw"]),
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{body_id}/{pose_id}: {exc}") from exc
        pose_vertices_out.append(vertices)
        pose_matrices_out.append(matrices)
        world_vertices = rotate_surface(vertices, pose["body_rotation_xyzw"])
        pose_records[pose_id] = {
            "label": pose["label"],
            "support_id": pose["support_id"],
            "body_rotation_xyzw": pose["body_rotation_xyzw"],
            "joint_rotations": rotations,
            "quality": quality,
            "anatomy": anatomy,
            "validation_vertices": world_vertices.reshape(-1, 3)[validation_indices].tolist(),
        }
    save_npz_deterministic(
        sidecar,
        rig_id=np.asarray(config["rig_id"]),
        surface_sha256=np.asarray(surface.sha256),
        bone_names=np.asarray(bone_names),
        part_names=np.asarray([part.name for part in surface.parts]),
        part_first_face=np.asarray([part.first_face for part in surface.parts], dtype=np.int32),
        part_face_count=np.asarray([part.face_count for part in surface.parts], dtype=np.int32),
        rest_vertices=surface.vertices.astype(np.float32),
        joint_indices=indices,
        joint_weights=weights,
        pose_ids=np.asarray(list(config["poses"])),
        pose_vertices=np.stack(pose_vertices_out).astype(np.float32),
        pose_matrices=np.stack(pose_matrices_out).astype(np.float32),
    )
    exports = export_meshes(surface, indices, weights, bone_names, armature)
    for part in parts.values():
        bpy.data.objects.remove(part, do_unlink=True)
    # Blender gave the export objects numeric suffixes while the source parts
    # still occupied these names. Restore the canonical node names after the
    # source objects are gone so both runtimes can use the same part contract.
    for obj, part in zip(exports, surface.parts, strict=True):
        obj.name = part.name
    export_glb(rigged, armature, exports)
    exported_surface = load_canonical_surface(rigged)
    if exported_surface.sha256 != surface.sha256:
        raise RuntimeError(
            f"{body_id}: rigged export changed canonical surface "
            f"{surface.sha256[:12]} -> {exported_surface.sha256[:12]}"
        )
    return {
        "body_id": body_id,
        "source_path": f"bodies/{source.name}",
        "source_asset_sha256": file_sha256(source),
        "surface_sha256": surface.sha256,
        "rigged_path": f"bodies/{rigged.name}",
        "rigged_asset_sha256": file_sha256(rigged),
        "sidecar_path": f"bodies/{sidecar.name}",
        "sidecar_sha256": file_sha256(sidecar),
        "automatic_weight_fallback_vertices": fallback,
        "max_lbs_error_m": max(errors.values()),
        "validation_vertex_indices": validation_indices.tolist(),
        "poses": pose_records,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    config_path = repo / "config" / "inkmap" / "body-rig.json"
    config = json.loads(config_path.read_text())
    selected = args.bodies or list(config["bodies"])
    unknown = set(selected) - set(config["bodies"])
    if unknown:
        raise SystemExit(f"unknown bodies: {sorted(unknown)}")
    records = {body_id: build_body(repo, config, body_id) for body_id in selected}
    catalog_path = repo / "config" / "inkmap" / "body-poses.json"
    existing = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    bodies = dict(existing.get("bodies", {}))
    bodies.update(records)
    catalog = {
        "schema_version": 1,
        "rig_id": config["rig_id"],
        "frame": config["frame"],
        "bones": [bone["name"] for bone in config["bones"]],
        "pose_ids": list(config["poses"]),
        "bodies": dict(sorted(bodies.items())),
    }
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(f"wrote {catalog_path}")
    for body_id, record in records.items():
        print(
            f"{body_id}: surface {record['surface_sha256'][:12]} rigged "
            f"{record['rigged_asset_sha256'][:12]} sidecar {record['sidecar_sha256'][:12]} "
            f"fallback={record['automatic_weight_fallback_vertices']} "
            f"lbs={record['max_lbs_error_m'] * 1000:.4f} mm"
        )
    print("INKMAP_RIG_OK")


if __name__ == "__main__":
    main()
