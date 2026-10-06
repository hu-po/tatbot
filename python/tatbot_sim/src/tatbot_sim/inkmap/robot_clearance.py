"""CPU clearance audit using collision geometry from the generated Tatbot URDF."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

from tatbot_sim.inkmap.scenario_scene import _collision_capsules, _scenario_support_boxes
from tatbot_sim.resolved import resolve
from tatbot_sim.urdf import build_tatbot_urdf

SURFACE_SAMPLE_VOXEL_M = 0.002
SURFACE_SAMPLE_ERROR_M = float(np.sqrt(3.0) * SURFACE_SAMPLE_VOXEL_M)


@dataclass(frozen=True)
class LocalCollisionGeometry:
    link: str
    center: np.ndarray
    half_size: np.ndarray
    surface_points: np.ndarray


def _numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()] if value else default, dtype=np.float64)


def _stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + 50 * count == len(raw):
            record = np.dtype([
                ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
            ])
            return np.frombuffer(raw, dtype=record, offset=84, count=count)["vertices"].reshape(-1, 3)
    vertices = []
    for line in raw.decode(errors="ignore").splitlines():
        words = line.split()
        if len(words) == 4 and words[0] == "vertex":
            vertices.append([float(value) for value in words[1:]])
    if not vertices:
        raise ValueError(f"cannot read collision mesh {path}")
    return np.asarray(vertices)


def _shape_points(geometry: ET.Element, root: Path) -> np.ndarray:
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.attrib["filename"]
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        return _stl_vertices(path) * _numbers(mesh.get("scale"), (1.0, 1.0, 1.0))
    box = geometry.find("box")
    if box is not None:
        half = _numbers(box.get("size"), (0.0, 0.0, 0.0)) / 2.0
    else:
        sphere = geometry.find("sphere")
        if sphere is not None:
            half = np.full(3, float(sphere.attrib["radius"]))
        else:
            cylinder = geometry.find("cylinder")
            if cylinder is None:
                raise ValueError("unsupported URDF collision geometry")
            radius = float(cylinder.attrib["radius"])
            half = np.asarray([radius, radius, float(cylinder.attrib["length"]) / 2.0])
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ])
    return signs * half


def _surface_samples(points: np.ndarray) -> np.ndarray:
    """Densify STL triangles enough for a stable mesh/proxy clearance audit."""
    if len(points) % 3:
        return points
    triangles = points.reshape(-1, 3, 3)
    samples = np.concatenate([
        triangles,
        triangles.mean(axis=1, keepdims=True),
        (triangles[:, 0:1] + triangles[:, 1:2]) / 2,
        (triangles[:, 1:2] + triangles[:, 2:3]) / 2,
        (triangles[:, 2:3] + triangles[:, 0:1]) / 2,
    ], axis=1)
    samples = np.unique(samples.reshape(-1, 3), axis=0)
    # Keep one deterministic representative per 2 mm cell. Distances below
    # are reduced by the cell diagonal, retaining a conservative lower bound.
    cells = np.floor(samples / SURFACE_SAMPLE_VOXEL_M).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return samples[np.sort(keep)]


def collision_geometries(urdf_path: str | None = None):
    # Resolve before caching: None must not cache a previously selected tool.
    return _collision_geometries(urdf_path or build_tatbot_urdf())


@lru_cache(maxsize=4)
def _collision_geometries(urdf_path: str) -> tuple[LocalCollisionGeometry, ...]:
    """Collision surfaces and broad-phase bounds from every URDF collision."""
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    boxes = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            points = _shape_points(geometry, path.parent)
            if geometry.find("mesh") is not None:
                points = _surface_samples(points)
            origin = collision.find("origin")
            xyz = _numbers(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _numbers(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            points = points @ euler2mat(*rpy, axes="sxyz").T + xyz
            low, high = points.min(axis=0), points.max(axis=0)
            boxes.append(LocalCollisionGeometry(
                link.attrib["name"],
                (low + high) / 2.0,
                (high - low) / 2.0,
                points,
            ))
    if not boxes:
        raise ValueError(f"{path}: no collision geometry")
    return tuple(boxes)


@lru_cache(maxsize=2)
def _full_chain(urdf_path: str):
    return pk.build_chain_from_urdf(Path(urdf_path).read_bytes()).to(
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _full_q(q_right: np.ndarray, chain, config) -> torch.Tensor:
    names = chain.get_joint_parameter_names()
    staged = config.staged_pose
    values = np.zeros((len(q_right), len(names)), dtype=np.float32)
    for column, name in enumerate(names):
        if name.startswith("joint_") and name[-1].isdigit():
            values[:, column] = q_right[:, int(name[-1])]
        elif name.endswith("carriage_joint"):
            values[:, column] = config.carriage_rest_m
        elif name.startswith("left/joint_") and name[-1].isdigit():
            values[:, column] = staged[int(name[-1])]
    return torch.as_tensor(values)


def _world_points_box_signed(
    points: np.ndarray, center: np.ndarray, rotation: np.ndarray, half: np.ndarray,
):
    local = np.einsum("tpi,ij->tpj", points - center, rotation)
    delta = np.abs(local) - half
    return np.linalg.norm(np.maximum(delta, 0.0), axis=-1) + np.minimum(delta.max(axis=-1), 0.0)


def _point_capsule_signed(points: np.ndarray, start: np.ndarray, end: np.ndarray, radius: float):
    axis = end - start
    fraction = np.clip(((points - start) @ axis) / max(axis @ axis, 1e-12), 0.0, 1.0)
    closest = start + fraction[..., None] * axis
    return np.linalg.norm(points - closest, axis=-1) - radius


def _segment_distance(a0, a1, b0, b1) -> float:
    """Minimum distance between two finite 3-D line segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = u @ u, u @ v, v @ v
    dd, ee = u @ w, v @ w
    denom = aa * cc - bb * bb
    s = 0.0 if denom < 1e-12 else np.clip((bb * ee - cc * dd) / denom, 0.0, 1.0)
    t = np.clip((bb * s + ee) / max(cc, 1e-12), 0.0, 1.0)
    s = np.clip((bb * t - dd) / max(aa, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(w + s * u - t * v))


def _transforms(q_right: np.ndarray, config):
    q_right = np.asarray(q_right, dtype=np.float32)
    chain = _full_chain(build_tatbot_urdf(config=config))
    return chain.forward_kinematics(_full_q(q_right, chain, config))


def non_tool_clearance(q_right: np.ndarray, scenario: dict, *, config=None) -> dict[str, float | str]:
    """Minimum sampled URDF collision-surface clearance to body/support proxies."""
    config = config or resolve(tool_id=scenario["robot"]["tool_id"])
    transforms = _transforms(q_right, config)
    body = _collision_capsules(scenario)
    supports = _scenario_support_boxes(scenario)
    minimum_robot = float("inf")
    minimum_robot_pair = "none"
    for geometry in collision_geometries(build_tatbot_urdf(config=config)):
        if geometry.link in ("tattoo_needle", "tattoo_pen") or geometry.link not in transforms:
            continue
        matrix = transforms[geometry.link].get_matrix().detach().numpy()
        centers = matrix[:, :3, :3] @ geometry.center[:, None]
        centers = centers[:, :, 0] + matrix[:, :3, 3]
        rotations = matrix[:, :3, :3]
        surface_points = np.einsum(
            "tij,pj->tpi", rotations, geometry.surface_points,
        ) + matrix[:, None, :3, 3]
        for capsule in body:
            clearance = float(_point_capsule_signed(
                surface_points, capsule.start, capsule.end, capsule.radius,
            ).min()) - SURFACE_SAMPLE_ERROR_M
            if clearance < minimum_robot:
                minimum_robot = clearance
                minimum_robot_pair = f"{geometry.link}:body/{capsule.name}"
        for support in supports:
            support_rotation = (
                np.eye(3) if support.quaternion_wxyz is None else quat2mat(support.quaternion_wxyz)
            )
            signed = _world_points_box_signed(
                surface_points, support.center, support_rotation, support.half_size,
            )
            clearance = float(signed.min()) - SURFACE_SAMPLE_ERROR_M
            if clearance < minimum_robot:
                minimum_robot = clearance
                minimum_robot_pair = f"{geometry.link}:support/{support.name}"

    return {"non_tool_robot_m": float(minimum_robot), "non_tool_pair": minimum_robot_pair}


def tool_terminal_exclusion_m(tool) -> float:
    """The distal span the shaft check leaves out: the tool's working point back
    to where its datasheet profile's final taper leaves the body. That tapered
    terminal assembly is what is meant to meet the skin."""
    return float(tool.protrusion_m - tool.profile[-2][0])


def tool_shaft_clearance(q_right: np.ndarray, scenario: dict, *, config=None) -> dict[str, float | str]:
    """Minimum shaft clearance outside the configured intentional terminal patch."""
    config = config or resolve(tool_id=scenario["robot"]["tool_id"])
    transforms = _transforms(q_right, config)
    body = _collision_capsules(scenario)
    mount = transforms["tool_mount"].get_matrix().detach().numpy()[:, :3, 3]
    tip = transforms["tattoo_needle"].get_matrix().detach().numpy()[:, :3, 3]
    tool_radius = config.tool.body_radius_m
    exclusion = tool_terminal_exclusion_m(config.tool)
    minimum_tool = float("inf")
    minimum_tool_pair = "none"
    for start, contact in zip(mount, tip, strict=True):
        shaft = contact - start
        shaft_length = np.linalg.norm(shaft)
        checked_end = contact - shaft / max(shaft_length, 1e-12) * min(exclusion, shaft_length)
        contact_distances = [
            _segment_distance(contact, contact, capsule.start, capsule.end)
            for capsule in body
        ]
        contacted = int(np.argmin(contact_distances))
        for capsule in body:
            # The nearest capsule is the intentional terminal contact. It is
            # convex, so an outward-facing straight shaft cannot re-enter it;
            # its coarse radius must not turn intended contact into collision.
            if capsule is body[contacted]:
                continue
            clearance = (
                _segment_distance(start, checked_end, capsule.start, capsule.end)
                - tool_radius
                - capsule.radius
            )
            if clearance < minimum_tool:
                minimum_tool = clearance
                minimum_tool_pair = f"tattoo_pen:body/{capsule.name}"
    return {"tool_shaft_m": float(minimum_tool), "tool_shaft_pair": minimum_tool_pair}
