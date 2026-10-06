"""Hands that hold the phantom, and bystanders in the background, from the same body model.

Visitors' hands are in the wrist camera's view whenever the phantom is being
moved, and people stand behind the table all day. Both come from the rig's
MHR/SOMA body: a right hand with part of its forearm (the part past the
wrist is a sleeve), and the whole body in its standing pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh
from tatbot_sim.inkmap.rig import load_body_rig

from tatbot_travel.phantom import _load_body

RIGHT = 2  # regions.json laterality code


@dataclass(frozen=True)
class HandMesh:
    vertices: np.ndarray  # hand frame: +x toward the fingertips, +z out of the back of the hand
    skin_faces: np.ndarray
    sleeve_faces: np.ndarray


def _weld(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    return mesh


@lru_cache(maxsize=1)
def build_hand(length_m: float = 0.30, sleeve_from_m: float = 0.11) -> HandMesh:
    """The reference right hand and forearm, ``length_m`` back from the fingertips."""
    vertices, faces, codes, sites = _load_body()
    tip = float(vertices[faces[codes % 4 == RIGHT]][..., 0].min())
    keep = vertices[faces][:, :, 0].max(axis=1) < tip + length_m
    mesh = _weld(vertices, faces[keep])
    palm_id = sites.index("palm") * 4 + RIGHT
    palm = vertices[faces[codes == palm_id]].reshape(-1, 3)
    origin = palm.mean(axis=0)
    x_axis = np.array([-1.0, 0.0, 0.0])  # the right arm points along the body's -x in the rest pose
    palm_normal = trimesh.Trimesh(vertices, faces[codes == palm_id], process=False).face_normals.mean(axis=0)
    z_axis = -(palm_normal - x_axis * (palm_normal @ x_axis))
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    local = (np.asarray(mesh.vertices) - origin) @ np.stack([x_axis, y_axis, z_axis], axis=1)
    centroid_x = local[np.asarray(mesh.faces)][:, :, 0].mean(axis=1)
    sleeve = centroid_x < -sleeve_from_m
    f = np.asarray(mesh.faces)
    return HandMesh(vertices=local, skin_faces=f[~sleeve], sleeve_faces=f[sleeve])


@lru_cache(maxsize=1)
def standing_body() -> tuple[np.ndarray, np.ndarray]:
    """The whole reference body in its standing pose, feet at z = 0, facing +y."""
    rig = load_body_rig()
    vertices = rig.posed("standing-neutral").vertices.reshape(-1, 3)
    mesh = _weld(vertices, np.arange(len(vertices)).reshape(-1, 3))
    v = np.asarray(mesh.vertices)
    v[:, 2] -= v[:, 2].min()
    return v, np.asarray(mesh.faces)
