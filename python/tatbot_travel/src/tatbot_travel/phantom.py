"""The practice arm, cut from the rig's body model.

The demo's phantom is a silicone forearm-and-hand; the rig's practice arm
ends at the wrist in a rounded stump. Their stand-in is the left arm of the
MHR/SOMA reference body (``web/inkmap/public/bodies``): everything beyond a
cut through the upper arm, capped where it was cut (the real phantom shows
foam there), optionally cut again at the wrist and domed, and expressed in
its own frame -- +x along the arm toward the fingertips, +z out of the back
of the forearm, origin on the arm's axis at the middle of the forearm.

A cylindrical texture map around the arm's centreline gives every vertex a
(u along, v around) coordinate, so drawn lines can be painted into a texture
and recovered on the surface as 3-D polylines for the expert to follow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tatbot_sim.inkmap.rig import PosedBody, load_body_rig

from tatbot_travel import assets

LEFT = 1  # regions.json laterality code
BODY_UP = np.array([0.0, 0.0, 1.0])  # the reference body's +z (regions.json "frame")


@dataclass(frozen=True)
class Phantom:
    """A closed arm mesh in the phantom frame, with its texture map and centreline."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) skin
    cap_faces: np.ndarray  # (C, 3) the cut face
    uv: np.ndarray  # (V, 2) u along the arm, v around it (0.5 = back of the forearm)
    x_range: tuple[float, float]  # extent along the arm
    forearm: tuple[float, float]  # wrist .. elbow stations, where lines are drawn
    centre_x: np.ndarray  # centreline stations
    centre_yz: np.ndarray  # (K, 2) centreline offsets at those stations
    radius: np.ndarray  # (K,) mean radius at those stations
    body: PosedBody  # shared SOMA surface, transformed into this phantom's frame
    face_indices: np.ndarray  # canonical SOMA face IDs, in skin face order
    ink_face_indices: np.ndarray  # forearm patch selected in canonical rest coordinates
    provenance: dict

    def centre_at(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.interp(x, self.centre_x, self.centre_yz[:, 0])
        z = np.interp(x, self.centre_x, self.centre_yz[:, 1])
        return np.stack([x, y, z], axis=-1)

    def radius_at(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.centre_x, self.radius)

    def skin_mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(self.vertices, np.vstack([self.faces, self.cap_faces]), process=False)

    def scaled(self, along: float, across: float) -> Phantom:
        """A longer/shorter, thicker/thinner phantom (per-episode shape spread)."""
        s = np.array([along, across, across])
        return Phantom(
            vertices=self.vertices * s, faces=self.faces, cap_faces=self.cap_faces, uv=self.uv,
            x_range=(self.x_range[0] * along, self.x_range[1] * along),
            forearm=(self.forearm[0] * along, self.forearm[1] * along),
            centre_x=self.centre_x * along, centre_yz=self.centre_yz * across, radius=self.radius * across,
            body=replace(self.body, vertices=self.body.vertices * s), face_indices=self.face_indices,
            ink_face_indices=self.ink_face_indices,
            provenance={**self.provenance, "scale_xyz": s.tolist()},
        )


def _load_body() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rig = load_body_rig()
    regions = json.loads(assets.body_regions().read_text())
    codes = np.asarray(regions["faces"])
    if len(codes) != len(rig.faces):
        raise ValueError("body atlas and mesh disagree on face count")
    return rig.indexed_vertices(), rig.faces, codes, regions["sites"]


def _site_range(vertices, faces, codes, sites, names) -> tuple[float, float]:
    ids = [sites.index(n) * 4 + LEFT for n in names]
    x = vertices[faces[np.isin(codes, ids)]][..., 0]
    return float(x.min()), float(x.max())


def _cut_arm(vertices, faces, x_cut: float, x_end: float | None = None):
    """Select in the rest frame, retaining the canonical vertex and face IDs."""
    keep = vertices[faces][:, :, 0].min(axis=1) > x_cut
    if x_end is not None:
        keep &= vertices[faces][:, :, 0].max(axis=1) < x_end
    face_ids = np.flatnonzero(keep)
    arm = trimesh.Trimesh(vertices, faces[face_ids], process=False)
    # Keep the largest face-connected piece (stray torso slivers past the cut).
    adjacency = arm.face_adjacency
    graph = coo_matrix((np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
                       shape=(len(arm.faces), len(arm.faces)))
    _, labels = connected_components(graph, directed=False)
    largest = np.bincount(labels).argmax()
    face_ids = face_ids[labels == largest]
    vertex_ids, inverse = np.unique(faces[face_ids], return_inverse=True)
    return vertex_ids, inverse.reshape(-1, 3), face_ids


def _cap(mesh: trimesh.Trimesh, dome_far_end: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Close every boundary loop with a fan about its centroid; returns (vertices, cap faces).

    ``dome_far_end`` lifts the fan's centre on the loop furthest along +x out of the arm, rounding that end.
    """
    vertices = np.asarray(mesh.vertices)
    unique, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return vertices, np.zeros((0, 3), dtype=int)
    n_comp, labels = connected_components(trimesh.graph.edges_to_coo(boundary), directed=False)
    used = {(int(a), int(b)) for tri in mesh.faces for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    centres, faces = [], []
    for comp in range(n_comp):
        loop = boundary[labels[boundary[:, 0]] == comp]
        if len(loop) < 3:
            continue
        centre = len(vertices) + len(centres)
        ring = vertices[np.unique(loop)]
        centres.append(ring.mean(axis=0))
        if dome_far_end and ring[:, 0].mean() >= vertices[:, 0].max() - 0.03:
            centres[-1] = centres[-1] + np.array([0.7 * float(np.linalg.norm(ring - ring.mean(0), axis=1).mean()),
                                                  0.0, 0.0])
        # Wind each cap triangle against the skin's use of the shared edge.
        faces += [(b, a, centre) if (int(a), int(b)) in used else (a, b, centre) for a, b in loop]
    return np.vstack([vertices, *centres]) if centres else vertices, np.asarray(faces, dtype=int)


def _frame(vertices: np.ndarray, forearm_mask: np.ndarray, forearm_x: tuple[float, float]) -> np.ndarray:
    """4x4 body_from_phantom: x along the forearm toward the hand, z the back of the forearm."""
    pts = vertices[forearm_mask]
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    x_axis = vt[0] if vt[0][0] > 0 else -vt[0]
    z_axis = BODY_UP - x_axis * (BODY_UP @ x_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    origin_x = 0.5 * (forearm_x[0] + forearm_x[1])
    origin = centre + x_axis * ((origin_x - centre[0]) / max(x_axis[0], 1e-6))
    transform = np.eye(4)
    transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform[:3, 3] = origin
    return transform


def _circle_centre(yz: np.ndarray) -> np.ndarray:
    """Least-squares (Kasa) circle centre of a cross-section's points."""
    a = np.column_stack([2 * yz, np.ones(len(yz))])
    b = (yz ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return sol[:2]


def _centreline(vertices: np.ndarray, n: int = 64, half_width: float = 0.02,
                smooth_stations: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-section centres along x: circle fits over wide slabs, then smoothed.

    The body mesh is coarse, so a slab's vertex mean is biased by how its few
    vertices happen to fall around the arm; a noisy centreline folds the
    cylindrical texture map. Circle fits over 4 cm slabs, Gaussian-smoothed
    along the arm, keep the chart one-to-one on the forearm.
    """
    xs = np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), n)
    centres = []
    for x in xs:
        slab = vertices[np.abs(vertices[:, 0] - x) <= half_width][:, 1:]
        centres.append(_circle_centre(slab) if len(slab) >= 8 else slab.mean(axis=0))
    centres = np.asarray(centres)
    k = np.arange(-3 * int(smooth_stations), 3 * int(smooth_stations) + 1)
    kernel = np.exp(-0.5 * (k / smooth_stations) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(centres, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(2)], axis=1)
    radius = np.array([np.median(np.linalg.norm(vertices[np.abs(vertices[:, 0] - x) <= half_width][:, 1:] - c,
                                                axis=1)) for x, c in zip(xs, smooth, strict=True)])
    return xs, smooth, radius


def _cylindrical_uv(vertices, x_range, centre_x, centre_yz) -> np.ndarray:
    u = (vertices[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    cy = np.interp(vertices[:, 0], centre_x, centre_yz[:, 0])
    cz = np.interp(vertices[:, 0], centre_x, centre_yz[:, 1])
    theta = np.arctan2(vertices[:, 1] - cy, vertices[:, 2] - cz)  # 0 on the back, +-pi underneath
    v = 0.5 + theta / (2.0 * np.pi)
    return np.stack([u, v], axis=1)


def _split_seam(vertices, faces, uv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices of faces that wrap across v = 0/1 so the map stays continuous."""
    vertices, uv, faces = list(vertices), list(uv), faces.copy()
    for f, tri in enumerate(faces):
        vs = np.array([uv[i][1] for i in tri])
        if vs.max() - vs.min() <= 0.5:
            continue
        for k, i in enumerate(tri):
            if uv[i][1] < 0.5:
                vertices.append(vertices[i])
                uv.append((uv[i][0], uv[i][1] + 1.0))
                faces[f, k] = len(vertices) - 1
    return np.asarray(vertices), faces, np.asarray(uv)


def _site_mask(points: np.ndarray, x_range: tuple[float, float]) -> np.ndarray:
    return (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])


@lru_cache(maxsize=8)
def build_phantom(length_m: float = 0.63, handless: bool = False, hand_pose: str = "open") -> Phantom:
    """The reference left arm, fingertips back ``length_m`` along the body's +x; ``handless`` ends it at the
    wrist in a rounded stump. A closed hand uses the shared, articulated SOMA pose cache."""
    if hand_pose not in ("open", "closed"):
        raise ValueError("hand_pose must be 'open' or 'closed'")
    rig = load_body_rig()
    vertices, faces, codes, sites = _load_body()
    pose_id = "left-fist-reference" if hand_pose == "closed" and not handless else None
    posed_vertices = rig.indexed_vertices(pose_id)
    tip = float(vertices[faces[codes % 4 == LEFT]][..., 0].max())
    forearm_x = _site_range(vertices, faces, codes, sites, ["forearm"])
    wrist_x = _site_range(vertices, faces, codes, sites, ["wrist"])
    hand_x = _site_range(vertices, faces, codes, sites, ["hand", "palm", "fingers", "thumb"])
    vertex_ids, skin_faces, face_ids = _cut_arm(vertices, faces, tip - length_m,
                                              x_end=hand_x[0] if handless else None)
    av = vertices[vertex_ids]
    body_from_phantom = _frame(av, _site_mask(av, forearm_x), forearm_x)
    to_phantom = np.linalg.inv(body_from_phantom)
    triangles = trimesh.transform_points(posed_vertices, to_phantom)[faces]
    body = PosedBody(body_id=rig.body_id, pose_id=pose_id or "rest",
                     surface_sha256=rig.surface_sha256, vertices=triangles,
                     body_from_rest=to_phantom, part_names=rig.part_names,
                     part_first_face=rig.part_first_face, part_face_count=rig.part_face_count)
    local = trimesh.transform_points(posed_vertices[vertex_ids], to_phantom)
    line_zone = local[_site_mask(av, (forearm_x[0], min(forearm_x[1], wrist_x[0])))]
    rest_triangles = trimesh.transform_points(vertices, to_phantom)[faces[face_ids]]
    ink_faces = face_ids[(rest_triangles[..., 0].max(axis=1) > line_zone[:, 0].min())
                        & (rest_triangles[..., 0].min(axis=1) < line_zone[:, 0].max())]
    centre_x, centre_yz, radius = _centreline(local[_site_mask(av, forearm_x)])
    capped, cap_faces = _cap(trimesh.Trimesh(local, skin_faces, process=False), dome_far_end=handless)
    x_range = (float(capped[:, 0].min()), float(capped[:, 0].max()))
    uv = _cylindrical_uv(capped, x_range, centre_x, centre_yz)
    skin_vertices, skin_faces, skin_uv = _split_seam(capped, skin_faces, uv)
    provenance = {"model_id": rig.body_id, "model_spec_sha256": rig.model_spec_sha256,
                  "identity_sha256": rig.identity_sha256, "topology_sha256": rig.topology_sha256,
                  "rest_surface_sha256": rig.surface_sha256, "pose_id": body.pose_id,
                  "pose_catalog_sha256": rig.catalog_sha256,
                  "posed_surface_sha256": (rig.catalog_record["poses"][pose_id]["surface_sha256"]
                                            if pose_id else rig.surface_sha256),
                  "phantom_from_body": to_phantom.tolist(), "scale_xyz": [1.0, 1.0, 1.0]}
    return Phantom(
        vertices=skin_vertices, faces=skin_faces, cap_faces=cap_faces, uv=skin_uv, x_range=x_range,
        forearm=(float(line_zone[:, 0].min()), float(line_zone[:, 0].max())),
        centre_x=centre_x, centre_yz=centre_yz, radius=radius,
        body=body, face_indices=face_ids, ink_face_indices=ink_faces, provenance=provenance,
    )


def obj_text(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None = None) -> str:
    """OBJ with smooth vertex normals (and texture coordinates when given)."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    normals = mesh.vertex_normals
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in normals]
    if uv is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
        lines += ["f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in tri) for tri in faces]
    else:
        lines += ["f " + " ".join(f"{i + 1}//{i + 1}" for i in tri) for tri in faces]
    return "\n".join(lines) + "\n"
