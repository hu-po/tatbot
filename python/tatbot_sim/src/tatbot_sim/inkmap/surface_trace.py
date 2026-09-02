"""Unfold a local body mesh patch and compile metric strokes to stable anchors."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from tatbot_sim.inkmap.rig import BodyRig, BodyRigError

TRACE_COMPILER_VERSION = 1


class SurfaceTraceError(ValueError):
    pass


@dataclass(frozen=True)
class SurfaceAnchor:
    face: int
    barycentric: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {"face": self.face, "barycentric": list(self.barycentric)}


@dataclass(frozen=True)
class SurfaceTrace:
    strokes: tuple[tuple[SurfaceAnchor, ...], ...]
    sha256: str

    def as_dict(self) -> dict:
        return {
            "compiler": "tatbot_sim.surface_trace",
            "compiler_version": TRACE_COMPILER_VERSION,
            "sha256": self.sha256,
            "strokes": [[anchor.as_dict() for anchor in stroke] for stroke in self.strokes],
        }


def _vertex_key(point) -> tuple[int, int, int]:
    return tuple(np.floor(np.asarray(point, dtype=np.float64) * 1e6 + 0.5).astype(np.int64))


def _barycentric_2d(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, point - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    good = np.abs(denominator) > 1e-16
    out = np.full((len(triangles), 3), np.nan)
    out[good, 1] = (d11[good] * d20[good] - d01[good] * d21[good]) / denominator[good]
    out[good, 2] = (d00[good] * d21[good] - d01[good] * d20[good]) / denominator[good]
    out[good, 0] = 1 - out[good, 1] - out[good, 2]
    return out


def _resample(points: np.ndarray, max_step_m: float) -> np.ndarray:
    pieces = [points[0]]
    for start, end in zip(points[:-1], points[1:], strict=True):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / max_step_m)))
        pieces.extend(start + (end - start) * (i / count) for i in range(1, count + 1))
    return np.asarray(pieces)


@dataclass
class UnfoldedPatch:
    seed_face: int
    body_first_face: int
    seed_triangle_uv: np.ndarray
    mesh_vertices: np.ndarray
    mesh_keys: list[list[tuple[int, int, int]]]
    face_indices: np.ndarray
    triangles_uv: np.ndarray
    adjacent: dict[int, set[int]]

    def _unfold_neighbor(self, face: int, triangle_uv: np.ndarray, neighbor: int) -> np.ndarray:
        local_face = face - self.body_first_face
        local_neighbor = neighbor - self.body_first_face
        known = {self.mesh_keys[local_face][i]: triangle_uv[i] for i in range(3)}
        shared = [key for key in self.mesh_keys[local_neighbor] if key in known]
        if len(shared) != 2:
            raise SurfaceTraceError("adjacent faces do not share exactly two vertices")
        ka, kb = shared
        qa, qb = known[ka], known[kb]
        kc = next(key for key in self.mesh_keys[local_neighbor] if key not in (ka, kb))
        source = {self.mesh_keys[local_neighbor][i]: self.mesh_vertices[local_neighbor][i] for i in range(3)}
        edge = qb - qa
        edge_length = np.linalg.norm(edge)
        unit = edge / edge_length
        da, db = np.linalg.norm(source[kc] - source[ka]), np.linalg.norm(source[kc] - source[kb])
        along = (da * da - db * db + edge_length * edge_length) / (2 * edge_length)
        height = np.sqrt(max(0.0, da * da - along * along))
        perp = np.array([-unit[1], unit[0]])
        known_third = next(key for key in self.mesh_keys[local_face] if key not in (ka, kb))
        offset = known[known_third] - qa
        side = np.sign(edge[0] * offset[1] - edge[1] * offset[0]) or 1.0
        lookup = {ka: qa, kb: qb, kc: qa + along * unit - side * height * perp}
        return np.stack([lookup[key] for key in self.mesh_keys[local_neighbor]])

    def _walk(self, target: np.ndarray, face: int, triangle_uv: np.ndarray,
              point: np.ndarray, barycentric: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        for _ in range(512):
            target_bary = _barycentric_2d(target, triangle_uv[None])[0]
            if np.all(target_bary >= -2e-7):
                target_bary = np.clip(target_bary, 0.0, 1.0)
                target_bary /= target_bary.sum()
                return face, triangle_uv, target_bary
            crossings = []
            for opposite in np.flatnonzero(target_bary < -2e-7):
                denominator = barycentric[opposite] - target_bary[opposite]
                if denominator > 1e-14:
                    crossings.append((float(barycentric[opposite] / denominator), int(opposite)))
            if not crossings:
                raise SurfaceTraceError("could not identify the next mesh edge in surface walk")
            _, opposite = min(crossings)
            edge_vertices = [index for index in range(3) if index != opposite]
            local_face = face - self.body_first_face
            edge_keys = {self.mesh_keys[local_face][index] for index in edge_vertices}
            neighbor = next((candidate for candidate in self.adjacent.get(face, set())
                             if edge_keys.issubset(self.mesh_keys[candidate - self.body_first_face])), None)
            if neighbor is None:
                raise SurfaceTraceError(f"surface trace reached open mesh edge on face {face}")
            t = min(value for value, index in crossings if index == opposite)
            intersection = point + t * (target - point)
            triangle_uv = self._unfold_neighbor(face, triangle_uv, neighbor)
            face = neighbor
            barycentric = _barycentric_2d(intersection, triangle_uv[None])[0]
            barycentric = np.clip(barycentric, 0.0, 1.0)
            barycentric /= barycentric.sum()
            point = intersection
        raise SurfaceTraceError("surface walk exceeded 512 crossed faces")

    def map_samples(self, points_m: np.ndarray) -> tuple[tuple[SurfaceAnchor, np.ndarray], ...]:
        mapped = []
        face = self.seed_face
        triangle_uv = self.seed_triangle_uv.copy()
        point = np.zeros(2)
        barycentric = _barycentric_2d(point, triangle_uv[None])[0]
        for target in points_m:
            face, triangle_uv, barycentric = self._walk(
                np.asarray(target), face, triangle_uv, point, barycentric,
            )
            point = np.asarray(target)
            b0, b1, b2 = barycentric
            mapped.append((SurfaceAnchor(face, (float(b0), float(b1), float(b2))), triangle_uv.copy()))
        return tuple(mapped)

    def anchors(self, points_m: np.ndarray) -> tuple[SurfaceAnchor, ...]:
        return tuple(anchor for anchor, _ in self.map_samples(points_m))


def anchor_frame(vertices: np.ndarray, face: int, barycentric, rotation_rad: float = 0.0):
    keys = [[_vertex_key(point) for point in triangle] for triangle in vertices]
    normals: dict[tuple[int, int, int], np.ndarray] = defaultdict(lambda: np.zeros(3))
    for triangle, triangle_keys in zip(vertices, keys, strict=True):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        for key in triangle_keys:
            normals[key] += normal
    bary = np.asarray(barycentric, dtype=np.float64)
    point = bary @ vertices[face]
    normal = sum(bary[i] * normals[keys[face][i]] / max(np.linalg.norm(normals[keys[face][i]]), 1e-20) for i in range(3))
    normal /= np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    v = up - np.dot(up, normal) * normal
    if np.dot(v, v) < 1e-8:
        v = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        if np.dot(v, v) < 1e-8:
            v = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    v /= np.linalg.norm(v)
    v = (v * np.cos(rotation_rad) + np.cross(normal, v) * np.sin(rotation_rad)
         + normal * np.dot(normal, v) * (1 - np.cos(rotation_rad)))
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    return point, normal, u, v


def unfold_body_patch(
    rig: BodyRig,
    anchor: SurfaceAnchor,
    rotation_rad: float,
    radius_m: float,
) -> UnfoldedPatch:
    """Unfold connected Body faces around an anchor into an isometric chart."""
    if radius_m <= 0:
        raise SurfaceTraceError("patch radius must be positive")
    body_index = rig.part_names.index("Body")
    body_start = int(rig.part_first_face[body_index])
    body_stop = body_start + int(rig.part_face_count[body_index])
    if not body_start <= anchor.face < body_stop:
        raise SurfaceTraceError("tattoo anchor must be on the Body mesh, not an eye or prop")
    vertices = np.asarray(rig.rest_vertices[body_start:body_stop], dtype=np.float64)
    local_seed = anchor.face - body_start
    keys = [[_vertex_key(point) for point in triangle] for triangle in vertices]
    edge_faces: dict[tuple, list[int]] = defaultdict(list)
    adjacent_local: dict[int, set[int]] = defaultdict(set)
    for face, triangle_keys in enumerate(keys):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge_faces[tuple(sorted((triangle_keys[i], triangle_keys[j])))].append(face)
    for edge, faces in edge_faces.items():
        if len(faces) > 2:
            raise SurfaceTraceError(f"non-manifold canonical body edge {edge}")
        if len(faces) == 2:
            adjacent_local[faces[0]].add(faces[1])
            adjacent_local[faces[1]].add(faces[0])

    _, normal, u, v = anchor_frame(vertices, local_seed, anchor.barycentric, rotation_rad)
    triangle = vertices[local_seed]
    e01 = triangle[1] - triangle[0]
    length = np.linalg.norm(e01)
    direction = np.array([np.dot(e01, u), np.dot(e01, v)])
    direction /= np.linalg.norm(direction)
    q0 = np.zeros(2)
    q1 = length * direction
    d02, d12 = np.linalg.norm(triangle[2] - triangle[0]), np.linalg.norm(triangle[2] - triangle[1])
    x = (d02 * d02 - d12 * d12 + length * length) / (2 * length)
    h = np.sqrt(max(0.0, d02 * d02 - x * x))
    perpendicular = np.array([-direction[1], direction[0]])
    orientation = np.sign(np.dot(np.cross(e01, triangle[2] - triangle[0]), normal)) or 1.0
    q2 = x * direction + orientation * h * perpendicular
    seed_uv = np.stack([q0, q1, q2])
    seed_uv -= np.asarray(anchor.barycentric) @ seed_uv

    unfolded: dict[int, np.ndarray] = {local_seed: seed_uv}
    queue = deque([local_seed])
    max_edge = float(max(np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1).max(),
                         np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1).max(),
                         np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1).max()))
    while queue:
        face = queue.popleft()
        face_uv = unfolded[face]
        known = {keys[face][i]: face_uv[i] for i in range(3)}
        for neighbor in adjacent_local[face]:
            if neighbor in unfolded:
                continue
            shared = [key for key in keys[neighbor] if key in known]
            if len(shared) != 2:
                raise SurfaceTraceError("adjacent faces do not share exactly two vertices")
            ka, kb = shared
            qa, qb = known[ka], known[kb]
            kc = next(key for key in keys[neighbor] if key not in (ka, kb))
            source = {keys[neighbor][i]: vertices[neighbor][i] for i in range(3)}
            edge = qb - qa
            edge_length = np.linalg.norm(edge)
            unit = edge / edge_length
            da, db = np.linalg.norm(source[kc] - source[ka]), np.linalg.norm(source[kc] - source[kb])
            along = (da * da - db * db + edge_length * edge_length) / (2 * edge_length)
            height = np.sqrt(max(0.0, da * da - along * along))
            perp = np.array([-unit[1], unit[0]])
            known_third = next(key for key in keys[face] if key not in (ka, kb))
            offset = known[known_third] - qa
            side = np.sign(edge[0] * offset[1] - edge[1] * offset[0]) or 1.0
            qc = qa + along * unit - side * height * perp
            lookup = {ka: qa, kb: qb, kc: qc}
            candidate = np.stack([lookup[key] for key in keys[neighbor]])
            if np.linalg.norm(candidate, axis=1).min() <= radius_m + max_edge:
                unfolded[neighbor] = candidate
                queue.append(neighbor)
    ordered = np.asarray(sorted(unfolded), dtype=np.int32)
    return UnfoldedPatch(
        seed_face=anchor.face,
        body_first_face=body_start,
        seed_triangle_uv=seed_uv,
        mesh_vertices=vertices,
        mesh_keys=keys,
        face_indices=ordered + body_start,
        triangles_uv=np.stack([unfolded[int(face)] for face in ordered]),
        adjacent={face + body_start: {neighbor + body_start for neighbor in adjacent_local[face] if neighbor in unfolded}
                  for face in unfolded},
    )


def compile_surface_trace(
    rig: BodyRig,
    placement: dict,
    metric_strokes: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    max_step_m: float = 5e-4,
) -> SurfaceTrace:
    if max_step_m <= 0:
        raise SurfaceTraceError("max trace step must be positive")
    source = placement["anchor"]
    b0, b1, b2 = source["barycentric"]
    anchor = SurfaceAnchor(int(source["face"]), (float(b0), float(b1), float(b2)))
    radius = max(float(np.linalg.norm(point)) for stroke in metric_strokes for point in stroke) + max_step_m * 2
    patch = unfold_body_patch(rig, anchor, float(placement["rotation_rad"]), radius)
    strokes = tuple(patch.anchors(_resample(np.asarray(stroke), max_step_m)) for stroke in metric_strokes)
    payload = [[item.as_dict() for item in stroke] for stroke in strokes]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SurfaceTrace(strokes=strokes, sha256=digest)


def anchors_to_points(posed_vertices: np.ndarray, trace: SurfaceTrace) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(posed_vertices)
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
        raise BodyRigError("posed vertices must have shape (faces, 3, 3)")
    return tuple(np.stack([np.asarray(anchor.barycentric) @ vertices[anchor.face] for anchor in stroke])
                 for stroke in trace.strokes)
