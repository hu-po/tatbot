"""Metric/UV charts that retain canonical SOMA face and barycentric addresses.

A chart supplies coordinates on existing triangles; it never creates a new
skin surface. Rendering and target sampling resolve the same addresses on a
``PosedBody``. This numerical path has no Torch or engine dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from tatbot_sim.inkmap.rig import BodyRigError, PosedBody


def barycentric_2d(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Broadcast points and chart triangles, preserving corner order."""
    a, b, c = np.moveaxis(triangles, -2, 0)
    ab, ac, ap = b - a, c - a, points - a
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    determinant = cross(ab, ac)
    with np.errstate(divide="ignore", invalid="ignore"):
        v, w = cross(ap, ac) / determinant, cross(ab, ap) / determinant
    return np.stack([1 - v - w, v, w], axis=-1)


class TriangleChart:
    """Locate chart points on canonical faces, including duplicated UV seams."""

    def __init__(self, body: PosedBody, face_indices: np.ndarray, triangles_uv: np.ndarray):
        self.body = body
        self.face_indices = np.asarray(face_indices)
        if (self.face_indices.ndim != 1 or not np.issubdtype(self.face_indices.dtype, np.integer)
                or np.any(self.face_indices < 0) or np.any(self.face_indices >= len(body.vertices))):
            raise BodyRigError("surface_coordinate_invalid: chart face is out of range or non-integer")
        self.triangles_uv = np.asarray(triangles_uv, dtype=np.float64)
        if self.triangles_uv.shape != (len(self.face_indices), 3, 2):
            raise BodyRigError("surface_coordinate_invalid: chart triangle shape")
        if not len(self.face_indices) or not np.isfinite(self.triangles_uv).all():
            raise BodyRigError("surface_coordinate_invalid: empty or non-finite chart")
        self.scale = np.maximum(np.ptp(self.triangles_uv.reshape(-1, 2), axis=0), 1e-8)
        self.tree = cKDTree(self.triangles_uv.mean(axis=1) / self.scale)

    def locate(self, points) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != 2 or not np.isfinite(values).all():
            raise BodyRigError("surface_coordinate_invalid: expected finite chart points")
        shape, flat = values.shape[:-1], values.reshape(-1, 2)
        count = min(24, len(self.face_indices))
        _, candidates = self.tree.query(flat / self.scale, k=count)
        candidates = np.asarray(candidates).reshape(len(flat), count)
        weights = barycentric_2d(flat[:, None], self.triangles_uv[candidates])
        inside = np.isfinite(weights).all(axis=-1) & (weights >= -1e-8).all(axis=-1)
        chosen = inside.argmax(axis=1)
        slots = candidates[np.arange(len(flat)), chosen]
        bary = weights[np.arange(len(flat)), chosen]
        # A long thin triangle may have a distant centroid. Search all faces
        # only for these points; nearest-centroid selection is never a fallback
        # surface projection or permission to move an address outside a face.
        for index in np.flatnonzero(~inside.any(axis=1)):
            weights = barycentric_2d(flat[index], self.triangles_uv)
            valid = np.flatnonzero(np.isfinite(weights).all(axis=-1) & (weights >= -1e-8).all(axis=-1))
            if not len(valid):
                raise BodyRigError(f"surface_coordinate_invalid: chart point {flat[index].tolist()} outside skin")
            slots[index] = valid[0]
            bary[index] = weights[valid[0]]
        bary = np.clip(bary, 0, 1)
        bary /= bary.sum(axis=-1, keepdims=True)
        return self.face_indices[slots].reshape(shape), bary.reshape(*shape, 3)

    def points(self, points) -> np.ndarray:
        faces, barycentric = self.locate(points)
        return self.body.points(faces, barycentric)

    def clipped(self, bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clip chart triangles to a rectangle while retaining surface addresses.

        Returns canonical face IDs, corner barycentrics, and chart triangles.
        Every new corner is an interpolation inside its original skin face.
        """
        face_ids, barycentric, uv = [], [], []
        for face, triangle in zip(self.face_indices, self.triangles_uv, strict=True):
            polygon = [(point, weight) for point, weight in zip(triangle, np.eye(3), strict=True)]
            for axis, limit, sign in ((0, bounds[0][0], 1), (0, bounds[0][1], -1),
                                      (1, bounds[1][0], 1), (1, bounds[1][1], -1)):
                result = []
                for end in range(len(polygon)):
                    a, ba = polygon[end - 1]
                    b, bb = polygon[end]
                    da, db = sign * (a[axis] - limit), sign * (b[axis] - limit)
                    if (da >= 0) != (db >= 0):
                        t = da / (da - db)
                        result.append((a + t * (b - a), ba + t * (bb - ba)))
                    if db >= 0:
                        result.append((b, bb))
                polygon = result
            for corner in range(1, len(polygon) - 1):
                points, weights = zip(polygon[0], polygon[corner], polygon[corner + 1], strict=True)
                points = np.asarray(points)
                area = np.linalg.det(np.stack([points[1] - points[0], points[2] - points[0]]))
                if abs(area) > 1e-14:
                    face_ids.append(face)
                    uv.append(points)
                    barycentric.append(weights)
        return np.asarray(face_ids, dtype=np.int32), np.asarray(barycentric), np.asarray(uv)
