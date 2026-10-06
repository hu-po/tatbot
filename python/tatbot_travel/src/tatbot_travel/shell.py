"""Forearm ink chart on the same articulated SOMA triangles as the phantom.

The texture map, rendered triangles and expert targets share canonical face
IDs and barycentric coordinates. Charting changes UVs, never the skin shape.
The small rendering offset prevents z-fighting; labels resolve the underlying
skin exactly through Inkmap's shared PosedBody/TriangleChart path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tatbot_sim.inkmap.triangle_chart import TriangleChart

from tatbot_travel.phantom import Phantom, build_phantom

SHELL_OFFSET_M = 0.0003


@dataclass(frozen=True)
class SkinShell:
    phantom: Phantom
    chart: TriangleChart
    x: np.ndarray
    theta: np.ndarray
    radius: np.ndarray
    corner_normals: np.ndarray

    def addresses(self, x, theta) -> tuple[np.ndarray, np.ndarray]:
        x, theta = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(theta, dtype=float))
        return self.chart.locate(np.stack([x, theta], axis=-1))

    def lift(self, x, theta) -> tuple[np.ndarray, np.ndarray]:
        faces, bary = self.addresses(x, theta)
        points = self.phantom.body.points(faces, bary)
        normals = np.einsum("...i,...ij->...j", bary, self.corner_normals[faces])
        return points, normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    @cached_property
    def points(self) -> np.ndarray:
        x, theta = np.meshgrid(self.x, self.theta, indexing="ij")
        return self.lift(x, theta)[0]

    @cached_property
    def normals(self) -> np.ndarray:
        x, theta = np.meshgrid(self.x, self.theta, indexing="ij")
        return self.lift(x, theta)[1]

    def radius_at(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.x, self.radius)

    def obj_text(self, offset: float = SHELL_OFFSET_M) -> str:
        faces, bary, chart = self.chart.clipped(((self.x[0], self.x[-1]), (-np.pi, np.pi)))
        ids = np.broadcast_to(faces[:, None], bary.shape[:-1])
        vertices = self.phantom.body.points(ids, bary)
        normals = np.einsum("fci,fij->fcj", bary, self.corner_normals[faces])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        vertices = (vertices + offset * normals).reshape(-1, 3)
        normals = normals.reshape(-1, 3)
        uv = (chart.reshape(-1, 2) - [self.x[0], -np.pi]) / [self.x[-1] - self.x[0], 2 * np.pi]
        lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
        lines += [f"vn {x:.7f} {y:.7f} {z:.7f}" for x, y, z in normals]
        lines += [f"vt {u:.9f} {v:.9f}" for u, v in uv]
        lines += ["f " + " ".join(f"{k + 1}/{k + 1}/{k + 1}" for k in tri)
                  for tri in np.arange(len(vertices)).reshape(-1, 3)]
        return "\n".join(lines) + "\n"


def _loops(segments: np.ndarray) -> list[list[int]]:
    """Chain plane-section segments into loops: each a list of (segment * 2 + entry end)."""
    ends = segments.reshape(-1, 3)
    partner = np.full(len(ends), -1)
    for a, b in cKDTree(ends).query_pairs(1e-7, output_type="ndarray"):
        if a // 2 != b // 2:
            partner[a], partner[b] = b, a
    seen = np.zeros(len(segments), dtype=bool)
    loops = []
    for first in range(len(segments)):
        if seen[first]:
            continue
        loop, entry = [], 2 * first
        while entry >= 0 and not seen[entry // 2]:
            seen[entry // 2] = True
            loop.append(entry)
            entry = partner[entry ^ 1]  # leave by the other end, enter the segment sharing it
        loops.append(loop)
    return loops


def _ring(mesh: trimesh.Trimesh, phantom: Phantom, x: float, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The skin's cross-section at ``x``, resampled evenly along the skin at chart angles ``theta``."""
    segments, faces = trimesh.intersections.mesh_plane(mesh, [1.0, 0.0, 0.0], [x, 0.0, 0.0], return_faces=True)
    ends = segments.reshape(-1, 3)
    loop = max(_loops(segments), key=lambda lp: np.linalg.norm(np.diff(ends[lp], axis=0), axis=1).sum())
    points, face_ids = ends[loop], faces[np.asarray(loop) // 2]
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
    normals = (bary[:, :, None] * mesh.vertex_normals[mesh.faces[face_ids]]).sum(axis=1)
    centre = phantom.centre_at(x)
    angle = np.unwrap(np.arctan2(points[:, 1] - centre[1], points[:, 2] - centre[2]))
    if angle[-1] < angle[0]:  # run the way the angle grows: from the back (+z) toward +y
        points, normals, angle = points[::-1], normals[::-1], angle[::-1]
    points, normals = np.vstack([points, points[:1]]), np.vstack([normals, normals[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    wrapped = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    k = int(np.argmin(np.abs(wrapped)))  # the loop point nearest the back; start the arclength there
    nxt = wrapped[(k + 1) % len(wrapped)]
    step = 1 if wrapped[k] < 0 and nxt >= 0 else -1  # interpolate toward the side the zero is on
    j = (k + step) % len(wrapped)
    t = float(np.clip(-wrapped[k] / (wrapped[j] - wrapped[k]), 0.0, 1.0)) if wrapped[j] != wrapped[k] else 0.0
    start = s[k] + step * t * np.linalg.norm(points[j] - points[k])
    target = np.mod(start + np.mod(theta, 2 * np.pi) / (2 * np.pi) * s[-1], s[-1])
    ring = np.stack([np.interp(target, s, points[:, d]) for d in range(3)], axis=1)
    ring_n = np.stack([np.interp(target, s, normals[:, d]) for d in range(3)], axis=1)
    return ring, ring_n / np.linalg.norm(ring_n, axis=1, keepdims=True), float(s[-1] / (2 * np.pi))


def chart_skin(phantom: Phantom, *, dx: float = 0.002, n_theta: int = 181,
               inset: float = 0.005, theta_origin: float = 0.0) -> SkinShell:
    """Chart this realization's existing forearm triangles, including its UV seam."""
    mesh = trimesh.Trimesh(phantom.vertices, phantom.faces, process=False)
    lo, hi = phantom.forearm
    x = np.arange(lo + inset, hi - inset + 1e-9, dx)
    theta = np.linspace(-np.pi, np.pi, n_theta)
    triangles = phantom.body.vertices[phantom.face_indices]
    # Curled fingers can overlap the wrist in posed x. Eligibility belongs to
    # the canonical forearm patch, never a fresh spatial cut through the pose.
    keep = (np.isin(phantom.face_indices, phantom.ink_face_indices)
            & (triangles[..., 0].max(axis=1) > x[0]) & (triangles[..., 0].min(axis=1) < x[-1]))
    faces = phantom.face_indices[keep]
    triangles = triangles[keep]
    centre = phantom.centre_at(triangles[..., 0])
    angles = np.arctan2(triangles[..., 1] - centre[..., 1], triangles[..., 2] - centre[..., 2])
    angles = (angles - theta_origin + np.pi) % (2 * np.pi) - np.pi
    seam = np.ptp(angles, axis=1) > np.pi
    angles = np.where(seam[:, None] & (angles < 0), angles + 2 * np.pi, angles)
    chart = np.stack([triangles[..., 0], angles], axis=-1)
    charts = np.concatenate([chart + [0, shift] for shift in (-2 * np.pi, 0, 2 * np.pi)])
    ids = np.tile(faces, 3)
    keep = (charts[..., 1].max(axis=1) > -np.pi) & (charts[..., 1].min(axis=1) < np.pi)
    atlas = TriangleChart(phantom.body, ids[keep], charts[keep])
    normals = np.zeros_like(phantom.body.vertices)
    normals[phantom.face_indices] = mesh.vertex_normals[mesh.faces]
    radii = [_ring(mesh, phantom, float(station), theta)[2] for station in x]
    return SkinShell(phantom=phantom, chart=atlas, x=x, theta=theta,
                     radius=np.asarray(radii), corner_normals=normals)


@lru_cache(maxsize=4)
def canonical_shell(dx: float = 0.002, n_theta: int = 181, inset: float = 0.005,
                    theta_origin: float = 0.0) -> SkinShell:
    """Reference chart for geometry tests; worlds always chart their own phantom."""
    return chart_skin(build_phantom(), dx=dx, n_theta=n_theta, inset=inset, theta_origin=theta_origin)
