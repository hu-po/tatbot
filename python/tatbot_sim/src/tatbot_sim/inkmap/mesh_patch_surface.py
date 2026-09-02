"""A posed, triangulated body patch implementing Tatbot's Surface contract."""

from __future__ import annotations

import numpy as np
import torch

from tatbot_sim.inkmap.contracts import validate_scenario
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.surface_trace import SurfaceAnchor, UnfoldedPatch, _vertex_key, unfold_body_patch
from tatbot_sim.surface import Surface


def _smooth_normals(vertices: np.ndarray) -> np.ndarray:
    accumulated: dict[tuple[int, int, int], np.ndarray] = {}
    keys = [[_vertex_key(point) for point in triangle] for triangle in vertices]
    for triangle, triangle_keys in zip(vertices, keys, strict=True):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        for key in triangle_keys:
            accumulated[key] = accumulated.get(key, np.zeros(3)) + normal
    out = np.zeros_like(vertices, dtype=np.float64)
    for face, triangle_keys in enumerate(keys):
        for corner, key in enumerate(triangle_keys):
            value = accumulated[key]
            out[face, corner] = value / max(np.linalg.norm(value), 1e-20)
    return out


def _closest_barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest point on a triangle, returned as normalized barycentrics."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return np.array([1.0, 0.0, 0.0])
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.array([1 - v, v, 0.0])
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.array([1 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.array([0.0, 1 - w, w])
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return np.array([1 - v - w, v, w])


class MeshPatchSurface(Surface):
    """Static named-pose skin over a local unfolded chart.

    Each batch element may own a different patch/body realization. ``env_view``
    repeats one realization and marks it sequential, so a whole trajectory is
    unfolded continuously across adjacent triangles in one call.
    """

    def __init__(
        self,
        patches: list[UnfoldedPatch],
        posed_vertices: list[np.ndarray],
        *,
        device: torch.device | str = "cpu",
        sequential: bool = False,
        width_m: float = 0.14,
        height_m: float = 0.185,
        cols: int = 336,
        rows: int = 444,
        normals: list[np.ndarray] | None = None,
    ):
        if len(patches) != len(posed_vertices) or not patches:
            raise ValueError("MeshPatchSurface needs one posed mesh per non-empty patch batch")
        super().__init__(width_m, height_m, cols, rows)
        self.patches = patches
        self.posed_vertices = [np.asarray(value, dtype=np.float64) for value in posed_vertices]
        self.normals = normals or [_smooth_normals(value) for value in self.posed_vertices]
        self._device_value = torch.device(device)
        self.sequential = sequential

    @property
    def batch_size(self) -> int:
        return len(self.patches)

    @property
    def _device(self):
        return self._device_value

    def _mapped(self, uv_m: np.ndarray):
        if len(uv_m) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} chart points, got {len(uv_m)}")
        if self.sequential and all(patch is self.patches[0] for patch in self.patches):
            return self.patches[0].map_samples(uv_m)
        return tuple(patch.map_samples(point[None])[0] for patch, point in zip(self.patches, uv_m, strict=True))

    def frame(self, uv_m: torch.Tensor):
        uv_np = uv_m.detach().cpu().numpy().astype(np.float64)
        points, derivatives_u, derivatives_v, normals = [], [], [], []
        mapped = self._mapped(uv_np)
        self._last_mapping = mapped
        for env, (anchor, triangle_uv) in enumerate(mapped):
            triangle = self.posed_vertices[env][anchor.face]
            bary = np.asarray(anchor.barycentric)
            points.append(bary @ triangle)
            chart_edges = np.stack([triangle_uv[1] - triangle_uv[0], triangle_uv[2] - triangle_uv[0]], axis=1)
            world_edges = np.stack([triangle[1] - triangle[0], triangle[2] - triangle[0]], axis=1)
            derivative = world_edges @ np.linalg.inv(chart_edges)
            derivatives_u.append(derivative[:, 0])
            derivatives_v.append(derivative[:, 1])
            normal = bary @ self.normals[env][anchor.face]
            normals.append(normal / np.linalg.norm(normal))
        values = [torch.as_tensor(np.asarray(value), dtype=uv_m.dtype, device=uv_m.device)
                  for value in (points, derivatives_u, derivatives_v, normals)]
        return tuple(values)

    def first_fundamental_form(self, uv_m: torch.Tensor) -> torch.Tensor:
        _, du, dv, _ = self.frame(uv_m)
        return torch.stack([
            torch.stack([(du * du).sum(-1), (du * dv).sum(-1)], dim=-1),
            torch.stack([(du * dv).sum(-1), (dv * dv).sum(-1)], dim=-1),
        ], dim=-2)

    def project(self, points_w: torch.Tensor, axis_w: torch.Tensor | None = None):
        points_np = points_w.detach().cpu().numpy().astype(np.float64)
        uv_values, distances, incidences = [], [], []
        for env, point in enumerate(points_np):
            patch = self.patches[env]
            if hasattr(self, "_last_mapping") and len(self._last_mapping) == len(points_np):
                anchor, triangle_uv = self._last_mapping[env]
                face = anchor.face
                triangle = self.posed_vertices[env][face]
                bary = _closest_barycentric(point, triangle)
                closest_point = bary @ triangle
                uv_values.append(bary @ triangle_uv)
            else:
                faces = patch.face_indices
                triangles = self.posed_vertices[env][faces]
                barycentrics = np.stack([_closest_barycentric(point, triangle) for triangle in triangles])
                closest = np.einsum("fi,fij->fj", barycentrics, triangles)
                choice = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
                bary = barycentrics[choice]
                face = int(faces[choice])
                closest_point = closest[choice]
                uv_values.append(bary @ patch.triangles_uv[choice])
            normal = bary @ self.normals[env][face]
            normal /= np.linalg.norm(normal)
            distances.append(np.dot(point - closest_point, normal))
            if axis_w is None:
                incidences.append(1.0)
            else:
                axis = axis_w[env].detach().cpu().numpy()
                incidences.append(np.clip(abs(np.dot(axis, normal)), 0.0, 1.0))
        dtype, device = points_w.dtype, points_w.device
        return (
            torch.as_tensor(np.asarray(uv_values), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(distances), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(incidences), dtype=dtype, device=device),
        )

    def env_view(self, i: int, n: int) -> "MeshPatchSurface":
        return MeshPatchSurface(
            [self.patches[i]] * n,
            [self.posed_vertices[i]] * n,
            device=self._device,
            sequential=True,
            width_m=self.width_m,
            height_m=self.height_m,
            cols=self.cols,
            rows=self.rows,
            normals=[self.normals[i]] * n,
        )

    def base_normal_np(self, i: int) -> np.ndarray:
        patch = self.patches[i]
        anchor = patch.map_samples(np.zeros((1, 2)))[0][0]
        bary = np.asarray(anchor.barycentric)
        normal = bary @ self.normals[i][anchor.face]
        return normal / np.linalg.norm(normal)


def mesh_patch_from_scenario(scenario: dict, *, device: torch.device | str = "cpu") -> MeshPatchSurface:
    """Rebuild the simulator surface only after scenario identities validate."""
    validate_scenario(scenario)
    rig = load_body_rig(scenario["body"]["id"])
    if scenario["body"]["surface_sha256"] != rig.surface_sha256:
        raise ValueError("scenario body surface does not match installed rig")
    if scenario["body"]["rig_sha256"] != rig.catalog_record["sidecar_sha256"]:
        raise ValueError("scenario rig checksum does not match installed rig")
    if scenario["pose"]["catalog_sha256"] != rig.catalog_sha256:
        raise ValueError("scenario pose catalog checksum does not match installed catalog")
    placement = scenario["placement"]
    source = placement["anchor"]
    anchor = SurfaceAnchor(int(source["face"]), tuple(float(value) for value in source["barycentric"]))
    size_m = np.asarray(placement["size_mm"], dtype=float) / 1000
    radius = float(np.linalg.norm(size_m / 2)) + 0.003
    patch = unfold_body_patch(rig, anchor, float(placement["rotation_rad"]), radius)
    posed = rig.posed(scenario["pose"]["id"], np.asarray(scenario["pose"]["world_from_body"]))
    metres_per_texel = 4.2e-4
    pixels_per_metre = max(1 / metres_per_texel, 128 / float(size_m.min()))
    cols = int(np.ceil(size_m[0] * pixels_per_metre))
    rows = int(np.ceil(size_m[1] * pixels_per_metre))
    return MeshPatchSurface(
        [patch], [posed.vertices], device=device,
        width_m=float(size_m[0]), height_m=float(size_m[1]), cols=cols, rows=rows,
    )
