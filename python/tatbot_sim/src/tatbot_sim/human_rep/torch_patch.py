"""Frozen differentiable fixed-topology surface patch experiment built only from Torch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from tatbot_sim.inkfield import differentiable_polyline_field


class TorchPatchError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class TorchPatchSample:
    points_m: torch.Tensor
    normals: torch.Tensor
    du: torch.Tensor
    dv: torch.Tensor
    metric: torch.Tensor
    barycentric: torch.Tensor


class TorchFixedPatch:
    """One preselected patch; face membership is never optimized or softened."""

    def __init__(
        self,
        rest_triangles_m: torch.Tensor,
        chart_triangles_m: torch.Tensor,
        *,
        face_indices: torch.Tensor | None = None,
        normal_triangles: torch.Tensor | None = None,
    ) -> None:
        rest = torch.as_tensor(rest_triangles_m)
        chart = torch.as_tensor(chart_triangles_m, dtype=rest.dtype, device=rest.device)
        if rest.ndim != 3 or rest.shape[1:] != (3, 3):
            raise TorchPatchError("surface_coordinate_invalid", "rest triangles need shape (F,3,3)")
        if chart.shape != (len(rest), 3, 2):
            raise TorchPatchError("surface_coordinate_invalid", "chart triangles need shape (F,3,2)")
        if not torch.isfinite(rest).all() or not torch.isfinite(chart).all():
            raise TorchPatchError("surface_coordinate_invalid", "patch contains non-finite values")
        if face_indices is None:
            indices = torch.arange(len(rest), dtype=torch.long, device=rest.device)
        else:
            indices = torch.as_tensor(face_indices, dtype=torch.long, device=rest.device)
        if indices.shape != (len(rest),) or torch.unique(indices).numel() != len(indices):
            raise TorchPatchError("surface_coordinate_invalid", "face indices must be unique")
        chart_edges = torch.stack([chart[:, 1] - chart[:, 0], chart[:, 2] - chart[:, 0]], dim=-1)
        determinant = torch.linalg.det(chart_edges)
        world_area = torch.linalg.norm(
            torch.linalg.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), dim=-1
        )
        if torch.any(determinant.abs() <= 1e-14) or torch.any(world_area <= 1e-14):
            raise TorchPatchError("surface_coordinate_invalid", "patch contains a degenerate triangle")
        world_edges = torch.stack([rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]], dim=-1)
        derivative = world_edges @ torch.linalg.inv(chart_edges)
        normal = torch.linalg.cross(derivative[:, :, 0], derivative[:, :, 1], dim=-1)
        normal = torch.nn.functional.normalize(normal, dim=-1)
        if normal_triangles is None:
            vertex_normals = normal[:, None, :].expand(-1, 3, -1)
        else:
            vertex_normals = torch.as_tensor(normal_triangles, dtype=rest.dtype, device=rest.device)
            if vertex_normals.shape != rest.shape or not torch.isfinite(vertex_normals).all():
                raise TorchPatchError(
                    "surface_coordinate_invalid",
                    "normal triangles must be finite and match the rest triangles",
                )
            lengths = torch.linalg.norm(vertex_normals, dim=-1)
            if torch.any(lengths <= 1e-14):
                raise TorchPatchError("surface_coordinate_invalid", "normal triangles contain a zero normal")
            vertex_normals = vertex_normals / lengths[..., None]
        self.rest_triangles_m = rest
        self.chart_triangles_m = chart
        self.face_indices = indices
        self._derivative = derivative
        self._vertex_normals = vertex_normals

    @classmethod
    def from_numpy(
        cls,
        rest_face_vertices_m: np.ndarray,
        chart_triangles_m: np.ndarray,
        face_indices: np.ndarray,
        *,
        normal_face_vertices: np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "TorchFixedPatch":
        faces = np.asarray(face_indices, dtype=np.int64)
        rest = np.asarray(rest_face_vertices_m, dtype=np.float64)[faces]
        return cls(
            torch.as_tensor(rest, dtype=dtype, device=device),
            torch.as_tensor(chart_triangles_m, dtype=dtype, device=device),
            face_indices=torch.as_tensor(faces, dtype=torch.long, device=device),
            normal_triangles=(
                None
                if normal_face_vertices is None
                else torch.as_tensor(np.asarray(normal_face_vertices)[faces], dtype=dtype, device=device)
            ),
        )

    def sample(self, chart_points_m: torch.Tensor, face_slots: torch.Tensor) -> TorchPatchSample:
        points = torch.as_tensor(
            chart_points_m,
            dtype=self.chart_triangles_m.dtype,
            device=self.chart_triangles_m.device,
        )
        slots = torch.as_tensor(face_slots, dtype=torch.long, device=points.device)
        if points.ndim != 2 or points.shape[-1] != 2 or slots.shape != (len(points),):
            raise TorchPatchError("surface_coordinate_invalid", "points need (N,2) and slots need (N,)")
        if torch.any(slots < 0) or torch.any(slots >= len(self.face_indices)):
            raise TorchPatchError("anchor_outside_domain", "a precomputed face slot is outside the patch")
        chart = self.chart_triangles_m[slots]
        edges = torch.stack([chart[:, 1] - chart[:, 0], chart[:, 2] - chart[:, 0]], dim=-1)
        local = torch.linalg.solve(edges, (points - chart[:, 0]).unsqueeze(-1)).squeeze(-1)
        barycentric = torch.stack([1 - local[:, 0] - local[:, 1], local[:, 0], local[:, 1]], dim=-1)
        if torch.any(barycentric.detach() < -1e-6) or torch.any(barycentric.detach() > 1 + 1e-6):
            raise TorchPatchError("anchor_outside_domain", "chart point left its precomputed triangle")
        rest = self.rest_triangles_m[slots]
        world = torch.einsum("ni,nij->nj", barycentric, rest)
        derivative = self._derivative[slots]
        du, dv = derivative[:, :, 0], derivative[:, :, 1]
        normals = torch.nn.functional.normalize(
            torch.einsum("ni,nij->nj", barycentric, self._vertex_normals[slots]),
            dim=-1,
        )
        metric = torch.stack(
            [
                torch.stack([(du * du).sum(-1), (du * dv).sum(-1)], dim=-1),
                torch.stack([(du * dv).sum(-1), (dv * dv).sum(-1)], dim=-1),
            ],
            dim=-2,
        )
        return TorchPatchSample(world, normals, du, dv, metric, barycentric)

    def addresses(
        self, chart_points_m: torch.Tensor, face_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.as_tensor(
            face_slots,
            dtype=torch.long,
            device=self.face_indices.device,
        )
        sample = self.sample(chart_points_m, slots)
        return self.face_indices[slots], sample.barycentric


def transform_chart_points(
    points_m: torch.Tensor,
    *,
    translation_m: torch.Tensor,
    rotation_rad: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply trainable translation, rotation, and anisotropic scale."""

    points = torch.as_tensor(points_m)
    translation = torch.as_tensor(translation_m, dtype=points.dtype, device=points.device)
    rotation = torch.as_tensor(rotation_rad, dtype=points.dtype, device=points.device)
    scale_value = torch.as_tensor(scale, dtype=points.dtype, device=points.device)
    if translation.shape != (2,) or scale_value.shape not in {(1,), (2,), ()}:
        raise TorchPatchError("surface_coordinate_invalid", "translation needs 2 values and scale 1 or 2")
    if torch.any(scale_value.detach() <= 0):
        raise TorchPatchError("surface_coordinate_invalid", "scale must be positive")
    scaled = points * scale_value
    c, s = torch.cos(rotation), torch.sin(rotation)
    matrix = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    return scaled @ matrix.T + translation


def render_chart_program(
    polylines_m: Iterable[torch.Tensor],
    *,
    width_m: float,
    height_m: float,
    rows: int = 128,
    cols: int = 128,
    radius_m: torch.Tensor | float = 0.0004,
) -> torch.Tensor:
    """Required InkField-based differentiable reference renderer."""

    shifted = [line + line.new_tensor([width_m / 2, height_m / 2]) for line in polylines_m]
    return differentiable_polyline_field(
        shifted,
        width_m=width_m,
        height_m=height_m,
        rows=rows,
        cols=cols,
        radius_m=radius_m,
    )


def relative_gradient_error(autograd: torch.Tensor, finite_difference: torch.Tensor) -> torch.Tensor:
    denominator = torch.maximum(
        torch.maximum(autograd.abs(), finite_difference.abs()),
        autograd.new_tensor(1e-9),
    )
    return (autograd - finite_difference).abs() / denominator


def angle_error_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def require_gradient(value: torch.Tensor, name: str) -> None:
    if not value.requires_grad or value.grad_fn is None:
        raise TorchPatchError("gradient_graph_broken", f"{name} is detached from its inputs")
    if not torch.isfinite(value).all():
        raise TorchPatchError("gradient_graph_broken", f"{name} contains non-finite values")


def proposed_hardening_gate(
    point_errors_m: torch.Tensor, *, p95_limit_m: float = 0.00025
) -> dict[str, float | str]:
    errors = torch.as_tensor(point_errors_m).detach().reshape(-1)
    if len(errors) == 0 or not torch.isfinite(errors).all():
        raise TorchPatchError("gradient_hardening_delta_exceeded", "hardening errors are empty or non-finite")
    p95 = float(torch.quantile(errors, 0.95))
    return {
        "p95_error_m": p95,
        "max_error_m": float(errors.max()),
        "proposed_limit_m": p95_limit_m,
        "status": "pass" if p95 <= p95_limit_m else "review",
    }
