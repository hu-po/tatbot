"""Frozen coupled-contact solver interface and hermetic shell reference.

The linear shell below is a verification fixture, not an admitted tissue
model.  Admission additionally requires a declared spatial-coupling residual
in qualified phantom held-out physical data and at least 20 percent improvement.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from tatbot_sim.human_rep.contracts import ContractError


@dataclass(frozen=True)
class ShellMesh:
    vertices_m: np.ndarray
    faces: np.ndarray
    fixed_vertices: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m)
        faces = np.asarray(self.faces)
        fixed = np.asarray(self.fixed_vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ContractError("contact_solver_unstable", "$.mesh.vertices_m", "expected finite (N>=4,3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 2:
            raise ContractError("contact_solver_unstable", "$.mesh.faces", "expected triangle indices")
        if not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer):
            raise ContractError("contact_solver_unstable", "$.mesh", "non-finite vertices or non-integer faces")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ContractError("contact_solver_unstable", "$.mesh.faces", "face index out of bounds")
        if fixed.shape != (len(vertices),) or fixed.dtype != np.bool_:
            raise ContractError("contact_solver_unstable", "$.mesh.fixed_vertices", "expected boolean (N,) mask")
        triangles = vertices[faces]
        doubled_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        if np.any(doubled_area <= 1e-12):
            raise ContractError("contact_solver_unstable", "$.mesh.faces", "degenerate triangle")
        if np.count_nonzero(fixed) < 2 or np.count_nonzero(~fixed) < 1:
            raise ContractError("contact_solver_unstable", "$.mesh.fixed_vertices", "insufficient boundary/free vertices")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            np.ascontiguousarray(self.vertices_m, dtype="<f8"),
            np.ascontiguousarray(self.faces, dtype="<i8"),
            np.ascontiguousarray(self.fixed_vertices, dtype=np.uint8),
        ):
            digest.update(value.tobytes())
            digest.update(b"\0")
        return digest.hexdigest()


def grid_shell(rows: int = 7, cols: int = 9, spacing_m: float = 0.005) -> ShellMesh:
    if rows < 3 or cols < 3 or spacing_m <= 0:
        raise ValueError("grid shell needs rows/cols >=3 and positive spacing")
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    vertices = np.stack(
        [
            (xx.ravel() - (cols - 1) / 2) * spacing_m,
            (yy.ravel() - (rows - 1) / 2) * spacing_m,
            np.zeros(rows * cols),
        ],
        axis=1,
    )
    index = np.arange(rows * cols).reshape(rows, cols)
    faces = np.concatenate(
        [
            np.stack([index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel()], axis=1),
            np.stack([index[:-1, :-1].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()], axis=1),
        ]
    )
    fixed = np.zeros(rows * cols, dtype=bool)
    fixed[index[0]] = True
    fixed[index[-1]] = True
    fixed[index[:, 0]] = True
    fixed[index[:, -1]] = True
    return ShellMesh(vertices, faces, fixed)


class LinearShellFixture:
    """Scalar normal-displacement membrane with graph coupling."""

    identifier = "compliant-shell-linear-fixture-v1"
    license_spdx = "Apache-2.0"

    def __init__(
        self,
        mesh: ShellMesh,
        *,
        local_stiffness_n_m: float,
        coupling_n_m: float,
    ) -> None:
        if not math.isfinite(local_stiffness_n_m) or local_stiffness_n_m <= 0:
            raise ContractError("contact_solver_unstable", "$.local_stiffness_n_m", "must be positive")
        if not math.isfinite(coupling_n_m) or coupling_n_m < 0:
            raise ContractError("contact_solver_unstable", "$.coupling_n_m", "must be nonnegative")
        self.mesh = mesh
        self.local_stiffness_n_m = float(local_stiffness_n_m)
        self.coupling_n_m = float(coupling_n_m)
        count = len(mesh.vertices_m)
        adjacency = np.zeros((count, count), dtype=np.float64)
        for triangle in mesh.faces:
            for left, right in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                adjacency[int(left), int(right)] = 1.0
                adjacency[int(right), int(left)] = 1.0
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        stiffness = self.local_stiffness_n_m * np.eye(count) + self.coupling_n_m * laplacian
        fixed = np.flatnonzero(mesh.fixed_vertices)
        stiffness[fixed, :] = 0.0
        stiffness[:, fixed] = 0.0
        stiffness[fixed, fixed] = 1.0
        self._stiffness = stiffness
        eigenvalues = np.linalg.eigvalsh(stiffness)
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
            raise ContractError("contact_solver_unstable", "$.system", "stiffness matrix is not positive definite")

    def _state(self, value: Any, path: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (len(self.mesh.vertices_m),) or not np.isfinite(array).all():
            raise ContractError("contact_solver_unstable", path, "expected finite scalar state per vertex")
        return array

    def force(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = self._state(displacement_m, "$.displacement_m").copy()
        if np.any(displacement < -1e-12):
            raise ContractError("contact_solver_unstable", "$.displacement_m", "negative normal displacement")
        if np.any(np.abs(displacement[self.mesh.fixed_vertices]) > 1e-12):
            raise ContractError("contact_solver_unstable", "$.boundary", "fixed vertex moved")
        return self._stiffness @ displacement

    def solve(self, force_n: np.ndarray) -> np.ndarray:
        force = self._state(force_n, "$.force_n").copy()
        force[self.mesh.fixed_vertices] = 0.0
        displacement = np.linalg.solve(self._stiffness, force)
        residual = self._stiffness @ displacement - force
        if not np.isfinite(displacement).all() or float(np.linalg.norm(residual)) > 1e-9:
            raise ContractError("contact_solver_unstable", "$.solve", "linear solve did not converge")
        if np.any(displacement < -1e-12):
            raise ContractError("contact_solver_unstable", "$.solve", "solution left the unilateral fixture domain")
        return displacement

    def gradient(self) -> np.ndarray:
        return self._stiffness.copy()

    def energy_j(self, displacement_m: np.ndarray) -> float:
        displacement = self._state(displacement_m, "$.displacement_m")
        energy = 0.5 * float(displacement @ self._stiffness @ displacement)
        if not math.isfinite(energy) or energy < -1e-14:
            raise ContractError("contact_solver_unstable", "$.energy", "non-finite or negative energy")
        return max(0.0, energy)


def verify_shell_fixture(solver: LinearShellFixture, *, seed: int = 9_042_026) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    force = np.zeros(len(solver.mesh.vertices_m), dtype=np.float64)
    free = np.flatnonzero(~solver.mesh.fixed_vertices)
    force[free] = rng.uniform(0.0, 0.02, len(free))
    started = time.perf_counter()
    first = solver.solve(force)
    runtime_s = time.perf_counter() - started
    second = solver.solve(force)
    reconstructed = solver.force(first)
    forward_error = float(np.max(np.abs(reconstructed - force)))
    gradient = solver.gradient()
    direction = rng.normal(size=len(force))
    direction[solver.mesh.fixed_vertices] = 0.0
    epsilon = 1e-7
    finite_difference = (solver.force(first + epsilon * direction) - solver.force(first)) / epsilon
    gradient_error = float(np.max(np.abs(finite_difference - gradient @ direction)))
    mesh_bytes = solver.mesh.vertices_m.nbytes + solver.mesh.faces.nbytes + solver.mesh.fixed_vertices.nbytes
    solver_bytes = gradient.nbytes
    checks = {
        "mesh_quality": True,
        "boundary_conditions": bool(np.all(first[solver.mesh.fixed_vertices] == 0)),
        "solver_convergence": forward_error <= 1e-9,
        "energy": solver.energy_j(first) >= 0,
        "gradient": gradient_error <= 1e-8,
        "forward_reference": forward_error <= 1e-9,
        "runtime": runtime_s <= 1.0,
        "memory": mesh_bytes + solver_bytes <= 10_000_000,
        "determinism": bool(np.array_equal(first, second)),
        "distribution_license": solver.license_spdx == "Apache-2.0",
    }
    return {
        "schema": "tatbot.coupled-contact-fixture-result/1",
        "solver": solver.identifier,
        "mesh_sha256": solver.mesh.sha256,
        "seed": seed,
        "vertices": len(solver.mesh.vertices_m),
        "faces": len(solver.mesh.faces),
        "forward_max_error_n": forward_error,
        "gradient_max_error_n_m": gradient_error,
        "energy_j": solver.energy_j(first),
        "runtime_s": runtime_s,
        "memory_bytes": mesh_bytes + solver_bytes,
        "output_sha256": hashlib.sha256(np.ascontiguousarray(first, dtype="<f8").tobytes()).hexdigest(),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "displacement_m": first.astype(float).tolist(),
        "force_n": force.astype(float).tolist(),
    }


def coupled_admission(
    *,
    p7_spatial_coupling_residual_available: bool,
    relative_primary_improvement: float | None,
    contact_regressions: Mapping[str, bool] | None,
) -> dict[str, Any]:
    if not p7_spatial_coupling_residual_available:
        return {
            "status": "not_admitted_pending_evidence",
            "candidate": None,
            "trigger": "missing qualified phantom spatial-coupling residual",
            "relative_primary_improvement": None,
            "required_relative_improvement": 0.20,
            "admitted_model": "rigid-contact-v1",
        }
    regressions = dict(contact_regressions or {})
    improvement = float(relative_primary_improvement or 0.0)
    admitted = improvement >= 0.20 and not any(regressions.values())
    return {
        "status": "admitted" if admitted else "archived_not_admitted",
        "candidate": "compliant-shell-v1",
        "trigger": "qualified phantom spatial-coupling residual",
        "relative_primary_improvement": improvement,
        "required_relative_improvement": 0.20,
        "contact_regressions": regressions,
        "admitted_model": "compliant-shell-v1" if admitted else "compliant-heightfield-v1",
    }
