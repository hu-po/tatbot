from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from tatbot_sim.human_rep.torch_patch import (
    TorchFixedPatch,
    TorchPatchError,
    angle_error_degrees,
    proposed_hardening_gate,
    relative_gradient_error,
    render_chart_program,
    require_gradient,
    transform_chart_points,
)
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.surface_trace import SurfaceAnchor, UnfoldedPatch, _vertex_key, unfold_body_patch
from tatbot_sim.repo import repo_root


def plane_patch() -> TorchFixedPatch:
    chart = torch.tensor(
        [[[0.0, 0.0], [0.04, 0.0], [0.0, 0.04]]],
        dtype=torch.float64,
    )
    rest = torch.cat([chart, torch.zeros((1, 3, 1), dtype=torch.float64)], dim=-1)
    return TorchFixedPatch(rest, chart, face_indices=torch.tensor([17]))


def test_torch_fixed_patch_matches_analytic_plane_point_frame_and_metric():
    patch = plane_patch()
    chart = torch.tensor([[0.01, 0.012], [0.02, 0.005]], dtype=torch.float64)
    sample = patch.sample(chart, torch.zeros(2, dtype=torch.long))
    np.testing.assert_allclose(sample.points_m.numpy(), [[0.01, 0.012, 0], [0.02, 0.005, 0]], atol=1e-12)
    np.testing.assert_allclose(sample.du.numpy(), [[1, 0, 0], [1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(sample.dv.numpy(), [[0, 1, 0], [0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(sample.metric.numpy(), np.repeat(np.eye(2)[None], 2, axis=0), atol=1e-12)
    face, bary = patch.addresses(chart, torch.zeros(2, dtype=torch.long))
    assert face.tolist() == [17, 17]
    np.testing.assert_allclose(bary.sum(dim=1).numpy(), 1, atol=1e-12)


def test_autograd_matches_central_difference_away_from_boundaries():
    base = torch.tensor([[0.004, 0.005], [0.012, 0.007], [0.016, 0.018]], dtype=torch.float64)
    parameters = torch.tensor([0.001, -0.0015, 0.12, 1.04], dtype=torch.float64, requires_grad=True)

    def objective(values):
        transformed = transform_chart_points(
            base,
            translation_m=values[:2],
            rotation_rad=values[2],
            scale=values[3],
        )
        return (transformed.square().sum(dim=1) * torch.tensor([0.7, 1.1, 1.4])).sum()

    loss = objective(parameters)
    require_gradient(loss, "analytic objective")
    auto = torch.autograd.grad(loss, parameters)[0]
    step = 1e-6
    finite = []
    for index in range(len(parameters)):
        delta = torch.zeros_like(parameters)
        delta[index] = step
        finite.append(
            (objective(parameters.detach() + delta) - objective(parameters.detach() - delta)) / (2 * step)
        )
    errors = relative_gradient_error(auto, torch.stack(finite))
    assert float(torch.quantile(errors, 0.5)) <= 0.01
    assert float(torch.quantile(errors, 0.95)) <= 0.05


def test_seeded_recovery_meets_translation_rotation_and_scale_targets():
    generator = torch.Generator().manual_seed(9042026)
    base = torch.rand((32, 2), generator=generator, dtype=torch.float64) * 0.025 - 0.0125
    truth = torch.tensor([0.0017, -0.0021, math.radians(7.0), 1.075], dtype=torch.float64)
    target = transform_chart_points(
        base,
        translation_m=truth[:2],
        rotation_rad=truth[2],
        scale=truth[3],
    )
    estimate = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [estimate], lr=0.8, max_iter=80, tolerance_grad=1e-12, tolerance_change=1e-14
    )

    def closure():
        optimizer.zero_grad()
        predicted = transform_chart_points(
            base,
            translation_m=estimate[:2],
            rotation_rad=estimate[2],
            scale=estimate[3],
        )
        loss = ((predicted - target) ** 2).mean() * 1e6
        loss.backward()
        return loss

    optimizer.step(closure)
    error = (estimate.detach() - truth).abs()
    assert float(torch.linalg.norm(error[:2])) <= 0.0005
    assert math.degrees(float(error[2])) <= 0.5
    assert float(error[3]) <= 0.005


def test_inkfield_reference_render_keeps_vector_gradients_connected():
    points = torch.tensor(
        [[-0.012, -0.005], [0.0, 0.008], [0.012, -0.004]],
        dtype=torch.float64,
        requires_grad=True,
    )
    image = render_chart_program([points], width_m=0.04, height_m=0.04, rows=48, cols=48)
    require_gradient(image, "ink field")
    weighted = image * torch.linspace(0, 1, image.numel(), dtype=image.dtype).reshape_as(image)
    weighted.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert float(points.grad.abs().sum()) > 0


def test_exact_hardening_delta_gate_reports_without_clamping():
    gate = proposed_hardening_gate(torch.tensor([0.0, 1e-5, 2e-5, 3e-5]))
    assert gate["status"] == "pass"
    assert gate["p95_error_m"] <= 0.00025


def _adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    owners: dict[tuple[int, int], list[int]] = {}
    for face, vertices in enumerate(faces):
        for left, right in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault(tuple(sorted((int(vertices[left]), int(vertices[right])))), []).append(face)
    adjacent: dict[int, set[int]] = {face: set() for face in range(len(faces))}
    for face_owners in owners.values():
        if len(face_owners) == 2:
            left, right = face_owners
            adjacent[left].add(right)
            adjacent[right].add(left)
    return adjacent


def _find_slots(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    slots = []
    for point in points:
        a = triangles[:, 0]
        edge = np.stack([triangles[:, 1] - a, triangles[:, 2] - a], axis=-1)
        local = np.linalg.solve(edge, (np.broadcast_to(point, a.shape) - a)[..., None]).squeeze(-1)
        bary = np.column_stack([1 - local.sum(axis=1), local])
        candidates = np.flatnonzero(np.all(bary >= -1e-10, axis=1))
        assert len(candidates) >= 1
        slots.append(int(candidates[0]))
    return np.asarray(slots, dtype=np.int64)


def _cylinder_patch():
    radius = 0.04
    v_nodes = np.linspace(-0.03012, 0.02988, 251)
    u_nodes = np.asarray([-0.007, 0.003])
    vertices = []
    charts = []
    normals = []
    for u in u_nodes:
        for v in v_nodes:
            angle = v / radius
            vertices.append([u, radius * np.sin(angle), radius * np.cos(angle)])
            charts.append([u, v])
            normals.append([0.0, np.sin(angle), np.cos(angle)])
    vertices = np.asarray(vertices)
    charts = np.asarray(charts)
    normals = np.asarray(normals)
    faces = []
    count_v = len(v_nodes)
    for index in range(len(v_nodes) - 1):
        a, b = index, index + 1
        c, d = count_v + index, count_v + index + 1
        faces.extend(((a, c, b), (b, c, d)))
    faces = np.asarray(faces, dtype=np.int64)
    world_triangles = vertices[faces]
    chart_triangles = charts[faces]
    normal_triangles = normals[faces]
    seed = int(_find_slots(np.asarray([[0.0, 0.0]]), chart_triangles)[0])
    bary = np.linalg.solve(
        np.stack(
            [
                chart_triangles[seed, 1] - chart_triangles[seed, 0],
                chart_triangles[seed, 2] - chart_triangles[seed, 0],
            ],
            axis=-1,
        ),
        -chart_triangles[seed, 0],
    )
    seed_bary = np.asarray([1 - bary.sum(), *bary])
    patch = UnfoldedPatch(
        seed_face=seed,
        body_first_face=0,
        seed_triangle_uv=chart_triangles[seed],
        mesh_vertices=world_triangles,
        mesh_keys=[[_vertex_key(point) for point in triangle] for triangle in world_triangles],
        face_indices=np.arange(len(faces), dtype=np.int32),
        triangles_uv=chart_triangles,
        adjacent=_adjacency(faces),
    )
    assert np.allclose(seed_bary @ chart_triangles[seed], 0, atol=1e-12)
    return radius, patch, world_triangles, chart_triangles, normal_triangles, v_nodes


def test_torch_numpy_and_analytic_cylinder_frames_match_over_sixty_mm():
    radius, patch, triangles, chart, normals, v_nodes = _cylinder_patch()
    centers = (v_nodes[:-1] + v_nodes[1:]) / 2
    selected = np.linspace(1, len(centers) - 2, 21, dtype=np.int64)
    points = np.column_stack([np.linspace(-0.0065, 0.0025, 21), centers[selected]])
    slots = _find_slots(points, chart)
    oracle = MeshPatchSurface(
        [patch] * len(points),
        [triangles] * len(points),
        normals=[normals] * len(points),
        sequential=True,
    )
    torch_patch = TorchFixedPatch.from_numpy(
        triangles,
        chart,
        np.arange(len(triangles)),
        normal_face_vertices=normals,
    )
    values = torch_patch.sample(torch.as_tensor(points), torch.as_tensor(slots))
    oracle_point, oracle_du, oracle_dv, oracle_normal = oracle.frame(torch.as_tensor(points))
    assert float(torch.linalg.norm(values.points_m - oracle_point, dim=1).max()) <= 0.00001
    assert float(angle_error_degrees(values.du, oracle_du).max()) <= 0.05
    assert float(angle_error_degrees(values.dv, oracle_dv).max()) <= 0.05
    assert float(angle_error_degrees(values.normals, oracle_normal).max()) <= 0.05
    np.testing.assert_allclose(
        values.metric,
        oracle.first_fundamental_form(torch.as_tensor(points)),
        rtol=1e-5,
        atol=1e-7,
    )

    angle = points[:, 1] / radius
    expected_point = np.column_stack([points[:, 0], radius * np.sin(angle), radius * np.cos(angle)])
    expected_normal = torch.as_tensor(np.column_stack([np.zeros(len(points)), np.sin(angle), np.cos(angle)]))
    expected_dv = torch.as_tensor(np.column_stack([np.zeros(len(points)), np.cos(angle), -np.sin(angle)]))
    assert float(torch.linalg.norm(values.points_m - torch.as_tensor(expected_point), dim=1).max()) <= 0.00001
    assert float(angle_error_degrees(values.normals, expected_normal).max()) <= 0.05
    assert float(angle_error_degrees(values.dv, expected_dv).max()) <= 0.05

    # The developed 60 mm chart remains intrinsic: summing the mesh geodesic
    # chords loses far less than the 1 mm acceptance budget.
    physical = np.linalg.norm(np.diff(triangles[::2, 0], axis=0), axis=1).sum()
    assert abs(physical - 0.06) <= 0.001


@pytest.mark.slow
def test_seeded_soma_patch_matches_numpy_and_keeps_gradients_and_hashes():
    rig = load_body_rig()
    atlas = __import__("json").loads(
        (repo_root() / "web/inkmap/public/bodies/mhr-soma-v1.regions.json").read_text()
    )
    anchor_value = atlas["regions"]["forearm:right"]["default_anchor"]
    anchor = SurfaceAnchor(anchor_value["face"], tuple(anchor_value["barycentric"]))
    patch = unfold_body_patch(rig, anchor, 0.0, 0.012)
    points = np.column_stack([np.linspace(0.0002, 0.004, 19), np.linspace(0.0001, 0.002, 19)])
    mapped = patch.map_samples(points)
    unique = []
    seen = set()
    for index, (item, _) in enumerate(mapped):
        if item.face not in seen:
            seen.add(item.face)
            unique.append(index)
    points = points[unique]
    mapped = tuple(mapped[index] for index in unique)
    face_ids = np.asarray([item.face for item, _ in mapped], dtype=np.int64)
    chart_triangles = np.stack([triangle for _, triangle in mapped])
    oracle = MeshPatchSurface(
        [patch] * len(points),
        [rig.rest_vertices] * len(points),
        sequential=True,
    )
    torch_patch = TorchFixedPatch(
        torch.as_tensor(rig.rest_vertices[face_ids], dtype=torch.float64),
        torch.as_tensor(chart_triangles, dtype=torch.float64),
        face_indices=torch.as_tensor(face_ids),
        normal_triangles=torch.as_tensor(oracle.normals[0][face_ids], dtype=torch.float64),
    )
    chart_points = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    slots = torch.arange(len(points))
    sample = torch_patch.sample(chart_points, slots)
    numpy_point, numpy_du, numpy_dv, numpy_normal = oracle.frame(chart_points.detach())
    assert float(torch.linalg.norm(sample.points_m.detach() - numpy_point, dim=1).max()) <= 0.00001
    assert float(angle_error_degrees(sample.normals.detach(), numpy_normal).max()) <= 0.05
    assert float(angle_error_degrees(sample.du.detach(), numpy_du).max()) <= 0.05
    assert float(angle_error_degrees(sample.dv.detach(), numpy_dv).max()) <= 0.05
    np.testing.assert_allclose(
        sample.metric.detach(),
        oracle.first_fundamental_form(chart_points.detach()),
        atol=1e-10,
    )

    immutable = (rig.topology_sha256, rig.identity_sha256, rig.surface_sha256, rig.catalog_sha256)
    loss = (sample.points_m.square().sum(dim=1) * torch.linspace(0.7, 1.3, len(points))).sum()
    require_gradient(loss, "seeded SOMA fixed patch")
    gradient = torch.autograd.grad(loss, chart_points)[0]
    step = 1e-7
    finite = torch.empty_like(gradient)
    for row in range(len(points)):
        for axis in range(2):
            delta = torch.zeros_like(chart_points)
            delta[row, axis] = step
            plus = torch_patch.sample(chart_points.detach() + delta, torch.as_tensor(slots)).points_m
            minus = torch_patch.sample(chart_points.detach() - delta, torch.as_tensor(slots)).points_m
            weights = torch.linspace(0.7, 1.3, len(points))
            finite[row, axis] = (
                (plus.square().sum(dim=1) * weights).sum() - (minus.square().sum(dim=1) * weights).sum()
            ) / (2 * step)
    errors = relative_gradient_error(gradient, finite)
    assert float(torch.quantile(errors, 0.5)) <= 0.01
    assert float(torch.quantile(errors, 0.95)) <= 0.05
    assert immutable == (rig.topology_sha256, rig.identity_sha256, rig.surface_sha256, rig.catalog_sha256)


def test_boundary_and_graph_breaks_are_named_refusals():
    patch = plane_patch()
    with pytest.raises(TorchPatchError, match="anchor_outside_domain"):
        patch.sample(torch.tensor([[0.041, 0.001]], dtype=torch.float64), torch.tensor([0]))
    with pytest.raises(TorchPatchError, match="gradient_graph_broken"):
        require_gradient(torch.tensor(1.0), "detached result")
