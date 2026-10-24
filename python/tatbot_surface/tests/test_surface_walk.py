"""Unfolding, the design-footprint check and trace compilation on the reference rig."""

from __future__ import annotations

import numpy as np
import pytest
from tatbot_surface.rig import load_body_rig
from tatbot_surface.surface_trace import (
    ChartSeam,
    SurfaceAnchor,
    SurfaceTraceError,
    UnfoldedPatch,
    anchor_frame,
    anchors_to_points,
    clip_to_box,
    compile_surface_trace,
    triangle_area_in_box,
    unfold_body_patch,
    validate_patch_footprint,
)

# The right forearm's region default anchor (config: regions.json "forearm:right").
FOREARM_FACE = 26_998
CENTRE = (1 / 3, 1 / 3, 1 / 3)


def test_box_clip_keeps_inside_corners_and_interpolates_weights():
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    inside = clip_to_box(list(zip(triangle, np.eye(3), strict=True)), ((-1, 3), (-1, 3)))
    assert len(inside) == 3
    np.testing.assert_array_equal([point for point, _ in inside], triangle)
    clipped = clip_to_box(list(zip(triangle, np.eye(3), strict=True)), ((0.5, 1.0), (0.25, 0.5)))
    for point, weights in clipped:
        np.testing.assert_allclose(weights @ triangle, point, atol=1e-15)
        assert 0.5 - 1e-15 <= point[0] <= 1.0 + 1e-15 and 0.25 - 1e-15 <= point[1] <= 0.5 + 1e-15
    assert clip_to_box(list(zip(triangle, np.eye(3), strict=True)), ((5, 6), (5, 6))) == []


@pytest.mark.parametrize(("bounds", "area"), [
    (((0.0, 1.0), (0.0, 1.0)), 1.0),       # the unit square lies under the hypotenuse x + y = 2
    (((-1.0, 0.5), (-1.0, 0.5)), 0.25),
    (((1.0, 3.0), (1.0, 3.0)), 0.0),       # touches the triangle only at (1, 1)
    (((-5.0, 5.0), (-5.0, 5.0)), 2.0),     # whole triangle
    (((1.5, 2.0), (0.0, 0.25)), 0.09375),  # x in [1.5, 2], y <= min(0.25, 2 - x)
])
def test_triangle_area_in_box_matches_hand_computed_areas(bounds, area):
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    assert triangle_area_in_box(triangle, bounds) == pytest.approx(area, abs=1e-15)
    assert triangle_area_in_box(triangle[::-1], bounds) == pytest.approx(area, abs=1e-15)


def test_anchor_frame_is_right_handed_with_normal_out_of_the_skin():
    rig = load_body_rig()
    vertices = np.asarray(rig.rest_vertices, dtype=np.float64)
    point, normal, u, v = anchor_frame(vertices, FOREARM_FACE, CENTRE, rotation_rad=0.3)
    frame = np.stack([u, v, normal])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(u, v), normal, atol=1e-12)
    triangle = vertices[FOREARM_FACE]
    np.testing.assert_allclose(point, np.mean(triangle, axis=0), atol=1e-12)
    face_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    assert np.dot(face_normal, normal) > 0


def _square_chart(**fields) -> UnfoldedPatch:
    """Two chart triangles tiling [-1, 1]^2 m, faces 0 and 1."""
    triangles = np.array([[[-1, -1], [1, -1], [1, 1]], [[-1, -1], [1, 1], [-1, 1]]], dtype=float)
    return UnfoldedPatch(seed_face=0, body_first_face=0, seed_triangle_uv=triangles[0],
                         mesh_vertices=np.zeros((2, 3, 3)), mesh_keys=[], face_indices=np.array([0, 1]),
                         triangles_uv=triangles, adjacent={0: {1}, 1: {0}}, **fields)


def test_footprint_accepts_a_forearm_design():
    rig = load_body_rig()
    patch = unfold_body_patch(rig, SurfaceAnchor(FOREARM_FACE, CENTRE), 0.0, 0.012)
    validate_patch_footprint(patch, 0.010, 0.010)
    with pytest.raises(SurfaceTraceError, match="design size must be positive"):
        validate_patch_footprint(patch, 0.0, 0.010)


def test_footprint_refuses_each_unflattenable_chart():
    validate_patch_footprint(_square_chart(), 2.0, 2.0)
    with pytest.raises(SurfaceTraceError, match="covers 50.0% of design area"):
        validate_patch_footprint(_square_chart(), 2.0, 4.0)  # the square is half of a 2 x 4 m design
    with pytest.raises(SurfaceTraceError, match="open surface edge at face 1"):
        validate_patch_footprint(_square_chart(boundary_faces=(1,)), 1.0, 1.0)
    seam = ChartSeam(face=0, neighbor=1, mismatch_m=1.0)
    with pytest.raises(SurfaceTraceError, match="faces 0 and 1 develop 1000.0 mm apart"):
        validate_patch_footprint(_square_chart(seams=(seam,)), 1.0, 1.0)
    far = _square_chart()
    far.triangles_uv = far.triangles_uv + 10.0
    with pytest.raises(SurfaceTraceError, match="no connected surface triangles"):
        validate_patch_footprint(far, 1.0, 1.0)


def test_a_compiled_trace_keeps_its_length_on_the_rest_surface_and_is_deterministic():
    rig = load_body_rig()
    placement = {"anchor": {"face": FOREARM_FACE, "barycentric": list(CENTRE)},
                 "rotation_rad": 0.0, "size_mm": [12, 4]}
    stroke = np.array([[-0.005, 0.0], [0.005, 0.0]])
    trace = compile_surface_trace(rig, placement, [stroke])
    again = compile_surface_trace(rig, placement, [stroke])
    assert trace.sha256 == again.sha256 and trace.strokes == again.strokes
    (points,) = anchors_to_points(rig.rest_vertices, trace)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    assert length == pytest.approx(0.010, rel=0.02)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.max() <= 5e-4
    as_dict = trace.as_dict()
    assert as_dict["compiler"] == "tatbot_sim.surface_trace" and as_dict["compiler_version"] == 2
