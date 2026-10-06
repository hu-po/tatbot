"""Surface addresses survive texture seams and chart clipping."""

import numpy as np
import pytest
from tatbot_sim.inkmap.rig import BodyRigError, PosedBody
from tatbot_sim.inkmap.triangle_chart import TriangleChart


def chart():
    body = PosedBody("test", "articulated", "a" * 64,
                     np.array([[[0, 0, 0], [1, 0, 1], [0, 1, 0]],
                               [[1, 0, 1], [1, 1, 2], [0, 1, 0]]], dtype=float),
                     np.eye(4), ("SOMA",), np.array([0]), np.array([2]))
    return TriangleChart(body, np.array([0, 1]), body.vertices[..., :2])


def test_chart_addresses_resolve_against_the_posed_surface():
    atlas = chart()
    points = np.array([[.2, .1], [.8, .9], [.5, .5]])
    faces, bary = atlas.locate(points)
    np.testing.assert_allclose(atlas.body.points(faces, bary)[..., :2], points)
    assert atlas.body.points(faces, bary)[1, 2] > 1
    with pytest.raises(BodyRigError, match="outside skin"):
        atlas.points([[1.1, .5]])
    with pytest.raises(BodyRigError, match="normalized"):
        atlas.body.point(0, [np.nan, 0, 1])
    with pytest.raises(BodyRigError, match="non-integer"):
        TriangleChart(atlas.body, np.array([0.5, 1]), atlas.triangles_uv)
    with pytest.raises(BodyRigError, match="out of range"):
        TriangleChart(atlas.body, np.array([0, 2]), atlas.triangles_uv)


def test_clipped_render_triangles_keep_canonical_addresses():
    atlas = chart()
    faces, bary, uv = atlas.clipped(((.15, .85), (.2, .8)))
    points = atlas.body.points(np.broadcast_to(faces[:, None], bary.shape[:-1]), bary)
    np.testing.assert_allclose(points[..., :2], uv, atol=1e-12)
    assert np.all(bary >= 0)
    np.testing.assert_allclose(bary.sum(axis=-1), 1)
    assert np.all(uv >= np.array([.15, .2]) - 1e-12)
    assert np.all(uv <= np.array([.85, .8]) + 1e-12)
