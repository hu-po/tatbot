"""The Rerun bridge's geometry (no rerun SDK needed) and its import."""
import numpy as np
import pytest
from tatbot_rerun import shapes


def test_import():
    import tatbot_rerun.node  # noqa: F401


def test_page_outlines_in_world():
    world_from_page = np.eye(4)
    world_from_page[:3, 3] = [0.3, -0.2, 0.05]
    page, clear = shapes.page_outlines(world_from_page, (0.100, 0.150), (0.062, 0.112))
    assert page.shape == (5, 3) and np.allclose(page[0], page[-1])
    assert np.allclose(page.min(0), [0.25, -0.275, 0.05]) and np.allclose(page.max(0), [0.35, -0.125, 0.05])
    assert np.allclose(clear.max(0) - clear.min(0), [0.062, 0.112, 0.0])


def test_program_strokes_follow_the_page_pose():
    rot = shapes.matrix([0.1, 0.0, 0.0], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg about z
    program = {"ops": [{"op": "stroke", "id": "s0", "points_m": [[0.01, 0.0], [0.01, 0.02]]},
                       {"op": "pause", "id": "p1"}, {"op": "stroke", "id": "s2", "points_m": []}]}
    strokes = shapes.program_strokes(program, rot)
    assert len(strokes) == 1
    assert np.allclose(strokes[0], [[0.1, 0.01, 0.0], [0.08, 0.01, 0.0]])


def test_transform_samples_normalize_their_quaternion_and_refuse_an_unset_rotation():
    quaternion = np.array([.1, -.2, .3, .4])
    np.testing.assert_allclose(shapes.matrix([.1, .2, .3], quaternion),
                               shapes.matrix([.1, .2, .3], -3*quaternion), atol=1e-15)
    with pytest.raises(ValueError, match='zero norm'):
        shapes.matrix([.1, .2, .3], [0, 0, 0, 0])


def test_tip_path_steps_and_chunks():
    path = shapes.TipPath(min_step_m=0.001, chunk=3)
    assert path.add([0, 0, 0]) and not path.add([0.0005, 0, 0])
    for x in (0.002, 0.004):
        assert path.add([x, 0, 0])
    assert path.index == 0 and path.array().shape == (3, 3)
    assert path.add([0.006, 0, 0])               # a full chunk closes; the next starts at its last point
    assert path.index == 1 and np.allclose(path.array()[:, 0], [0.004, 0.006])
    path.clear()
    assert path.index == 0 and path.array().shape == (0, 3)
