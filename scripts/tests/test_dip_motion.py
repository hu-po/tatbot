"""The cap entry geometry both stacks now share.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_dip_motion.py

Until 2026-09-09 the arm placed the hover and the plunge along the palette's
own axis while the simulator assumed world Z. They agreed while the rack was
level and nothing compared them, so the simulator's reference sat 23 degrees
off the axis it commanded at every cap and the charge was credited anyway.
These pin the convention that removed the second implementation.
"""

from __future__ import annotations

import dip_motion  # noqa: E402
import numpy as np
import pytest


def test_the_tool_hovers_against_the_axis_and_plunges_along_it():
    poses = dip_motion.dip_poses([0.2, -0.05, 0.06], [0, 0, -1], hover_m=0.02, plunge_m=0.003)
    assert np.allclose(poses.above, [0.2, -0.05, 0.08]), "hover is above a level cap"
    assert np.allclose(poses.bottom, [0.2, -0.05, 0.057]), "and the plunge is inside it"
    assert np.allclose(poses.transit, poses.above), "no lift asked for, none added"
    assert np.allclose(poses.tool_axis, [0, 0, -1])
    assert np.allclose(poses.outward_normal, [0, 0, 1]), "the cap mouth faces up"


def test_a_tilted_rack_moves_both_points_off_world_z():
    """The case the two implementations could not have agreed on. A cap on a
    tilted rack is entered along its own normal, not straight down."""
    axis = np.array([0.3, 0.0, -1.0])
    axis /= np.linalg.norm(axis)
    poses = dip_motion.dip_poses([0.2, 0.0, 0.06], axis, hover_m=0.02, plunge_m=0.004)
    assert poses.above[0] < 0.2, "the hover leans back over the rim"
    assert poses.bottom[0] > 0.2, "and the plunge follows the cap in"
    assert np.linalg.norm(poses.above - poses.rim) == pytest.approx(0.02)
    assert np.linalg.norm(poses.bottom - poses.rim) == pytest.approx(0.004)
    # both offsets are purely along the axis: no residual sideways component
    for point, expected in ((poses.above, -0.02), (poses.bottom, 0.004)):
        delta = point - poses.rim
        assert np.allclose(delta - np.dot(delta, poses.axis) * poses.axis, 0, atol=1e-12)
        assert np.dot(delta, poses.axis) == pytest.approx(expected)


def test_a_transit_lift_is_added_above_the_hover():
    poses = dip_motion.dip_poses([0.2, 0.0, 0.06], [0, 0, -1],
                                 hover_m=0.02, plunge_m=0.003, lift_m=0.03)
    assert np.allclose(poses.transit - poses.above, [0, 0, 0.03])


def test_a_level_rack_enters_straight_down():
    assert np.allclose(dip_motion.palette_entry_axis(np.eye(4)), [0, 0, -1])


def test_the_entry_axis_follows_the_rig_transform():
    """A rack rotated in the rig is entered along its own -Z, carried into the
    arm base frame — which is the whole reason this is not a constant."""
    transform = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    axis = dip_motion.palette_entry_axis(transform)
    assert np.allclose(axis, transform[:3, :3] @ [0, 0, -1])
    assert np.linalg.norm(axis) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [
    {"hover_m": -0.001}, {"plunge_m": float("nan")}, {"lift_m": -1.0},
])
def test_a_nonsense_distance_is_refused(bad):
    kwargs = {"hover_m": 0.02, "plunge_m": 0.003} | bad
    with pytest.raises(ValueError):
        dip_motion.dip_poses([0.2, 0.0, 0.06], [0, 0, -1], **kwargs)


def test_an_axis_without_direction_is_refused():
    with pytest.raises(ValueError, match="no direction"):
        dip_motion.dip_poses([0.2, 0.0, 0.06], [0, 0, 0], hover_m=0.02, plunge_m=0.003)
