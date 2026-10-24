"""The station touch dips aim by: its ball height against the overhead fix, and a re-touch keeping the camera's height.
2026-10-15: a re-touch's own height, 5.4 mm under the palette camera's, put the 3RL through a cap's floor."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tatbot_motion import station

REPO = Path(__file__).resolve().parents[3]
PALETTE = REPO / "urdf/palette.urdf"
POSE = np.array([[-1.0, -0.028, 0.0, 0.289], [0.028, -1.0, 0.0, 0.327], [0.0, 0.0, 1.0, 0.030], [0, 0, 0, 1.0]])
ROTATION = np.diag([1.0, -1.0, -1.0])
TCP = np.array([-0.0005, -0.00697, 0.074495])


@pytest.fixture
def fix():
    return station.StationFix("right", POSE, (POSE @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3],
                              "2026-10-15T21:17:09Z", "overhead_tag", {"registration_sha256": "r" * 64})


def touch(fix, dz, **kw):
    """A touch with the ball dz under the fix in height: 2.4-2.8 mm with the camera's height (2026-10-14/15)."""
    return station.StationTouch("right", "lutin-3rl-bugpin", fix.ball + [0.0017, 0.009, -dz], ROTATION, TCP,
                                [0.0] * 7, fix, kw.pop("run", "ros-calib-test"), fix.measured_utc, **kw)


def place(t, fix):
    return station.touched_palette(t, fix, tool_id="lutin-3rl-bugpin", tcp_m=t.tcp_m, joint_offsets=t.joint_offsets,
                                   palette_urdf=PALETTE)


def test_a_dip_aims_by_a_ball_height_near_the_fix_and_refuses_one_far_from_it(fix):
    placed = place(touch(fix, 0.0026), fix)
    ball = (placed @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3]
    np.testing.assert_allclose(ball, touch(fix, 0.0026).ball, atol=1e-12)
    with pytest.raises(station.StaleStationError, match=r"-7\.9 mm from the overhead fix in height"):
        place(touch(fix, 0.0079), fix)


def test_the_overheads_registration_places_its_views_by_a_touch_whatever_its_height(fix):
    """tatbot_calib.overhead aims by the touch while the registration it makes is what stands the two heights apart
    (2026-10-22: 9 mm, the cube-fitted registration's tilt)."""
    far = touch(fix, -0.009)
    placed = station.touched_palette(far, fix, tool_id="lutin-3rl-bugpin", tcp_m=far.tcp_m,
                                     joint_offsets=far.joint_offsets, palette_urdf=PALETTE, check_height=False)
    np.testing.assert_allclose((placed @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3], far.ball, atol=1e-12)
    with pytest.raises(station.StaleStationError, match="in height"):
        place(far, fix)


def test_a_camera_measured_height_is_held_only_to_a_gross_bound(fix):
    """2026-10-16: the overhead hung lower read the ball 5.2 mm over the palette camera's height, which matched 10-14's;
    a touch-only height that far off is still refused."""
    camera = "the palette camera, a calib height run"
    place(touch(fix, 0.0054, height_source=camera), fix)
    with pytest.raises(station.StaleStationError, match=r"at most 8\.0"):
        place(touch(fix, 0.0085, height_source=camera), fix)
    with pytest.raises(station.StaleStationError, match=r"at most 4\.0"):
        place(touch(fix, 0.0054), fix)


def test_a_retouch_keeps_the_camera_height_of_the_touch_it_replaces(fix):
    camera = touch(fix, 0.0028, height_source="the palette camera, 20261014T154230Z")
    retouch = replace(touch(fix, 0.0079, run="retouch"), ball=touch(fix, 0.0079).ball + [0.0007, 0.0011, 0.0])
    kept, note = station.carry_height(retouch, camera)
    assert kept.ball[2] == camera.ball[2] and kept.ball[0] == retouch.ball[0] and kept.ball[1] == retouch.ball[1]
    assert kept.height_source.endswith("carried over touch retouch") and "kept the camera's ball height" in note
    place(kept, fix)   # the carried height aims; the touch's own would have been refused
    moved = replace(retouch, ball=retouch.ball + [0.004, 0.0, 0.0])
    same, note = station.carry_height(moved, camera)
    assert same is moved and "does not carry" in note and "calib height" in note
    other = replace(retouch, tool_id="lutin-5rl-mast")
    assert station.carry_height(other, camera)[0] is other
    assert station.carry_height(retouch, touch(fix, 0.0028)) == (retouch, "")
    assert station.carry_height(retouch, None) == (retouch, "")
