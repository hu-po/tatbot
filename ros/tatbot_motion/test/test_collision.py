"""The pink arm as the overhead camera placed it on the demo table (2026-09-29, evening): its bodies, its tool and its
wrist cameras in one scene, the ways that fold it onto itself refused, and its bodies measured to the station's parts.
The blue arm is the teleop leader alone and is not modelled."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("coal")
from tatbot_motion import load_motion  # noqa: E402
from tatbot_motion.collision import Guard, _capsules  # noqa: E402

# world <- base (the D555's colour optical frame is the world), from its registration that evening
REGISTRATION = [[-0.879673, -0.475165, -0.019833, 0.310598], [-0.408991, 0.777131, -0.478325, -0.043666],
                [0.242696, -0.412658, -0.877959, 0.847485], [0.0, 0.0, 0.0, 1.0]]
LANDED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0])


@pytest.fixture(scope="module")
def guard():
    return Guard.from_registrations(None, {"right": REGISTRATION}, load_motion(), {"right": LANDED})


def _way(a, b, n=200):
    return np.linspace(a, b, n)


def test_without_a_registration_nothing_is_checked(tmp_path):
    missing = {"registration": {"right": str(tmp_path / "none.json")}}
    guard, text = Guard.from_stack(None, missing, load_motion())
    assert guard is None and "no arm registration" in text


def test_the_registered_arm_its_tool_and_both_wrist_cameras_are_in_the_scene(tmp_path):
    path = tmp_path / "right.json"
    path.write_text(json.dumps({"world_from_arm_base": REGISTRATION}))
    guard, text = Guard.from_stack(None, {"registration": {"right": str(path)}}, load_motion())
    assert guard.scene.arms == ["right"] and "kept off itself" in text
    bodies = {body.name.split("/")[1] for body in guard.scene.geom.geometryObjects}
    assert {"tool_0", "realsense_link_0", "realsense_lower_link_0", "wrist_cube"} <= bodies
    assert not len(guard.scene.geom.collisionPairs)   # nothing across arms


def test_a_profile_becomes_capsules_as_wide_as_each_segments_wider_end():
    """A step (two radii at one z) leaves no zero-length capsule, and the segment above it is as wide as the step."""
    got = _capsules([(0.0, 0.01), (0.02, 0.005), (0.02, 0.004), (0.05, 0.001)])
    assert np.allclose(got, [(0.01, 0.01, 0.01), (0.035, 0.015, 0.005)])


# 2026-09-30, a ros-draw run: at its third post-draw inspection view the pink arm folded its wrist
# tag cube into its own upper arm, from the flight recorder: the wrist commanded to -2.351 rad, stopped at -1.782.
HIT = np.array([-0.06, 1.502, 1.211, -1.352, -0.342, -1.782, 0.002])
HIT_TARGET = np.array([-0.06, 1.502, 1.211, -1.352, -0.285, -2.351, 0.002])
GOOD_VIEWS = [np.array(q) for q in ([-0.028, 0.711, 0.179, -0.195, -0.282, 1.266, 0.002],   # 45ae's first two views
                                    [0.135, 1.342, 0.891, -0.811, -0.078, 0.988, 0.002])]


def test_the_wrist_cube_is_a_box_from_its_tags_and_checked_against_the_links_it_folds_onto(guard):
    """The URDF gives the wrist tag cube no collision: the scene builds it from the three tags' calibrated frames, and
    keeps it off every link from link_5 up, not the parts it rides with. Links within two joints are not paired:
    their hulls overlap about the compact wrist in every pose."""
    scene = guard.scene
    cube = next(body for body in scene.geom.geometryObjects if body.name == "right/wrist_cube")
    assert 0.060 < 2 * float(cube.geometry.halfSide[0]) < 0.080
    pairs = {tuple(sorted(scene.geom.geometryObjects[k].name.split("/")[1] for k in pair))
             for pair in scene.self_pairs["right"]}
    assert {("link_2_0", "wrist_cube"), ("link_3_0", "wrist_cube"), ("link_5_0", "wrist_cube")} <= pairs
    assert ("link_6_0", "wrist_cube") not in pairs and ("link_3_0", "link_5_0") not in pairs


def test_the_way_that_folded_the_cube_into_the_arm_is_refused_and_good_views_are_not(guard):
    hit = guard.scene.self_clearance("right", HIT)
    assert hit.distance_m < -0.005 and {hit.body_a.split("/")[1], hit.body_b.split("/")[1]} == {"link_2_0", "wrist_cube"}
    for start in (LANDED, *GOOD_VIEWS):
        for end in (HIT, HIT_TARGET):
            why = guard.refusal("right", _way(start, end))
            assert why is not None and "its own" in why and "wrist_cube" in why, why
    for view in GOOD_VIEWS:
        assert guard.scene.self_clearance("right", view).raw_m > guard.self_margin_m
        assert guard.refusal("right", _way(LANDED, view)) is None
    # the landed pose folds the arm onto itself in the model (link_2 and link_5 overlap): ways leave it and reach it
    assert guard.scene.self_clearance("right", LANDED).raw_m < guard.self_margin_m
    for view in GOOD_VIEWS:
        assert guard.refusal("right", _way(view, LANDED)) is None


def test_one_pose_says_how_far_the_arm_stands_from_itself(guard):
    """Guard.self_gap: for a view to choose (tatbot_session inspect.aim), one arm's joints alone, m; negative inside."""
    assert guard.self_gap("right", HIT) < 0.0
    assert all(guard.self_gap("right", view) > 0.0 for view in GOOD_VIEWS)
    assert guard.self_gap("right", LANDED) >= 0.0          # its own rest is no collision


def test_an_arms_bodies_are_measured_to_fixed_parts_its_pads_counted_its_tool_left_out(guard):
    """Scene.parts_clearance: the station's parts as spheres and posts in the world; a body's pad makes it larger;
    the tool's own approach is its halo's check. 2026-09-30: a gripper's pad hung over the roof tag's box."""
    import pinocchio as pin

    scene = guard.scene
    scene._q[scene._iq["right"]] = LANDED
    pin.updateGeometryPlacements(scene.model, scene.data, scene.geom, scene.gdata, scene._q)
    carriage = next(i for i, b in enumerate(scene.geom.geometryObjects) if b.name == "right/carriage_left_0")
    tool = next(i for i, b in enumerate(scene.geom.geometryObjects) if b.name.startswith("right/tool_"))
    at = np.asarray(scene.gdata.oMg[carriage].translation)
    parts = [("box", "sphere", at, 0.002)]
    bare = scene.parts_clearance("right", [LANDED], parts)
    padded = scene.parts_clearance("right", [LANDED], parts, {"carriage": 0.03})
    assert padded.distance_m == pytest.approx(bare.distance_m - 0.03) and padded.body_b == "box"
    assert padded.body_a.split("/")[1].startswith("carriage")
    at_tool = np.asarray(scene.gdata.oMg[tool].translation)
    assert scene.parts_clearance("right", [LANDED], [("x", "sphere", at_tool, 0.001)]).body_a.split("/")[1][:4] != "tool"
    post = scene.parts_clearance("right", [LANDED], [("p", "post", at + [0, 0, 0.3], 0.01, 0.05, [0, 0, 1])])
    assert np.isfinite(post.distance_m)
