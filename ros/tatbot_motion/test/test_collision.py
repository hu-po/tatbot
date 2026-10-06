"""The two arms as the overhead camera placed them on the demo table (2026-09-29, evening), and the view the pink arm
planned there, whose own wrist cables caught on the landed blue arm: the model puts the pink tool 87 mm from the blue laser
prop at it, and the guard refuses the way to it while it lets the arm back away from it."""
from __future__ import annotations

import re

import numpy as np
import pytest

pytest.importorskip("coal")
from tatbot_motion import load_motion  # noqa: E402
from tatbot_motion.collision import Guard, _capsules  # noqa: E402

# world <- base for each arm (the D555's colour optical frame is the world), from their registrations that evening
REGISTRATIONS = {
    "right": [[-0.879673, -0.475165, -0.019833, 0.310598], [-0.408991, 0.777131, -0.478325, -0.043666],
              [0.242696, -0.412658, -0.877959, 0.847485], [0.0, 0.0, 0.0, 1.0]],
    "left": [[0.462902, -0.886125, 0.022442, -0.313322], [-0.755237, -0.407529, -0.513358, 0.324075],
             [0.464045, 0.220686, -0.857881, 0.634123], [0.0, 0.0, 0.0, 1.0]]}
LANDED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0])
SNAG = np.array([0.718, 1.8, 1.905, -1.229, -0.712, 0.995, 0.002])     # the pink mount view that met the cable
CLEAR = np.array([0.947, 1.091, 0.991, -0.453, -0.311, 1.384, 0.002])  # a later view that did not


@pytest.fixture(scope="module")
def guard():
    return Guard.from_registrations(None, REGISTRATIONS, load_motion(), {"right": LANDED, "left": LANDED})


def _way(a, b, n=200):
    return np.linspace(a, b, n)


def test_the_snag_view_is_inside_the_margin_and_a_later_view_is_not(guard):
    scene = guard.scene
    assert {"right/tool_0", "left/tool_0"} <= {body.name for body in scene.geom.geometryObjects}
    at_rest = scene.clearance({"right": LANDED, "left": LANDED})
    snag = scene.clearance({"right": SNAG, "left": LANDED})
    clear = scene.clearance({"right": CLEAR, "left": LANDED})
    assert at_rest.raw_m > 0.30
    assert 0.07 < snag.raw_m < 0.12 and snag.distance_m < guard.margin_m, snag
    assert {snag.body_a.split("/")[1][:4], snag.body_b.split("/")[1][:4]} == {"tool"}   # the pen against the laser
    assert clear.distance_m > guard.margin_m, clear


def test_wrists_and_tools_are_bubbled_out_and_a_pair_keeps_the_larger_bubble(guard):
    """The cable loops hang about the wrist and the tool: those bodies carry the loops' reach, the rest none. A pair
    keeps the larger of its two bubbles, not their sum: two wrists that pass a loop apart are clear (the later view,
    143 mm, did not catch)."""
    scene, reach = guard.scene, load_motion()["collision"]["wrist_m"]
    bodies = scene.geom.geometryObjects
    padded = {re.sub(r"_\d+$", "", body.name.split("/")[1]) for body, pad in zip(bodies, scene.pad, strict=True)
              if pad > 0}
    assert padded == {"link_5", "link_6", "carriage_left", "carriage_right", "realsense_link", "realsense_mount_d405",
                      "tool"}
    snag = scene.clearance({"right": SNAG, "left": LANDED})
    assert snag.distance_m == pytest.approx(snag.raw_m - reach)             # two bubbled tools: one reach, not two
    clear = scene.clearance({"right": CLEAR, "left": LANDED})
    assert clear.raw_m < 2 * reach + guard.margin_m < clear.raw_m + reach   # two reaches would have refused it


def test_the_guard_refuses_the_way_to_the_snag_and_lets_the_arm_back_away(guard):
    """A straight joint move from rest to the snag's view dips to 77 mm, nearer than the view itself (83 mm): the
    way is refused, and so is the same line run back, which from the view gets nearer first. Along the stretch where
    the clearance falls steadily into the margin, entering is refused and leaving is not."""
    why = guard.refusal("right", _way(LANDED, SNAG), {})
    # 77 mm with the ballpoint's body, 80 with the 3RL's slimmer cartridge: the fitted tool's capsules set it
    assert why is not None and re.search(r"comes (7\d|8[0-2]) mm", why) and "assumed landed" in why and "not sent" in why
    assert guard.refusal("right", _way(SNAG, LANDED), {}) is not None
    assert guard.refusal("right", _way(CLEAR, CLEAR, 2), {}) is None       # holding at the later view
    rows = _way(LANDED, SNAG)
    gaps = np.array([guard.scene.clearance({"right": q, "left": LANDED}).distance_m for q in rows])
    inside = int(np.argmax(gaps < guard.margin_m)) + 5
    assert np.all(np.diff(gaps[: inside + 1]) < 0)                          # a steady fall into the margin
    assert guard.refusal("right", rows[: inside + 1], {}) is not None       # entering it: refused
    assert guard.refusal("right", rows[inside::-1], {}) is None             # already inside and leaving: sent


def test_the_other_arm_is_where_the_stack_measures_it(guard):
    joints, sources = guard.others("right", {"left": SNAG})
    assert np.allclose(joints["left"], SNAG) and sources == ["the left arm as measured"]
    joints, sources = guard.others("right", {})
    assert np.allclose(joints["left"], LANDED) and sources == ["the left arm assumed landed"]


def test_without_both_registrations_nothing_is_checked(tmp_path):
    missing = {"registration": {"right": str(tmp_path / "none.json")}}
    guard, text = Guard.from_stack(None, missing, load_motion())
    assert guard is None and "no registration for the left and right arm" in text


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
    for arm in ("right", "left"):
        cube = next(body for body in scene.geom.geometryObjects if body.name == f"{arm}/wrist_cube")
        assert 0.060 < 2 * float(cube.geometry.halfSide[0]) < 0.080
        pairs = {tuple(sorted(scene.geom.geometryObjects[k].name.split("/")[1] for k in pair))
                 for pair in scene.self_pairs[arm]}
        assert {("link_2_0", "wrist_cube"), ("link_3_0", "wrist_cube"), ("link_5_0", "wrist_cube")} <= pairs
        assert ("link_6_0", "wrist_cube") not in pairs and ("link_3_0", "link_5_0") not in pairs
    cube = {i for i, body in enumerate(scene.geom.geometryObjects) if body.name.endswith("wrist_cube")}
    assert not any(pair.first in cube or pair.second in cube for pair in scene.geom.collisionPairs)   # own arm only


def test_the_way_that_folded_the_cube_into_the_arm_is_refused_and_good_views_are_not(guard):
    hit = guard.scene.self_clearance("right", HIT)
    assert hit.distance_m < -0.005 and {hit.body_a.split("/")[1], hit.body_b.split("/")[1]} == {"link_2_0", "wrist_cube"}
    for start in (LANDED, *GOOD_VIEWS):
        for end in (HIT, HIT_TARGET):
            why = guard.refusal("right", _way(start, end), {})
            assert why is not None and "its own" in why and "wrist_cube" in why, why
    for view in GOOD_VIEWS:
        assert guard.scene.self_clearance("right", view).raw_m > guard.self_margin_m
        assert guard.refusal("right", _way(LANDED, view), {}) is None
    # the landed pose folds the arm onto itself in the model (link_2 and link_5 overlap): ways leave it and reach it
    assert guard.scene.self_clearance("right", LANDED).raw_m < guard.self_margin_m
    for view in GOOD_VIEWS:
        assert guard.refusal("right", _way(view, LANDED), {}) is None


def test_one_pose_says_how_far_the_arm_stands_from_itself(guard):
    """Guard.self_gap: for a view to choose (tatbot_session inspect.aim), one arm's joints alone, m; negative inside."""
    assert guard.self_gap("right", HIT) < 0.0
    assert all(guard.self_gap("right", view) > 0.0 for view in GOOD_VIEWS)
    assert guard.self_gap("right", LANDED) >= 0.0          # its own rest is no collision


AWAY = LANDED + [-1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # the blue arm turned away from the pink arm's views


def test_a_way_in_flight_is_reserved_and_the_other_arm_waits_for_it(guard):
    """2026-09-30: the two arms move at once in one stack. Where the blue arm stands at dispatch says nothing of
    where its goal takes it next: its way in flight is reserved, the pink way that would meet it waits (only a way
    in flight blocks it, and it ends), and once released the pink way goes."""
    guard.release("left")
    guard.release("right")
    away = guard.scene.clearance({"right": SNAG, "left": AWAY})
    assert away.distance_m > guard.margin_m, away                   # the snag view is clear of the blue arm turned away
    assert guard.reserve("left", _way(AWAY, LANDED), {"right": LANDED}) == (None, False)   # blue goes back to rest
    why, waits = guard.reserve("right", _way(LANDED, SNAG), {"left": AWAY})
    assert why is not None and waits and "in flight" in why and "waiting" in why
    # every row against every row: the later view, clear of the blue arm standing at rest, is not clear of its whole
    # way back there, and waits too; holding at rest is clear
    assert guard.reserve("right", _way(LANDED, CLEAR), {"left": AWAY})[1]
    assert guard.reserve("right", _way(LANDED, LANDED, 2), {"left": AWAY}) == (None, False)
    guard.release("right")
    guard.release("left")
    assert guard.reserve("right", _way(LANDED, SNAG), {"left": AWAY}) == (None, False)     # released: it goes
    guard.release("right")


def test_a_landing_holds_its_way_and_a_standing_refusal_does_not_wait(guard):
    guard.release("left")
    guard.hold_way("left", _way(AWAY, LANDED))
    why, waits = guard.reserve("right", _way(LANDED, SNAG), {"left": AWAY})
    assert why is not None and waits
    guard.release("left")
    why, waits = guard.reserve("right", _way(LANDED, SNAG), {"left": LANDED})   # the blue arm standing at rest
    assert why is not None and not waits and "not sent" in why


def test_a_long_way_is_sampled_to_a_bounded_sweep_with_its_slack(guard):
    rows, slack = guard._sweep(_way(LANDED, LANDED + [3.0, 0, 0, 0, 0, 0, 0], 2000))
    assert len(rows) <= 60 and slack > 0.0 and np.allclose(rows[-1][0], 3.0)
    rows, slack = guard._sweep(_way(LANDED, LANDED + [0.2, 0, 0, 0, 0, 0, 0], 200))
    assert slack < 1e-4 and len(rows) <= 22        # rows 10 mrad apart: the step path_clearance takes


def _body_world(guard, arm, q, name):
    import pinocchio as pin

    scene = guard.scene
    scene._q[scene._iq[arm]] = q[: len(scene._iq[arm])]
    pin.updateGeometryPlacements(scene.model, scene.data, scene.geom, scene.gdata, scene._q)
    i = next(k for k, body in enumerate(scene.geom.geometryObjects) if body.name == name)
    return np.array(scene.gdata.oMg[i].translation)


def test_a_way_into_the_palette_another_arms_run_holds_waits_and_its_holder_goes(guard):
    """2026-09-30: while the blue arm's run holds the palette (tatbot_session.lease), a pink way into its zone waits
    for the lease to end; the holder's own ways, and pink ways that keep out, go."""
    guard.release("left")
    guard.release("right")
    at = _body_world(guard, "right", SNAG, "right/tool_0")
    pose = np.eye(4)
    pose[:3, 3] = at - [0.0, 0.0, 0.05]
    zone = {"arm": "left", "run": "r1", "world_from_zone": pose.tolist(), "radius_m": 0.03, "height_m": 0.10}
    guard.zone_source = lambda: zone
    try:
        gap = guard.scene.zone_clearance("right", [SNAG], pose, 0.03, 0.10)
        assert gap.raw_m <= 0.0 and gap.body_a.startswith("right/")      # inside it: coal gives the depth, negative
        why, waits = guard.reserve("right", _way(AWAY, SNAG), {"left": AWAY})
        assert why is not None and waits and "palette" in why and "left arm's run r1" in why
        assert guard.reserve("right", _way(LANDED, LANDED, 2), {"left": AWAY}) == (None, False)   # keeping out
        guard.release("right")
        assert guard.reserve("left", _way(AWAY, AWAY, 2), {"right": LANDED}) == (None, False)     # its holder
        guard.release("left")
        zone["arm"] = "right"                                  # pink's own run holds it: pink goes
        assert guard.reserve("right", _way(AWAY, SNAG), {"left": AWAY}) == (None, False)
        guard.release("right")
    finally:
        guard.zone_source = None


def test_an_arms_bodies_are_measured_to_fixed_parts_its_pads_counted_its_tool_left_out(guard):
    """Scene.parts_clearance: the station's parts as spheres and posts in the world; a body's pad makes it larger;
    the tool's own approach is its halo's check. 2026-09-30: the blue gripper's pad hung over the roof tag's box."""
    import pinocchio as pin

    scene = guard.scene
    scene._q[scene._iq["left"]] = LANDED
    pin.updateGeometryPlacements(scene.model, scene.data, scene.geom, scene.gdata, scene._q)
    carriage = next(i for i, b in enumerate(scene.geom.geometryObjects) if b.name == "left/carriage_right_0")
    tool = next(i for i, b in enumerate(scene.geom.geometryObjects) if b.name.startswith("left/tool_"))
    at = np.asarray(scene.gdata.oMg[carriage].translation)
    parts = [("box", "sphere", at, 0.002)]
    bare = scene.parts_clearance("left", [LANDED], parts)
    padded = scene.parts_clearance("left", [LANDED], parts, {"carriage": 0.03})
    assert padded.distance_m == pytest.approx(bare.distance_m - 0.03) and padded.body_b == "box"
    assert padded.body_a.split("/")[1].startswith("carriage")
    at_tool = np.asarray(scene.gdata.oMg[tool].translation)
    assert scene.parts_clearance("left", [LANDED], [("x", "sphere", at_tool, 0.001)]).body_a.split("/")[1][:4] != "tool"
    post = scene.parts_clearance("left", [LANDED], [("p", "post", at + [0, 0, 0.3], 0.01, 0.05, [0, 0, 1])])
    assert np.isfinite(post.distance_m)
