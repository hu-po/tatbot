"""The tip fit on synthetic contacts: the touch plans put the wall where they say, opposed side pairs at upright yaws
recover the tip across its axis and the ball with its length and axis held, a side radius per pair axis takes the
arm's give, and leave-one-attitude-out predicts held-out contacts."""
from __future__ import annotations

import math

import numpy as np
import pytest
from tatbot_calib import halo, solve, station

TIP = np.array([0.0046, -0.0057, 0.1325])            # the end-plane centre in the tool mount
BALL = np.array([0.220, -0.228, 0.120])
SIDE = 0.0089
HALO = halo.Halo(0.0090, 0.0077, 0.00835, 0.006)   # the laser's nose, as its datasheet has it
YAWS = (0.0, 30.0, 60.0, 90.0, -30.0)
HELD = {"sigma_tip": np.array([0.010, 0.010, 1e-7]), "sigma_axis": 1e-7, "axis0": [0.0, 0.0, 1.0]}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def pose_frame(q):
    """A synthetic arm whose joints are the mount pose: xyz, then URDF roll-pitch-yaw."""
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rpy_matrix(*q[3:6]), q[:3]
    return out


def pose_joints(mount):
    r = mount[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return np.array([*mount[:3, 3], math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0]), 0.0])


def _mount(a_w, rotation):
    mount = np.eye(4)
    mount[:3, :3], mount[:3, 3] = rotation, a_w - rotation @ TIP
    return mount


def side_contacts(rng, give=lambda yaw, axis: 0.0, noise=5e-5):
    """Opposed pairs along fixed base directions at each upright yaw, the arm giving `give` before the trip."""
    contacts = []
    for yaw in YAWS:
        rotation = halo.tool_pose(np.zeros(3), math.radians(yaw))[:3, :3]
        u = rotation[:, 2]
        d, p = halo._across(u, np.array([1.0, 0.0, 0.0]))
        for axis, direction in (("d", d), ("d", -d), ("p", p), ("p", -p)):
            a_w = BALL - (SIDE - give(yaw, axis) + rng.normal(0, noise)) * direction + HALO.side_height_m * u
            contacts.append(solve.Contact("side", pose_joints(_mount(a_w, rotation)), f"{yaw:.0f}", axis))
    return contacts


def test_touch_plans_put_the_wall_where_their_contact_will_be():
    rotation = halo.tool_pose(np.zeros(3), math.radians(30.0))[:3, :3]
    for plan in halo.side_touches(BALL, rotation, np.array([1.0, 0.0, 0.0]), HALO, SIDE):
        end = plan.start[:3, 3] + plan.prior_m * plan.direction          # the end plane's centre at contact
        rel = BALL - end
        axis = rotation[:, 2]
        assert np.linalg.norm(np.cross(rel, axis)) == pytest.approx(SIDE)
        assert rel @ axis == pytest.approx(-HALO.side_height_m)           # the ball 6 mm up the wall
        assert abs(plan.direction @ axis) < 1e-12


def test_upright_yaws_recover_the_tip_across_its_axis_and_the_ball():
    contacts = side_contacts(np.random.default_rng(3))
    got = solve.fit(contacts, pose_frame, tip0=TIP + [0.002, -0.003, 0.0], ball0=BALL + [0.004, -0.003, 0.002],
                    side0=0.0093, **HELD)
    assert np.linalg.norm(got.tip[:2] - TIP[:2]) < 1e-4                  # across the axis: 0.1 mm
    assert got.tip[2] == pytest.approx(TIP[2], abs=1e-6)                 # along it: held
    assert np.linalg.norm(got.ball[:2] - BALL[:2]) < 2e-4 and got.side_radius == pytest.approx(SIDE, abs=5e-5)
    held = solve.leave_one_group_out(contacts, pose_frame, tip0=TIP, ball0=BALL, side0=SIDE, **HELD)
    assert len(held) == len(YAWS) and max(held.values()) < 3e-4


def test_a_side_radius_per_pair_axis_takes_the_arms_give_and_the_midpoints_keep_the_tip():
    """Round 18 (2026-09-30, pink at v11): the pairs read half-spans 1.6 mm across the heading and 2.7 mm along it,
    the arm's give before the probe trips differing by direction. One side radius cannot fit that; one per
    attitude and pair axis does, and its midpoints recover the tip."""
    yaws = {yaw: n for n, yaw in enumerate(YAWS)}
    contacts = side_contacts(np.random.default_rng(5),
                             give=lambda yaw, axis: 0.0008 + 0.0001 * yaws[yaw] if axis == "d" else 0.0001)
    start = {"tip0": TIP + [0.002, -0.003, 0.0], "ball0": BALL + [0.004, -0.003, 0.002], "side0": 0.0093, **HELD}
    assert solve.fit(contacts, pose_frame, **start).rms_m["side"] > 2e-4
    got = solve.fit(contacts, pose_frame, per_axis=True, **start)
    assert got.rms_m["side"] < 1e-4 and len(got.side_radii) == 2 * len(YAWS)
    assert np.linalg.norm(got.tip[:2] - TIP[:2]) < 1e-4
    assert got.side_radii[("0", "d")] == pytest.approx(SIDE - 0.0008, abs=1e-4)
    held = solve.leave_one_group_out(contacts, pose_frame, per_axis=True, **start)
    assert len(held) == len(YAWS) and max(held.values()) < 3e-4
    # an attitude skipped after one touch holds out nothing
    stray = solve.Contact("side", contacts[0].q, "S4 +45/+0", "d")
    assert "S4 +45/+0" not in solve.leave_one_group_out([*contacts, stray], pose_frame, per_axis=True, **start)
    assert "0/d" in got.as_dict()["side_radii_m"]


def test_a_route_past_the_palette_camera_is_caught_and_one_beside_the_ball_is_not():
    base_from_palette = np.eye(4)
    base_from_palette[:3, 3] = [0.221, -0.264, 0.0656]          # the ball 55.6 mm over the rail tops
    zones = station.parts(base_from_palette)
    ball = base_from_palette[:3, 3] + [0.0, 0.0, 0.0556]
    rotation = halo.tool_pose(np.zeros(3), math.radians(30.0))[:3, :3]
    here = ball + [0.0, 0.0, 0.035]
    for plan in halo.side_touches(ball, rotation, np.array([1.0, 0.0, 0.0]), HALO, SIDE):
        assert halo.meets(halo.route(here, plan, 0.030), rotation[:, 2], zones, HALO) is None
    # the same touches about a "ball" found on the palette camera: every route meets it
    wrong = next(centre for name, centre, _, _ in zones if name == "palette_camera")
    met = {halo.meets(halo.route(here, plan, 0.030), rotation[:, 2], zones, HALO)
           for plan in halo.side_touches(wrong, rotation, np.array([1.0, 0.0, 0.0]), HALO, SIDE)}
    assert None not in met and "palette_camera" in met
