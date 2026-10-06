"""The op choreography of ros/README.md section 4.4 on the generated right-arm description."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tatbot_motion as tm
from tatbot_motion import timelaw as tl

SPEED = 0.0035


@pytest.fixture(scope="module")
def kin():
    return tm.Kinematics.from_repo(arm="right")


@pytest.fixture(scope="module")
def motion():
    """motion.yaml with knots at the control rate, so the checks below see every planned tick, and the
    pen-down path on the page (pen.press.lift_m is a bench pressure knob, not choreography)."""
    motion = tm.load_motion()
    pen = {**motion["pen"], "mode": tm.PRESS, tm.PRESS: {**motion["pen"][tm.PRESS], "lift_m": 0.0}}
    return {**motion, "knot_rate_hz": motion["control_rate_hz"], "pen": pen}


@pytest.fixture(scope="module")
def page():
    """A fixed page in right/base_link, z up out of the paper: the old rig's, not stack.yaml's configured
    page. The checks below are tuned to it: _pose_over's seed reaches it on a well-conditioned IK branch,
    page x runs along base x, and the touch's tracking lag decays within 0.1 s of the slow leg."""
    base_from_page = np.eye(4)
    base_from_page[:3, 3] = [0.371, -0.231, 0.0334]
    return base_from_page


def _pose_over(kin, page, xy, height, q=None):
    q = np.array([-0.5, 1.0, 1.0, -0.3, 0.0, 0.0, 0.002]) if q is None else q
    start = kin.fk(q)
    target = np.eye(4)
    target[:3, :3] = tm.plan.pen_rotation(page, start[:3, :3])
    target[:3, 3] = (page @ [xy[0], xy[1], height, 1.0])[:3]
    return kin.solve(target, q)


def _height(page, tip):
    return (tip - page[:3, 3]) @ page[:3, 2]


def _stroke(points, **extra):
    return {"op": "stroke", "id": "s0001", "ink": "black", "closed": False, "continues": False,
            "points_m": points, **extra}


SQUARE = [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01]]


def _segments(phase):
    """The phases in order, runs collapsed."""
    return [int(p) for i, p in enumerate(phase) if i == 0 or p != phase[i - 1]]


def test_kinematics_jacobian_matches_finite_differences(kin):
    q = np.array([0.2, 0.9, 1.1, -0.3, 0.4, 0.2, 0.003])
    jac = kin.jacobian(q)
    base = kin.fk(q)
    step = 1e-7
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = step
        moved = kin.fk(q + dq)
        np.testing.assert_allclose((moved[:3, 3] - base[:3, 3]) / step, jac[:3, j], atol=1e-5)
        omega = tl.rotation_log(moved[:3, :3] @ base[:3, :3].T)
        np.testing.assert_allclose(omega[0] * omega[1] / step, jac[3:, j], atol=1e-5)
    assert kin.joint_names[-1] == "right/left_carriage_joint"
    assert kin.lower[6] == pytest.approx(-0.006) and kin.upper[6] == pytest.approx(0.040)


def test_kinematics_names_a_missing_frame():
    from tatbot_description import robot_description

    urdf = robot_description(arms=("right",)).replace('"right/tcp"', '"right/pen"')
    with pytest.raises(ValueError, match="right/tcp"):
        tm.Kinematics(urdf, "right")


def test_a_lagging_tool_axis_is_a_plan_error(kin, motion, page):
    strict = {**motion, "clik": {**motion["clik"], "max_orientation_error_rad": 1e-9}}
    q = _pose_over(kin, page, (-0.01, -0.01), 0.02)
    with pytest.raises(tm.PlanError, match="tool axis"):
        tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q, kin=kin, motion=strict, speed_m_s=SPEED)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q, kin=kin, motion=motion, speed_m_s=SPEED)
    assert traj.info["max_orientation_error_rad"] <= motion["clik"]["max_orientation_error_rad"]


def test_the_pen_presses_or_rides_its_stroke_over_the_touched_page(kin, motion, page):
    """A touch finds the paper with the tip at the top of its stroke. press: the path presses pen.press.lift_m into
    the page and dwells pen.press.settle_s at touchdown; ride: it rides pen.ride.fraction of the fitted tool's
    stroke over it and dwells pen.ride.settle_s. A tool without the stroke facts, or whose tip sits inside its tube
    at the top of the stroke, cannot ride."""
    tool = SimpleNamespace(tool_id="pen", stroke_m=0.0035, tip_out_at_top_m=0.002)
    assert tm.pen_down(motion) == tm.PenDown(tm.PRESS, 0.0, motion["pen"]["press"]["settle_s"])
    riding = {**motion, "pen": {**motion["pen"], "mode": tm.RIDE}}
    riding["pen"][tm.RIDE] = dict(riding["pen"][tm.RIDE], fraction=0.5)
    ride = tm.pen_down(riding, tool)
    assert ride.machine and ride.height_m == pytest.approx(0.00175) and ride.settle_s == 0.0
    assert ride.describe().startswith("ride: the path 1.75 mm over the touched page, 0.50 of the 3.5 mm stroke")
    for bad in (None, SimpleNamespace(tool_id="old", stroke_m=0.0035, tip_out_at_top_m=None),
                SimpleNamespace(tool_id="needles", stroke_m=0.0035, tip_out_at_top_m=-0.0015)):
        with pytest.raises(ValueError):
            tm.pen_down(riding, bad)
    # a needle cartridge's touches and gauge meet its tube's end: the path is that end's height, and the needles
    # pass it by their reach, once recorded
    tube = {"calibration": {"contact_reference": "tube_end"}}
    needles = SimpleNamespace(tool_id="3rl", stroke_m=0.004, tip_out_at_top_m=None, raw=dict(tube, needle_reach_mm=3.0))
    needled = tm.pen_down(riding, needles)
    assert needled.machine and needled.height_m == pytest.approx(0.002) and needled.reach_m == pytest.approx(0.003)
    assert "the needles 1.00 mm past the page at its bottom" in needled.describe()
    unknown = tm.pen_down(riding, SimpleNamespace(tool_id="3rl", stroke_m=0.004, tip_out_at_top_m=None, raw=tube))
    assert unknown.reach_m is None and "needles" not in unknown.describe()
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    pressed = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                      pen=ride)
    assert np.abs(_height(page, traj.tip[traj.phase == tm.PHASE_DRAW]) - 0.00175).max() < 1e-4

    def settled(t):
        return np.count_nonzero(t.phase == tm.PHASE_SETTLE) / motion["knot_rate_hz"]

    assert settled(pressed) == pytest.approx(motion["pen"]["press"]["settle_s"], abs=0.02)
    assert settled(traj) <= 0.01   # no dwell at touchdown


def test_the_hover_pen_draws_the_path_over_the_page_with_the_machine_off(kin, motion, page, tmp_path):
    """hover rehearses a drawing without a mark: the path pen.hover.height_m over the page, no tool facts needed,
    the machine off; motion.yaml refuses a hover under the page or above the stroke's standoff."""
    hovering = {**motion, "pen": {**motion["pen"], "mode": tm.HOVER}}
    hovering["pen"][tm.HOVER] = {"height_m": 0.006, "settle_s": 0.0}
    hover = tm.pen_down(hovering)
    assert not hover.machine and hover.height_m == 0.006
    assert hover.describe() == "hover: the path 6.00 mm over the touched page, the tattoo machine off"
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=hovering, speed_m_s=SPEED,
                      pen=hover)
    assert np.abs(_height(page, traj.tip[traj.phase == tm.PHASE_DRAW]) - 0.006).max() < 1e-4
    text = tm.motion_path().read_text()
    for bad in ("height_m: 0.0\n", "height_m: 0.012\n"):
        path = tmp_path / "motion.yaml"
        path.write_text(text.replace("height_m: 0.006", bad.strip(), 1))
        with pytest.raises(ValueError, match="pen.hover.height_m"):
            tm.load_motion(path)


def test_stroke_choreography(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    assert _segments(traj.phase) == [tm.PHASE_TRAVEL, tm.PHASE_DESCEND, tm.PHASE_SETTLE, tm.PHASE_DRAW, tm.PHASE_LIFT]
    dt = 1.0 / motion["control_rate_hz"]
    np.testing.assert_allclose(np.diff(traj.t), dt)
    assert traj.t[0] == 0.0
    np.testing.assert_allclose(traj.q[0], q0, atol=1e-6)  # the first tick tracks the seed
    assert np.all(traj.qd[-1] == 0.0)
    speed = np.linalg.norm(np.gradient(traj.tip, dt, axis=0), axis=1)
    height = np.array([_height(page, tip) for tip in traj.tip])
    draw = traj.phase == tm.PHASE_DRAW
    # pen down: on the page (the CLIK model error), at no more than the program speed, tcp z into the paper
    assert np.abs(height[draw]).max() < 1e-4
    assert speed[draw].max() <= SPEED * 1.02
    for q in traj.q[draw][::200]:
        np.testing.assert_allclose(kin.fk(q)[:3, 2], -page[:3, 2], atol=1e-3)
    # arc runs 0 .. the square's length while drawing and is NaN otherwise
    assert np.all(np.isnan(traj.arc_m[~draw]))
    assert traj.arc_m[draw][0] < 1e-6 and traj.arc_m[draw][-1] == pytest.approx(0.08, abs=1e-9)
    assert np.all(np.diff(traj.arc_m[draw]) >= 0.0)
    # descent: the last 5 mm at 3 mm/s; settle holds 1 s at the start; the lift ends 10 mm up
    final = (traj.phase == tm.PHASE_DESCEND) & (height < motion["approach"]["final_m"] - 2e-4)
    assert speed[final].max() <= motion["approach"]["final_m_s"] * 1.05
    settle = traj.phase == tm.PHASE_SETTLE
    assert np.count_nonzero(settle) * dt == pytest.approx(motion["pen"]["press"]["settle_s"], abs=2 * dt)
    assert height[-1] == pytest.approx(motion["approach"]["standoff_m"], abs=1e-4)
    assert speed[traj.phase == tm.PHASE_TRAVEL].max() <= motion["tip_speed"]["pen_up_max_m_s"]
    # joints within their caps
    pen_down = np.abs(traj.qd[draw, :6]).max()
    assert pen_down <= motion["joint_speed"]["pen_down_rad_s"]
    assert np.abs(traj.qd[:, :6]).max() <= motion["joint_speed"]["pen_up_rad_s"]
    assert traj.info["max_model_error_m"] < 1e-4


def test_explicit_ending_lift_and_incoming_continuation_are_independent(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.02)
    first = tm.plan_op(_stroke([[-0.01, 0.0], [0.0, 0.0]]), base_from_page=page, q_seed=q0, kin=kin,
                       motion=motion, speed_m_s=SPEED, lift_at_end=False)
    assert tm.PHASE_LIFT not in set(first.phase.tolist())
    assert _segments(first.phase)[-1] == tm.PHASE_DRAW
    second = tm.plan_op(_stroke([[0.0, 0.0], [0.0, 0.01]], continues=True), base_from_page=page, q_seed=first.q[-1], kin=kin,
                        motion=motion, speed_m_s=SPEED, pen_down_at_start=True)
    assert _segments(second.phase) == [tm.PHASE_DRAW, tm.PHASE_LIFT]
    assert second.info["joined"]
    heights = [_height(page, tip) for tip in second.tip[second.phase == tm.PHASE_DRAW]]
    assert np.abs(heights).max() < 1e-4


def test_resume_from_an_arc_position(kin, motion, page):
    op = _stroke(SQUARE)
    # the tip is on the path 30 mm in (after a latch): join pen down and draw the remaining 50 mm
    q_at = _pose_over(kin, page, (0.01, 0.0), 0.0)
    traj = tm.plan_op(op, base_from_page=page, q_seed=q_at, kin=kin, motion=motion, speed_m_s=SPEED,
                      from_arc_m=0.03, pen_down_at_start=True)
    assert traj.info["joined"] and _segments(traj.phase) == [tm.PHASE_DRAW, tm.PHASE_LIFT]
    draw = traj.arc_m[traj.phase == tm.PHASE_DRAW]
    assert draw[0] == pytest.approx(0.03) and draw[-1] == pytest.approx(0.08)
    # the tip 2 mm off its path: lift off the page, then travel, descend and settle at the arc point
    q_off = _pose_over(kin, page, (0.01, 0.002), 0.0)
    traj = tm.plan_op(op, base_from_page=page, q_seed=q_off, kin=kin, motion=motion, speed_m_s=SPEED,
                      from_arc_m=0.03, pen_down_at_start=True)
    assert _segments(traj.phase) == [tm.PHASE_LIFT, tm.PHASE_TRAVEL, tm.PHASE_DESCEND, tm.PHASE_SETTLE,
                                     tm.PHASE_DRAW, tm.PHASE_LIFT]
    settle = traj.tip[traj.phase == tm.PHASE_SETTLE][-1]
    np.testing.assert_allclose(settle, (page @ [0.01, 0.0, 0.0, 1.0])[:3], atol=1e-4)
    # never dragged: every pen-up row before the descent stays above the page
    heights = np.array([_height(page, tip) for tip in traj.tip])
    travel = traj.phase == tm.PHASE_TRAVEL
    assert heights[travel].min() >= motion["approach"]["standoff_m"] - 1.5e-3


def test_pause_lifts_to_the_standoff_or_holds(kin, motion, page):
    on_page = _pose_over(kin, page, (0.0, 0.0), 0.0)
    traj = tm.plan_op({"op": "pause", "id": "p0001", "reason": "swap pen: red"}, base_from_page=page,
                      q_seed=on_page, kin=kin, motion=motion, speed_m_s=SPEED)
    assert _segments(traj.phase) == [tm.PHASE_LIFT]
    assert _height(page, traj.tip[-1]) == pytest.approx(motion["approach"]["standoff_m"], abs=1e-4)
    held = tm.plan_op({"op": "pause", "id": "p0001"}, base_from_page=page, q_seed=traj.q[-1], kin=kin, motion=motion,
                      speed_m_s=SPEED)
    assert _segments(held.phase) == [tm.PHASE_PAUSED]
    assert np.abs(held.q - traj.q[-1]).max() < 1e-6
    with pytest.raises(NotImplementedError):
        tm.plan_op({"op": "dip", "id": "d0002"}, base_from_page=page, q_seed=on_page, kin=kin, motion=motion,
                   speed_m_s=SPEED)


def test_travel_returns_the_carriage_to_its_bias(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    q0[6] = 0.030  # after a trip retract
    target = kin.fk(_pose_over(kin, page, (0.02, 0.02), 0.02))
    traj = tm.plan_travel(q_seed=q0, base_from_tcp=target, kin=kin, motion=motion)
    assert traj.q[-1, 6] == pytest.approx(motion["carriage"]["bias_m"], abs=1e-6)
    assert np.abs(traj.qd[:, 6]).max() <= motion["carriage"]["travel_m_s"] * 1.01
    np.testing.assert_allclose(traj.tip[-1], target[:3, 3], atol=2e-4)
    assert set(traj.phase.tolist()) == {tm.PHASE_TRAVEL}


def test_a_slow_joint_cap_slows_the_travel_until_it_fits(kin, motion, page):
    slow = {**motion, "joint_speed": {**motion["joint_speed"], "pen_up_rad_s": 0.02}}
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    target = kin.fk(_pose_over(kin, page, (0.03, -0.04), 0.05))
    fast = tm.plan_travel(q_seed=q0, base_from_tcp=target, kin=kin, motion=motion)
    slowed = tm.plan_travel(q_seed=q0, base_from_tcp=target, kin=kin, motion=slow)
    assert slowed.info["slowdowns"] and slowed.duration_s > fast.duration_s
    assert np.abs(slowed.qd[:, :6]).max() <= 0.02 * 1.001


def test_touch_descends_fast_then_slow_with_the_wiggle(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.012)
    start = kin.fk(q0)
    down = -page[:3, 2]
    traj = tm.plan_touch(q_seed=q0, start_base_from_tcp=start, direction=down, max_travel_m=0.02, kin=kin,
                         motion=motion, prior_distance_m=0.012)
    assert _segments(traj.phase) == [tm.PHASE_SETTLE, tm.PHASE_DESCEND, tm.PHASE_TOUCH]
    dt = 1.0 / motion["control_rate_hz"]
    along = (traj.tip - start[:3, 3]) @ down
    touch = traj.phase == tm.PHASE_TOUCH
    fast = traj.phase == tm.PHASE_DESCEND
    assert along[fast].max() <= 0.012 - motion["touch"]["slow_above_m"] + 1e-4
    assert along[-1] == pytest.approx(0.02, abs=1e-4)
    axial = np.gradient(along, dt)
    assert axial[touch][40:].max() <= motion["touch"]["slow_m_s"] * 1.05  # the FK tip trails the switch by ~0.1 s
    assert axial[fast].max() <= motion["touch"]["fast_m_s"] * 1.01
    lateral = traj.tip - start[:3, 3] - np.outer(along, down)
    assert np.linalg.norm(lateral[touch], axis=1).max() == pytest.approx(motion["touch"]["wiggle_amplitude_m"], rel=0.05, abs=1e-5)
    assert np.linalg.norm(lateral[~touch], axis=1).max() < 1e-4
    assert tm.guard_arm_time_s(traj) == pytest.approx(traj.t[np.argmax(touch)])
    # without a prior the whole search is slow
    blind = tm.plan_touch(q_seed=q0, start_base_from_tcp=start, direction=down, max_travel_m=0.04, kin=kin,
                          motion=motion)
    assert tm.PHASE_DESCEND not in set(blind.phase.tolist())
    assert ((blind.tip[-1] - start[:3, 3]) @ down) == pytest.approx(motion["touch"]["search_m"], abs=1e-4)


def test_touch_travels_to_a_distant_start_first(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    start = kin.fk(_pose_over(kin, page, (0.02, 0.0), 0.012))
    traj = tm.plan_touch(q_seed=q0, start_base_from_tcp=start, direction=-page[:3, 2], max_travel_m=0.02, kin=kin,
                         motion=motion, prior_distance_m=0.012)
    assert _segments(traj.phase)[:2] == [tm.PHASE_TRAVEL, tm.PHASE_SETTLE]


def test_touch_travels_through_its_via_points(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.012)
    here = kin.fk(q0)[:3, 3]
    start = kin.fk(_pose_over(kin, page, (0.02, 0.0), 0.012))
    up = np.array([here[0], here[1], here[2] + 0.02])
    over = np.array([start[0, 3], start[1, 3], up[2]])
    traj = tm.plan_touch(q_seed=q0, start_base_from_tcp=start, direction=-page[:3, 2], max_travel_m=0.02, kin=kin,
                         motion=motion, prior_distance_m=0.012, via=[up, over])
    travel = traj.tip[traj.phase == tm.PHASE_TRAVEL]
    for point in (up, over, start[:3, 3]):
        assert np.linalg.norm(travel - point, axis=1).min() < 2e-4
    across = travel[(travel[:, 2] > up[2] - 1e-4)]
    assert len(across) and np.ptp(across[:, 0]) > 0.015   # the move across stays up on the clearance plane


def test_a_probe_touch_goes_no_farther_past_the_expected_contact_than_the_stylus_allows(motion):
    probe = motion["probe"]
    assert tm.probe_travel_m(np.array([0.0, 0.0, -1.0]), 0.004, motion) == pytest.approx(0.004 + probe["axial_cap_m"])
    assert tm.probe_travel_m(np.array([0.0, 2.0, 0.0]), 0.004, motion) == pytest.approx(0.004 + probe["lateral_cap_m"])
    # a normal at 40 deg elevation: the stylus's axial budget binds first
    slant = np.array([np.cos(np.radians(40.0)), 0.0, -np.sin(np.radians(40.0))])
    past = tm.probe_travel_m(slant, 0.004, motion) - 0.004
    assert past * np.sin(np.radians(40.0)) == pytest.approx(probe["axial_cap_m"])
    assert past * np.cos(np.radians(40.0)) < probe["lateral_cap_m"]


def test_to_knots_resamples_the_plan(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    knots = tm.to_knots(traj, 100.0)
    assert np.all(np.diff(knots.t) > 0.0) and np.diff(knots.t).max() <= 0.01 + 1e-12
    assert knots.t[0] == 0.0 and knots.t[-1] == traj.t[-1]
    np.testing.assert_allclose(knots.q[::1][:-1], traj.q[::4][:len(knots.t) - 1], atol=1e-12)
    np.testing.assert_allclose(knots.q[-1], traj.q[-1])
    assert np.all(knots.qd[-1] == 0.0)
    assert set(knots.phase.tolist()) == set(traj.phase.tolist())
    # JTC's cubic between knots stays within 5 um of the 400 Hz plan
    for k in range(0, len(knots.t) - 1, 50):
        h = knots.t[k + 1] - knots.t[k]
        u = 0.5
        mid = ((2 * u**3 - 3 * u**2 + 1) * knots.q[k] + (u**3 - 2 * u**2 + u) * h * knots.qd[k]
               + (-2 * u**3 + 3 * u**2) * knots.q[k + 1] + (u**3 - u**2) * h * knots.qd[k + 1])
        exact = traj.q[int(round((knots.t[k] + 0.5 * h) * motion["control_rate_hz"]))]
        assert np.abs(mid - exact)[:6].max() * 0.45 < 5e-6


def test_planners_return_knots_at_the_configured_rate(kin, page, motion):
    motion = {**motion, "knot_rate_hz": tm.load_motion()["knot_rate_hz"]}
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    assert np.diff(traj.t).max() == pytest.approx(1.0 / motion["knot_rate_hz"])
    assert traj.info["knot_rate_hz"] == motion["knot_rate_hz"] and np.all(traj.qd[-1] == 0.0)


def test_dispatch_drift(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    assert tm.dispatch_drift(traj, q0, kin, motion)["ok"]
    moved = q0.copy()
    moved[0] += 0.015
    pen_up = tm.dispatch_drift(traj, moved, kin, motion)
    assert not pen_up["pen_down"] and pen_up["joint_rad"] == pytest.approx(0.015, abs=1e-6) and not pen_up["ok"]
    moved[0] = q0[0] + 0.004  # ~1.4 mm at the tip over a 0.35 m lever
    drift = tm.dispatch_drift(traj, moved, kin, motion)
    assert drift["ok"] and not tm.dispatch_drift(traj, moved, kin, motion, pen_down=True)["ok"]
    carriage = q0.copy()
    carriage[6] += 0.001  # the carriage reading is not the seed: the plan's carriage stands
    assert tm.dispatch_drift(traj, carriage, kin, motion)["tip_m"] < 1e-6


def test_planning_a_stroke_takes_a_fraction_of_its_duration(kin, motion, page):
    x = np.linspace(-0.02, 0.02, 400)
    tentacle = np.stack([x, 0.01 * np.sin(x / 0.04 * 4 * math.pi)], axis=1).tolist()
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    begin = time.perf_counter()
    traj = tm.plan_op(_stroke(tentacle), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    elapsed = time.perf_counter() - begin
    print(f"planned {traj.duration_s:.1f} s ({traj.info['stroke_m'] * 1e3:.0f} mm) in {elapsed:.2f} s")
    assert elapsed < 0.25 * traj.duration_s


def test_closed_stroke_and_dot(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    ring = tm.plan_op(_stroke(SQUARE[:-1], closed=True), base_from_page=page, q_seed=q0, kin=kin, motion=motion,
                      speed_m_s=SPEED)
    assert ring.info["stroke_m"] == pytest.approx(0.08)
    dot = tm.plan_op(_stroke([[0.005, 0.005]]), base_from_page=page, q_seed=q0, kin=kin, motion=motion,
                     speed_m_s=SPEED)
    assert _segments(dot.phase)[-2:] == [tm.PHASE_DRAW, tm.PHASE_LIFT]


def test_a_trim_moves_the_pen_along_the_page_normal(kin, motion, page):
    """A stroke plan carries its depth axis (the page normal) and dq_dh, so composing a trim moves every knot's
    tip that far along the normal on the plan's own joints: within 0.1 um at a 0.1 mm step, 20 um at the 2 mm
    limit, the tool's rotation held."""
    q0 = _pose_over(kin, page, (-0.01, -0.01), 0.01)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    np.testing.assert_allclose(traj.axis, np.broadcast_to(page[:3, 2], traj.axis.shape), atol=1e-12)
    rows = slice(None, None, 37)
    for h, tol in ((0.0001, 1e-7), (-0.002, 2e-5)):
        moved = tm.compose(traj, tm.Trim(h))
        for qa, qb in zip(traj.q[rows], moved.q[rows], strict=True):
            a, b = kin.fk(qa), kin.fk(qb)
            np.testing.assert_allclose(b[:3, 3] - a[:3, 3], h * page[:3, 2], atol=tol)
            assert tl.rotation_angle(a[:3, :3].T @ b[:3, :3]) < 1e-5
        np.testing.assert_allclose(moved.tip - traj.tip, h * traj.axis)
        assert moved.info["trim"] == tm.Trim(h) and np.all(moved.q[:, 6] == traj.q[:, 6])


def test_a_trim_change_eases_in_ahead_and_leaves_the_past_alone(kin, motion, page):
    """A change mid-goal: the composed goal is the running one until the change starts, then a quintic ramp no
    faster than pen.trim speed_m_s, its velocity feed-forward the composed joints' own rate."""
    cfg = motion["pen"]["trim"]
    q0 = _pose_over(kin, page, (-0.01, -0.01), 0.01)
    traj = tm.plan_op(_stroke(SQUARE), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    running = tm.Trim(-0.0003)
    trim = running.to(5.0, -0.0003 - 0.002, cfg)
    t0, span, a, b = trim.ramps[-1]
    assert (t0, a, b) == (5.0, -0.0003, pytest.approx(-0.0023)) and span == pytest.approx(1.875, rel=1e-6)
    assert trim.settled_s() == pytest.approx(6.875) and trim.end_m == pytest.approx(-0.0023)
    assert running.to(1.0, -0.0004, cfg).ramps[-1][1] == cfg["min_s"]
    before, after = tm.compose(traj, running), tm.compose(traj, trim)
    early = traj.t < t0
    np.testing.assert_array_equal(after.q[early], before.q[early])
    np.testing.assert_array_equal(after.qd[early], before.qd[early])
    t = np.linspace(0.0, 8.0, 801)
    assert np.abs(np.diff(trim.value(t))).max() <= cfg["speed_m_s"] * 0.01 + 1e-12
    ramp = (traj.t > t0) & (traj.t < t0 + span)
    rate = np.gradient(after.q, traj.t, axis=0)
    assert np.abs(rate[ramp] - after.qd[ramp]).max() < 0.02 * np.abs(after.qd[ramp]).max()
    assert after.qd[-1].tolist() == [0.0] * 7


def test_only_planned_strokes_carry_a_depth_axis(kin, motion, page):
    q0 = _pose_over(kin, page, (0.0, 0.0), 0.03)
    above = kin.fk(q0)
    above[2, 3] += 0.01
    travel = tm.plan_travel(q_seed=q0, base_from_tcp=above, kin=kin, motion=motion)
    assert travel.axis is None and travel.dq_dh is None
    with pytest.raises(ValueError, match="no depth axis"):
        tm.compose(travel, tm.Trim(0.0001))


def test_a_hover_stops_over_the_strokes_start_and_the_stroke_descends_from_it(kin, motion, page):
    """Stencil tracking's stop: at rest at the standoff over the start, pen rotation; the stroke from there
    turns straight down."""
    op = _stroke([[0.004, 0.002], [0.014, 0.002]])
    q0 = _pose_over(kin, page, (-0.02, -0.01), 0.03)
    hover = tm.plan_hover(op, base_from_page=page, q_seed=q0, kin=kin, motion=motion)
    standoff = motion["approach"]["standoff_m"]
    start = (page @ [0.004, 0.002, tm.pen_down(motion).height_m + standoff, 1.0])[:3]
    assert np.linalg.norm(hover.tip[-1] - start) < 2e-4
    assert np.allclose(hover.qd[-1], 0.0) and hover.axis is None
    assert set(_segments(hover.phase)) <= {tm.PHASE_TRAVEL}
    end = kin.fk(hover.q[-1])
    np.testing.assert_allclose(end[:3, 2], -page[:3, 2], atol=1e-3)            # the drawing rotation
    stroke = tm.plan_op(op, base_from_page=page, q_seed=hover.q[-1], kin=kin, motion=motion, speed_m_s=SPEED)
    travel = stroke.phase == tm.PHASE_TRAVEL
    assert not travel.any() or np.linalg.norm(stroke.tip[travel] - start, axis=1).max() < 1e-3
