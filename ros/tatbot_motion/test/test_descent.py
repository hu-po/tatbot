"""A needle stroke's descent and lift on the arm that draws (the xArm), as motion.yaml and stack.yaml ship them: the
speed the running needles first meet the skin at, at the trims the depth keys reach and over skin that stands
higher than the gauge read it, and how far the pen-up legs put the tip off its command on a plant as late as the rig."""
import numpy as np
import pytest
import tatbot_motion as tm
import tool_spec
from numpy.lib.stride_tricks import sliding_window_view
from tatbot_description import arm_model, load_stack, robot_description

SPEED = 0.0035
GAUGE_LOW_M = 0.0010      # the gauge reads the skin about this much low (motion.yaml pen.ride)
SCATTER_M = 0.0005        # one gauge reading's scatter on silicone (stack.yaml page.surface.frames)
STROKE = {"op": "stroke", "id": "s0001", "ink": "black", "closed": False, "continues": False,
          "points_m": [[0.004, 0.002], [0.024, 0.002]]}


@pytest.fixture(scope="module")
def kin():
    return tm.Kinematics(robot_description(arms=("xarm",)), "xarm")


@pytest.fixture(scope="module")
def motion():
    return tm.load_motion()


@pytest.fixture(scope="module")
def stack():
    return load_stack()


@pytest.fixture(scope="module")
def pen(motion):
    """The 3RL riding: its tube's end over the gauged skin, its needles' reach past that end."""
    ride = {**motion, "pen": {**motion["pen"], "mode": tm.RIDE}}
    return tm.pen_down(ride, tool_spec.load_tool("lutin-3rl-bugpin"))


@pytest.fixture(scope="module")
def page(kin):
    """A page under the arm's rest pose, z up out of the skin."""
    base_from_page = np.eye(4)
    base_from_page[:3, 3] = kin.fk(np.array(arm_model.for_arm("xarm").rest))[:3, 3] - [0.0, 0.0, 0.10]
    return base_from_page


@pytest.fixture(scope="module")
def plans(kin, motion, pen, page):
    """(the hover over the stroke's start from 200 mm away, the stroke from that hover, the stroke in one move from
    there): what stencil tracking's hover stop sends, and what a run without it does."""
    rest = np.array(arm_model.for_arm("xarm").rest)
    away = kin.fk(rest)
    away[:3, :3] = tm.plan.pen_rotation(page, away[:3, :3])
    away[:3, 3] = (page @ [0.004, -0.198, 0.030, 1.0])[:3]
    q0 = kin.solve(away, rest)
    hover = tm.plan_hover(STROKE, base_from_page=page, q_seed=q0, kin=kin, motion=motion, pen=pen)
    plan = {"base_from_page": page, "kin": kin, "motion": motion, "speed_m_s": SPEED, "pen": pen}
    return hover, tm.plan_op(STROKE, q_seed=hover.q[-1], **plan), tm.plan_op(STROKE, q_seed=q0, **plan)


def _eased(traj, trim_m, motion):
    """`traj` with a trim eased in from none, as ArmExecutor.trimmed sends a stroke after a hover."""
    by = tm.slow_from_s(traj, motion["approach"]["final_m"])
    return tm.compose(traj, tm.Trim(0.0).to(0.0, trim_m, motion["pen"]["trim"], by))


def _contact_m_s(sent, page, pen, skin_over_m):
    """The pen's speed toward the skin when its needles' lowest reach first meets skin standing `skin_over_m` over
    the gauged skin (the page); 0 for needles that never reach it."""
    tube = (sent.tip - page[:3, 3]) @ page[:3, 2]
    down = -np.gradient(tube, sent.t)
    met = np.flatnonzero((tube - pen.reach_m <= skin_over_m) & (sent.phase != tm.PHASE_LIFT))
    return float(down[met[0]]) if len(met) else 0.0


def test_the_needles_first_meet_the_skin_at_the_slow_speed_at_every_trim_the_keys_reach(motion, stack, pen, page, plans):
    """The descent's last approach.final_m runs at final_m_s so that running needles meet the skin gently. They meet
    it (trim + skin error - 1) mm over the stop for the 3RL, so the stretch has to cover the deepest trim the keys
    take (page.surface.rigid.trim_stop_m) over skin as far over its gauged height as the gauge, the surface's field
    and one reading can put it together; and a trim eased in from a hover has to be in before it."""
    approach, surface = motion["approach"], stack["page"]["surface"]
    deepest = -float(surface["rigid"]["trim_stop_m"])
    worst = GAUGE_LOW_M + float(surface["max_loo_m"]) + SCATTER_M
    assert worst - (pen.height_m + deepest - pen.reach_m) < approach["final_m"]   # 4.7 mm of the 5
    _, from_hover, one_move = plans
    for trim in (0.0, -0.001, -0.0025, deepest):
        for skin in (0.0, GAUGE_LOW_M, worst):
            for what, sent in (("eased in from its hover", _eased(from_hover, trim, motion)),
                               ("already in", tm.compose(one_move, tm.Trim(trim)))):
                speed = _contact_m_s(sent, page, pen, skin)
                assert speed <= approach["final_m_s"] * 1.05, (trim, skin, what, speed)


def test_a_trim_eased_in_from_a_hover_is_in_before_the_descents_slow_stretch(motion, page, plans):
    """At a key press's 2 mm/s a -2.5 mm trim eased over 2.3 s of the 2.4 s descent and added its rate to the slow
    stretch (5 mm/s commanded where the plan said 3, on every stroke of the 2026-11-12 rehearsals). From a hover the
    pen is in the air until its descent enters that stretch, and the trim is in by then; a change that would be in
    sooner at the keys' speed keeps it, and so does a plan that comes down nowhere."""
    approach, cfg = motion["approach"], motion["pen"]["trim"]
    _, from_hover, one_move = plans
    by = tm.slow_from_s(from_hover, approach["final_m"])
    height = (from_hover.tip - from_hover.tip[from_hover.phase == tm.PHASE_DRAW][0]) @ page[:3, 2]
    at = int(np.searchsorted(from_hover.t, by))
    assert height[at] == pytest.approx(approach["final_m"], abs=2e-4) and 0.3 < by < 1.5
    assert tm.slow_from_s(one_move, approach["final_m"]) > by            # the travel before it is time in the air too
    deep = tm.Trim(0.0).to(0.0, -0.0033, cfg, by)
    assert deep.settled_s() == pytest.approx(by) and deep.settled_s() < tm.Trim(0.0).to(0.0, -0.0033, cfg).settled_s()
    sent = tm.compose(from_hover, deep)
    slow = (sent.phase == tm.PHASE_DESCEND) & (sent.t >= by)
    down = -np.gradient((sent.tip - page[:3, 3]) @ page[:3, 2], sent.t)
    assert down[slow].max() <= approach["final_m_s"] * 1.05
    assert down.max() <= approach["descend_m_s"]                         # the plan's descent and the ease together
    np.testing.assert_allclose(sent.tip[slow] - from_hover.tip[slow], -0.0033 * from_hover.axis[slow], atol=1e-9)
    small = tm.Trim(0.0).to(0.0, -0.0002, cfg, by)
    assert small == tm.Trim(0.0).to(0.0, -0.0002, cfg) and small.ramps[-1][1] == cfg["min_s"]
    down_at = int(np.flatnonzero(from_hover.phase == tm.PHASE_DRAW)[0])
    held = tm.dwell(from_hover, down_at, down_at, 1.0, float(from_hover.t[1] - from_hover.t[0]))
    assert tm.slow_from_s(held, approach["final_m"]) is None


def test_a_lift_is_gentle_over_its_own_first_stretch_not_the_descents(kin, motion, page, plans):
    """The needles leave the skin under the touchdown budget over approach.lift_first_m; the descent's longer
    final_m does not slow a lift."""
    approach, corners = motion["approach"], motion["corners"]
    assert approach["lift_first_m"] < approach["final_m"]
    _, from_hover, _ = plans
    q = from_hover.q[np.flatnonzero(from_hover.phase == tm.PHASE_DRAW)[-1]]
    lift = tm.plan_lift(q_seed=q, direction=page[:3, 2], distance_m=approach["standoff_m"], kin=kin, motion=motion)
    rise = (lift.tip - lift.tip[0]) @ page[:3, 2]
    up = np.gradient(rise, lift.t)
    first = rise < approach["lift_first_m"]
    assert np.all(up[first] <= np.sqrt(2.0 * corners["touchdown_m_s2"] * np.maximum(rise[first], 0.0)) * 1.05 + 1e-4)
    same = {**motion, "approach": {**approach, "lift_first_m": approach["final_m"]}}
    slower = tm.plan_lift(q_seed=q, direction=page[:3, 2], distance_m=approach["standoff_m"], kin=kin, motion=same)
    assert lift.duration_s < slower.duration_s - 0.2


def _answered(traj, delay_s, lag_s, dt):
    """(t, the joints sent, the joints a plant answers with): `traj` held at its end a second, sent at the control
    rate, and a plant that follows it a pure `delay_s` late and then a first-order `lag_s` (the fake arm's model)."""
    t = np.arange(0.0, traj.t[-1] + 1.0, dt)
    sent = np.stack([np.interp(t, traj.t, traj.q[:, j]) for j in range(traj.q.shape[1])], axis=1)
    late = np.stack([np.interp(t - delay_s, t, sent[:, j]) for j in range(sent.shape[1])], axis=1)
    out, gain = late.copy(), dt / (lag_s + dt)
    for k in range(1, len(out)):
        out[k] = out[k - 1] + gain * (late[k] - out[k - 1])
    return t, sent, out


def _snag_off_m(kin, traj, delay_s, lag_s, command_delay_s, hold_s, dt):
    """The most the driver's snag guard would see held `hold_s` over `traj`: the measured tip from the command sent
    command_delay_s before it (tatbot_hardware core.cpp)."""
    t, sent, measured = _answered(traj, delay_s, lag_s, dt)
    answering = np.stack([np.interp(t - command_delay_s, t, sent[:, j]) for j in range(sent.shape[1])], axis=1)
    off = np.array([np.linalg.norm(kin.fk(a)[:3, 3] - kin.fk(b)[:3, 3])
                    for a, b in zip(measured[::4], answering[::4], strict=True)])
    return float(sliding_window_view(off, max(1, round(hold_s / (4 * dt)))).min(axis=1).max())


def test_the_pen_up_legs_stay_inside_the_snag_guard_on_a_plant_as_late_as_the_rigs(kin, motion, stack, plans):
    """The snag guard holds the arm when its tip stands safety.snag.off_m from its command for hold_s, and takes the
    arm model's command_delay_s out first. A plant that answers at exactly that delay shows nothing, so the legs are
    answered here by plants as early and as late as the arm was on the rig: over the lifts and the travels of the
    2026-11-12 rehearsals each move fitted a pure delay of 92.5 to 102.5 ms (95 the median), and the guard's
    quantity stayed under 0.6 mm held 0.2 s. The hover's travel reaches the pen-up cap, so this is where a faster
    budget would show first; a plant 20 ms late is past the threshold, which is what makes it a test."""
    model, snag = arm_model.for_arm("xarm"), stack["safety"]["snag"]
    dt = 1.0 / float(motion["control_rate_hz"])
    hover, from_hover, _ = plans
    peak = np.linalg.norm(np.gradient(hover.tip, hover.t, axis=0), axis=1).max()
    assert peak > 0.95 * motion["tip_speed"]["pen_up_max_m_s"]
    plants = [(model.fake_plant.delay_s, model.fake_plant.lag_s), (0.0925, 0.0), (0.1025, 0.0), (0.0875, 0.007)]
    for traj in (hover, from_hover):
        for delay, lag in plants:
            off = _snag_off_m(kin, traj, delay, lag, model.command_delay_s, snag["hold_s"], dt)
            assert off < 0.75 * snag["off_m"], (delay, lag, off)
    late = _snag_off_m(kin, hover, model.command_delay_s + 0.020, 0.0, model.command_delay_s, snag["hold_s"], dt)
    assert late > snag["off_m"]
