"""plan_op and plan_hover on a curved page (tatbot_motion.surface): on a flat surface exactly the plane's plan; on a
forearm's curve every drawing knot at the pen's height along its own normal, the pen along that normal."""
import numpy as np
import pytest
import tatbot_motion as tm
from tatbot_motion import timelaw as tl
from tatbot_motion.surface import PageSurface, fit_height

SPEED = 0.0035
R = 0.040


@pytest.fixture(scope="module")
def kin():
    return tm.Kinematics.from_repo(arm="right")


@pytest.fixture(scope="module")
def motion():
    motion = tm.load_motion()
    pen = {**motion["pen"], "mode": tm.PRESS, tm.PRESS: {**motion["pen"][tm.PRESS], "lift_m": 0.0}}
    return {**motion, "knot_rate_hz": motion["control_rate_hz"], "pen": pen}


@pytest.fixture(scope="module")
def page():
    base_from_page = np.eye(4)
    base_from_page[:3, 3] = [0.371, -0.231, 0.0334]
    return base_from_page


@pytest.fixture(scope="module")
def forearm(page):
    """A forearm of radius 40 mm under the page, its crest along page y through the page's origin."""
    xy = np.stack(np.meshgrid(np.linspace(-0.022, 0.022, 23), np.linspace(-0.03, 0.03, 21)), -1).reshape(-1, 2)
    return PageSurface(page, fit_height(np.c_[xy, np.sqrt(R ** 2 - xy[:, 0] ** 2) - R], degree=4))


def _seed(kin, page):
    q = np.array([-0.5, 1.0, 1.0, -0.3, 0.0, 0.0, 0.002])
    target = np.eye(4)
    target[:3, :3] = tm.plan.pen_rotation(page, kin.fk(q)[:3, :3])
    target[:3, 3] = (page @ [0.0, 0.0, 0.01, 1.0])[:3]
    return kin.solve(target, q)


def _stroke(points):
    return {"op": "stroke", "id": "s0001", "closed": False, "continues": False, "points_m": points}


ACROSS = [[-0.012, 0.004], [0.012, 0.004]]   # round the arm: two points, 24 mm apart


def test_a_flat_surface_plans_exactly_the_planes_stroke(kin, motion, page):
    q0 = _seed(kin, page)
    plane = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED)
    flat = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                      surface=PageSurface(page))
    assert np.array_equal(plane.q, flat.q) and np.array_equal(plane.tip, flat.tip) and "surface" not in flat.info


def test_a_stroke_round_a_forearm_rides_its_curve_along_each_points_normal(kin, motion, page, forearm):
    q0 = _seed(kin, page)
    traj = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                      surface=forearm)
    drawing = traj.phase == tm.plan.PHASE_DRAW
    gaps = np.array([forearm.height_over(tip)[0] for tip in traj.tip[drawing]])
    assert traj.info["surface"] and np.max(np.abs(gaps)) < 1e-4           # on the skin all the way round (CLIK)
    heights = (traj.tip[drawing] - page[:3, 3]) @ page[:3, 2]
    assert heights.max() - heights[0] > 0.0015   # over the crest, 1.8 mm above the chord a plane's line would cut
    for k in np.flatnonzero(drawing)[::20]:
        normal = forearm.height_over(traj.tip[k])[1]
        tool = kin.fk(traj.q[k])[:3, 2]
        assert np.degrees(np.arccos(np.clip(-tool @ normal, -1, 1))) < 1.0   # the pen along the skin's normal
        assert np.allclose(traj.axis[k], normal, atol=0.02)                # and its depth axis
    angle = np.degrees(np.arccos(np.clip(traj.axis[drawing][0] @ traj.axis[drawing][-1], -1, 1)))
    assert 30.0 < angle < 40.0                                              # 24 mm of arc at 40 mm: 34 deg
    uv = forearm.chart.to_chart((np.linalg.inv(page) @ np.c_[traj.tip[drawing], np.ones(drawing.sum())].T).T[:, :2])
    assert np.isclose(uv[0, 0], -0.012, atol=2e-4) and np.isclose(uv[-1, 0], 0.012, atol=2e-4)   # print mm, not shadow


def test_the_approach_and_the_hover_come_down_the_starts_normal(kin, motion, page, forearm):
    q0 = _seed(kin, page)
    traj = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                      surface=forearm)
    start, n0 = forearm.lift([ACROSS[0]], 0.0)
    descent = traj.tip[traj.phase == tm.plan.PHASE_DESCEND]
    off = descent - start[0]
    across = off - np.outer(off @ n0[0], n0[0])
    assert np.max(np.linalg.norm(across, axis=1)) < 2e-4                    # straight down the normal
    hover = tm.plan_hover(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, surface=forearm)
    end = hover.tip[-1]
    assert np.isclose(forearm.height_over(end)[0], motion["approach"]["standoff_m"], atol=1e-4)
    assert np.allclose(hover.axis[-1], n0[0], atol=1e-6)
    assert np.degrees(tl.rotation_angle(kin.fk(hover.q[-1])[:3, :3].T @ kin.fk(traj.q[traj.phase ==
                                                                           tm.plan.PHASE_SETTLE][0])[:3, :3])) < 1.0


def test_joins_is_plan_ops_own_join_on_a_curve(kin, motion, page, forearm):
    """tatbot_motion.joins answers what plan_op does with a pen-down start on the surface: a continuation from its
    chain's end joins and draws; the same tip 5 mm up the normal (lifted after a stop) is a descent."""
    first = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=_seed(kin, page), kin=kin, motion=motion,
                       speed_m_s=SPEED, surface=forearm, lift_at_end=False)
    cont = {**_stroke([ACROSS[1], [0.012, 0.012]]), "continues": True}
    for q, up in ((first.q[-1], 0.0), (None, 0.005)):
        tip = first.tip[-1] + up * forearm.height_over(first.tip[-1])[1]
        q = q if q is not None else kin.solve(np.r_[np.c_[kin.fk(first.q[-1])[:3, :3], tip], [[0, 0, 0, 1]]],
                                              first.q[-1])
        traj = tm.plan_op(cont, base_from_page=page, q_seed=q, kin=kin, motion=motion, speed_m_s=SPEED,
                          surface=forearm, pen_down_at_start=True)
        assert traj.info["joined"] == tm.joins(cont, kin.fk(q)[:3, 3], base_from_page=page, motion=motion,
                                               surface=forearm) == (up == 0.0)


def test_with_turn_pen_off_the_pen_holds_the_pages_square_along_the_curve(kin, motion, page, forearm):
    """turn_pen False: the tip rides the curve as before, along each point's normal at the pen's height, but the pen
    holds the page's square (the pose the gauge holds measured the skin in) through the stroke and its hover."""
    q0 = _seed(kin, page)
    turned = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                        surface=forearm)
    square = tm.plan_op(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, speed_m_s=SPEED,
                        surface=forearm, turn_pen=False)
    drawing = square.phase == tm.plan.PHASE_DRAW
    gaps = np.array([forearm.height_over(tip)[0] for tip in square.tip[drawing]])
    assert np.max(np.abs(gaps)) < 1e-4                                      # still on the skin all the way round
    for k in np.flatnonzero(drawing)[::20]:
        assert -kin.fk(square.q[k])[:3, 2] @ page[:3, 2] > np.cos(np.radians(0.5))   # square to the page
        assert np.allclose(square.axis[k], forearm.height_over(square.tip[k])[1], atol=0.02)   # depth along the normal
    assert np.allclose(turned.tip[turned.phase == tm.plan.PHASE_DRAW][[0, -1]], square.tip[drawing][[0, -1]], atol=1e-4)
    hover = tm.plan_hover(_stroke(ACROSS), base_from_page=page, q_seed=q0, kin=kin, motion=motion, surface=forearm,
                          turn_pen=False)
    assert -kin.fk(hover.q[-1])[:3, 2] @ page[:3, 2] > np.cos(np.radians(0.5))


def test_a_liftoff_with_no_foot_under_the_tip_rises_the_standoff_up_the_pages_normal(motion, page, forearm):
    """A steep field's normal search can run off it (2026-10-21, the pen back from a dip): a non-finite gap or normal
    lifts the whole standoff along the page's normal instead of failing as a leg with no length."""
    tip, rot = (page @ [0.0, 0.0, 0.004, 1.0])[:3], np.eye(3)

    class Lost:
        base = forearm.base

        def height_over(self, point):
            return float("nan"), np.full(3, np.nan)

    legs = tm.plan._surface_liftoff(tip, rot, Lost(), motion, 1.0)
    standoff = float(motion["approach"]["standoff_m"])
    assert len(legs) == 1 and np.allclose(legs[0].p[-1], tip + standoff * page[:3, 2], atol=1e-6)
    assert np.allclose(legs[0].axis, page[:3, 2])
