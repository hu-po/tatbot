"""Palette dips with the real URDF and CLIK: the cap from the current assets, the tool as a body of revolution, the
dwell over the declared ink, and every phase planned and checked at its control ticks."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tatbot_motion import Kinematics, load_motion
from tatbot_motion import plan as pplan
from tatbot_motion.dip import (
    Cap,
    Dip,
    Tool,
    check_tick,
    plan_approach,
    plan_axial,
    plan_return,
    read_cap,
    verify_return,
)

REPO = Path(__file__).resolve().parents[3]
DOWN = np.diag([1.0, -1.0, -1.0])   # a tcp pointing down the base's -z


def settings(**changes):
    return {'above_ink_m': .0005, 'dwell_s': .4, 'hover_m': .004, 'speed_m_s': .01, 'wall_margin_m': .0015,
            'mm_per_dip': 40.0, **changes}


def cone():
    """A cartridge-like body: 16 mm across far up, tapering to a 1.25 mm end."""
    return Tool.from_profile([[-.01, .016], [.038, .016], [.073, .00125]])


def make_dip(*, cap=None, tool=None, level=.006, rotation=DOWN, **changes):
    cap = cap or Cap('inkcap_large_1', np.eye(4), .0005, .014, .007)
    return Dip(cap, tool or cone(), settings(**changes.pop('recipe', {})), level, rotation, **changes)


def test_inside_dimensions_and_support_frame_come_from_current_assets():
    cap = read_cap(REPO, 'inkcap_medium_1', 'right', np.eye(4))
    assert (cap.inner_floor_m, cap.rim_m, cap.bore_radius_m) == pytest.approx((.0005, .010, .005))
    np.testing.assert_allclose(cap.base_from_cap[:3, 3], [-.025846187, .025772555, .022])
    for slot, arm in [('M1', 'right'), ('inkcap_medium_1', 'left')]:
        with pytest.raises(ValueError, match='canonical cap slot'):
            read_cap(REPO, slot, arm, np.eye(4))


@pytest.mark.parametrize('mutation', ['parent', 'moving', 'duplicate'])
def test_cap_does_not_ignore_a_changed_urdf_joint(tmp_path, mutation):
    import xml.etree.ElementTree as ET

    (tmp_path / 'config').mkdir()
    (tmp_path / 'urdf').mkdir()
    (tmp_path / 'config/palette.yaml').write_bytes((REPO / 'config/palette.yaml').read_bytes())
    root = ET.parse(REPO / 'urdf/palette.urdf').getroot()
    joint = next(j for j in root.iter('joint') if j.find('child').get('link') == 'inkcap_medium_1')
    if mutation == 'parent':
        joint.find('parent').set('link', 'palette_camera')
    elif mutation == 'moving':
        joint.set('type', 'revolute')
    else:
        root.append(copy.deepcopy(joint))
    ET.ElementTree(root).write(tmp_path / 'urdf/palette.urdf')
    with pytest.raises(ValueError, match='fixed directly|exactly one'):
        read_cap(tmp_path, 'inkcap_medium_1', 'right', np.eye(4))


def test_the_tool_is_measured_up_from_its_end():
    tool = cone()
    assert tool.heights[0] == 0 and tool.radius_at(0) == pytest.approx(.00125)
    assert tool.radius_at(.035) == pytest.approx(.016) and tool.widest(.002) == pytest.approx(tool.radius_at(.002))
    pose = np.eye(4)
    pose[:3, :3] = DOWN
    pose[2, 3] = .05
    assert tool.lowest(pose) == pytest.approx(.05)   # pointing down, its end is lowest
    pose[:3, :3] = Rotation.from_rotvec([math.radians(30), 0, 0]).as_matrix() @ DOWN
    assert tool.lowest(pose) < .05                   # tilted, a wider ring hangs lower than the end
    with pytest.raises(ValueError, match='increasing'):
        Tool.from_profile([[.02, .01], [.01, .001]])


def test_the_dwell_holds_the_tools_end_just_over_the_declared_ink():
    dip = make_dip(level=.006)
    assert dip.dwell_height_m == pytest.approx(.0005 + .006 + .0005)
    assert dip.local(dip.dwell) == pytest.approx((dip.dwell_height_m, 0.0))
    assert dip.local(dip.hover) == pytest.approx((.014 + .004, 0.0))
    np.testing.assert_allclose(dip.dwell[:3, 2], [0, 0, -1], atol=1e-12)   # down the cap's axis
    assert dip.in_cap(dip.dwell) and not dip.in_cap(dip.hover)
    # the cone 7.5 mm into a 14 mm bore: its radius there plus the wall margin leaves the rest
    assert dip.slack_m == pytest.approx(.007 - float(dip.tool.radius_at(.014 - dip.dwell_height_m)) - .0015)


def test_a_tilted_cap_turns_the_tool_down_its_own_axis():
    support = np.eye(4)
    support[:3, :3] = Rotation.from_rotvec([.1, 0, 0]).as_matrix()
    dip = make_dip(cap=Cap('inkcap_large_1', support, .0005, .014, .007))
    np.testing.assert_allclose(dip.dwell[:3, 2], -support[:3, 2], atol=1e-12)


@pytest.mark.parametrize(('changes', 'message'), [
    ({'level': 0.0}, 'over the inner floor'),
    ({'level': .0136}, 'over the inner floor|stays over the rim'),
    ({'level': .013}, 'stays over the rim'),
    ({'cap': Cap('inkcap_small_1', np.eye(4), .0005, .010, .0035), 'level': .002}, 'bore beyond its wall margin'),
    ({'recipe': {'above_ink_m': -.001}}, 'nonnegative'),
    ({'reach_m': .0004}, 'never touch it'),
    ({'reach_m': .009, 'level': .002}, "fill it deeper"),
])
def test_a_dip_that_cannot_work_is_refused_before_anything_moves(changes, message):
    with pytest.raises(ValueError, match=message):
        make_dip(**changes)


def test_the_3rl_nozzle_fits_every_cap_size_filled_well():
    import sys

    sys.path.insert(0, str(REPO / 'scripts' / 'lib'))
    import tool_spec
    from tatbot_contracts.dip import resolve_dip

    sheet = tool_spec.load_tool('lutin-3rl-bugpin', REPO)
    tool, recipe = Tool.from_profile(sheet.profile), resolve_dip(sheet.raw['dip'], 'nighthawk_black')
    slack = []
    for slot in ('inkcap_large_1', 'inkcap_medium_1', 'inkcap_small_1'):
        cap = read_cap(REPO, slot, 'right', np.eye(4))
        for level in (.0005, .7 * (cap.rim_m - cap.inner_floor_m)):   # nearly empty, and filled well
            slack.append(Dip(cap, tool, recipe, level, DOWN).slack_m)
    assert min(slack) > 0.0005
    assert slack[0] > slack[2] > slack[4]   # the small cap leaves the least room


@pytest.fixture(scope='module')
def kin():
    return Kinematics.from_repo(arm='right')


@pytest.fixture(scope='module')
def motion():
    return load_motion()


@pytest.fixture
def reachable(kin):
    """A cap where the right arm reaches it, a page beside it, and the arm at the page standoff."""
    support = np.eye(4)
    support[:3, 3] = [.371, -.231, .025]
    q0 = np.array([-.5, 1., 1., -.3, 0, 0, .002])
    dip = make_dip(cap=Cap('inkcap_large_1', support, .0005, .014, .007), rotation=kin.fk(q0)[:3, :3])
    start = dip.hover.copy()
    start[:3, :3] = Rotation.from_rotvec([.12, .2, -.1]).as_matrix() @ start[:3, :3]
    start[:3, 3] = [.30, -.10, .04]
    page = start.copy()
    page[:3, :3] = start[:3, :3] @ DOWN
    page[:3, 3] -= .001 * page[:3, 2]
    return dip, kin.solve(start, q0), page


def run(dip, plan, *, check=None):
    stages = []

    def seen(stage, rows):
        stages.append((stage, np.array(rows)))
        if check:
            check(stage, rows)

    return plan(seen), stages


def test_approach_lifts_rises_turns_crosses_and_comes_down_the_axis(kin, motion, reachable):
    dip, seed, page = reachable
    top = .035
    traj, stages = run(dip, lambda check: plan_approach(dip, q_seed=seed, kin=kin, motion=motion, base_from_page=page,
                                                        station_top_m=top, check=check))
    assert [stage for stage, _ in stages] == ['page_lift', 'rise', 'align', 'transit', 'hover']
    assert list(traj.info['stages_s']) == [s for s, _ in stages] and traj.info['dip_phase'] == 'approach'
    assert np.isnan(traj.arc_m).all() and not (traj.phase == pplan.PHASE_DRAW).any()
    np.testing.assert_allclose(traj.q[:, 6], seed[6], atol=1e-12)
    poses = {stage: [kin.fk(q) for q in rows] for stage, rows in stages}
    lift = poses['page_lift']
    assert page[:3, 2] @ (lift[-1][:3, 3] - page[:3, 3]) >= motion['approach']['standoff_m'] - 1e-4
    for stage in ('align', 'transit'):   # nothing turns or crosses until the whole tool is over the station
        assert min(dip.tool.lowest(p) for p in poses[stage]) > top
    end = kin.fk(traj.q[-1])
    assert dip.local(end)[1] < 1e-3 and dip.local(end)[0] == pytest.approx(dip.hover_height_m, abs=1e-3)
    np.testing.assert_allclose(traj.info['return_tcp'], lift[-1], atol=2e-3)


def test_the_axial_phases_go_down_and_up_the_axis_and_keep_the_carriage(kin, motion, reachable):
    dip, _, _ = reachable
    seed = kin.solve(dip.hover, np.array([-.5, 1., 1., -.3, 0, 0, .002]))
    for phase, feedback in (('descend', pplan.PHASE_DESCEND), ('dwell', pplan.PHASE_SETTLE),
                            ('retract', pplan.PHASE_LIFT)):
        traj, stages = run(dip, lambda check, p=phase, q=seed: plan_axial(dip, p, q_seed=q, kin=kin, motion=motion,
                                                                          check=check))
        assert set(traj.phase) == {feedback} and traj.info['dip_phase'] == phase
        np.testing.assert_allclose(traj.q[:, 6], seed[6], atol=1e-12)
        positions = np.array([kin.fk(q)[:3, 3] for q in traj.q])
        np.testing.assert_allclose((positions - positions[0])[:, :2], 0, atol=motion['clik']['max_model_error_m'])
        assert sum(len(rows) - 1 for _, rows in stages) + 1 == traj.info['samples'] > len(traj.q)
        seed = traj.q[-1]
        if phase == 'descend':
            assert dip.local(kin.fk(seed))[0] == pytest.approx(dip.dwell_height_m, abs=2e-4)
        if phase == 'dwell':
            assert traj.duration_s >= dip.settings['dwell_s']
    assert dip.local(kin.fk(seed))[0] == pytest.approx(dip.hover_height_m, abs=2e-4)


def test_an_interrupted_descent_retracts_along_the_axis_and_cannot_dwell(kin, motion, reachable):
    dip, _, _ = reachable
    seed = kin.solve(dip.hover, np.array([-.5, 1., 1., -.3, 0, 0, .002]))
    down = plan_axial(dip, 'descend', q_seed=seed, kin=kin, motion=motion)
    stopped = down.q[len(down.q) * 3 // 4]
    with pytest.raises(ValueError, match='from its dwell height'):
        plan_axial(dip, 'dwell', q_seed=stopped, kin=kin, motion=motion)
    up = plan_axial(dip, 'retract', q_seed=stopped, kin=kin, motion=motion)
    np.testing.assert_allclose(kin.fk(up.q[-1])[:2, 3], kin.fk(stopped)[:2, 3], atol=motion['clik']['max_model_error_m'])
    assert not dip.in_cap(kin.fk(up.q[-1]))


def test_the_return_leaves_up_the_axis_and_restores_the_page_standoff(kin, motion, reachable):
    dip, seed, page = reachable
    entered = plan_approach(dip, q_seed=seed, kin=kin, motion=motion, base_from_page=page, station_top_m=.035,
                            check=lambda *a: None)
    target = np.asarray(entered.info['return_tcp'])
    traj, stages = run(dip, lambda check: plan_return(dip, q_seed=entered.q[-1], kin=kin, motion=motion,
                                                      return_tcp=target, station_top_m=.035, check=check))
    assert [stage for stage, _ in stages] == ['cap_exit', 'transit', 'over_page', 'page_return']
    first = kin.fk(entered.q[-1])
    for q in stages[0][1]:
        pose = kin.fk(q)
        assert dip.local(pose)[1] < dip.local(first)[1] + motion['clik']['max_model_error_m']
    verify_return(kin.fk(traj.q[-1]), target, motion)
    assert traj.info['return_tcp'] == target.tolist()


def test_ticks_off_the_axis_under_the_dwell_or_low_over_the_station_are_faults(reachable):
    dip, _, _ = reachable
    off = dip.dwell.copy()
    off[:3, 3] += [dip.slack_m + .0002, 0, 0]
    with pytest.raises(ValueError, match="off the cap's axis"):
        check_tick(dip, 'dwell', off, .035)
    check_tick(dip, 'cap_exit', off, .035)   # leaving rises from wherever the tool is
    low = dip.dwell.copy()
    low[:3, 3] -= dip.cap.axis * .0015
    with pytest.raises(ValueError, match='under its dwell height'):
        check_tick(dip, 'descend', low, .035)
    settled = dip.dwell.copy()
    settled[:3, 3] -= dip.cap.axis * .0006   # the arm settles a little into its dwell: going in and leaving pass
    check_tick(dip, 'dwell', settled, .035)
    check_tick(dip, 'retract', low, .035)   # leaving starts wherever the dwell left the tool
    floor = dip.at(dip.cap.inner_floor_m + .0002)
    with pytest.raises(ValueError, match="cap's floor"):
        check_tick(dip, 'retract', floor, .035)
    with pytest.raises(ValueError, match="station's top"):
        check_tick(dip, 'transit', dip.hover, dip.hover[2, 3])
    check_tick(dip, 'dwell', dip.dwell, .035)


def test_a_refused_check_or_an_unworkable_cap_never_reaches_the_controller(kin, motion, reachable, monkeypatch):
    dip, seed, page = reachable
    published = []
    monkeypatch.setattr(pplan, 'to_knots', lambda *a: published.append(a))

    def refuse(stage, rows):
        raise RuntimeError('the station refused the approach')

    with pytest.raises(RuntimeError, match='station refused'):
        plan_approach(dip, q_seed=seed, kin=kin, motion=motion, base_from_page=page, station_top_m=.035, check=refuse)
    with pytest.raises(ValueError, match='finite station top'):
        plan_approach(dip, q_seed=seed, kin=kin, motion=motion, base_from_page=page, station_top_m=math.nan,
                      check=lambda *a: None)
    support = dip.cap.base_from_cap.copy()
    support[:3, :3] = np.diag([1., -1., -1.])
    upside_down = make_dip(cap=Cap('inkcap_large_1', support, .0005, .014, .007))
    with pytest.raises(ValueError, match='upward cap axis'):
        plan_approach(upside_down, q_seed=seed, kin=kin, motion=motion, base_from_page=page, station_top_m=.035,
                      check=lambda *a: None)
    assert not published


def test_a_travel_the_clik_cannot_track_is_planned_again_slower(kin, motion, reachable, monkeypatch):
    from tatbot_motion import dip as dips
    from tatbot_motion.clik import PlanError

    dip, seed, page = reachable
    real, speeds = pplan._solve, []

    def lagging(build, *args, **kwargs):
        legs = build({})
        speeds.append(float(np.linalg.norm(np.diff(legs[1].p, axis=0), axis=1).max()))
        if len(speeds) == 1:
            raise PlanError('the tip trails its reference by 1.51 mm at t=5.30 s')
        return real(build, *args, **kwargs)

    monkeypatch.setattr(pplan, '_solve', lagging)
    plan_approach(dip, q_seed=seed, kin=kin, motion=motion, base_from_page=page, station_top_m=.035, check=lambda *a: None)
    assert len(speeds) == 2 and speeds[1] < 0.6 * speeds[0]
    monkeypatch.setattr(pplan, '_solve', lambda *a, **k: (_ for _ in ()).throw(PlanError('joint 3 reaches its limit')))
    with pytest.raises(PlanError, match='limit'):   # only a lagging tip is worth a slower try
        dips.plan_axial(dip, 'descend', q_seed=seed, kin=kin, motion=motion)
