"""Native wrist replay: metric units, distortion, shared FK bias and explicit unavailable measurements."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from rgbd_geometry import depth_points, page_height_from_depth, wrist_clearance, wrist_frame_age
from tatbot_cli import ros_depth


def plane():
    xy = np.random.default_rng(1).uniform([-.045, -.065], [.045, .065], (600, 2))
    pts = np.c_[xy, .004 + .01*xy[:, 0] - .02*xy[:, 1]]
    return page_height_from_depth(pts, np.eye(4))


def test_relative_clearance_cancels_a_common_upstream_rigid_error_only():
    fit = plane()
    depth, tcp = np.eye(4), np.eye(4)
    depth[:3, 3] = [.02, -.01, .12]
    tcp[:3, 3] = [.01, -.01, .025]
    args = {'frame_age_s': .05, 'joint_age_s': .01}
    a = wrist_clearance(fit, np.eye(4), depth, tcp, **args)
    from scipy.spatial.transform import Rotation
    bias = np.eye(4)
    bias[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
    bias[:3, 3] = [.025, -.018, .010]
    shifted = {**fit, 'normal': (bias[:3, :3] @ fit['normal']).tolist()}
    b = wrist_clearance(shifted, bias, bias@depth, bias@tcp, **args)
    assert a['relative_clearance_m'] == pytest.approx(b['relative_clearance_m'])
    assert a['local_paper_height_m'] == pytest.approx(.0043)
    assert a['systematic_sigma_m'] == .008 and not a['used_for_drawing']
    assert not a['adoption_supported'] and a['reason'] == 'camera_to_tool_unqualified'
    tcp[2, 3] += .002
    c = wrist_clearance(fit, np.eye(4), depth, tcp, **args)
    assert c['relative_clearance_m']-a['relative_clearance_m'] == pytest.approx(.002*fit['normal'][2])


@pytest.mark.parametrize('fit,age,joint,reason', [(None,.1,.1,'rejected_page_plane'),
                                               (plane(),.7,.1,'stale_depth'),
                                               (plane(),.1,.7,'stale_joints'),
                                               (plane(),float('nan'),.1,'stale_depth')])
def test_unavailable_depth_is_never_a_used_measurement(fit, age, joint, reason):
    out = wrist_clearance(fit, np.eye(4), np.eye(4), np.eye(4), frame_age_s=age, joint_age_s=joint)
    assert out['reason'] == reason and not out['used_for_drawing']
    assert 'relative_clearance_m' not in out


def test_clearance_outside_observed_patch_is_explicit():
    tcp = np.eye(4)
    tcp[0, 3] = .1
    out = wrist_clearance(plane(), np.eye(4), np.eye(4), tcp, frame_age_s=.1, joint_age_s=.1)
    assert not out['local_supported'] and out['reason'] == 'tip_outside_plane_support'


def test_prior_rejection_is_preserved_on_the_same_points():
    xy = np.random.default_rng(2).uniform([-.04, -.06], [.04, .06], (600, 2))
    pts = np.c_[xy, np.full(len(xy), .035)]
    diagnostic = {}
    assert page_height_from_depth(pts, np.eye(4), diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'depth_outside_prior_band'
    assert diagnostic['prior_band_points'] == 0 and diagnostic['page_xy_points'] == 600
    pts[:, :2] += 1
    assert page_height_from_depth(pts, np.eye(4), diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'insufficient_page_xy_depth'


def test_native_deprojection_uses_declared_distortion_and_metres():
    depth = np.full((16, 24), .12)
    k = np.array([[25.,0,12],[0,25.,8],[0,0,1]])
    intr = {'fx':25.,'fy':25.,'ppx':12.,'ppy':8.,'model':'distortion.brown_conrady','coeffs':[.1,0,0,0,0]}
    a = depth_points(depth, k, intrinsics=intr)
    b = depth_points(depth, k)
    assert np.allclose(a[:,2], .12) and not np.allclose(a[:,:2], b[:,:2])
    intr['model'] = 'unknown'
    with pytest.raises(ValueError, match='unsupported'):
        depth_points(depth, k, intrinsics=intr)


@pytest.mark.parametrize('missing,exposure_lag,reason', [(False,0.,'accepted'),
                                                       (True,0.,'insufficient_page_xy_depth'),
                                                       (False,1.,'stale_depth')])
def test_capture_preserves_paired_raw_units_rgb_and_measurement_lifetime(tmp_path, missing, exposure_lag, reason):
    stamp = time.monotonic()
    raw = np.full((80, 120), 0 if missing else 1960, np.uint16)
    k = np.array([[150.,0,60],[0,150.,40],[0,0,1]])
    intr = {'fx':150.,'fy':150.,'ppx':60.,'ppy':40.,'model':'distortion.brown_conrady','coeffs':[0.]*5}
    cam = SimpleNamespace(k_depth=k, raw_depth=[raw], depth_metadata={'intrinsics':intr,'depth_units_m':.0001},
                          capture_timestamps=[{'received_monotonic_s':stamp,'received_at_ns':time.time_ns(),
                                               'depth_ms':time.time_ns()*1e-6-exposure_lag*1000,'depth_domain':'timestamp_domain.global_time','joints':{'q':[0.]*7,'measured_monotonic_s':stamp-.01}}])
    optical = np.diag([1.,-1.,-1.,1.])
    optical[2,3] = .2
    tcp = np.eye(4)
    tcp[2,3] = .020
    kin = SimpleNamespace(frame=lambda q, frame: optical, fk=lambda q:tcp)
    rgb = np.full((80,120,3),127,np.uint8)
    stem = tmp_path/'capture'
    fit, obs, result = ros_depth.retain(cam, [rgb], [raw*.0001], kin, np.zeros(7), np.eye(4), stem, 'right',
                                      reference={'captured_at_monotonic_s':stamp+.05})
    saved = np.load(stem.with_suffix('.npz'))
    assert np.array_equal(saved['depth_raw'][0],raw) and np.array_equal(saved['rgb_bgr'][0],rgb)
    assert obs['reason'] == reason
    if reason == 'accepted':
        assert fit['offset_m'] == pytest.approx(.004)
        assert result['relative_clearance_m'] == pytest.approx(.016)
    else:
        assert fit is None and 'relative_clearance_m' not in result
    meta = json.loads(stem.with_suffix('.json').read_text())
    assert meta['reference_lifetime'].startswith('this capture only')
    assert not meta['clearance']['used_for_drawing']


def test_exposure_staleness_is_not_hidden_by_a_fresh_host_receipt():
    stamp = {'received_at_ns': 10_000_000_000, 'received_monotonic_s': 20.,
             'depth_ms': 9000., 'depth_domain': 'timestamp_domain.global_time'}
    age, reason = wrist_frame_age([stamp], 20.1)
    assert age == pytest.approx(1.1) and reason is None
    report = wrist_clearance(plane(), np.eye(4), np.eye(4), np.eye(4), frame_age_s=age, joint_age_s=.01)
    assert report['reason'] == 'stale_depth' and not report['used_for_drawing']
    stamp['depth_domain'] = 'timestamp_domain.hardware_clock'
    assert wrist_frame_age([stamp], 20.1) == (None, 'unmapped_depth_clock')
    assert wrist_frame_age([], 20.1) == (None, 'missing_depth_timestamps')
    assert wrist_frame_age([{}], 20.1) == (None, 'missing_depth_timestamps')
