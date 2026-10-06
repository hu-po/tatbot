"""Rigid blank interiors inherit only measured, enclosing original anchor evidence."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest
from surface_attachment import SurfaceAttachmentObserver, measured_xyz
from surface_attachment_inputs import pose, render_fixture
from surface_rigid_support import RigidPatchSupport

QUERY_PIXELS = np.array([[140, 110], [180, 110], [180, 130], [140, 130]])


@pytest.fixture(scope='module', params=['plane', 'curved'])
def scene(request):
    first, _ = render_fixture(shape=request.param, appearance='sparse-border')
    observer = SurfaceAttachmentObserver('paper', 'original', (10, 10, 300, 220))
    original = observer.observe(first, now_ns=first['capture_timestamp_ns'])
    assert original['accepted'], original['reason']
    points, valid = measured_xyz(first['rgbd'], QUERY_PIXELS, .003)
    assert valid.all()
    support = RigidPatchSupport(observer.reference['rgbd'], points,
        observer.material_support._points, observer.reference['digest'],
        observer.material_support.identity)
    return request.param, first, observer, original, support, points


def test_blank_interior_uses_real_original_anchor_evidence(scene):
    _, first, _, original, support, _ = scene
    assert first['rgbd']['gray'][105:135, 130:190].std() == 0
    result = support.evaluate(first['rgbd'], np.eye(4), original['material_support'])
    assert result['accepted'], result
    assert all(p['status'] == 'rigid_inferred' for p in result['points'])
    assert result['motion_authority'] is False and result['deformation_qualified'] is False
    assert result['anchor_evidence']['supported_count'] >= 3
    expected = hashlib.sha256(json.dumps(support._anchors.tolist(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert result['profile']['anchor_inventory_sha256'] == expected


def test_rigid_motion_pause_and_original_recovery(scene):
    shape, _, _, _, _, _ = scene
    first, _ = render_fixture(shape=shape, appearance='sparse-border')
    observer = SurfaceAttachmentObserver('paper', 'original', (10, 10, 300, 220))
    seed = observer.observe(first, now_ns=first['capture_timestamp_ns'])
    points, _ = measured_xyz(first['rgbd'], QUERY_PIXELS, .003)
    support = RigidPatchSupport(observer.reference['rgbd'], points, observer.material_support._points,
        seed['reference_digest'], observer.material_support.identity)
    before = support.identity
    for index, dark in enumerate([False, True, False], 1):
        frame, _ = render_fixture(shape=shape, appearance='sparse-border',
            world_from_material=pose(xyz=(.01, 0, .32), angles_deg=(0, 0, 3)),
            dark=dark, sequence=index, stamp=1_000_000_000+index*100_000_000)
        observed = observer.observe(frame, now_ns=frame['capture_timestamp_ns'])
        if dark:
            assert not observed['accepted']
            continue
        assert observed['accepted'], observed['reason']
        result = support.evaluate(frame['rgbd'], observed['transform_reference_to_camera'],
                                  observed['material_support'])
        assert result['accepted'], result
        assert support.identity == before and observed['reference_digest'] == seed['reference_digest']


def test_new_ink_inside_clear_paper_does_not_replace_original_anchors(scene):
    shape = scene[0]
    first, _ = render_fixture(shape=shape, appearance='sparse-border')
    observer = SurfaceAttachmentObserver('paper', 'original', (10, 10, 300, 220))
    seed = observer.observe(first, now_ns=first['stamp'])
    query, valid = measured_xyz(first['rgbd'], QUERY_PIXELS, .003)
    assert valid.all()
    support = RigidPatchSupport(observer.reference['rgbd'], query, observer.reference['xyz'],
                                seed['reference_digest'], observer.material_support.identity)
    frame, _ = render_fixture(shape=shape, appearance='sparse-border', sequence=1, stamp=1_100_000_000)
    cv2.polylines(frame['rgbd']['gray'], [np.array([[135, 105], [181, 105], [181, 135], [135, 135]])],
                  True, 25, thickness=2)
    result = observer.observe(frame, now_ns=frame['stamp'])
    assert result['accepted'], result['reason']
    inferred = support.evaluate(frame['rgbd'], result['transform_reference_to_camera'], result['material_support'])
    assert inferred['accepted']
    assert result['reference_digest'] == seed['reference_digest']
    assert observer.reference['rgbd']['gray'][105:136, 135:182].std() == 0


def altered_report(original, keep=(), contradict=None):
    report = copy.deepcopy(original['material_support'])
    for point in report['points']:
        if point['id'] not in keep:
            point.update(status='unknown', supported=False, current_pixel=None)
        if point['id'] == contradict:
            point.update(status='contradictory', supported=False, current_pixel=[100., 100.])
    report['supported_count'] = sum(p['supported'] for p in report['points'])
    report['contradictory_count'] = sum(p['status'] == 'contradictory' for p in report['points'])
    report['accepted'] = report['supported_count'] == report['point_count']
    return report


def test_lost_and_contradictory_anchors_cannot_infer_blank_material(scene):
    _, first, _, original, support, _ = scene
    for report in [altered_report(original), altered_report(original, contradict=0)]:
        result = support.evaluate(first['rgbd'], np.eye(4), report)
        assert not result['accepted'] and result['supported_count'] == 0


def test_same_side_anchors_do_not_enclose_drawing(scene):
    _, first, _, original, support, _ = scene
    keep = [p['id'] for p in original['material_support']['points']
            if p['supported'] and p['reference_pixel'][0] < 125]
    result = support.evaluate(first['rgbd'], np.eye(4), altered_report(original, keep))
    assert not result['accepted']
    assert result['reason'] in ('drawing_outside_supported_anchor_hull', 'ill_conditioned_rigid_anchors',
                                'insufficient_rigid_anchors')


def test_depth_loss_and_foreground_occlusion_pause_inferred_drawing(scene):
    _, first, _, original, support, _ = scene
    for depth in (0., .1):
        rgbd = {**first['rgbd'], 'depth_m': first['rgbd']['depth_m'].copy()}
        rgbd['depth_m'][105:135, 130:190] = depth
        result = support.evaluate(rgbd, np.eye(4), original['material_support'])
        assert not result['accepted'] and result['supported_count'] == 0


@pytest.mark.parametrize('mutation', ['identity', 'pose', 'bool-id', 'missing-pixel', 'authority'])
def test_malformed_or_rebound_anchor_report_refused(scene, mutation):
    _, first, _, original, support, _ = scene
    report = copy.deepcopy(original['material_support'])
    if mutation == 'identity':
        report['request_sha256'] = 'f'*64
    elif mutation == 'pose':
        report['transform_reference_to_camera'][0][3] = .001
    elif mutation == 'bool-id':
        report['points'][0]['id'] = False
    elif mutation == 'missing-pixel':
        del report['points'][0]['current_pixel']
    else:
        report['motion_authority'] = True
    with pytest.raises(ValueError):
        support.evaluate(first['rgbd'], np.eye(4), report)


def test_reference_queries_cannot_bridge_disconnected_background(scene):
    _, first, observer, _, _, points = scene
    rgbd = {**observer.reference['rgbd'], 'depth_m': first['rgbd']['depth_m'].copy()}
    rgbd['depth_m'][:, 157:163] = 0
    anchors, valid = measured_xyz(rgbd, np.array([[100, 80], [100, 160], [220, 80], [220, 160]]), .003)
    assert valid.all()
    with pytest.raises(ValueError, match='disconnected'):
        RigidPatchSupport(rgbd, points, anchors,
            observer.reference['digest'], observer.material_support.identity)
