"""Shared roof-tag measurement with actual geometry/optics and synthetic capture transport."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tatbot_bridge import capture
from tatbot_bridge import station as producer
from tatbot_description import repo_root
from tatbot_motion import station

REPO = repo_root()


@pytest.fixture
def observations(tmp_path, monkeypatch):
    from fiducials import load_inventory, tag_model_corners
    from fiducials.detector import FiducialDetector

    repo = tmp_path/'repo'
    for rel in ('config/fiducials.json', 'urdf/palette.urdf'):
        path = repo/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO/rel).read_bytes())
    pose = station.rpy_matrix([.25, .12, .02], [0, 0, .24])
    base = np.diag([1., -1., -1., 1.])
    base[:3, :3] = Rotation.from_rotvec([.22, -.13, 0]).as_matrix() @ base[:3, :3]
    base[2, 3] = 1.
    registration = repo/'registration.json'
    registration.write_text(json.dumps({'schema': 'tatbot.arm-registration/1', 'arm': 'right',
                                        'calibration_id': 'synthetic-bundle', 'world_from_arm_base': base.tolist()}))
    target = load_inventory(repo/'config/fiducials.json').target('palette')
    model = tag_model_corners(target.edge_m)
    k = np.array([[800., 0, 640], [0, 800., 360], [0, 0, 1.]])
    tag = base @ pose @ station.palette_from_tag(repo/'urdf/palette.urdf')
    camera_corners = (tag @ np.c_[model, np.ones(4)].T)[:3].T
    corners = cv2.projectPoints(camera_corners, np.zeros(3), np.zeros(3), k, np.zeros(5))[0].reshape(4, 2)
    metadata = {'calibration_id': 'synthetic-bundle', 'attributes': {'intrinsics': json.dumps({
        'schema': 'tatbot.camera-intrinsics/1', 'fx': 800, 'fy': 800, 'ppx': 640, 'ppy': 360})}}
    state = SimpleNamespace(repo=repo, pose=pose, base=base, registration=registration, shots=[],
                            corners=corners, target=target, detections=True, closed=False, fault=False,
                            image=np.full((720, 1280, 3), 100, np.uint8), depth=None, metadata=metadata)

    class Camera:
        def __init__(self, repo):
            assert repo == state.repo

        def capture(self, after):
            if state.fault:
                raise RuntimeError('synthetic capture unavailable')
            shot = {'stamp_ns': max(after, time.time_ns()), 'image': state.image,
                    'metadata': copy.deepcopy(state.metadata), 'depth_m': state.depth}
            state.shots.append(shot)
            return shot

        def close(self):
            state.closed = True

    detector = SimpleNamespace(detect=lambda *args: [SimpleNamespace(tag_id=target.ids[0], corners_px=state.corners)]
                               if state.detections else [])
    monkeypatch.setattr(capture, 'Camera', Camera)
    monkeypatch.setattr(FiducialDetector, 'from_inventory', lambda *a, **k: detector)
    return state


def observe(state, tmp_path, shots=3):
    return producer.observe(state.repo, 'right', shots, tmp_path/'evidence', state.registration)


def test_shared_producer_places_the_station_and_preserves_exposure_evidence(observations, tmp_path):
    s = observations
    fix = observe(s, tmp_path)
    np.testing.assert_allclose(fix.base_from_palette, s.pose, atol=1e-7)
    assert fix.detail['shots'] == 3 and fix.source == 'overhead_tag'
    assert fix.detail['registration_sha256'] == hashlib.sha256(s.registration.read_bytes()).hexdigest()
    assert station.parse_utc(fix.measured_utc) == pytest.approx(s.shots[0]['stamp_ns']/1e9, abs=.001)
    assert s.closed and (tmp_path/'evidence/overhead.jpg').is_file()
    assert json.loads((tmp_path/'evidence/station.json').read_text()) == fix.as_dict()
    rows = [json.loads(row) for row in (tmp_path/'evidence/shots.jsonl').read_text().splitlines()]
    assert all(row['placed_by'] == 'corners' and row['rms_px'] < .001 for row in rows)


def test_aligned_depth_places_the_tag_without_apparent_size_bias(observations, tmp_path):
    s = observations
    s.corners = s.corners.mean(axis=0) + 1.015*(s.corners-s.corners.mean(axis=0))
    tag = s.base @ s.pose @ station.palette_from_tag(s.repo/'urdf/palette.urdf')
    ys, xs = np.mgrid[:s.image.shape[0], :s.image.shape[1]]
    rays = np.stack([(xs-640)/800, (ys-360)/800, np.ones_like(xs)], axis=-1)
    s.depth = ((tag[:3, 2] @ tag[:3, 3]) / (rays @ tag[:3, 2])).astype(np.float32)
    fix = observe(s, tmp_path)
    np.testing.assert_allclose(fix.base_from_palette[:3, 3], s.pose[:3, 3], atol=.0001)
    assert fix.detail['placed_by_depth'] == 3


@pytest.mark.parametrize('change', ['missing', 'schema', 'arm', 'bundle', 'singular', 'tag', 'shots'])
def test_unidentified_or_unusable_observation_inputs_refuse_before_capture(observations, tmp_path, change):
    s = observations
    value = json.loads(s.registration.read_text())
    if change == 'missing':
        s.registration.unlink()
    elif change == 'tag':
        path = s.repo/'urdf/palette.urdf'
        tree = ET.parse(path)
        tree.getroot().set('tag_pose_status', 'nominal')
        tree.write(path)
    elif change != 'shots':
        key = {'schema': 'schema', 'arm': 'arm', 'bundle': 'calibration_id', 'singular': 'world_from_arm_base'}[change]
        value[key] = np.zeros((4, 4)).tolist() if change == 'singular' else (None if change == 'bundle' else 'wrong')
        s.registration.write_text(json.dumps(value))
    with pytest.raises((producer.UnmeasuredError, ValueError)):
        observe(s, tmp_path, shots=1 if change == 'shots' else 3)
    assert not s.shots


@pytest.mark.parametrize('change,match', [('bundle', 'camera bundle'), ('capture', 'unavailable'), ('hidden', 'tag unseen'),
                                        ('dark', 'room may need a light')])
def test_capture_failures_close_transport_and_retain_rows(observations, tmp_path, change, match):
    s = observations
    if change == 'bundle':
        s.metadata['calibration_id'] = 'another-bundle'
    elif change == 'capture':
        s.fault = True
    else:
        s.detections = False
        if change == 'dark':
            s.image.fill(0)
    with pytest.raises((producer.UnmeasuredError, RuntimeError), match=match):
        observe(s, tmp_path)
    assert s.closed and (tmp_path/'evidence/shots.jsonl').is_file()
    assert not (tmp_path/'evidence/station.json').exists()


def test_calibration_delegates_to_the_same_producer_with_its_effective_registration(observations, tmp_path, monkeypatch):
    from tatbot_calib import cli, register

    s = observations
    monkeypatch.setattr(cli, '_stack', lambda repo: {'registration': {'right': str(s.registration)}})
    fix = cli.measure(s.repo, 'right', 3, tmp_path/'evidence')
    np.testing.assert_allclose(fix.base_from_palette, s.pose, atol=1e-7)
    assert register.Camera is capture.Camera and cli.UnmeasuredError is producer.UnmeasuredError


def test_bridge_session_calibration_dependencies_remain_acyclic():
    packages = {p.parent.name: ET.parse(p).getroot() for p in (REPO/'ros').glob('*/package.xml')}
    graph = {name: [n.text for n in p.findall('exec_depend') if n.text in packages] for name, p in packages.items()}

    def visit(name, path):
        assert name not in path, f'package cycle: {path+[name]}'
        for dependency in graph[name]:
            visit(dependency, path+[name])

    for name in graph:
        visit(name, [])
    assert 'tatbot_motion' in graph['tatbot_bridge']
    assert 'tatbot_bridge' in graph['tatbot_session'] and 'tatbot_bridge' in graph['tatbot_calib']
