"""The stencil observer: a fiducial-tile print placed by its markers in a fixed RGB-D view, other references refused
by name, and the worker protocol stencild speaks."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'scripts/lib'), str(REPO/'scripts/vision')]
import stencil_generate  # noqa: E402
import stencil_observer  # noqa: E402
import stencil_reference  # noqa: E402
from test_fiducial_stencil import forearm_pose, overhead_view  # noqa: E402

OBSERVER = REPO/'scripts/vision/stencil_observer.py'
CALIBRATION = {'bundle_id': 'b' * 64}
ROBOT = {'calibration_id': 'b' * 64, 'frame': 'world', 'world_from_base': np.eye(4).tolist()}


@pytest.fixture
def tile_reference(tmp_path, capsys):
    """The silicone tile's seed 0, generated: its tracking.json."""
    assert stencil_generate.main(['--preset', 'silicone', '--seed', '0', '--output', str(tmp_path / 'tile')]) == 0
    capsys.readouterr()
    return tmp_path / 'tile' / 'tracking.json'


def observed(monkeypatch, tile_reference, camera_from_target, root_from_camera):
    """One observer turn over a synthetic overhead RGB-D view of the tile."""
    image, depth, rays, _ = overhead_view('silicone', camera_from_target)
    frame = {'image': image, 'depth_m': depth, 'timestamp_ns': 1_000_000_000}
    monkeypatch.setattr(stencil_observer, 'rgbd_frame', lambda f: ({'depth_m': f['depth_m'], 'rays': rays}, []))
    surface = stencil_observer.LiveSurface({'references': [str(tile_reference)], 'calibration': CALIBRATION,
                                            'robot_world': ROBOT, 'observer_epoch': 'test-1'})
    surface.capture_reader = lambda path, calibration, robot: {'overhead_depth_color': (frame, root_from_camera)}
    return surface.request({'captures': {'overhead-depth': 'capture'}, 'errors': {}})


def test_a_tile_is_placed_by_its_markers(monkeypatch, tile_reference):
    camera_from_target = np.eye(4)
    camera_from_target[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.05, 0.3]))[0]
    camera_from_target[:3, 3] = [0.02, -0.01, 0.75]
    root_from_camera = np.eye(4)
    root_from_camera[:3, 3] = [0.1, 0.4, 0.6]
    reply = observed(monkeypatch, tile_reference, camera_from_target, root_from_camera)
    [target] = reply['targets']
    assert target['source'] == 'measured' and target['support']['method'] == 'fiducial_markers'
    assert target['support']['markers']['read'] >= 20
    pose = np.asarray(target['world_from_target'])
    truth = root_from_camera @ camera_from_target
    assert np.linalg.norm(pose[:3, 3] - truth[:3, 3]) < 0.0015
    assert target['translation_sigma_m'] >= stencil_observer.TRANSLATION_SIGMA_M
    assert reply['observer_epoch'] == 'test-1'
    manifest = json.loads(tile_reference.read_text())
    assert reply['inventory_sha256'] == hashlib.sha256(json.dumps([manifest['reference_id']]).encode()).hexdigest()


def test_a_small_marker_tile_is_read_in_a_window_about_its_last_placement(monkeypatch, tmp_path, capsys):
    """The forearm tile, read upsampled: after a placement the next turn reads the window about it, and a print
    moved out of that window is found again in the whole frame. The silicone tile keeps no window."""
    assert stencil_generate.main(['--preset', 'forearm', '--seed', '0', '--output', str(tmp_path / 'tile')]) == 0
    capsys.readouterr()
    surface = stencil_observer.LiveSurface({'references': [str(tmp_path / 'tile' / 'tracking.json')],
                                            'calibration': CALIBRATION, 'robot_world': ROBOT})
    [pattern] = surface.windows
    windows, real = [], stencil_observer.tile_stencil.rgbd_pose
    monkeypatch.setattr(stencil_observer.tile_stencil, 'rgbd_pose',
                        lambda *args, **kwargs: windows.append(kwargs.get('roi_xyxy')) or real(*args, **kwargs))
    moved = forearm_pose(0, z=0.5)
    moved[:3, 3] += [0.2, 0.0, 0.0]
    for pose in (forearm_pose(0, z=0.5), forearm_pose(0, z=0.5), moved):
        image, depth, rays, _ = overhead_view('forearm', pose, look='transfer')
        frame = {'image': image, 'depth_m': depth, 'timestamp_ns': 1_000_000_000}
        monkeypatch.setattr(stencil_observer, 'rgbd_frame', lambda f, rays=rays: ({'depth_m': f['depth_m'],
                                                                                  'rays': rays}, []))
        surface.capture_reader = lambda path, calibration, robot, frame=frame: {
            'overhead_depth_color': (frame, np.eye(4))}
        [target] = surface.request({'captures': {'overhead-depth': 'capture'}, 'errors': {}})['targets']
        assert target['source'] == 'measured'
        assert np.linalg.norm(np.asarray(target['world_from_target'])[:3, 3] - pose[:3, 3]) < 0.001
    first, second = windows[1], windows[2]
    assert windows[0] is None and first == second and windows[3] is None   # read in the window, then the frame
    assert (first[2] - first[0]) * (first[3] - first[1]) < 0.1 * 1280 * 720
    assert surface.windows[pattern]['overhead_depth_color'] != first


def test_no_view_is_a_lost_target_with_the_reason(tile_reference):
    surface = stencil_observer.LiveSurface({'references': [str(tile_reference)], 'calibration': CALIBRATION,
                                            'robot_world': ROBOT})
    assert surface.windows == {}   # the silicone tile's 18 mm markers: the whole frame, plain reading
    reply = surface.request({'captures': {}, 'errors': {'overhead-depth': 'no fresh set'}})
    [target] = reply['targets']
    assert target['source'] == 'lost' and target['world_from_target'] is None
    assert target['support']['reason'] == 'markers: no fixed RGB-D view; overhead-depth: no fresh set'


def test_a_reference_that_is_not_a_tile_is_refused_by_name(tmp_path, tile_reference):
    manifest = json.loads(tile_reference.read_text())
    manifest.pop('tile')
    manifest['generator'] = {'generator_version': 'coded-fol-1', 'svg_sha256': '0' * 64}
    manifest['pattern_id'] = 'stencil-' + hashlib.sha256(json.dumps(manifest['generator'], sort_keys=True)
                                                         .encode()).hexdigest()
    manifest['reference_id'] = stencil_reference.reference_digest(manifest)
    (tile_reference.parent / 'tracking.json').write_text(json.dumps(manifest))
    surface = stencil_observer.LiveSurface({'references': [str(tile_reference)], 'calibration': CALIBRATION,
                                            'robot_world': ROBOT})
    [target] = surface.request({'captures': {}, 'errors': {}})['targets']
    assert target['pattern_id'] == manifest['pattern_id'] and target['source'] == 'lost'
    assert 'not a fiducial-tile print' in target['support']['reason']


def test_the_worker_answers_stencild(tmp_path):
    binding = tmp_path / 'binding.json'
    binding.write_text(json.dumps({'references': [], 'calibration': CALIBRATION, 'robot_world': ROBOT,
                                   'observer_epoch': 'e', 'evidence_kind': 'live-rgbd', 'publish_targets_only': True}))
    worker = subprocess.run([sys.executable, str(OBSERVER), str(binding), str(tmp_path)], text=True, timeout=60,
                            input=json.dumps({'captures': {}, 'errors': {}}) + '\n'
                            + json.dumps({'kind': 'support'}) + '\n', capture_output=True, check=True)
    ready, reply, refused = (json.loads(line) for line in worker.stdout.splitlines())
    assert ready == {'ready': True}
    assert reply['targets'] == [] and reply['motion_authority'] is False and 'timings_ms' in reply
    assert 'not an observer product' in refused['error']


def test_each_marker_id_names_its_four_anchor_points_at_their_own_depth(monkeypatch, tile_reference):
    """The drawing session judges a moved sheet by the same markers' corners as the overhead sees them
    (tatbot_session.geometry.page_marks): support.markers.ids[k] names anchor_points_m[4k:4k+4], in the layout's
    corner order, each on its corner's camera ray (the world is the overhead's colour optical frame) at the depth
    about its own pixel. No plane is published."""
    camera_from_target = np.eye(4)
    camera_from_target[:3, :3] = cv2.Rodrigues(np.array([0.65, -0.05, 0.3]))[0]
    camera_from_target[:3, 3] = [0.02, -0.01, 0.67]
    [target] = observed(monkeypatch, tile_reference, camera_from_target, np.eye(4))['targets']
    tile = stencil_observer.tile_stencil.load(tile_reference.parent)
    ids = target['support']['markers']['ids']
    points = np.asarray(target['support']['anchor_points_m']).reshape(-1, 4, 3)
    assert len(points) == len(ids) == len(set(ids)) >= 2 and 'plane' not in target['support']
    normal, origin = camera_from_target[:3, 2], camera_from_target[:3, 3]
    for i, seen in zip(ids, points, strict=True):
        truth = stencil_observer.tile_stencil.target_corners(tile, i) @ camera_from_target[:3, :3].T + origin
        assert np.abs(seen[:, :2] / seen[:, 2:] - truth[:, :2] / truth[:, 2:]).max() < 1.5e-3   # ~1 px at 645 px
        depth = (normal @ origin) / (seen / seen[:, 2:] @ normal)   # the skin's depth along each corner's ray
        assert np.abs(seen[:, 2] - depth).max() < 1e-4


def test_each_marker_names_where_it_was_read_in_the_overhead_image(monkeypatch, tile_reference):
    """The viewer draws the print's kept markers on the overhead's colour image: support.anchor_pixels[4k:4k+4] are
    marker markers.ids[k]'s corners where the anchor camera read them (pixel centres at integers), each within a
    pixel of the print's corner projected; a lost print reads none."""
    camera_from_target = np.eye(4)
    camera_from_target[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.05, 0.3]))[0]
    camera_from_target[:3, 3] = [0.02, -0.01, 0.7]
    [target] = observed(monkeypatch, tile_reference, camera_from_target, np.eye(4))['targets']
    tile = stencil_observer.tile_stencil.load(tile_reference.parent)
    ids, support = target['support']['markers']['ids'], target['support']
    pixels = np.asarray(support['anchor_pixels']).reshape(-1, 4, 2)
    assert support['anchor_camera'] == 'overhead_depth_color' and len(pixels) == len(ids) >= 10
    k = np.array([[645.0, 0, 639.5], [0, 645.0, 359.5], [0, 0, 1]])   # overhead_view's
    for i, read in zip(ids, pixels, strict=True):
        corners = stencil_observer.tile_stencil.target_corners(tile, i) @ camera_from_target[:3, :3].T
        projected = (corners + camera_from_target[:3, 3]) @ k.T
        assert np.abs(read - projected[:, :2] / projected[:, 2:]).max() < 1.0
    lost = stencil_observer.target('p', {'reference_id': 'r', 'seed': 0}, None, 'no view', 'b')
    assert lost['support']['anchor_pixels'] == []
