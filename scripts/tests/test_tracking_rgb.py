"""Retained fixed RGB uses measured registration and rejects changed evidence."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'scripts/lib'), str(REPO/'scripts/vision')]
from tracking_rgb import registered_frame  # noqa: E402


def fixture(tmp_path):
    capture = tmp_path/'capture'
    capture.mkdir()
    pixels = bytes([11, 22, 33])*24
    (capture/'tracking.pixels').write_bytes(pixels)
    profile = {'stream': 'color', 'width': 6, 'height': 4, 'fps_num': 15, 'fps_den': 1, 'format': 'rgb8'}
    entry = {'metadata': {'sensor_name': 'fixed', 'calibration_id': 'measured', 'profile': profile,
                         'timestamps': {'normalized_unix_ns': 100}},
             'payload_file': 'tracking.pixels', 'payload_bytes': len(pixels), 'sha256': hashlib.sha256(pixels).hexdigest()}
    camera = {'profile': dict(profile, format='h264'), 'intrinsics': {'width': 6, 'height': 4,
              'fx': 8., 'fy': 8., 'cx': 2.5, 'cy': 1.5},
              'distortion': {'model': 'brown_conrady', 'coefficients': [0.]*5},
              'world_from_camera': {'rotation': np.eye(3).ravel().tolist(), 'translation_m': [.1, .2, .3]}}
    robot = {'calibration_id': 'measured', 'world_from_base': np.eye(4).tolist()}
    robot['world_from_base'][0][3] = .4
    manifest = {'schema': 'tatbot.stencil-tracking-capture/1', 'producer': {'node': 'capture-owner', 'run_id': 'test'},
                'geometry_calibration_id': 'measured', 'wrist_capture_window': {'after_ns': 90, 'before_ns': 110},
                'frames': {'fixed': entry}}
    (tmp_path/'overhead-calibration.json').write_text(json.dumps({'bundle_id': 'measured', 'cameras': {'fixed': camera}}))
    (tmp_path/'overhead-robot-world.json').write_text(json.dumps(robot))
    (capture/'tracking-1.json').write_text(json.dumps(manifest))
    return entry, camera, robot, manifest


def test_registered_fixed_frame_preserves_optics_color_timestamp_and_root_chain(tmp_path):
    import capture_geometry
    entry, camera, robot, manifest = fixture(tmp_path)
    frame, matrix = registered_frame(entry, camera, robot, manifest, 'measured',
                                     lambda entry: capture_geometry._verified_overhead_payload(tmp_path/'capture', entry))
    np.testing.assert_array_equal(frame['image'][0, 0], [33, 22, 11])
    np.testing.assert_allclose(matrix[:3, 3], [-.3, .2, .3])
    assert frame['timestamp_ns'] == 100
    assert frame['camera_model']['ppx'] == 2.5
    assert 'depth_m' not in frame


@pytest.mark.parametrize('joints', [[0.]*6, [.2, .3, .5, -.2, .15, .4]])
def test_fixed_rgb_and_depth_roundtrip_solver_root_fk_with_arm_mount(tmp_path, joints):
    """A point seen by a camera must return to solver FK, with one arm mount."""
    import arm_kinematics as dk
    import capture_geometry
    from calib_synth import pose_joint_values, vector_to_rotation

    entry, camera, robot, manifest = fixture(tmp_path)
    root_from_wrist = dk.urdf_chain().link_pose('right/link_6', pose_joint_values({'joints': joints}, 'right'))
    assert np.linalg.norm(dk.BASE_IN_ROOT) > .1  # A zero mount would hide the original bug.
    world_from_root = np.eye(4)
    world_from_root[:3, :3] = vector_to_rotation(np.array([.4, -.2, .6]))
    world_from_root[:3, 3] = [.2, -.1, .5]
    robot['world_from_base'] = world_from_root.tolist()  # Actual solver's serialized convention.
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = vector_to_rotation(np.array([-.2, .1, -.3]))
    world_from_camera[:3, 3] = [-.3, .25, -.5]
    camera['world_from_camera'] = {'rotation': world_from_camera[:3, :3].ravel().tolist(),
                                   'translation_m': world_from_camera[:3, 3].tolist()}
    calibration = {'bundle_id': 'measured', 'cameras': {
        'overhead_depth_depth': camera, 'overhead_depth_color': camera}}
    _, _, root_from_depth = capture_geometry._overhead_registration(calibration, robot)
    _, root_from_rgb = registered_frame(entry, camera, robot, manifest, 'measured',
                                         lambda _: (tmp_path/'capture/tracking.pixels').read_bytes())
    # Match the solver equation Z @ root_from_link @ point_in_link, including
    # translation AND orientation, then independently invert each camera sighting.
    points_link = np.array([[0., 0., 0., 1.], [.03, .02, .01, 1.], [-.02, .01, .04, 1.]]).T
    points_root = root_from_wrist @ points_link
    points_world = world_from_root @ points_root
    points_camera = np.linalg.solve(world_from_camera, points_world)
    for matrix in (root_from_rgb, root_from_depth):
        np.testing.assert_allclose(matrix @ points_camera, points_root, atol=1e-12)


@pytest.mark.parametrize('source_format', ['yuyv', None, 'bgr8'])
def test_bus_color_conversion_requires_the_original_calibrated_format(tmp_path, source_format):
    entry, camera, robot, manifest = fixture(tmp_path)
    camera['profile']['format'] = 'yuyv'
    entry['metadata']['attributes'] = {'bus_source_format': source_format}
    data = (tmp_path/'capture/tracking.pixels').read_bytes()
    if source_format != 'yuyv':
        with pytest.raises(ValueError, match='format differs'):
            registered_frame(entry, camera, robot, manifest, 'measured', lambda _: data)
    else:
        frame, _ = registered_frame(entry, camera, robot, manifest, 'measured', lambda _: data)
        np.testing.assert_array_equal(frame['image'][0, 0], [33, 22, 11])
        assert frame['color_metadata']['attributes']['bus_source_format'] == 'yuyv'
