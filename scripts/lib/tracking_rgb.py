"""Checksum-bound fixed RGB scan observations, with accepted robot registration."""

import hashlib
import json

import numpy as np
from robot_world import root_from_world
from stencil_cross_camera import rigid_transform
from visiond_wire import decode_video


def registered_frame(entry, camera, robot, manifest, calibration_id, read_payload):
    """Decoded stream formats share optics only with the exact calibrated profile."""
    metadata = entry['metadata']
    if (metadata.get('calibration_id') != calibration_id
            or manifest.get('geometry_calibration_id') != calibration_id
            or robot.get('calibration_id') != calibration_id):
        raise ValueError('tracking capture and robot registration calibration IDs differ')
    profile, expected = metadata['profile'], camera['profile']
    if any(profile[key] != expected[key] for key in ('stream', 'width', 'height', 'fps_num', 'fps_den')):
        raise ValueError('tracking profile differs from calibration')
    decoded_video = expected['format'] == 'h264' and profile['format'] in ('rgb8', 'bgr8', 'y8')
    converted_yuyv = (expected['format'] == 'yuyv' and profile['format'] == 'rgb8'
                      and metadata.get('attributes', {}).get('bus_source_format') == 'yuyv')
    if profile['format'] != expected['format'] and not (decoded_video or converted_yuyv):
        raise ValueError('tracking format differs from calibrated stream')
    if not 0 < int(profile['width'])*int(profile['height']) <= 16_000_000:
        raise ValueError('tracking image exceeds pixel budget')
    stamp = metadata['timestamps']['normalized_unix_ns']
    window = manifest['wrist_capture_window']
    if not window['after_ns'] <= stamp <= window['before_ns']:
        raise ValueError('tracking exposure outside capture window')
    intr, distortion = camera['intrinsics'], camera['distortion']
    models = {'none': 'None', 'brown_conrady': 'BrownConrady'}
    model = {'schema': 'tatbot.camera-intrinsics/1', 'width': intr['width'], 'height': intr['height'],
             'fx': intr['fx'], 'fy': intr['fy'], 'ppx': intr['cx'], 'ppy': intr['cy'],
             'distortion_model': models[distortion['model']], 'distortion_coefficients': distortion['coefficients']}
    pose = camera['world_from_camera']
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = np.asarray(pose['rotation']).reshape(3, 3)
    world_from_camera[:3, 3] = pose['translation_m']
    matrix = root_from_world(robot) @ rigid_transform(world_from_camera)
    return {'image': decode_video(read_payload(entry), profile), 'timestamp_ns': stamp,
            'color_metadata': metadata, 'camera_model': model,
            'source_identity': {'producer': manifest['producer'], 'profile': profile, 'calibration_id': calibration_id,
                                'camera_model': model, 'registration_sha256': hashlib.sha256(
                                    json.dumps(robot, sort_keys=True).encode()).hexdigest()}}, matrix
