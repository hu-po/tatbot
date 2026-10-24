"""Decode registered, checksum-bound live captures from camera owners (the stencil observer's input)."""

import json
from pathlib import Path

import schemas
from rgbd_geometry import require_calibrated_model, rgbd_frame
from tracking_rgb import registered_frame
from view_assets import bound_read, read
from visiond_wire import decode_depth


def load_frames(path):
    path = Path(path)
    manifest = json.loads(read(path, 256*1024))
    if not schemas.is_schema(manifest, schemas.SURFACE, 'live-capture') or not 1 <= len(manifest['frames']) <= 5:
        raise ValueError('invalid live display capture')
    frames = {}
    for name, entry in manifest['frames'].items():
        if name != entry['metadata']['sensor_name']:
            raise ValueError('live capture sensor identity changed')
        data = bound_read(path.parent, entry['payload_file'], entry['sha256'])
        if len(data) != entry['payload_bytes']:
            raise ValueError('live capture payload length changed')
        frames[name] = (entry, data)
    return manifest, frames


def capture(path, calibration, robot):
    manifest, frames = load_frames(Path(path))
    views = {}
    for name, (entry, data) in frames.items():
        if entry['metadata']['profile']['format'] == 'z16':
            continue
        frame, matrix = registered_frame(entry, calibration['cameras'][name], robot, manifest,
                                         calibration['bundle_id'], lambda _, data=data: data)
        views[name] = (frame, matrix)
    for entry, data in frames.values():
        if entry['metadata']['profile']['format'] != 'z16':
            continue
        frame, _ = views[entry['metadata']['attributes']['aligned_to']]
        add_depth(frame, entry['metadata'], data, calibration, manifest)
    return views


def add_depth(frame, metadata, data, calibration, manifest):
    name, color = metadata['sensor_name'], metadata['attributes']['aligned_to']
    cameras = calibration['cameras']
    if any(cameras[name][key] != cameras[color][key] for key in ('intrinsics', 'distortion', 'world_from_camera')):
        raise ValueError('bound aligned RGB-D calibration differs')
    if metadata['calibration_id'] != calibration['bundle_id'] or metadata['profile'] != cameras[name]['profile']:
        raise ValueError('depth calibration identity or profile differs')
    window = manifest['wrist_capture_window']
    if not window['after_ns'] <= metadata['timestamps']['normalized_unix_ns'] <= window['before_ns']:
        raise ValueError('depth exposure outside display window')
    frame.update(depth_m=decode_depth(data, metadata['profile']).astype(float)*float(metadata['attributes']['depth_units_m']),
                 depth_metadata=metadata)
    _, warnings = rgbd_frame(frame)
    if warnings:
        raise ValueError('; '.join(warnings))
    attributes = metadata['attributes']
    require_calibrated_model(json.loads(attributes['intrinsics']), cameras[name], attributes.get('device_serial'))
