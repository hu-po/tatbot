"""The vision frame-set wire reader (tatbot-vision-frame-set v1).

Read by the ROS station calibration. The board RGB-D witness
retention that shared this module went with the five-camera calibration
pipeline (2026-09-29).
"""
from __future__ import annotations

import json

MAX_WIRE_BYTES = 16 * 1024 * 1024


def unpack(blob, max_bytes=MAX_WIRE_BYTES):
    if not 4 < len(blob) <= max_bytes:
        raise ValueError('RGB-D reply exceeds the wire budget')
    size = int.from_bytes(blob[:4], 'big')
    if not 0 < size <= min(len(blob)-4, 256*1024):
        raise ValueError('invalid RGB-D header length')
    header = json.loads(blob[4:4+size])
    if header.get('magic') != 'tatbot-vision-frame-set' or header.get('version') != 1:
        raise ValueError('unknown RGB-D wire version')
    frames, cursor = {}, 4+size
    for frame in header['frames']:
        name = frame['metadata']['sensor_name']
        if name in frames or len(frame['payload']) != 1:
            raise ValueError('duplicate sensor or invalid payload')
        payload = next(iter(frame['payload'].values()))
        length = payload['bytes']
        if type(length) is not int or not 0 < length <= len(blob)-cursor:
            raise ValueError('invalid RGB-D payload size')
        frames[name] = (frame['metadata'], payload, blob[cursor:cursor+length])
        cursor += length
    if cursor != len(blob):
        raise ValueError('RGB-D reply contains trailing bytes')
    return header, frames
