"""The frame-set wire reader accepts one well-formed reply and nothing else."""
from __future__ import annotations

import copy
import json

import pytest
from board_rgbd_evidence import unpack  # noqa: E402

STAMP = 10_000_000_000


def wire(change=None):
    attributes = {'capture_epoch': 'epoch', 'device_serial': 'device', 'physical_arm': 'left',
                  'capture_owner_role': 'left-wrist-cameras',
                  'intrinsics': json.dumps({'width': 2, 'height': 2, 'fx': 2, 'fy': 2, 'ppx': 1, 'ppy': 1})}
    metadata = {'sequence': 1, 'profile': {'width': 2, 'height': 2, 'fps_num': 30, 'fps_den': 1},
                'timestamps': {'normalized_unix_ns': STAMP}, 'attributes': attributes}
    color, depth = copy.deepcopy(metadata), copy.deepcopy(metadata)
    color['sensor_name'], depth['sensor_name'] = 'wrist_color', 'wrist_depth'
    color['profile']['format'], depth['profile']['format'] = 'rgb8', 'z16'
    depth['attributes'].update(aligned_to='wrist_color', depth_units_m='0.0001')
    header = {'magic': 'tatbot-vision-frame-set', 'version': 1,
              'envelope': {'producer': {'node': 'camera-host'}},
              'frames': [{'metadata': color, 'payload': {'Encoded': {'format': 'jpeg', 'bytes': 4}}},
                         {'metadata': depth, 'payload': {'Depth': {'width': 2, 'height': 2, 'bytes': 8}}}]}
    if change:
        change(header)
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(4, 'big')+encoded+b'JPEG'+bytes(8)


def test_a_well_formed_reply_unpacks_by_sensor():
    header, frames = unpack(wire())
    assert header['version'] == 1 and set(frames) == {'wrist_color', 'wrist_depth'}
    assert frames['wrist_color'][2] == b'JPEG' and frames['wrist_depth'][2] == bytes(8)


def test_corrupt_wire_is_rejected():
    for blob in [wire()[:-1], wire()+b'extra', b'\xff'*5]:
        with pytest.raises(ValueError):
            unpack(blob)
