"""The frame-set wire: the reader accepts one well-formed reply and nothing else, and pack writes the golden
fixture byte for byte (the Rust decoders read the same file)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from frame_set_wire import pack, unpack  # noqa: E402

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


# --- pack: the wrist tap's encoder, held byte for byte by the golden fixture -------------------------------

FIXTURES = Path(__file__).parent / 'fixtures'


def golden():
    doc = json.loads((FIXTURES / 'frame_set_wire_golden.json').read_text())
    inputs = doc['inputs']
    frames = [(f['metadata'], f['payload'], bytes.fromhex(f['data_hex'])) for f in inputs['frames']]
    return doc, frames, {k: inputs[k] for k in ('sequence', 'timestamp_ns', 'producer')}


def test_pack_reproduces_the_golden_fixture_byte_for_byte():
    doc, frames, args = golden()
    blob = pack(frames, **args)
    assert blob == (FIXTURES / 'frame_set_wire_golden.bin').read_bytes()
    assert hashlib.sha256(blob).hexdigest() == doc['expected']['sha256'] and len(blob) == doc['expected']['bytes']


def test_the_golden_fixture_unpacks_to_its_expected_values():
    doc, frames, _ = golden()
    expected = doc['expected']
    blob = (FIXTURES / 'frame_set_wire_golden.bin').read_bytes()
    header, decoded = unpack(blob)
    assert int.from_bytes(blob[:4], 'big') == expected['header_bytes']
    assert {k: header[k] for k in ('magic', 'version', 'sequence', 'timestamp_basis', 'timestamp_ns',
                                   'maximum_skew_ns', 'envelope')} == {k: expected[k] for k in (
        'magic', 'version', 'sequence', 'timestamp_basis', 'timestamp_ns', 'maximum_skew_ns', 'envelope')}
    assert [f['metadata']['sensor_name'] for f in header['frames']] == expected['frame_order']
    for name, (metadata, payload, data) in decoded.items():
        want = expected['frames'][name]
        assert metadata == want['metadata'] and {next(iter(want['payload'])): payload} == want['payload']
        assert hashlib.sha256(data).hexdigest() == want['data_sha256']
    raw = decoded['realsense2_depth'][2]
    assert [int.from_bytes(raw[i:i+2], 'little') for i in range(0, len(raw), 2)] == \
        expected['realsense2_depth_z16_row_major']
    # the round trip: what unpack returns packs back to the same bytes
    again = [(m, {'Encoded' if 'format' in p else 'Depth': p}, d) for m, p, d in decoded.values()]
    assert pack(again, sequence=header['sequence'], timestamp_ns=header['timestamp_ns'],
                producer=header['envelope']['producer']) == blob


def test_pack_refuses_what_a_strict_decoder_would():
    _, frames, args = golden()
    (depth_meta, depth_payload, depth_data), (color_meta, color_payload, color_data) = frames

    def refused(frames=frames, **change):
        with pytest.raises(ValueError):
            pack(frames, **{**args, **change})

    refused(frames=[])
    refused(frames=[frames[1], frames[1]])                                 # one sensor twice
    refused(frames=[(depth_meta, depth_payload, depth_data[:-2])])         # z16 short of its profile
    refused(frames=[(color_meta, {'Video': {'format': 'bgr8', 'bytes': 36}}, bytes(36))])   # not a bus payload
    float_fps = copy.deepcopy(color_meta)
    float_fps['profile']['fps_num'] = 30.0                                  # serde reads u32, not 30.0
    refused(frames=[(float_fps, color_payload, color_data)])
    unknown = {**copy.deepcopy(color_meta), 'extra': 1}
    refused(frames=[(unknown, color_payload, color_data)])
    z16_jpeg = copy.deepcopy(color_meta)
    z16_jpeg['profile']['format'] = 'z16'
    refused(frames=[(z16_jpeg, color_payload, color_data)])
    refused(producer={**args['producer'], 'host': 'x'})                    # the bus envelope denies unknown fields
    refused(timestamp_ns=-1)
