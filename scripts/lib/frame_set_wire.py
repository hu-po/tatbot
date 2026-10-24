"""The vision frame-set wire (tatbot-vision-frame-set v1): `unpack`, read by the ROS station calibration and the
Rerun bridge, and `pack`, the ROS wrist tap's encoder.

The Rust producer owns the format (rust/visiond/src/transport.rs encode_frame_set; the bus envelope in
frame_bus.rs). `pack` writes a bus frame set, JPEG colour and z16 depth, the way serde writes it: compact JSON,
each struct's fields in declaration order, map keys (attributes, the envelope) sorted, payloads in sensor-name
order. Its checks are the strict decoders' (transport::decode_frame_set, frame_bus::decode, `unpack`), so a set
it packs is one they accept: a bus subscriber that met a bad set would stop. Golden fixture:
scripts/tests/fixtures/frame_set_wire_golden.bin.
"""
from __future__ import annotations

import json

MAGIC, VERSION = 'tatbot-vision-frame-set', 1
SCHEMA = 'tatbot.frame-set/1'                 # the bus envelope's schema
BASIS = 'normalized_unix_ns'
MAX_WIRE_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 256 * 1024                 # unpack's bound; Rust allows 4 MiB
MAX_FRAMES = 16
METADATA = ('sensor_name', 'sensor_kind', 'sequence', 'profile', 'timestamps', 'dropped_before', 'calibration_id',
            'flags', 'attributes')
PROFILE = ('stream', 'width', 'height', 'fps_num', 'fps_den', 'format')
TIMESTAMPS = ('source_ns', 'source_domain', 'rtp_timestamp', 'pipeline_pts_ns', 'pipeline_dts_ns',
              'host_monotonic_ns', 'host_unix_ns', 'normalized_unix_ns')
PRODUCER = ('node', 'pid', 'run_id', 'sha')
JPEG_COLOURS = ('rgb8', 'bgr8', 'y8')        # a JPEG's profile format: the pixels it decodes back to
SENSOR_KINDS = ('po_e', 'real_sense')                                    # serde snake_case of the Rust enums
PIXEL_FORMATS = ('h264', 'jpeg', 'bgr8', 'rgb8', 'yuyv', 'z16', 'y8')
TIMESTAMP_DOMAINS = ('unknown', 'camera_ntp', 'rtp', 'real_sense_global', 'real_sense_hardware', 'host_monotonic',
                     'host_unix')


def unpack(blob, max_bytes=MAX_WIRE_BYTES):
    if not 4 < len(blob) <= max_bytes:
        raise ValueError('RGB-D reply exceeds the wire budget')
    size = int.from_bytes(blob[:4], 'big')
    if not 0 < size <= min(len(blob)-4, MAX_HEADER_BYTES):
        raise ValueError('invalid RGB-D header length')
    header = json.loads(blob[4:4+size])
    if header.get('magic') != MAGIC or header.get('version') != VERSION:
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


def frame_metadata(sensor_name: str, *, sequence: int, stream: str, width: int, height: int, fmt: str,
                   unix_ns: int, monotonic_ns: int = 0, fps: int = 30, attributes=None,
                   sensor_kind: str = 'real_sense') -> dict:
    """One frame's FrameMetadata (rust/visiond/src/lib.rs), stamped by the host's wall clock (`unix_ns`, which is
    also its normalized time) and with string `attributes`."""
    return {'sensor_name': sensor_name, 'sensor_kind': sensor_kind, 'sequence': sequence,
            'profile': {'stream': stream, 'width': width, 'height': height, 'fps_num': fps, 'fps_den': 1,
                        'format': fmt},
            'timestamps': {'source_ns': unix_ns, 'source_domain': 'host_unix', 'rtp_timestamp': None,
                           'pipeline_pts_ns': None, 'pipeline_dts_ns': None, 'host_monotonic_ns': monotonic_ns,
                           'host_unix_ns': unix_ns, 'normalized_unix_ns': unix_ns},
            'dropped_before': 0, 'calibration_id': None, 'flags': [], 'attributes': dict(attributes or {})}


def jpeg(data: bytes) -> dict:
    return {'Encoded': {'format': 'jpeg', 'bytes': len(data)}}


def depth(width: int, height: int, data: bytes) -> dict:
    return {'Depth': {'width': width, 'height': height, 'bytes': len(data)}}


def _count(value, what: str, low: int = 0) -> int:
    if type(value) is not int or value < low:
        raise ValueError(f'{what} must be an integer >= {low}, not {value!r}')
    return value


def _fields(value, fields, what: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f'{what} fields are {sorted(value or ())}, not {sorted(fields)}')
    return {key: value[key] for key in fields}


def _variants(metadata: dict) -> None:
    """The enum and optional fields serde would refuse to read."""
    stamps = metadata['timestamps']
    checks = {'sensor_kind': (metadata['sensor_kind'], SENSOR_KINDS),
              'profile format': (metadata['profile']['format'], PIXEL_FORMATS),
              'source_domain': (stamps['source_domain'], TIMESTAMP_DOMAINS)}
    for what, (value, allowed) in checks.items():
        if value not in allowed:
            raise ValueError(f'{what} {value!r} is not one of {allowed}')
    for key in ('source_ns', 'rtp_timestamp', 'pipeline_pts_ns', 'pipeline_dts_ns', 'normalized_unix_ns'):
        if stamps[key] is not None and type(stamps[key]) is not int:
            raise ValueError(f'{key} is an integer or null')
    if metadata['calibration_id'] is not None and not isinstance(metadata['calibration_id'], str):
        raise ValueError('calibration_id is a string or null')
    if not isinstance(metadata['flags'], list) or not all(isinstance(f, str) for f in metadata['flags']):
        raise ValueError('flags are a list of strings')


def _metadata(metadata: dict) -> dict:
    """The metadata in serde's field order, checked as FrameRecord::validate checks it."""
    out = _fields(metadata, METADATA, 'metadata')
    out['profile'] = profile = _fields(out['profile'], PROFILE, 'profile')
    out['timestamps'] = stamps = _fields(out['timestamps'], TIMESTAMPS, 'timestamps')
    if not str(out['sensor_name']).strip() or not str(profile['stream']).strip():
        raise ValueError('a frame needs a sensor and a stream name')
    for key in ('width', 'height', 'fps_num', 'fps_den'):
        _count(profile[key], f'profile {key}', 1)
    if not 0.1 <= profile['fps_num'] / profile['fps_den'] <= 240.0:
        raise ValueError('stream FPS must be between 0.1 and 240 Hz')
    _count(stamps['host_unix_ns'], 'host_unix_ns', 1)
    _count(stamps['host_monotonic_ns'], 'host_monotonic_ns')
    _count(out['sequence'], 'sequence')
    _count(out['dropped_before'], 'dropped_before')
    _variants(out)
    attributes = out['attributes']
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in attributes.items()):
        raise ValueError('frame attributes are strings')
    out['attributes'] = dict(sorted(attributes.items()))
    return out


def _payload(metadata: dict, payload: dict, data: bytes) -> dict:
    """The payload descriptor, checked as frame_bus::decode checks a bus frame: JPEG colour or z16 depth of the
    metadata's size."""
    profile = metadata['profile']
    variant, fields = next(iter(payload.items())) if isinstance(payload, dict) and len(payload) == 1 else ('', {})
    if not isinstance(fields, dict) or fields.get('bytes') != len(data) or not data:
        raise ValueError(f"{metadata['sensor_name']}: the payload descriptor does not match its {len(data)} bytes")
    if variant == 'Encoded' and set(fields) == {'format', 'bytes'} and fields['format'] == 'jpeg':
        if profile['format'] not in JPEG_COLOURS:
            raise ValueError(f"{metadata['sensor_name']}: a JPEG's profile format is one of {JPEG_COLOURS}")
        return {'Encoded': {'format': 'jpeg', 'bytes': len(data)}}
    if variant == 'Depth' and set(fields) == {'width', 'height', 'bytes'}:
        size = (fields['width'], fields['height'])
        if size != (profile['width'], profile['height']) or profile['format'] != 'z16' \
                or len(data) != 2 * profile['width'] * profile['height']:
            raise ValueError(f"{metadata['sensor_name']}: z16 depth of its profile's size")
        return {'Depth': {'width': profile['width'], 'height': profile['height'], 'bytes': len(data)}}
    raise ValueError(f"{metadata['sensor_name']}: a bus frame is a JPEG or z16 depth, not {payload}")


def _envelope(producer: dict, sequence: int, wall_ns: int, basis: str) -> dict:
    """tatbot_bus::Envelope (deny_unknown_fields) as serde_json::Value writes it: keys sorted."""
    producer = _fields(producer, PRODUCER, 'producer')
    if not all(isinstance(producer[k], str) for k in ('node', 'run_id', 'sha')):
        raise ValueError('producer node, run_id and sha are strings')
    _count(producer['pid'], 'producer pid')
    return {'payload': {'wire_version': VERSION}, 'producer': producer, 'schema': SCHEMA, 'seq': sequence,
            'stamp': {'basis': basis, 'mono_ns': 0, 'wall_ns': wall_ns}}


def pack(frames, *, sequence: int, timestamp_ns: int, producer: dict | None = None, basis: str = BASIS,
         maximum_skew_ns: int = 0, max_bytes: int = MAX_WIRE_BYTES) -> bytes:
    """One frame set from [(metadata, payload, data)] (payload: jpeg() or depth()). With `producer` ({node, pid,
    run_id, sha}) it carries the bus envelope (schema tatbot.frame-set/1, seq = `sequence`, stamped
    `timestamp_ns`), without one it is a bare transport set. ValueError when a strict decoder would refuse it."""
    _count(sequence, 'sequence')
    _count(timestamp_ns, 'timestamp_ns')
    _count(maximum_skew_ns, 'maximum_skew_ns')
    checked = sorted(((_metadata(m), p, bytes(d)) for m, p, d in frames), key=lambda f: f[0]['sensor_name'])
    names = [m['sensor_name'] for m, _, _ in checked]
    if not 0 < len(names) <= MAX_FRAMES or len(set(names)) != len(names):
        raise ValueError(f'a frame set holds 1-{MAX_FRAMES} distinct sensors, not {names}')
    header = {'magic': MAGIC, 'version': VERSION, 'sequence': sequence, 'timestamp_basis': basis,
              'timestamp_ns': timestamp_ns, 'maximum_skew_ns': maximum_skew_ns,
              'frames': [{'metadata': m, 'payload': _payload(m, p, d)} for m, p, d in checked]}
    if producer is not None:
        header['envelope'] = _envelope(producer, sequence, timestamp_ns, basis)
    encoded = json.dumps(header, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    blob = b''.join([len(encoded).to_bytes(4, 'big'), encoded, *(d for _, _, d in checked)])
    if len(encoded) > MAX_HEADER_BYTES or len(blob) > max_bytes:
        raise ValueError(f'frame set of {len(blob)} bytes (header {len(encoded)}) exceeds the wire budget')
    return blob
