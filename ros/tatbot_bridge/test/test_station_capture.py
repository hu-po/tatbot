"""Owner/freshness/alignment checks use the actual frame-set decoder and synthetic bus replies."""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_bridge import capture


def packet(*, owner='synthetic-owner', stamp=None, color=True, aligned=True):
    frames, bodies = [], []
    metadata = {'profile': {'width': 4, 'height': 3, 'format': 'bgr8'},
                'timestamps': {'normalized_unix_ns': time.time_ns() if stamp is None else stamp},
                'attributes': {}}
    if color:
        frames.append({'metadata': {**metadata, 'sensor_name': capture.COLOR},
                       'payload': {'color': {'bytes': 36, 'format': 'bgr8'}}})
        bodies.append(np.full((3, 4, 3), 100, np.uint8).tobytes())
    depth = copy.deepcopy(metadata)
    depth.update(sensor_name=capture.DEPTH, profile={'width': 4, 'height': 3, 'format': 'z16'},
                 attributes={'aligned_to': capture.COLOR if aligned else 'other', 'depth_units_m': .0001})
    frames.append({'metadata': depth, 'payload': {'depth': {'bytes': 24, 'format': 'z16'}}})
    bodies.append(np.full((3, 4), 1000, dtype='<u2').tobytes())
    header = json.dumps({'magic': 'tatbot-vision-frame-set', 'version': 1,
                         'envelope': {'producer': {'node': owner}}, 'frames': frames}).encode()
    return len(header).to_bytes(4, 'big')+header+b''.join(bodies)


def camera(packets):
    calls = []
    packets = iter(packets)

    def get(topic, **kwargs):
        calls.append((topic, kwargs))
        raw = next(packets)
        return [SimpleNamespace(err=None, ok=SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: raw)))]

    obj = capture.Camera.__new__(capture.Camera)
    obj.owner = 'synthetic-owner'
    obj.session = SimpleNamespace(get=get)
    return obj, calls


def test_capture_decodes_the_identified_new_aligned_pair():
    after = time.time_ns()-1_000_000_000
    c, calls = camera([packet()])
    value = c.capture(after)
    assert value['stamp_ns'] >= after and value['image'].shape == (3, 4, 3)
    assert 'original_packet' not in value
    np.testing.assert_allclose(value['depth_m'], .1)
    assert calls[0][0] == capture.OVERHEAD_TOPIC
    assert json.loads(calls[0][1]['payload'])['after_ns'] == after


def test_retained_packet_is_the_exact_owner_reply_before_decoding():
    raw = packet()
    c, _ = camera([raw])
    shot = c.capture(time.time_ns()-1_000_000_000, retain_original=True)
    assert shot['original_packet'] == raw
    assert shot['image'].tobytes() != raw


def test_old_exposure_is_skipped_until_the_owner_supplies_a_new_pair():
    after = time.time_ns()-1_000_000_000
    c, calls = camera([packet(stamp=after-1), packet(stamp=after+1)])
    assert c.capture(after)['stamp_ns'] == after+1 and len(calls) == 2


@pytest.mark.parametrize('settings,match', [({'owner': 'foreign-owner'}, 'not the owner'),
                                          ({'color': False}, 'no overhead_depth_color'),
                                          ({'aligned': False}, 'not aligned')])
def test_foreign_missing_or_unaligned_capture_is_refused(settings, match):
    c, _ = camera([packet(**settings)])
    with pytest.raises(RuntimeError, match=match):
        c.capture(time.time_ns()-1_000_000_000)


def test_missing_overhead_owner_refuses_without_opening_a_bus(monkeypatch):
    from tatbot_bridge import stack
    from tatbot_cli import nodes
    from tatbot_description import repo_root

    monkeypatch.setattr(nodes, 'nodes_with', lambda *a: [])
    monkeypatch.setattr(stack, 'open_bus', lambda *a: pytest.fail('unowned capture opened a bus'))
    with pytest.raises(RuntimeError, match='exactly one overhead-depth'):
        capture.Camera(repo_root())
