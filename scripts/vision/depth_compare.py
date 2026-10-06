#!/usr/bin/env python3
"""Replay retained scan bursts against filter candidates and optional depth truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from depth_quality import filter_depth  # noqa: E402


def compare(capture, candidates, truth=None):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise ValueError('comparison requires 1..32 filter candidates')
    results, intrinsics = [], {}
    with np.load(capture, allow_pickle=False) as data:
        for role in ('wrist_upper', 'wrist_lower'):
            if f'raw_depth_{role}' not in data:
                continue
            raw = data[f'raw_depth_{role}']
            units = float(data[f'units_m_{role}'])
            if raw.dtype != np.uint16 or raw.ndim != 3 or not 8 <= len(raw) <= 255:
                raise ValueError('comparison needs an exact retained Z16 burst')
            intrinsics[role] = data[f'intrinsics_{role}'].tolist()
            for options in candidates:
                started = time.monotonic()
                depth, report = filter_depth(data[f'depth_{role}'], units,
                    data[f'valid_{role}'], len(raw), data[f'temporal_mad_m_{role}'], options=options)
                valid = depth > 0
                item = {'role': role, 'options': options, 'quality': report,
                        'filter_s': time.monotonic()-started,
                        'temporal_mad_p95_mm': float(np.percentile(data[f'temporal_mad_m_{role}'][valid], 95))*1e3
                        if valid.any() else None, 'absolute_error': None}
                if truth is not None:
                    expected = np.asarray(truth[f'depth_m_{role}'], float)
                    if expected.shape != depth.shape:
                        raise ValueError('truth dimensions differ from capture')
                    supported = valid & np.isfinite(expected) & (expected > 0)
                    error = depth[supported]*units-expected[supported]
                    item['absolute_error'] = {'samples': int(len(error)),
                        'signed_bias_mm': float(np.median(error))*1e3 if len(error) else None,
                        'p95_mm': float(np.percentile(np.abs(error), 95))*1e3 if len(error) else None}
                results.append(item)
    if not results:
        raise ValueError('capture has no replayable wrist burst')
    return {'schema': 'tatbot.depth-comparison/1', 'capture_sha256': hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
            'camera_intrinsics': intrinsics, 'results': results, 'hardware_authority': False,
            'ground_truth': 'caller supplied' if truth is not None else 'absent; stability is not accuracy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True, help='JSON list of filter candidates')
    parser.add_argument('--ground-truth', type=Path)
    args = parser.parse_args()
    import tatbot_runlog
    run = tatbot_runlog.init('depth-compare')
    with run:
        candidates = json.loads(args.settings.read_text())
        if args.ground_truth:
            with np.load(args.ground_truth, allow_pickle=False) as truth:
                report = compare(args.capture, candidates, truth)
            report['ground_truth_sha256'] = hashlib.sha256(args.ground_truth.read_bytes()).hexdigest()
        else:
            report = compare(args.capture, candidates)
        report['settings_sha256'] = hashlib.sha256(args.settings.read_bytes()).hexdigest()
        (run.dir/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
