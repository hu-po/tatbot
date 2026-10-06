#!/usr/bin/env python3
"""Inspect or position one parked wrist through the guarded calibration owner."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/lib'))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import arm_calibration as recipe  # noqa: E402
import tatbot_runlog  # noqa: E402
from arm_guide_owner import CaptureError, Owner, validate_native_build  # noqa: E402
from tatbot_digest import sha256_file  # noqa: E402
from tool_spec import read_workspace  # noqa: E402


def run(args, log):
    inspection = getattr(args, 'inspect', False)
    overhead = getattr(args, 'capture_overhead', False)
    if (inspection or overhead) and not getattr(args, 'operator_go', False):
        raise CaptureError('inspection or overhead capture needs --operator-go recording authorization for the native hold')
    arm = recipe.selected_arm(REPO, args.arm)
    if read_workspace(REPO).get(arm.arm_id, {}).get('tool_id') != args.ee_tool:
        raise CaptureError('stated tool does not match this arm’s fitted tool')
    binary = REPO / 'rust/target/release/tatbot-arm-guide'
    checksum = sha256_file(binary)
    build = json.loads(subprocess.check_output([str(binary), '--build-info'], text=True, timeout=10))
    receipt = {'binary_sha256': checksum, 'build': build}
    validate_native_build(receipt, recipe.native_source_digests(REPO), platform.machine())
    if sha256_file(binary) != checksum:
        raise CaptureError('native binary changed during validation')
    recipe.write_json(log.dir / 'native-build.json', receipt)
    device = os.environ.get('TATBOT_ESTOP_DEVICE')
    if not device:
        raise CaptureError('run through tatbot calib pose with the resolved hardware profile')
    device = recipe.native_estop_device(REPO, device)
    argv = [str(binary), '--run-dir', str(log.dir / 'arm'), '--run-id', log.run_id,
            '--controller-role', arm.controller_role, '--backend', 'trossen',
            '--profile-dir', str(REPO / 'config/trossen'), '--estop-device', device,
            '--tool-datasheet', str(REPO / 'config/tools' / (args.ee_tool + '.yaml'))]
    camera = _overhead_camera() if overhead else None
    try:
        _owned_hold(argv, build, args, log, arm, camera)
    finally:
        if camera is not None:
            camera.close()
    _bind_captures(args, log)


def _owned_hold(argv, build, args, log, arm, camera):
    owner = Owner(argv, log.dir / 'owner.log')
    try:
        ready = owner.wait('ready')
        if ready.get('build') != build:
            raise CaptureError('native owner build changed before launch')
        if ready.get('tool_datasheet_sha256') != sha256_file(REPO / 'config/tools' / (args.ee_tool + '.yaml')):
            raise CaptureError('tool datasheet changed before launch')
        owner.send({'cmd': 'connect'})
        connected = owner.wait('connected')
        recipe.write_json(log.dir / 'connected.json', connected)
        logging.info(json.dumps({'event': 'connected', 'arm': args.arm, 'state': connected}))
        if getattr(args, 'inspect', False):
            _inspect_current(owner, args, log, arm.arm_id)
        else:
            _position_wrist(owner, args, log)
        if camera is not None:
            _capture_overhead(camera, log)
            owner.send({'cmd': 'status'})
            status = owner.wait('status')
            recipe.write_json(log.dir / 'overhead-status.json', status)
            if status.get('worker', {}).get('fault'):
                raise CaptureError('arm fault during overhead capture')
        owner.send({'cmd': 'release'})
        released = owner.wait('released')
        recipe.write_json(log.dir / 'released.json', released)
        logging.info(json.dumps({'event': 'released', 'arm': args.arm, 'state': released}))
    finally:
        # EOF follows the owner's existing stop, idle and verified release path.
        code = owner.close()
        if code != 0:
            raise CaptureError(f'calibration owner failed to close cleanly: {code}')


def _bind_captures(args, log):
    if getattr(args, 'capture_wrist', False):
        binding = recipe.bind_wrist_capture(log.dir / 'wrist/capture-1.npz', log.dir / 'arm/telemetry.bin')
        recipe.write_json(log.dir / 'wrist-pose-binding.json', binding)
    if getattr(args, 'capture_overhead', False):
        binding = recipe.bind_owner_packet(log.dir / 'overhead/owner-packet.bin', log.dir / 'arm/telemetry.bin')
        recipe.write_json(log.dir / 'overhead-pose-binding.json', binding)


def _overhead_camera():
    import cv2  # noqa: F401 - preflight before an arm owner is opened

    # These shared capture libraries import no ROS node or arm executor.
    for root in ('tatbot_bridge', 'tatbot_description'):
        sys.path.insert(0, str(REPO / 'ros' / root))
    from tatbot_bridge.capture import Camera

    return Camera(REPO)


def _capture_overhead(camera, log):
    import cv2
    import numpy as np

    requested = time.time_ns()
    shot = camera.capture(requested, retain_original=True)
    out = log.dir / 'overhead'
    out.mkdir()
    packet = out / 'owner-packet.bin'
    packet.write_bytes(shot['original_packet'])
    if not cv2.imwrite(str(out / 'color.png'), shot['image']):
        raise CaptureError('could not retain overhead colour preview')
    np.save(out / 'decoded-color-bgr.npy', shot['image'], allow_pickle=False)
    if shot['depth_m'] is not None:
        np.save(out / 'aligned-depth-m.npy', shot['depth_m'], allow_pickle=False)
    recipe.write_json(out / 'capture.json', {
        'schema': 'tatbot.overhead-owner-capture/1', 'requested_after_ns': requested,
        'stamp_ns': shot['stamp_ns'], 'color_metadata': shot['metadata'],
        'depth_metadata': shot['depth_metadata'], 'original_packet': packet.name,
        'original_packet_sha256': sha256_file(packet), 'decoded_color_order': 'BGR',
        'world_registration': 'unmeasured', 'physical_accuracy_bound_m': None})


def _position_wrist(owner, args, log):
    owner.send({'cmd': 'wrist', 'angle_rad': math.radians(args.wrist_deg),
                'base_rad': None if args.base_deg is None else math.radians(args.base_deg)})
    reached = owner.wait('wrist', timeout=120)
    recipe.write_json(log.dir / 'wrist.json', reached)
    logging.info(json.dumps({'event': 'wrist_reached', 'arm': args.arm, 'pose': reached,
                            'hold_seconds': args.hold_seconds}))
    _hold_current(owner, args.hold_seconds, log)


def _hold_current(owner, seconds, log):
    until = time.monotonic() + seconds
    states = []
    while time.monotonic() < until:
        owner.send({'cmd': 'status'})
        status = owner.wait('status')
        if status.get('worker', {}).get('fault'):
            raise CaptureError('arm fault while holding the inspection pose')
        states.append(status)
        time.sleep(min(0.2, max(0, until-time.monotonic())))
    recipe.write_json(log.dir / 'hold-status.json', {'seconds': seconds, 'states': states})


def _inspect_current(owner, args, log, arm_id):
    _hold_current(owner, args.hold_seconds, log)
    owner.send({'cmd': 'status'})
    status = owner.wait('status')
    if status.get('worker', {}).get('fault'):
        raise CaptureError('arm fault during measured-state inspection')
    recipe.write_json(log.dir / 'status.json', status)
    logging.info(json.dumps({'event': 'measured_inspection', 'arm': args.arm, 'state': status}))
    if getattr(args, 'capture_wrist', False):
        import vision_capture

        cameras = vision_capture.open_cameras(False, True, arm_id)
        try:
            # Original camera evidence has no asserted pose until telemetry is paired.
            vision_capture.capture(cameras, log.dir / 'wrist', 1, [float('nan')]*6,
                                   float('nan'), time.time(), vision_capture.MIN_FRAMES)
        finally:
            cameras.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('pink', 'blue'))
    parser.add_argument('--ee-tool', required=True)
    parser.add_argument('--wrist-deg', type=float)
    parser.add_argument('--base-deg', type=float)
    parser.add_argument('--hold-seconds', default=5.0, type=float)
    parser.add_argument('--capture-wrist', action='store_true', help='retain original RGB-D from the existing wrist owner during inspection')
    parser.add_argument('--capture-overhead', action='store_true', help='retain original overhead RGB-D during the native hold')
    parser.add_argument('--inspect', action='store_true', help='retain measured state at the current pose, then idle release')
    parser.add_argument('--operator-go', action='store_true', help='record authorization for inspection or a captured pose')
    args = parser.parse_args()
    if args.capture_overhead and not args.operator_go:
        parser.error('--capture-overhead requires --operator-go')
    if args.inspect:
        if not args.operator_go or args.wrist_deg is not None or args.base_deg is not None:
            parser.error('inspection needs --operator-go and accepts no wrist/base target')
    elif args.capture_wrist:
        parser.error('--capture-wrist requires --inspect')
    elif args.wrist_deg is None or not -180 <= args.wrist_deg <= 180:
        parser.error('positioning needs a wrist angle within +/-180 degrees')
    if not 0 <= args.hold_seconds <= 120:
        parser.error('wrist angle must be within +/-180 degrees and hold within 0..120 seconds')
    if args.base_deg is not None and not -30 <= args.base_deg <= 30:
        parser.error('base angle must be within +/-30 degrees')
    with tatbot_runlog.init('calib-inspect' if args.inspect else 'calib-pose', meta=vars(args)) as log:
        run(args, log)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
