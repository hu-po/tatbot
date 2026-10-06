"""Replayable wrist RGB-D evidence in the existing ROS camera owner; no motion or calibration adoption."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rgbd_geometry import depth_points, page_height_from_depth, wrist_clearance, wrist_frame_age


def observe(io):
    q, at = io.observed()
    return {'q': q.tolist(), 'measured_monotonic_s': at}


def retain(cam, images, depths, kin, q, page, out, arm, *, reference=None, overhead=None):
    """Keep all paired native frames, optical calibration and the exact FK/reference used by the fit."""
    evaluated = (reference or {}).get('captured_at_monotonic_s', time.monotonic())
    depth_frame = f'{arm}/realsense_depth_optical_frame'
    base_from_depth = kin.frame(q, depth_frame)
    base_from_tcp = kin.fk(q)
    stack = np.asarray(depths, float)
    stack[stack <= 0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.)
    pts = depth_points(median, cam.k_depth, intrinsics=cam.depth_metadata['intrinsics'])
    base = pts @ base_from_depth[:3, :3].T + base_from_depth[:3, 3]
    diagnostics = {}
    fit = page_height_from_depth(base, page, diagnostics=diagnostics)
    times = cam.capture_timestamps
    joint_age = max((abs(t['received_monotonic_s']-t['joints']['measured_monotonic_s'])
                     for t in times if t.get('joints')), default=float('inf'))
    age, time_reason = wrist_frame_age(times, evaluated)
    result = wrist_clearance(fit, page, base_from_depth, base_from_tcp, frame_age_s=float('inf') if age is None else age, joint_age_s=joint_age)
    if time_reason:
        result['reason'] = time_reason
    if time_reason or result['reason'] in ('stale_depth', 'stale_joints'):
        diagnostics['fit_reason'] = diagnostics['reason']
        diagnostics['reason'], fit = result['reason'], None
    diagnostics.update(depth_pixels=int(median.size), valid_depth_samples=int(np.isfinite(stack).sum()),
                       depth_samples=int(stack.size))
    if out is None:
        return fit, diagnostics, result
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'depth_m': np.asarray(depths), 'depth_raw': np.asarray(cam.raw_depth), 'rgb_bgr': np.asarray(images),
              'q': q, 'base_from_depth': base_from_depth, 'base_from_tcp': base_from_tcp,
              'base_from_page': page, 'k_depth': cam.k_depth,
              'base_from_color': kin.frame(q, f'{arm}/realsense_color_optical_frame')}
    samples = [t['joints']['q'] for t in times if t.get('joints')]
    if samples:
        arrays.update(q_frames=np.asarray(samples),
                      base_from_depth_frames=np.asarray([kin.frame(np.asarray(sample), depth_frame) for sample in samples]),
                      base_from_tcp_frames=np.asarray([kin.fk(np.asarray(sample)) for sample in samples]))
    if overhead is not None:
        arrays['overhead_rgb_bgr'] = overhead['image']
        if overhead['depth_m'] is not None:
            arrays['overhead_depth_m'] = overhead['depth_m']
    np.savez_compressed(out.with_suffix('.npz'), **arrays)
    meta = {'schema': 'tatbot.wrist-page-depth/1', 'frame': depth_frame, 'camera': cam.depth_metadata,
            'timestamps': times, 'height_fit': fit, 'observation': diagnostics, 'clearance': result, 'evaluated_monotonic_s': evaluated,
            'reference': reference, 'overhead': None if overhead is None else overhead['metadata'],
            'overhead_depth_metadata': None if overhead is None else overhead.get('depth_metadata'),
            'coordinates': 'native depth optical frame; FK shares arm error with the TCP',
            'reference_lifetime': 'this capture only; no carried correction or calibration adoption'}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    return fit, diagnostics, result


def stationary(node, io, kin, page, out_dir, args):
    """Capture at the current held pose; motion remains exclusively in native Touch/Draw actions."""
    from rosidl_runtime_py.convert import message_to_ordereddict
    from tatbot_bridge import capture
    from tatbot_description import names, repo_root
    from tatbot_interfaces.msg import Page, SafetyState
    from tatbot_session import config, inspect, runtime

    pages, safety = [], []
    subscriptions = [node.create_subscription(Page, names.PAGE_TOPIC, pages.append, 10),
                     node.create_subscription(SafetyState, names.SAFETY_TOPIC,
                                              lambda msg: safety.append(msg) if msg.arm == args.arm else None, 10)]
    camera = overhead_camera = None
    try:
        io.wait(lambda: bool(safety and pages), timeout=3.)
        if not safety or not safety[-1].machine_off:
            raise RuntimeError('stationary depth capture requires fresh native machine-off evidence')
        before = time.time_ns()
        camera = inspect.WristCamera(serial=args.serial, depth=True)
        images, depths = camera.grab_both(args.frames, observe=lambda: observe(io))
        q = io.measured()
        reference = {'run': args.run, 'page': page, 'runtime': json.loads(runtime.workspace_record(repo_root(None)).read_text()),
                     'tool': config.tool_datasheet(repo_root(None), config.fitted_tool(repo_root(None), args.arm)).raw,
                     'safety': message_to_ordereddict(safety[-1]), 'tracked_page': message_to_ordereddict(pages[-1]) if pages else None}
        reference['captured_at_monotonic_s'] = time.monotonic()
        tracked = reference['tracked_page']
        if tracked is not None:
            stamp = tracked['pose']['header']['stamp']
            reference['tracked_page_age_s'] = time.time()-stamp['sec']-stamp['nanosec']*1e-9
            if tracked['source'] != Page.SOURCE_MEASURED or reference['tracked_page_age_s'] > 5.:
                reference['tracked_page_fallback'] = 'lost/stale tracked pose; raw D555 capture is a separate comparison'
        overhead = None
        try:
            overhead_camera = capture.Camera(repo_root(None))
            overhead = overhead_camera.capture(before)
            if overhead['depth_m'] is None:
                reference['overhead_fallback'] = 'missing_overhead_depth'
        except (RuntimeError, ValueError, OSError) as error:
            reference['overhead_fallback'] = str(error)
        stem = out_dir/'depth'
        _, _, result = retain(camera, images, depths, kin, q, np.asarray(page['used']), stem, args.arm,
                              reference=reference, overhead=overhead)
        print(json.dumps({'ok': True, 'capture': str(stem.with_suffix('.npz')), 'clearance': result}))
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        if camera is not None:
            camera.close()
        if overhead_camera is not None:
            overhead_camera.close()
