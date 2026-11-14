"""Shared timing evidence API; no motion or controller ownership."""
from tatbot_contracts import ros_pipeline as pipeline
from tatbot_contracts.ros_timing import (  # noqa: F401 -- existing native API
    SCHEMA,
    begin,
    finish,
    summarize,
    time,
)


def start(directory, clock, *, attempt=None, simulated=False, hardware='unknown'):
    import logging

    try:
        return pipeline.Trace(directory, clock, attempt=attempt, hardware=hardware,
                              clock_domain='simulation' if simulated else 'host_monotonic')
    except (OSError, RuntimeError) as error:
        logging.getLogger(__name__).warning('pipeline timing unavailable: %s', error)
        return None


def finish_pipeline(trace, directory):
    """Shutdown reporting runs outside motion; diagnostics never change the draw outcome."""
    if trace is None:
        return None
    status = trace.close()
    try:
        result = pipeline.write_report(directory)
    except (OSError, ValueError, TypeError) as error:
        return {'recording_complete': False, 'writer': status, 'error': str(error)}
    return {**result, 'writer': status}


def _diagnostic(function):
    """Malformed optional telemetry cannot interrupt the controller/camera path."""
    import functools

    @functools.wraps(function)
    def observe(owner, *args, **kwargs):
        try:
            return function(owner, *args, **kwargs)
        except Exception as error:
            trace = getattr(getattr(owner, 'ex', owner), 'pipeline_trace', None)
            if trace is not None:
                trace.error = f'{function.__name__}: {error}'
            return None
    return observe


@_diagnostic
def measurement(tracker, job, capture, camera, name, op):
    trace = getattr(tracker.ex, 'pipeline_trace', None)
    if trace is None:
        return
    captures = [] if capture is None else capture
    keys = ('received_at_ns', 'received_monotonic_s', 'depth_ms', 'depth_domain', 'color_ms', 'color_domain',
            'color_frame_number', 'depth_frame_number')
    frames = [{k: frame[k] for k in keys if k in frame} for frame in captures]
    receipts = [f['received_monotonic_s'] for f in frames if 'received_monotonic_s' in f]
    # Real camera receipt and WallClock share host monotonic. ROS/device clocks are not converted.
    received = max(receipts) if receipts and trace.domain == 'host_monotonic' else None
    role = tracker.views[camera - 1]['role'] if camera else 'session'
    identity = trace.emit('measurement', arm=tracker.ex.arm, camera=camera, camera_role=role, reading=name, op=op,
                          frames=frames, frame_receipt_t_s=received,
                          oldest_frame_receipt_t_s=min(receipts) if received is not None else None,
                          pose_observed_t_s=job.get('pose_observed_t_s'))
    if identity is None:
        return
    job[pipeline.INFO] = {'measurement': identity, 'submitted_host_mono_s': pipeline.time.monotonic()}


@_diagnostic
def applied(tracker, row, camera):
    context = row.pop(pipeline.INFO, None)
    trace = getattr(tracker.ex, 'pipeline_trace', None)
    if trace is None or context is None:
        return
    verdict = row.get('field')
    taken = bool(row.get('accepted') and verdict in ('added', 'moved'))
    trace.emit('measurement_applied', arm=tracker.ex.arm, measurement=context['measurement'],
               camera=camera, applied=taken, verdict=verdict, worker=context)
    if taken:
        tracker._timing_applied[camera] = context['measurement']
    row[pipeline.INFO] = {'measurement': context['measurement'], 'applied': taken}


def snapshot(tracker, field):
    _snapshot(tracker, field)
    return field


@_diagnostic
def _snapshot(tracker, field):
    trace = getattr(tracker.ex, 'pipeline_trace', None)
    if trace is not None:
        field.pipeline_source = pipeline.snapshot(trace, tracker.ex.arm, tracker._timing_applied.values())


def plan(ex, op, snap, build):
    return pipeline.plan(getattr(ex, 'pipeline_trace', None), ex.arm, op['id'],
                         getattr(snap, 'pipeline_source', None), build)


@_diagnostic
def sent(io, traj, first, *, scheduled_start=None):
    trace = getattr(io, 'pipeline_trace', None)
    if trace is None:
        return None
    from tatbot_session.ports import goal_tag

    context = traj.info.get(pipeline.INFO, {})
    previous = getattr(io, '_timing_goal', None)
    identity = trace.emit('goal_sent', arm=io.arm, op=io.op_id, tag=goal_tag(io.op_id, traj, first),
                          replacement_of=previous, scheduled_start_t_s=scheduled_start,
                          contains_draw=any(int(phase) == 5 for phase in traj.phase[first:]), **context)
    io._timing_goal = identity
    return identity


@_diagnostic
def accepted(io, identity, taken, goal_id=None):
    trace = _goal_trace(io, identity)
    if trace is not None:
        trace.emit('goal_accepted', goal=identity, accepted=bool(taken),
                   goal_uuid=bytes(goal_id.uuid).hex() if goal_id is not None else None)


@_diagnostic
def feedback(io, identity, trajectory, message, scheduled_start=None):
    trace = _goal_trace(io, identity)
    if trace is None or (scheduled_start is not None and trace.clock.now() < scheduled_start):
        return
    import numpy as np

    fb = message.feedback
    duration = fb.desired.time_from_start
    elapsed = duration.sec + duration.nanosec * 1e-9
    index = min(max(int(np.searchsorted(trajectory.t, elapsed, side='right')) - 1, 0), len(trajectory.t) - 1)
    stamp = fb.header.stamp
    trace.feedback(identity, phase=int(trajectory.phase[index]), elapsed_s=elapsed,
                   ros_stamp_ns=stamp.sec * 1_000_000_000 + stamp.nanosec)


@_diagnostic
def result(io, identity, response):
    trace = _goal_trace(io, identity)
    if trace is not None:
        trace.result(identity, status=int(response.status), error_code=int(response.result.error_code))


def _goal_trace(io, identity):
    trace = getattr(io, 'pipeline_trace', None)
    return trace if trace is not None and identity is not None and identity.startswith(trace.attempt + ':') else None
