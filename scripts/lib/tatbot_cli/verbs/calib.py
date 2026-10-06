"""calib — native arm measurements and retained overhead registration.

The five-camera calibration pipeline that lived here (`tatbot calib <phases>`,
list/review/solve/preview/apply/catalog/destination/receive and its `vision
calib …` spellings) was removed on 2026-09-29 with the overhead PoE cameras and
the calibration conductor node it needed. Tool-tip calibration on the demo rig
is the ROS 2 stack's station probe (`tatbot ros calib run`); `vision touchoff`
still solves a tip from a flight log.

register-fit consumes original native holds through the shared pure registration
fitter. It supplies no capture executor or calibration adoption path.
"""

from __future__ import annotations

from tatbot_cli.registry import MOTION_AUTO, OFFLINE, Plan, verb
from tatbot_cli.verbs._common import sh, tool_flag


def _joint_measure_args(p):
    p.add_argument('--resume-run', help='local prior measurement run ID with verified idle release')
    p.add_argument('--skip-blocked-run', help='linked no-progress run to mark unfinished and continue at the next joint')


@verb(effects=('read_files', 'write_files', 'robot_motion'), noun='calib', verb='joint-measure',
      tier=MOTION_AUTO, role='arm', auto_hop=True, needs_tool=True, launch_id=True,
      args=_joint_measure_args,
      wraps=('scripts/il_joint_payload_measure.sh', 'scripts/vision/joint_payload_measure.py'), doc='docs/teleop_tuning.md',
      summary='measure the right carriage and rotary joints sequentially in free air; apply no settings',
      example=(),
      invariants=('One owned right-arm driver with physical E-stop, carriage cap and trip retract.',
                  'Carriage excursions are 0.2 mm; each rotary step is at most 0.02 rad.',
                  'All feedback is retained; no controller golden or touch constant is changed.'))
def calib_joint_measure(ctx, ns, rest):
    resume = ('--resume-run', ns.resume_run) if ns.resume_run else ()
    blocked = ('--skip-blocked-run', ns.skip_blocked_run) if ns.skip_blocked_run else ()
    return sh(ctx, 'scripts/il_joint_payload_measure.sh', *tool_flag(ctx), *resume, *blocked)


def _pose_args(p):
    p.add_argument('--arm', required=True, choices=('pink', 'blue'))
    p.add_argument('--wrist-deg', required=True, type=float)
    p.add_argument('--base-deg', type=float, help='optional absolute base yaw, within +/-30 degrees')
    p.add_argument('--hold-seconds', default=5.0, type=float)
    p.add_argument('--capture-overhead', action='store_true', help='retain original overhead RGB-D during the native hold')
    p.add_argument('--operator-go', action='store_true', help='record authorization for a pose with overhead capture')


@verb(effects=('read_files', 'write_files', 'network', 'robot_motion'), noun='calib', verb='pose',
      tier=MOTION_AUTO, role='arm', auto_hop=True, needs_tool=True, launch_id=True,
      args=_pose_args, wraps=('scripts/il_wrist_pose.sh', 'scripts/vision/wrist_pose.py'), doc='docs/fiducials.md',
      summary='rotate one parked wrist slowly for camera inspection, then idle and release',
      example=('--arm', 'pink', '--wrist-deg', '180'),
      invariants=('Wrist roll and optional base yaw change; the other joints return to their configured parked targets.',
                  'Uses the calibrated arm worker, native build identity, exclusive lease and physical E-stop.',
                  'Wrist travel is at most a quarter turn; optional base yaw is within 30 degrees; then idle release.'))
def calib_pose(ctx, ns, rest):
    from tatbot_cli.cli import UsageError

    if ns.capture_overhead and not ns.operator_go:
        raise UsageError('overhead pose capture requires --operator-go declaring this run is covered by operator authorization')
    return sh(ctx, 'scripts/il_wrist_pose.sh', '--arm', ns.arm, '--wrist-deg', str(ns.wrist_deg),
              '--hold-seconds', str(ns.hold_seconds),
              *(['--capture-overhead'] if ns.capture_overhead else []),
              *(['--operator-go'] if ns.operator_go else []),
              *(['--base-deg', str(ns.base_deg)] if ns.base_deg is not None else []), *tool_flag(ctx))


def _inspect_args(p):
    p.add_argument('--arm', required=True, choices=('pink', 'blue'))
    p.add_argument('--operator-go', action='store_true', required=True,
                   help='record authorization for a measured hold at the current pose, then idle release')
    p.add_argument('--hold-seconds', default=5.0, type=float, help='bounded current-pose hold before the final measurement')
    p.add_argument('--capture-wrist', action='store_true', help='subscribe to the existing wrist owner during the measured hold')
    p.add_argument('--capture-overhead', action='store_true', help='retain original overhead RGB-D during the native hold')


@verb(effects=('read_files', 'write_files', 'network', 'robot_motion'), noun='calib', verb='inspect',
      tier=MOTION_AUTO, role='arm', auto_hop=True, needs_tool=True, launch_id=True,
      args=_inspect_args, wraps=('scripts/il_wrist_pose.sh', 'scripts/vision/wrist_pose.py'), doc='docs/fiducials.md',
      summary='retain measured joints at the current parked pose, then release to idle without a new pose target',
      example=('--arm', 'blue', '--operator-go'),
      invariants=('Connection takes a measured position hold; this is not a passive network probe.',
                  'Uses the existing calibrated arm owner, exact native build identity, exclusive lease and physical E-stop.',
                  'No wrist, base, approach, staging or landing target is selected; release verifies idle.'))
def calib_inspect(ctx, ns, rest):
    return sh(ctx, 'scripts/il_wrist_pose.sh', '--arm', ns.arm, '--inspect', '--operator-go',
              '--hold-seconds', str(ns.hold_seconds), *(['--capture-wrist'] if ns.capture_wrist else []),
              *(['--capture-overhead'] if ns.capture_overhead else []), *tool_flag(ctx))


def _register_fit_args(p):
    p.add_argument('--python', required=True, help='existing interpreter with NumPy, OpenCV and PyYAML; no installation')
    p.add_argument('--arm', required=True, choices=('blue', 'pink'))
    p.add_argument('--capture', required=True, nargs='+', help='completed native hold directories with original overhead packets')
    p.add_argument('--bundle', required=True, help='camera optics bundle matching the original packets')
    p.add_argument('--setup-id', required=True, help='label for this explicitly selected physical setup window')
    p.add_argument('--out', required=True, help='new directory for the unadopted candidate and evidence')
    p.add_argument('--seat-report', action='store_true',
                   help='compare shared rigid/seat models by distinct view; interpreter also needs SciPy')


@verb(effects=('read_files', 'write_files', 'start_process'), noun='calib', verb='register-fit',
      tier=OFFLINE, visibility='public', args=_register_fit_args, doc='docs/fiducials.md',
      wraps=('scripts/vision/native_registration.py',),
      summary='fit an unadopted overhead registration from original native holds and shared geometry',
      example=('--python', '/path/to/validated/python', '--arm', 'blue', '--capture', '/path/to/hold',
               '--bundle', '/path/to/calibration.json', '--setup-id', 'table-setup', '--out', '/tmp/registration-fit'),
      invariants=('Reads original packets and rebinds their exposures to measured native joints and carriage.',
                  'Uses the shared URDF, target detector, registration solve and quality gates.',
                  'Never connects an arm, starts a camera owner or adopts a transform.'))
def calib_register_fit(ctx, ns, rest):
    return Plan(argv=[ns.python, ctx.path('scripts/vision/native_registration.py'),
                      '--arm', ns.arm, '--capture', *ns.capture, '--bundle', ns.bundle,
                      '--setup-id', ns.setup_id, '--out', ns.out,
                      *(['--seat-report'] if ns.seat_report else [])])


__all__ = ["calib_joint_measure", "calib_pose", "calib_inspect", "calib_register_fit"]
