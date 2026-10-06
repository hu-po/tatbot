"""One bounded material-to-native candidate path for contact and standoff tools.

The tools supply their geometry and interaction policy. This stage owns the
shared material/phase assembly, native acceptance, and standoff retry record;
it does not grant dispatch authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stroke_operation as operation
from tatbot_digest import sha256_file


@dataclass(frozen=True)
class CandidateGeometry:
    cursor: operation.StrokeCursor
    work: operation.WorkPolicy
    before: list[operation.PhasePart]
    after_work: Callable
    period_s: float
    assembler: Callable
    contact_report: Callable[[operation.PlannedStroke], dict] | None = None


@dataclass(frozen=True)
class StandoffPolicy:
    seed: np.ndarray
    binary: Path
    directory: Path
    index: int
    approach: list[str]
    retract: list[str]
    scales: tuple[float, ...]
    constants_sha: str
    period_s: float
    frame: str
    arm_prefix: str
    write_samples: Callable[[Path, object, tuple[int, int]], None]
    validate: Callable[[object, dict, np.ndarray, tuple[int, int]], tuple]


@dataclass(frozen=True)
class CandidateOutcome:
    planned: operation.PlannedStroke
    preflight: dict | None = None


def _standoff_request(policy, path):
    return operation.NativeRequest(
        policy.seed, path, policy.write_samples,
        native_plan=lambda target, joints, period: operation.executor_plan(
            policy.binary, target, joints, period, policy.constants_sha,
            arm_prefix=policy.arm_prefix),
        constants_sha=policy.constants_sha, validate_policy=policy.validate)


def _standoff_preflight(policy, scale, attempt, planned):
    samples = planned.samples
    first, end = planned.work_range
    accepted, speed, acceleration, slope_error, clearance = planned.native.policy
    if not accepted:
        return None, (f'scale {scale}: speed {speed:.3f} rad/s, '
                      f'acceleration {acceleration:.3f} rad/s^2, '
                      f'slope error {slope_error:.4f} rad/s')
    plan = planned.native.native.plan
    report = {'schema': 'tatbot.draw-preflight/1', 'ok': True, 'executor_check': 'accepted',
              **planned.material_origin,
              'samples_sha256': sha256_file(planned.native.samples_path),
              'constants_sha': policy.constants_sha, 'frame': policy.frame,
              'period_s': policy.period_s, 'sample_count': samples.n,
              'material_s': (end-first)*policy.period_s,
              'phases': {'approach': [0, first], 'stroke': [first, end], 'retract': [end, samples.n]},
              'initial_seed': policy.seed.tolist(),
              'final_joints': np.asarray(plan['positions'])[-1].tolist(),
              'stroke_index': policy.index, 'approach': policy.approach,
              'retract': policy.retract, 'speed_scale': scale,
              'duration_s': samples.n*policy.period_s,
              'max_joint_speed_rad_s': speed,
              'max_joint_acceleration_rad_s2': acceleration,
              'max_feedforward_slope_error_rad_s': slope_error,
              'max_model_error_mm': plan['max_model_error_mm'],
              'max_orientation_error_rad': plan['max_orientation_error_rad'],
              'planner_sha256': sha256_file(policy.binary),
              'plan': f'{policy.index}-{attempt}.plan.json',
              'contact_establishment_required': False, 'hardware_authority': False}
    if clearance is not None:
        report['measured_shape_clearance'] = clearance
    return report, None


def _plan_geometry(geometry, *, native=None, contact_resources=None):
    return operation.plan_next(
        geometry.cursor, geometry.work,
        before=geometry.before, after_work=geometry.after_work,
        period_s=geometry.period_s, assembler=geometry.assembler,
        native=native, contact_report=geometry.contact_report,
        contact_resources=contact_resources)


def plan_next(geometry: CandidateGeometry | Callable[[float], CandidateGeometry], *,
              standoff: StandoffPolicy | None = None,
              contact_resources: operation.ContactResources | None = None):
    """Plan one selected original interval under an explicit tool policy.

    Standoff may retry native refusals or bounded dynamics at slower scales.
    Contact has one candidate: its resource callbacks retain their setup,
    native validation, then receipt/ink commit order in ``operation.plan_next``.
    """
    if standoff is None:
        if not isinstance(geometry, CandidateGeometry) or geometry.work.kind != 'contact':
            raise ValueError('contact candidate needs measured contact geometry')
        return CandidateOutcome(_plan_geometry(geometry,
                                               contact_resources=contact_resources))
    if contact_resources is not None:
        raise ValueError('next candidate accepts one interaction policy')
    if not callable(geometry) or not standoff.scales:
        raise ValueError('standoff candidate needs bounded scale geometry')
    refusals = []
    for attempt, scale in enumerate(standoff.scales):
        attempt_geometry = geometry(scale)
        if (not isinstance(attempt_geometry, CandidateGeometry)
                or attempt_geometry.work.kind != 'standoff'
                or attempt_geometry.period_s != standoff.period_s
                or attempt_geometry.cursor.chunk_index != standoff.index):
            raise ValueError('standoff candidate geometry differs from the native policy')
        path = standoff.directory/f'{standoff.index}-{attempt}.csv'
        try:
            planned = _plan_geometry(attempt_geometry, native=_standoff_request(standoff, path))
        except operation.ExecutorRefusalError as refusal:
            (standoff.directory/f'{standoff.index}-{attempt}.stderr').write_text(refusal.stderr)
            refusals.append(f'scale {scale}: {refusal.stderr.strip() or f"exit {refusal.returncode}"}')
            continue
        (standoff.directory/f'{standoff.index}-{attempt}.stderr').write_text(planned.native.native.stderr)
        (standoff.directory/f'{standoff.index}-{attempt}.plan.json').write_text(planned.native.native.stdout)
        report, refusal = _standoff_preflight(standoff, scale, attempt, planned)
        if report is not None:
            return CandidateOutcome(planned, report)
        refusals.append(refusal)
    raise ValueError(f'stroke {standoff.index}: the planner or the dynamics bounds refused every speed scale; '
                     + '; '.join(refusals))
