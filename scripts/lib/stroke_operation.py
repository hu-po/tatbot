"""Shared bounded motion candidate assembly and native-plan structure checks.

Geometry and interaction policies supply rows. This module binds each work
phase to its original material candidate before the arm planner sees it.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from stroke_material import MaterialCandidate, MaterialRows, material_address, plan_material_rows


class ExecutorRefusalError(ValueError):
    """The native planner rejected a candidate; a caller may try a slower one."""

    def __init__(self, stderr, returncode):
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f'executor preflight refused: {stderr.strip() or f"exit {returncode}"}')


@dataclass(frozen=True)
class ExecutorPlan:
    plan: dict
    stdout: str
    stderr: str


def _check_plan_binding(plan, seed, period_s, constants_sha):
    if not isinstance(plan, dict):
        raise ValueError('executor returned a mismatched plan')
    try:
        plan_seed = np.asarray(plan.get('seed', []), float)
    except (TypeError, ValueError) as error:
        raise ValueError('executor returned a mismatched plan') from error
    if (plan.get('schema') != 'tatbot.joint-plan/1' or plan.get('hardware_authority') is not False
            or plan.get('constants_sha') != constants_sha or plan.get('period_s') != period_s
            or plan_seed.shape != (7,) or not np.isfinite(plan_seed).all()
            or not np.allclose(plan_seed, seed, atol=1e-12, rtol=0)):
        raise ValueError('executor returned a mismatched plan')


def _check_tool_binding(plan, arm_prefix, tip):
    declared = plan.get('tool_model')
    if not isinstance(declared, dict):
        raise ValueError('executor returned a mismatched tool model')
    try:
        actual = np.asarray(declared.get('tip_in_link6'), float)
        axis = np.asarray(declared.get('carriage_axis_in_link6'), float)
    except (TypeError, ValueError) as error:
        raise ValueError('executor returned a mismatched tool model') from error
    if (declared.get('arm_prefix') != arm_prefix or declared.get('source') != 'bound-input'
            or actual.shape != (3,) or axis.shape != (3,) or not np.isfinite(actual).all()
            or not np.allclose(actual, tip, atol=1e-12, rtol=0)
            or not np.array_equal(axis, [0., 1., 0.])):
        raise ValueError('executor returned a mismatched tool model')


def executor_plan(binary, samples, seed, period_s, constants_sha, *, arm_prefix=None,
                  tool_tip_in_link6=None, runner=None):
    """Run the one offline C++ planner and bind its response to exact inputs.

    Tool policies supply samples and optional physical WXAI prefix. An explicit
    tip binds an independently resolved model; the CSV cannot select it alone.
    The default compiled tool remains unchanged. A planner
    refusal can trigger a bounded retry; malformed or mismatched success cannot.
    """
    seed = np.asarray(seed, float)
    if seed.shape != (7,) or not np.isfinite(seed).all() or not np.isfinite(period_s) or period_s <= 0:
        raise ValueError('executor preflight needs finite seven-axis seed and period')
    argv = [str(Path(binary)), str(Path(samples)), str(period_s), *map(str, seed), '--json']
    if arm_prefix is not None:
        argv += ['--arm-prefix', arm_prefix]
    if tool_tip_in_link6 is not None:
        tip = np.asarray(tool_tip_in_link6, float)
        if (not arm_prefix or tip.shape != (3,) or not np.isfinite(tip).all()
                or not .05 <= np.linalg.norm(tip) <= .45):
            raise ValueError('bound tool needs an explicit arm prefix and finite 50..450 mm tip')
        argv += ['--tool-tip-in-link6', *map(str, tip)]
    result = (runner or subprocess.run)(argv, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode:
        raise ExecutorRefusalError(result.stderr, result.returncode)
    plan = json.loads(result.stdout)
    _check_plan_binding(plan, seed, period_s, constants_sha)
    if tool_tip_in_link6 is not None:
        _check_tool_binding(plan, arm_prefix, tip)
    return ExecutorPlan(plan, result.stdout, result.stderr)


@dataclass(frozen=True)
class FinalizedCandidate:
    samples_path: Path
    native: ExecutorPlan
    positions: np.ndarray
    policy: object


@dataclass(frozen=True)
class StrokeCursor:
    """One local execution cursor bound to the original design interval.

    A successor keeps its source offset as lineage, while its session chunk
    and count name the successor program before any native work is accepted.
    """

    candidate: MaterialCandidate
    chunk_index: int
    source_chunk_count: int
    placement_id: str | None = None
    source_chunk_origin: int | None = None


@dataclass(frozen=True)
class WorkPolicy:
    """Tool-specific sampled work geometry; transit is supplied as phases."""

    distances: np.ndarray
    kind: str
    orientation: object
    metric_vertices: np.ndarray | None = None
    projector: object = None
    placement: object = None
    normalize_normals: bool = False
    pen: int = 0
    stroke_index: int | None = None


@dataclass(frozen=True)
class PhasePart:
    kind: str
    part: tuple
    stroke_index: int | None = None


@dataclass(frozen=True)
class PlannedStroke:
    samples: object
    rows: MaterialRows
    work_range: tuple[int, int]
    phases: MotionPhases
    material_origin: dict
    native: FinalizedCandidate | None = None
    report: dict | None = None


@dataclass(frozen=True)
class NativeRequest:
    """One checked native draw request for the already assembled material rows."""

    seed: np.ndarray
    samples_path: Path
    write_samples: Callable[[Path, object, tuple[int, int]], None]
    native_plan: Callable[[Path, np.ndarray, float], ExecutorPlan]
    constants_sha: str
    require_pen: bool = False
    validate_policy: Callable[[object, dict, np.ndarray, tuple[int, int]], object] | None = None


class ContactResources(Protocol):
    """The existing contact scan/dip setup and post-native ink/receipt effects."""

    def prepare_draw(self, planned: PlannedStroke) -> NativeRequest: ...

    def commit_draw(self, planned: PlannedStroke) -> None: ...


def finalize_candidate(samples, seed, samples_path, *, write_samples, native_plan,
                       constants_sha, require_pen=False, validate_policy=None):
    """Bind one policy-built motion candidate to a checked native solution.

    The producer owns geometry and material rows; its writer retains the source
    CSV, and its native adapter selects the built planner and physical profile.
    Policy checks (envelope, dynamics, shape) run only on the aligned solution.
    A returned candidate remains offline evidence, never dispatch authority.
    """
    seed = np.asarray(seed, float)
    if (seed.shape != (7,) or not np.isfinite(seed).all()
            or not np.isfinite(samples.period_s) or samples.period_s <= 0):
        raise ValueError('candidate needs finite seven-axis seed and period')
    samples_path = Path(samples_path)
    write_samples(samples_path, samples)
    native = native_plan(samples_path, seed, samples.period_s)
    if not isinstance(native, ExecutorPlan):
        raise TypeError('native planner must return an ExecutorPlan')
    _check_plan_binding(native.plan, seed, samples.period_s, constants_sha)
    positions = checked_joint_positions(native.plan, samples, require_pen=require_pen)
    policy = validate_policy(samples, native.plan, positions) if validate_policy is not None else None
    return FinalizedCandidate(samples_path, native, positions, policy)


class MotionPhases:
    """Ordered approach/work/retract phases with exact material row ranges."""

    def __init__(self):
        self.parts = []
        self.segments = []
        self.material_ranges = []
        self.row_count = 0

    def add(self, kind, part, *, stroke_index=None, dip_index=None):
        count = len(part[0])
        # A contact descent can already be at its travel waypoint. Preserve
        # its zero-row phase in the report and its checked phase indices.
        start = self.row_count
        self.parts.append(part)
        self.row_count += count
        self.segments.append({'kind': kind, 'start': start, 'stop': self.row_count,
                              'stroke_index': stroke_index, 'dip_index': dip_index})
        return start, self.row_count

    def add_material(self, rows, kind, *, stroke_index=None):
        if not isinstance(rows, MaterialRows):
            raise TypeError('material phase needs aligned sampled material rows')
        return self.add_material_part(rows.candidate, rows.part(), kind, stroke_index=stroke_index)

    def add_material_part(self, candidate, part, kind, *, stroke_index=None):
        if not isinstance(candidate, MaterialCandidate):
            raise TypeError('material phase needs a source material candidate')
        if len(part[0]) < 2:
            raise ValueError('material phase needs sampled motion rows')
        start, stop = self.add(kind, part, stroke_index=stroke_index)
        self.material_ranges.append((candidate, start, stop))
        return start, stop

    def assemble(self, period_s, assembler):
        if not self.material_ranges:
            raise ValueError('operation has no material phase')
        samples = assembler(period_s, self.parts)
        if samples.n != self.row_count:
            raise ValueError('assembled motion differs from phase row accounting')
        return samples


def plan_work_phase(phases, candidate, distances, kind, *, stroke_index=None,
                    metric_vertices=None, projector=None, placement=None,
                    orientation, normalize_normals=False, pen=0):
    """Project one original material interval and bind its sampled work rows.

    Contact and standoff supply different surface, orientation, timing and pen
    policies. Both pass through this same projection and ordered phase boundary;
    approach, retreat and native acceptance remain separate policy work.
    """
    if not isinstance(phases, MotionPhases):
        raise TypeError('work phase needs ordered motion phases')
    rows = plan_material_rows(
        candidate, distances, metric_vertices=metric_vertices, projector=projector,
        placement=placement, orientation=orientation,
        normalize_normals=normalize_normals, pen=pen)
    start, stop = phases.add_material(rows, kind, stroke_index=stroke_index)
    return rows, (start, stop)


def _next_origin(cursor, work, native, contact_report, contact_resources):
    if not isinstance(cursor, StrokeCursor) or not isinstance(work, WorkPolicy):
        raise TypeError('next stroke needs a material cursor and work policy')
    if (work.kind, work.pen) not in (('contact', 1), ('standoff', 0)):
        raise ValueError('next stroke needs a matching contact or standoff work policy')
    origin = material_address(cursor.candidate, cursor.chunk_index,
                              cursor.source_chunk_count, cursor.placement_id)
    if cursor.source_chunk_origin is not None:
        if type(cursor.source_chunk_origin) is not int or cursor.source_chunk_origin < 0:
            raise ValueError('next stroke needs a nonnegative source chunk origin')
        origin['source_chunk_origin'] = cursor.source_chunk_origin
    if (native is not None or contact_resources is not None) and 'placement_id' not in origin:
        raise ValueError('native next stroke requires an original design placement')
    if native is not None and contact_resources is not None:
        raise ValueError('next stroke accepts one native policy')
    if contact_resources is not None and work.kind != 'contact':
        raise ValueError('contact resources require contact work')
    if contact_resources is not None and contact_report is None:
        raise ValueError('contact resources need a checked contact report')
    return origin


def _accept_native(planned, native, contact_resources):
    request = contact_resources.prepare_draw(planned) if contact_resources is not None else native
    if request is None:
        if contact_resources is not None:
            raise TypeError('contact resources must supply a native request')
        return planned
    if not isinstance(request, NativeRequest):
        raise TypeError('next stroke needs a typed native request')
    finalized = finalize_candidate(
        planned.samples, request.seed, request.samples_path,
        write_samples=lambda path, sampled: request.write_samples(path, sampled, planned.work_range),
        native_plan=request.native_plan, constants_sha=request.constants_sha,
        require_pen=request.require_pen,
        validate_policy=(None if request.validate_policy is None else
                         lambda sampled, plan, positions: request.validate_policy(
                             sampled, plan, positions, planned.work_range)))
    accepted = replace(planned, native=finalized)
    if contact_resources is not None:
        contact_resources.commit_draw(accepted)
    return accepted


def plan_next(cursor: StrokeCursor, work: WorkPolicy, *, before, after_work,
              period_s, assembler, native: NativeRequest | None = None,
              contact_report: Callable[[PlannedStroke], dict] | None = None,
              contact_resources: ContactResources | None = None):
    """Project one interval, assemble its phases and validate its native plan.

    Contact resources stage scan/dip/approach before native Draw acceptance
    and commit ink and receipts only afterward. The caller publishes a
    candidate only after this function returns.
    """
    origin = _next_origin(cursor, work, native, contact_report, contact_resources)
    phases = MotionPhases()
    for phase in before:
        if not isinstance(phase, PhasePart):
            raise TypeError('next stroke needs typed transit phases')
        phases.add(phase.kind, phase.part, stroke_index=phase.stroke_index)
    rows, work_range = plan_work_phase(
        phases, cursor.candidate, work.distances, work.kind,
        stroke_index=work.stroke_index, metric_vertices=work.metric_vertices,
        projector=work.projector, placement=work.placement,
        orientation=work.orientation, normalize_normals=work.normalize_normals,
        pen=work.pen)
    for phase in after_work(rows):
        if not isinstance(phase, PhasePart):
            raise TypeError('next stroke needs typed interaction phases')
        phases.add(phase.kind, phase.part, stroke_index=phase.stroke_index)
    samples = phases.assemble(period_s, assembler)
    planned = PlannedStroke(samples, rows, work_range, phases, origin)
    if contact_report is not None:
        planned = replace(planned, report=contact_report(planned))
    return _accept_native(planned, native, contact_resources)


def checked_joint_positions(plan, samples, *, require_pen=False):
    """Both tool interactions require one finite, row-aligned native solution.

    The contact planner also exports pen and count fields. Those are mandatory
    for its accounting; a standoff planner's simpler JSON lacks them.
    """
    positions = np.asarray(plan['positions'], float)
    if positions.shape != (samples.n, 7) or not np.isfinite(positions).all():
        raise ValueError('executor returned invalid joint samples')
    if require_pen and (plan['sample_count'] != samples.n
                        or not np.array_equal(plan['pen'], samples.pen > 0)):
        raise ValueError('executor plan differs from source samples')
    return positions
