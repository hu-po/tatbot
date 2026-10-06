"""Lower flat-paper factory intent through the production motion compiler.

This compiles reference primitives. The sampled plane and start are simulated geometry,
never measured contact evidence or hardware authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import arm_kinematics as dk
import numpy as np
import pen_path as dp
import stroke_operation as operation
from surface_model import HeightFieldSurface, PlaneChart
from tatbot_contracts.observations import FOLLOWER_JOINTS

from tatbot_sim.native_reference import ReferenceBatch, compile_joint_plan


def _tool_model(config):
    model = dk.ArmModel(workspace=config.workspace, repo=config.repo)
    model.assert_cpp_wxai_compatible()
    tip = dk.tool_tip_in_link6(config.geometry.tcp_offset_m, chain=model.chain)
    if not np.allclose(tip, model.tcp_in_link6(), atol=1e-9, rtol=0):
        raise ValueError('simulated tool TCP differs from the production planner model')
    return model


def read_config(path: str, config) -> dict:
    """Explicit motion settings; no second set of factory motion defaults."""
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, dict) or value.get('schema') != 'tatbot.draw-config/1'
            or value.get('tool') != config.tool.tool_id or value.get('dips')):
        raise ValueError('production-flat requires a matching draw config without legacy dips')
    speed = value.get('draw_speed_mm_s')
    if type(speed) not in (float, int) or not np.isfinite(speed) or speed <= 0:
        raise ValueError('production-flat requires an explicit positive draw_speed_mm_s')
    if (config.tool.tool_id != 'lutin-ballpoint-dot' or config.substrate.name != 'paper_pad'
            or config.dr.surface.profile != 'flat' or config.dr.surface.enabled
            or config.scenario_path or np.any(config.geometry.calibration_delta_m)):
        raise ValueError('production-flat supports nominal ballpoint geometry on an undisplaced paper plane')
    config.validate_sources()
    _tool_model(config)
    return value


def plane_from_world(surface, index, *, model):
    points, normals = surface.frame_np(index, np.array([[0., 0.], [.01, 0.], [0., .01]]))
    normal = np.asarray(normals[0], float)
    normal /= np.linalg.norm(normal)
    u = np.asarray(points[1] - points[0], float)
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    if np.dot(v, points[2] - points[0]) <= 0:
        raise ValueError('simulated surface chart has an inconsistent orientation')
    field = HeightFieldSurface(PlaneChart(model.root_from_base(points[0]), np.stack([u, v, normal], axis=1)),
                               np.zeros((5, 5)), surface.width_m, surface.height_m)
    field.count[:] = 1  # the simulated plane is fully defined, with no scan holes
    field.anchor_uv = np.zeros(2)
    field.anchor_point = model.root_from_base(points[0])
    return field


def compile_strokes(field, strokes, speeds, contact, hold, cfg, *, max_contact_s, output, model):
    """The same Cartesian chunks, CSV parser and native planner as a session."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    field.to_npz(output / 'surface.npz')
    (output / 'draw.json').write_text(json.dumps(cfg, sort_keys=True))
    seed = np.array([*hold['joints'], hold['carriage_m']], dtype=float)
    initial = seed.copy()
    current_hold = hold
    commands, targets, markers, reports = [], [], [], []
    offset = 0

    def preflight(samples, report):
        nonlocal seed, current_hold, offset
        checked = dp.preflight(samples, field, cfg, model.tool_axis_in_link6, current_hold, model=model)
        csv = output / f'stroke-{len(reports)}.csv'
        dp.write_samples_csv(csv, samples, 'path', model.tcp_in_link6(),
                             {'carriage_ik': int(bool(cfg.get('path', {}).get('carriage_ik', False)))}, model=model)
        native = compile_joint_plan(csv, seed, samples.period_s, model=model)
        positions = operation.checked_joint_positions(native, samples, require_pen=True)
        plan_path = csv.with_suffix('.joint-plan.json')
        plan_path.write_text(json.dumps(native))
        reports.append({'start_tick': offset + 1, 'stop_tick': offset + samples.n,
                        'samples_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
                        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                        'plan_file': plan_path.name, 'compiler': report, 'preflight': checked})
        commands.append(positions)
        targets.append(samples.p.copy())
        markers.append(samples.pen > 0)
        seed = positions[-1].copy()
        current_hold = {'tip': samples.p[-1].tolist(), 'rotation': samples.R[-1].tolist()}
        offset += samples.n
        return True

    for candidate, samples, report in dp.iter_chunk_candidates(field, strokes, speeds, contact, hold, dp.C.period_s,
            config=cfg, max_contact_s=max_contact_s, preflight=preflight, model=model):
        if (report['source_stroke'] != candidate.source_stroke
                or report['source_arc_range_m'] != list(candidate.arc_range_m)):
            raise ValueError('production-flat stroke differs from its material candidate')
        del candidate, samples, report
    return initial, np.concatenate(commands), np.concatenate(targets), np.concatenate(markers), reports


def _pose(joints, *, model):
    point, rotation, _ = model.fk_tcp(joints[:6], joints[6])
    return {'joints': joints[:6].tolist(), 'carriage_m': float(joints[6]),
            'tip': point.tolist(), 'rotation': rotation.tolist()}


def planned_drawing_start(measured, *, model: dk.ArmModel | None = None):
    """Plan the nearest admitted carriage setup at the recorded lifted TCP.

    The original pose remains measured evidence. The existing native approach
    moves to this separate planned seed before the checked drawing stream.
    """
    model = dk.ArmModel() if model is None else model
    carriage = float(measured['carriage_m'])
    limits = dp.C.carriage_ik
    if not limits.pen_up_min_m <= carriage <= limits.pen_up_max_m:
        raise ValueError('measured start carriage is outside the pen-up envelope')
    target = min(limits.max_m, max(limits.min_m, carriage))
    if target == carriage:
        return measured, None
    joints = model.solve_ik(np.asarray(measured['tip']), np.asarray(measured['rotation']),
                            np.asarray(measured['joints']), target)
    point, rotation, _ = model.fk_tcp(joints, target)
    planned = dict(measured, joints=joints.tolist(), carriage_m=target,
                   tip=point.tolist(), rotation=rotation.tolist(), operator_contact=False,
                   planned_carriage_setup=True)
    position_error = float(np.linalg.norm(point - measured['tip']))
    if position_error > dk.IK_POSITION_TOL_M:
        raise ValueError('planned carriage setup does not preserve the lifted TCP')
    return planned, {'measured_carriage_m': carriage, 'planned_carriage_m': target,
                     'carriage_displacement_m': target-carriage,
                     'lifted_tcp_target_preserved': True, 'position_error_m': position_error,
                     'position_tolerance_m': dk.IK_POSITION_TOL_M, 'contact_reference_changed': False,
                     'motion_owner': 'existing native approach', 'hardware_authority': False}


def from_intent(plan, world, cfg, output, *, max_ticks, artifact_root=None):
    """Keep sampled UV geometry; replace synthetic motion with native commands."""
    from tatbot_sim.judge import strokes_from_plan_paths

    if plan.dips and any(plan.dips):
        raise ValueError('production-flat does not yet support palette primitives')
    world.config.validate_sources()
    model = _tool_model(world.config)
    expert = world.expert
    order = [world.ik_names.index(name) for name in FOLLOWER_JOINTS]
    seeds, commands, targets, surface_points, normals, provenance = [], [], [], [], [], []
    lengths = []
    for index, path in enumerate(plan.paths):
        strokes = [stroke.points for stroke in strokes_from_plan_paths(path)]
        field = plane_from_world(world.base.surface, index, model=model)
        first, normal = world.base.surface.frame_np(index, strokes[0][:1])
        guess = expert.solve_pose(first, world.positions()[index:index + 1], normals=normal)
        contact = _pose(guess[0, order].cpu().numpy().astype(float), model=model)
        hold_point = first[0] + dp.approach_standoff_m(cfg) * normal[0]
        hold_guess = expert.solve_pose(hold_point[None], guess, normals=normal)
        hold, _ = planned_drawing_start(_pose(hold_guess[0, order].cpu().numpy().astype(float), model=model), model=model)
        folder = Path(output) / f'env-{index}'
        initial, positions, points, pen, chunks = compile_strokes(
            field, strokes, [cfg['draw_speed_mm_s'] * .001] * len(strokes), contact, hold, cfg,
            max_contact_s=float(world.config.tool.raw.get('stroke', {}).get('max_s', 6)), output=folder, model=model)
        if len(positions) > max_ticks:
            raise dp.DrawRefusal('duration', 'native reference exceeds the requested episode horizon')
        uv, _ = field.project(model.root_from_base(points))
        plane_points, _, _, plane_normals = field.frame(uv)
        seeds.append(initial)
        commands.append(positions)
        targets.append(points)
        surface_points.append(model.base_from_root(plane_points))
        normals.append(plane_normals)
        lengths.append(len(positions))
        provenance.append({'geometry_source': 'simulated-plane', 'chunks': chunks,
                           'tool_model': {'arm_prefix': model.prefix, 'tip_in_link6': model.tcp_in_link6().tolist(),
                                          'carriage_axis_in_link6': model.carriage_axis_in_link6.tolist(),
                                          'tip_source': model.tip_source, 'hardware_authority': False},
                           'pen_down_count': int(pen.sum()),
                           'draw_config': cfg,
                           'surface_sha256': hashlib.sha256((folder / 'surface.npz').read_bytes()).hexdigest(),
                           'draw_config_sha256': hashlib.sha256((folder / 'draw.json').read_bytes()).hexdigest(),
                           'reference_directory': str(folder.relative_to(artifact_root) if artifact_root else folder)})
    horizon = max(lengths)

    def padded(rows):
        return np.stack([np.concatenate([row, np.repeat(row[-1:], horizon - len(row), axis=0)]) for row in rows])

    reference = ReferenceBatch(np.stack(seeds), padded(commands), dp.C.period_s, tuple(provenance))
    return replace(plan, n_app=0, q_raised=None, draw_horizon=horizon,
                   targets=padded(targets), pen_normals=padded(normals),
                   surface_points=padded(surface_points), surface_normals=padded(normals),
                   lean_profiles=[np.zeros((horizon, 2)) for _ in lengths], lengths=np.asarray(lengths),
                   dip_mask=None, dip_credits=None, native_reference=reference,
                   intended_targets=None, intended_lengths=None)
