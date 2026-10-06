"""Pin scripts/lib/pen_path.py: time law, retiming, surface lift, preflight, samples files.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_pen_path.py

The surface here is a small local fake of the surface_model.HeightFieldSurface
API (frame / project / count / width_m / height_m / chart / anchor_to) — a plane
and a 40 mm cylinder with closed-form geometry — so this module pins the path
compiler independently of the mapper.
"""

from __future__ import annotations

import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import arm_kinematics as dk  # noqa: E402
import ballpoint_fixture
import numpy as np
import pen_path as dp  # noqa: E402
import pytest

PERIOD = 0.0025
TOOL_AXIS = np.array([math.sqrt(0.5), -math.sqrt(0.5), 0.0])
CONFIG = {
    "schema": "tatbot.draw-config/1", "tool": "lutin-ballpoint-dot",
    "design": {"kind": "spiral", "radius_mm": 6, "turns": 3, "rotation_deg": 0},
    "duration_s": 30, "ease_s": 2, "scan_only": False,
    "orbit": {"standoff_mm": 120, "tilt_deg": 15, "poses": 5, "speed_mm_s": 10},
    "map": {"cell_mm": 1.0, "extent_mm": 60, "chart": "auto"},
    "lean_budget_deg": 20,
}


class FakePlane:
    """Plane chart: point = c + u e_u + v e_v, normal n. Cells all filled unless told otherwise."""

    def __init__(self, center, e_u, e_v, width_m=0.06, height_m=0.06, cell_m=0.001):
        self.center = np.asarray(center, float)
        self.e_u = np.asarray(e_u, float)
        self.e_v = np.asarray(e_v, float)
        self.n = np.cross(self.e_u, self.e_v)
        self.width_m = width_m
        self.height_m = height_m
        rows = int(round(height_m / cell_m)) + 1
        cols = int(round(width_m / cell_m)) + 1
        self.count = np.ones((rows, cols), dtype=np.int32)
        self.chart = SimpleNamespace(kind="plane", radius_m=float("nan"))

    def frame(self, uv):
        uv = np.asarray(uv, float)
        point = self.center + uv[:, :1] * self.e_u + uv[:, 1:] * self.e_v
        n = np.repeat(self.n[None], len(uv), axis=0)
        return point, np.repeat(self.e_u[None], len(uv), 0), np.repeat(self.e_v[None], len(uv), 0), n

    def project(self, points):
        q = np.asarray(points, float) - self.center
        return np.stack([q @ self.e_u, q @ self.e_v], axis=1), q @ self.n

    def anchor_to(self, point):
        uv, dist = self.project(np.asarray(point, float)[None])
        shifted = FakePlane(self.center + dist[0] * self.n, self.e_u, self.e_v, self.width_m, self.height_m)
        shifted.count = self.count
        return shifted, float(dist[0]), uv[0]


class FakeCylinder:
    """Cylinder chart: u along the axis e_u, v = arc length around from the crest; n outward at the crest."""

    def __init__(self, center, e_u, e_v, radius_m, width_m=0.06, height_m=0.06, cell_m=0.001):
        self.center = np.asarray(center, float)
        self.e_u = np.asarray(e_u, float)
        self.e_v = np.asarray(e_v, float)
        self.n = np.cross(self.e_u, self.e_v)
        self.radius_m = radius_m
        self.width_m = width_m
        self.height_m = height_m
        rows = int(round(height_m / cell_m)) + 1
        cols = int(round(width_m / cell_m)) + 1
        self.count = np.ones((rows, cols), dtype=np.int32)
        self.chart = SimpleNamespace(kind="cylinder", radius_m=radius_m)

    def frame(self, uv):
        uv = np.asarray(uv, float)
        phi = uv[:, 1] / self.radius_m
        s, c = np.sin(phi)[:, None], np.cos(phi)[:, None]
        point = self.center + uv[:, :1] * self.e_u + self.radius_m * (s * self.e_v + (c - 1.0) * self.n)
        normal = s * self.e_v + c * self.n
        d_dv = c * self.e_v - s * self.n
        return point, np.repeat(self.e_u[None], len(uv), 0), d_dv, normal

    def project(self, points):
        q = np.asarray(points, float) - self.center
        u = q @ self.e_u
        y = q @ self.e_v
        z = q @ self.n + self.radius_m
        phi = np.arctan2(y, z)
        return np.stack([u, self.radius_m * phi], axis=1), np.hypot(y, z) - self.radius_m


def _contact(surface_center_root, normal, lean_deg=0.0):
    """Contact pose in base with the tool axis lean_deg off -normal (rotated about base x)."""
    r_c = dk.align_rotation(TOOL_AXIS, -np.asarray(normal, float))
    if lean_deg:
        r_c = dk.axis_rotation([1.0, 0.0, 0.0], math.radians(lean_deg)) @ r_c
    tip = dk.base_from_root(surface_center_root)
    return {"schema": "tatbot.draw-pose/1", "frame": "right/base_link", "period_s": PERIOD,
            "tip": tip.tolist(), "rotation": r_c.tolist(), "tool": "lutin-ballpoint-dot"}


def _hold(contact, normal, standoff_m=0.12):
    tip = np.asarray(contact["tip"]) + standoff_m * np.asarray(normal)
    return dict(contact, tip=tip.tolist())


CENTER_ROOT = np.array([0.35, -0.05, 0.02])


# --- time law and spiral --------------------------------------------------------

def test_time_law_totals_and_continuity():
    length = 0.0572
    t, s, sdot = dp.time_law(length, 120.0, 2.0, PERIOD)
    assert len(t) == 48000
    assert t[0] == pytest.approx(PERIOD) and t[-1] == pytest.approx(120.0)
    assert s[-1] == pytest.approx(length, abs=1e-12)
    assert np.all(np.diff(s) >= -1e-15)
    cruise = length / 118.0
    assert sdot.max() == pytest.approx(cruise, rel=1e-12)
    assert sdot[-1] == pytest.approx(0.0, abs=1e-9)
    # speed is continuous at the ease boundaries and the distance integrates it
    assert np.abs(np.diff(sdot)).max() < cruise * 0.01
    assert np.abs(np.diff(s) / PERIOD - 0.5 * (sdot[1:] + sdot[:-1])).max() < cruise * 0.01
    with pytest.raises(ValueError):
        dp.time_law(length, 3.0, 2.0, PERIOD)


def test_resample_by_arclength_hits_vertices():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    points, tangents = dp.resample_polyline_by_arclength(poly, np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert np.allclose(points, [[0, 0], [0.5, 0], [1, 0], [1, 1], [1, 2]])
    assert np.allclose(tangents[1], [1, 0]) and np.allclose(tangents[-1], [0, 1])


# --- compile on surfaces ----------------------------------------------------------


def test_transported_rotations_carry_the_contact_normal():
    rng = np.random.default_rng(5)
    n_c = np.array([0.0, 0.0, 1.0])
    r_c = dk.align_rotation(TOOL_AXIS, -n_c)
    normals = rng.normal(size=(50, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rotations = dp.transported_rotations(normals, n_c, r_c)
    for i in range(50):
        assert np.allclose(rotations[i] @ rotations[i].T, np.eye(3), atol=1e-12)
        assert np.allclose(rotations[i] @ TOOL_AXIS, -normals[i], atol=1e-12)
        assert np.allclose(rotations[i], dk.align_rotation(n_c, normals[i]) @ r_c, atol=1e-12)


# --- files ------------------------------------------------------------------------


# --- orbit --------------------------------------------------------------------------


def _renamed_arm_repo(root: Path, camera_count: int) -> Path:
    """A renamed logical ID over the installed right chain, with one or two mounted views."""
    (root / 'config').mkdir(parents=True)
    arms = json.loads((dk.REPO / 'config/arms.json').read_text())
    arms['arms']['pink'] = arms['arms'].pop('right')
    (root / 'config/arms.json').write_text(json.dumps(arms))
    for name in ('workspace.yaml',):
        (root / 'config' / name).symlink_to(dk.REPO / 'config' / name)
    (root / 'config/tools').symlink_to(dk.REPO / 'config/tools', target_is_directory=True)
    (root / 'config/trossen').symlink_to(dk.REPO / 'config/trossen', target_is_directory=True)
    (root / 'urdf').mkdir()
    urdf = (dk.REPO / 'urdf/tatbot.urdf').read_text()
    vision = (dk.REPO / 'rust/visiond/config/vision.toml').read_text()
    assert vision.count('arm = "right"') == 1
    vision = vision.replace('arm = "right"', 'arm = "pink"', 1)
    if camera_count == 2:
        # Both roles name independently mounted optical links on the same arm.
        vision = vision.replace('arm = "pink"', 'arm = "pink"\noptical_frame = "right/realsense_color_optical_frame"\n'
                                'depth_optical_frame = "right/realsense_depth_optical_frame"', 1)
        vision += '''\n[[cameras.realsense]]
name = "fixture_second_wrist"
serial = "fixture-second"
role = "wrist_aux"
group = "d405"
arm = "pink"
owner_role = "realsense"
optical_frame = "right/aux_color_optical_frame"
depth_optical_frame = "right/aux_depth_optical_frame"
[cameras.realsense.color]
width = 640
height = 480
fps_num = 30
fps_den = 1
[cameras.realsense.depth]
width = 640
height = 480
fps_num = 30
fps_den = 1
'''
        extra = '''
<joint name="right/aux_color_joint" type="fixed">
  <parent link="right/realsense_color_optical_frame"/>
  <child link="right/aux_color_optical_frame"/>
  <origin xyz="0.010 0 0" rpy="0 0 0"/>
</joint>
<link name="right/aux_color_optical_frame"/>
<joint name="right/aux_depth_joint" type="fixed">
  <parent link="right/realsense_depth_optical_frame"/>
  <child link="right/aux_depth_optical_frame"/>
  <origin xyz="0.010 0 0" rpy="0 0 0"/>
</joint>
<link name="right/aux_depth_optical_frame"/>
'''
        urdf = urdf.replace('</robot>', extra + '</robot>')
    (root / 'urdf/tatbot.urdf').write_text(urdf)
    path = root / 'rust/visiond/config/vision.toml'
    path.parent.mkdir(parents=True)
    path.write_text(vision)
    return root


def _third_wxai_repo(root: Path) -> Path:
    """Add a configured third WXAI chain beside the installed left/right chains."""
    registry = json.loads((dk.REPO / 'config/arms.json').read_text())
    registry['arms']['third'] = {**registry['arms']['right'], 'urdf_prefix': 'third',
                                 'workspace_section': 'third', 'profile_ip_field': 'third_ip',
                                 'controller_config': 'config/trossen/third.yaml',
                                 'sdk_end_effector': 'wxai_v0_third'}
    config_dir = root / 'config'
    config_dir.mkdir(parents=True)
    (config_dir / 'arms.json').write_text(json.dumps(registry))
    workspace = (dk.REPO / 'config/workspace.yaml').read_text()
    right_section = workspace.split('\nright:\n', 1)[1].split('\nleft:\n', 1)[0]
    (config_dir / 'workspace.yaml').write_text(
        workspace + '\nthird:\n' + right_section.replace('tip_frame: right/tool_mount',
                                                         'tip_frame: third/tool_mount', 1))
    (config_dir / 'tools').symlink_to(dk.REPO / 'config/tools', target_is_directory=True)
    trossen = config_dir / 'trossen'
    trossen.mkdir()
    (trossen / 'third.yaml').write_bytes((dk.REPO / 'config/trossen/follower.yaml').read_bytes())
    urdf = root / 'urdf/tatbot.urdf'
    urdf.parent.mkdir()
    tree = ET.parse(dk.REPO / 'urdf/tatbot.urdf')
    robot = tree.getroot()
    for element in list(robot):
        if element.tag not in {'link', 'joint'} or not element.get('name', '').startswith('right/'):
            continue
        clone = deepcopy(element)
        for node in clone.iter():
            for key, value in node.attrib.items():
                node.set(key, value.replace('right/', 'third/'))
        robot.append(clone)
    tree.write(urdf, encoding='unicode')
    return root


def test_configured_third_prefix_cpp_preflight_is_explicit_and_renamed_id_is_stable(tmp_path):
    binary = dk.REPO / 'cpp/teleop/build/path_plan_check'
    if not binary.is_file():
        pytest.skip('C++ path_plan_check is not built on this node')
    witness = np.array([0.173762112856, 1.544403791428, 0.826848268509,
                        -0.061226826161, 0.121118485928, 1.642061471939])
    carriage = 0.002

    def hold_file(model, name):
        tip, rotation, _ = model.fk_tcp(witness, carriage)
        samples = dp.assemble(PERIOD, [dp.hold_rows(tip, rotation, 2 * PERIOD, PERIOD) + (0, None)])
        path = tmp_path / name
        dp.write_samples_csv(path, samples, 'orbit', model.tcp_in_link6(), model=model)
        return path

    third = dk.ArmModel('third', repo=_third_wxai_repo(tmp_path / 'third_repo'))
    third.assert_cpp_wxai_compatible()
    assert np.allclose(third.carriage_axis_in_link6, [0., 1., 0.])
    np.testing.assert_allclose(third.joint_limits(), dk.ArmModel().joint_limits())
    csv = hold_file(third, 'third.csv')
    assert 'arm,third\nframe,third/base_link\n' in csv.read_text()
    argv = [str(binary), str(csv), str(PERIOD), *map(str, witness), str(carriage)]
    unbound = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert unbound.returncode == 3 and 'arm must be right or left' in unbound.stderr
    accepted = subprocess.run([*argv, '--json', '--arm-prefix', third.prefix],
                              capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)['sample_count'] == 2
    wrong = subprocess.run([*argv, '--arm-prefix', 'left'], capture_output=True, text=True, check=False)
    assert wrong.returncode == 3 and 'does not match expected WXAI arm prefix' in wrong.stderr
    bad_tip_csv = tmp_path / 'third_bad_tip.csv'
    lines = ['tip_x_m,0.6' if line.startswith('tip_x_m,') else line
             for line in csv.read_text().splitlines()]
    bad_tip_csv.write_text('\n'.join(lines) + '\n')
    bad_tip = subprocess.run([str(binary), str(bad_tip_csv), str(PERIOD), *map(str, witness),
                              str(carriage), '--arm-prefix', third.prefix],
                             capture_output=True, text=True, check=False)
    assert bad_tip.returncode == 3 and 'declared tip model' in bad_tip.stderr

    renamed = dk.ArmModel('pink', repo=_renamed_arm_repo(tmp_path / 'renamed_check', 1),
                          workspace=ballpoint_fixture.workspace())
    assert renamed.prefix == 'right'
    renamed_csv = hold_file(renamed, 'pink.csv')
    normal_csv = hold_file(dk.ArmModel(workspace=ballpoint_fixture.workspace()), 'right.csv')
    assert renamed_csv.read_bytes() == normal_csv.read_bytes()
    normal_argv = [str(binary), str(renamed_csv), str(PERIOD), *map(str, witness), str(carriage)]
    normal = subprocess.run(normal_argv, capture_output=True, text=True, check=False)
    assert normal.returncode == 0, normal.stderr
    explicit = subprocess.run([*normal_argv, '--arm-prefix', renamed.prefix],
                              capture_output=True, text=True, check=False)
    assert explicit.returncode == 0, explicit.stderr
    lines = renamed_csv.read_text().splitlines()
    lines = ['tip_x_m,0.3' if line.startswith('tip_x_m,') else line for line in lines]
    altered = tmp_path / 'pink_wrong_tip.csv'
    altered.write_text('\n'.join(lines) + '\n')
    invalid = subprocess.run([str(binary), str(altered), str(PERIOD), *map(str, witness),
                              str(carriage), '--arm-prefix', renamed.prefix],
                             capture_output=True, text=True, check=False)
    assert invalid.returncode == 3 and 'ballpoint constant' in invalid.stderr


def test_native_planner_nominal_tool_requires_external_model_binding(tmp_path):
    import stroke_operation
    import tool_spec
    from fleet_release import planner_binary

    binary = planner_binary(dk.REPO)
    if not binary.is_file():
        pytest.skip('C++ path_plan_check is not built on this node')
    model = dk.ArmModel(workspace={'right': {'tool_id': 'lutin-ballpoint-dot'}})
    model.assert_cpp_wxai_compatible()
    assert model.tip_source == 'datasheet nominal'
    assert tool_spec.tip_offset_m(model.workspace, 'right') is None
    tip = model.tcp_in_link6()
    assert np.linalg.norm(tip - dk.BALLPOINT_TIP_IN_LINK6) > 1e-4
    witness = np.array([.173762112856, 1.544403791428, .826848268509,
                        -.061226826161, .121118485928, 1.642061471939])
    carriage = .002
    point, rotation, _ = model.fk_tcp(witness, carriage)
    samples = dp.assemble(PERIOD, [dp.hold_rows(point, rotation, 2 * PERIOD, PERIOD) + (0, None)])
    path = tmp_path/'nominal.csv'
    dp.write_samples_csv(path, samples, 'orbit', tip, model=model)
    seed = np.array([*witness, carriage])
    with pytest.raises(stroke_operation.ExecutorRefusalError, match='ballpoint constant'):
        stroke_operation.executor_plan(binary, path, seed, PERIOD, dp.SHA)
    accepted = stroke_operation.executor_plan(binary, path, seed, PERIOD, dp.SHA,
                                              arm_prefix=model.prefix, tool_tip_in_link6=tip).plan
    assert accepted['hardware_authority'] is False
    np.testing.assert_allclose(accepted['positions'], np.tile(seed, (samples.n, 1)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(accepted['tool_model']['tip_in_link6'], tip, atol=1e-12, rtol=0)
    with pytest.raises(stroke_operation.ExecutorRefusalError, match='independently bound'):
        stroke_operation.executor_plan(binary, path, seed, PERIOD, dp.SHA,
                                       arm_prefix=model.prefix, tool_tip_in_link6=tip + [.000001, 0., 0.])
    invalid = subprocess.run([str(binary), str(path), str(PERIOD), *map(str, seed),
                              '--tool-tip-in-link6', *map(str, tip)], capture_output=True, text=True, check=False)
    assert invalid.returncode == 2
    for values in (['nan', '0', '0'], ['.6', '0', '0'], ['.2suffix', '0', '0']):
        invalid = subprocess.run([str(binary), str(path), str(PERIOD), *map(str, seed),
                                  '--arm-prefix', model.prefix, '--tool-tip-in-link6', *values],
                                 capture_output=True, text=True, check=False)
        assert invalid.returncode == 3


# --- square design and the no-scan lift (2026-09-02: the former teleop square) --------------


# --- ink dips inside the path (2026-09-03) ------------------------------------------------


def test_session_lookahead_samples_only_requested_material(monkeypatch):
    surface = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, surface.n)
    hold = _hold(contact, surface.n, standoff_m=0.01)
    config = {**CONFIG, "ease_s": 0.05, "path": {"approach_mm": 10.0}}
    args = (surface, [np.array([[0.0, 0.0], [0.012, 0.0]])], [0.001], contact, hold, PERIOD)
    kwargs = {"config": config, "max_contact_s": 2.0, "preflight": lambda *_: True}
    original = list(dp.iter_chunk_candidates(*args, **kwargs))
    plan_original = dp.operation.plan_next
    sampled = []
    def plan_candidate(*args, **kwargs):
        sampled.append(1)
        return plan_original(*args, **kwargs)
    monkeypatch.setattr(dp.operation, "plan_next", plan_candidate)
    accepted = []
    batch = list(dp.iter_chunk_candidates(*args, **{**kwargs,
        "first_chunk": 2, "end_chunk": 4,
        "preflight": lambda _, report: accepted.append(report["chunk_index"]) or True}))
    assert len(sampled) == 2 and accepted == [2, 3]
    for (candidate, samples, report), (old_candidate, before, old_report) in zip(batch, original[2:4], strict=True):
        assert candidate.arc_range_m == old_candidate.arc_range_m
        for key in ("chunk_index", "source_chunk_count", "source_arc_range_m", "source_uv_start", "source_uv_end"):
            assert report[key] == old_report[key]
        np.testing.assert_array_equal(samples.p[samples.pen > 0], before.p[before.pen > 0])
    for end in (True, 2, -1, 3.5, len(original) + 1):
        with pytest.raises(dp.DrawRefusal, match="look-ahead"):
            list(dp.iter_chunk_candidates(*args, **kwargs, first_chunk=2, end_chunk=end))
    assert len(sampled) == 2
    successor = list(dp.iter_chunk_candidates(*args, **{**kwargs,
        "first_chunk": 2, "end_chunk": 4, "chunk_origin": 2}))
    for local, ((_, _, report), (_, _, old_report)) in enumerate(zip(
            successor, original[2:4], strict=True)):
        assert (report["chunk_index"], report["source_chunk_count"],
                report["source_chunk_origin"]) == (local, len(original) - 2, 2)
        for key in ("source_stroke", "source_arc_range_m", "source_uv_start", "source_uv_end"):
            assert report[key] == old_report[key]
    with pytest.raises(dp.DrawRefusal, match="source chunk origin"):
        list(dp.iter_chunk_candidates(*args, **kwargs, first_chunk=2, end_chunk=4,
                                        chunk_origin=3))


def test_session_candidate_uses_original_placement_before_preflight():
    surface = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, surface.n)
    hold = _hold(contact, surface.n, standoff_m=0.01)
    seen = []
    candidates = list(dp.iter_chunk_candidates(
        surface, [np.array([[0., 0.], [.006, 0.]])], [.001], contact, hold, PERIOD,
        config={**CONFIG, 'ease_s': 0.05, 'path': {'approach_mm': 10.0}},
        max_contact_s=2.0, placement_ids=['design-placement'],
        preflight=lambda _, report: seen.append((report['placement_id'], report['source_stroke'])) or True))
    assert len(candidates) > 1
    assert seen == [('design-placement', 0)] * len(candidates)
    with pytest.raises(dp.DrawRefusal, match='placement IDs'):
        list(dp.iter_chunk_candidates(
            surface, [np.array([[0., 0.], [.006, 0.]])], [.001], contact, hold, PERIOD,
            config=CONFIG, max_contact_s=2.0, placement_ids=[''], preflight=lambda *_: True))


# --- corner-aware retiming (2026-09-14) ---------------------------------------

def _hatch_stroke():
    """A hatch pair: 10 mm out, a 160 deg turnaround, 10 mm back 0.5 mm over, plus a 90 deg hook."""
    return np.array([[0.0, 0.0], [0.010, 0.0], [0.0, 0.0005], [0.0, 0.004]])


def test_corner_speed_limits_isolated_corner_and_dense_arc():
    retime = dp.Retime()
    # an isolated 90 deg corner between 5 mm chords: accel * window / theta
    poly = np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.005]])
    arc, turn = dp.polyline_turn_angles(poly)
    caps = dp.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    assert caps[0] == caps[2] == 0.003
    assert caps[1] == pytest.approx(retime.corner_accel_m_s2 * retime.window_s / (math.pi / 2), rel=1e-6)
    # a dense arc of radius 0.2 mm sampled every 2 deg: about sqrt(accel * radius)
    radius = 0.0002
    angles = np.radians(np.arange(0.0, 180.0, 2.0))
    arc_poly = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    poly = np.concatenate([[[radius + 0.005, 0.0]], arc_poly[::-1], [[-radius - 0.005, 0.0]]])
    arc, turn = dp.polyline_turn_angles(poly)
    caps = dp.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    middle = caps[len(caps) // 2]
    assert 0.5 * math.sqrt(retime.corner_accel_m_s2 * radius) < middle < 1.5 * math.sqrt(retime.corner_accel_m_s2 * radius)
    # repeated vertices neither hide nor invent a corner
    doubled = np.repeat(np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.005]]), 2, axis=0)
    _, turn2 = dp.polyline_turn_angles(doubled)
    assert np.count_nonzero(turn2) == 1 and turn2.max() == pytest.approx(math.pi / 2)


def test_accel_limited_speeds_respect_budget_and_rest_ends():
    arc = np.linspace(0.0, 0.03, 3001)
    caps = np.full(len(arc), 0.010)
    v = dp.accel_limited_speeds(arc, caps, 0.010, v_start=0.0, v_end=0.0)
    assert v[0] == 0.0 and v[-1] == 0.0 and v.max() == pytest.approx(0.010)
    implied = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * np.diff(arc))
    assert np.abs(implied).max() <= 0.010 + 1e-9
    # the S-curve variant never accelerates harder and reaches the plateau with zero acceleration
    s_curve = dp.accel_limited_speeds(arc, caps, 0.010, v_start=0.0, v_end=0.0, jerk=0.05)
    assert np.all(s_curve <= v + 1e-12)
    implied_s = (s_curve[1:] ** 2 - s_curve[:-1] ** 2) / (2.0 * np.diff(arc))
    plateau = np.flatnonzero(s_curve >= 0.010 - 1e-12)
    assert len(plateau) and abs(implied_s[plateau[0] - 1]) < 0.002


def test_corner_time_law_keeps_the_legacy_law_on_a_straight_stroke():
    straight = np.array([[0.0, 0.0], [0.012, 0.0]])
    t, s, sdot = dp.time_law(0.012, 0.012 / 0.003 + 2.0, 2.0, PERIOD)
    t2, s2, sdot2, info = dp.corner_time_law(straight, 0.003, 2.0, PERIOD, dp.Retime())
    np.testing.assert_array_equal(t2, t)
    np.testing.assert_array_equal(s2, s)
    assert info["added_s"] == 0.0 and info["limited_vertices"] == 0


def _polyline_deviation_mm(points, polyline):
    """Largest distance (mm) from any point to the polyline: every retimed sample must lie on it."""
    points, poly = np.asarray(points, float), np.asarray(polyline, float)
    seg = np.diff(poly, axis=0)
    length2 = np.einsum("ij,ij->i", seg, seg)
    a, seg, length2 = poly[:-1][length2 > 0], seg[length2 > 0], length2[length2 > 0]
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nij,ij->ni", rel, seg) / length2[None, :], 0.0, 1.0)
    diff = rel - t[:, :, None] * seg[None, :, :]
    return float(np.sqrt(np.einsum("nij,nij->ni", diff, diff).min(axis=1)).max()) * 1e3


def _angular_rates(rotations, period_s):
    """Body-rate magnitude between consecutive rotations (rad/s)."""
    r = np.asarray(rotations, float)
    step = np.einsum("nji,njk->nik", r[:-1], r[1:])
    angle = np.arccos(np.clip((np.einsum("nii->n", step) - 1.0) / 2.0, -1.0, 1.0))
    return np.concatenate([angle / period_s, angle[-1:] / period_s])


def test_corner_time_law_slows_corners_preserves_geometry_and_reports_the_cost():
    stroke = _hatch_stroke()
    retime = dp.Retime()
    length = dp.polyline_length(stroke)
    t, s, sdot, info = dp.corner_time_law(stroke, 0.003, 2.0, PERIOD, retime)
    legacy_t, legacy_s, _ = dp.time_law(length, length / 0.003 + 2.0, 2.0, PERIOD)
    assert info["limited_vertices"] == 2 and info["added_s"] == pytest.approx(t[-1] - legacy_t[-1])
    assert t[-1] > legacy_t[-1]
    assert s[0] >= 0.0 and s[-1] == length and np.all(np.diff(s) >= 0.0)
    # every retimed sample is a point of the polyline
    points, _ = dp.resample_polyline_by_arclength(stroke, s)
    assert _polyline_deviation_mm(points, stroke) < 1e-9
    # the tip slows through both corners below the corner caps, and the ends still rest
    speed = np.gradient(s, PERIOD)
    arc, turn = dp.polyline_turn_angles(stroke)
    caps = dp.corner_speed_limits(arc, turn, 0.003, retime.corner_accel_m_s2, retime.window_s)
    for vertex in (1, 2):
        row = int(np.argmin(np.abs(s - arc[vertex])))
        assert speed[row] < caps[vertex] * 1.5 + 5e-5
    assert speed[0] < 1e-4 and abs(speed[-1]) < 1e-4
    assert speed.max() <= 0.003 + 1e-5
    # the tangential acceleration on the ramps stays within the budget (finite differences, away from vertices)
    accel = np.gradient(speed, PERIOD)
    away = np.ones(len(s), dtype=bool)
    for vertex in (1, 2):
        away &= np.abs(s - arc[vertex]) > 2e-5
    assert np.abs(accel[away]).max() < retime.corner_accel_m_s2 * 1.2


def test_pen_up_leg_is_one_continuous_profile_on_the_exact_polyline():
    retime = dp.Retime()
    points = np.array([[0.30, 0.0, 0.15], [0.32, 0.01, 0.15], [0.32, 0.01, 0.125], [0.32, 0.01, 0.12]])
    r0 = np.eye(3)
    r1 = dk.axis_rotation([0.0, 0.0, 1.0], 0.2)
    p, r, chord = dp.pen_up_leg(points, r0, r1, [0.020, 0.010, 0.003], retime, PERIOD,
                                gentle_end=True, include_start=True, omega_max=math.radians(8.0))
    assert _polyline_deviation_mm(p, points) < 1e-9
    np.testing.assert_allclose(p[0], points[0], atol=1e-12)
    np.testing.assert_allclose(p[-1], points[-1], atol=1e-12)
    assert sorted(set(chord.tolist())) == [0, 1, 2]
    speed = np.linalg.norm(np.gradient(p, PERIOD, axis=0), axis=1)
    assert speed[0] < 1e-4 and speed[-1] < 1e-4
    # chord caps hold, and the tip never stops between the chords (no intermediate rest)
    assert speed[chord == 0].max() <= 0.020 + 1e-5
    assert speed[chord == 1].max() <= 0.010 + 1e-5
    assert speed[chord == 2].max() <= 0.003 + 1e-5
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    interior = speed[(arc > 0.001) & (arc < arc[-1] - 0.001)]
    assert interior.min() > 1e-4
    # the rotation completes within the first chord and is then held; its rate rests at the chord ends
    np.testing.assert_allclose(r[chord > 0], np.repeat(r1[None], np.count_nonzero(chord > 0), axis=0), atol=1e-12)
    omega = _angular_rates(r, PERIOD)
    assert omega.max() <= math.radians(8.0) + 1e-6
    boundary = int(np.argmax(chord > 0))
    assert omega[boundary - 2:boundary + 2].max() < 1e-3
    # the gentle touchdown: the last 5 mm decelerate no harder than the touchdown budget
    accel = np.linalg.norm(np.gradient(np.gradient(p, PERIOD, axis=0), PERIOD, axis=0), axis=1)
    last = chord == 2
    assert accel[last][10:-3].max() < retime.touchdown_accel_m_s2 * 1.3
    # a leg whose first chord has no length but turns is refused (the caller turns in place first)
    with pytest.raises(ValueError):
        dp.pen_up_leg(np.array([points[1], points[1], points[2]]), r0, r1, [0.010, 0.003], retime, PERIOD)


def test_pen_up_leg_keeps_identical_rotations_at_a_chunk_boundary():
    # Retained from a refused inclined-paper compile: trace(R @ R.T) rounds
    # below three even though both chunk endpoints have the same rotation.
    rotation = np.array([
        [0.776835225299586, 0.614897105249878, 0.13575191596829006],
        [0.10145927991036406, 0.09054160402183661, -0.9907109732213646],
        [-0.621476505793913, 0.7833924737297858, 0.007948889841669395],
    ])
    points = np.array([[0.30, 0.0, 0.15], [0.30, 0.0, 0.15],
                       [0.30, 0.0, 0.125], [0.30, 0.0, 0.12]])
    assert dk.rotation_angle(rotation @ rotation.T) > 1e-9
    p, r, chord = dp.pen_up_leg(points, rotation, rotation.copy(), [0.02, 0.01, 0.003],
                                dp.Retime(), PERIOD, gentle_end=True, include_start=True)
    np.testing.assert_array_equal(r, np.broadcast_to(rotation, r.shape))
    np.testing.assert_array_equal(p[[0, -1]], points[[0, -1]])
    assert set(chord) == {1, 2}
    # A real turn still needs a nonzero travel chord.
    turned = dk.axis_rotation([0.0, 0.0, 1.0], 1e-6) @ rotation
    with pytest.raises(ValueError, match="cannot turn in place"):
        dp.pen_up_leg(points, rotation, turned, [0.02, 0.01, 0.003], dp.Retime(), PERIOD)


def test_retime_settings_default_legacy_and_validation():
    assert dp.retime_settings({"path": {"timing": "legacy"}}, PERIOD) is None
    default = dp.retime_settings({}, PERIOD)
    assert isinstance(default, dp.Retime) and default.window_s == pytest.approx(dp.FEEDFORWARD_WINDOW_TICKS * PERIOD)
    custom = dp.retime_settings({"path": {"corner_accel_mm_s2": 5, "angular_accel_rad_s2": 0,
                                          "lean_deadband_blend_deg": 0, "orientation_smoothing_mm": 0}}, PERIOD)
    assert custom.corner_accel_m_s2 == pytest.approx(0.005) and custom.angular_accel_rad_s2 == 0.0
    assert custom.deadband_blend_rad == 0.0 and custom.orientation_smoothing_m == 0.0
    for bad in ({"path": {"timing": "fast"}}, {"path": {"corner_accel_mm_s2": 0}},
                {"path": {"pen_up_accel_mm_s2": -1}}, {"path": {"lean_deadband_blend_deg": 60}},
                {"path": {"angular_accel_rad_s2": float("nan")}}):
        with pytest.raises(dp.DrawRefusal):
            dp.retime_settings(bad, PERIOD)


def test_lean_deadband_blend_is_c1_and_never_leans_past_the_deadband():
    n_c = np.array([0.0, 0.0, 1.0])
    r_c = dk.align_rotation(TOOL_AXIS, -n_c)
    swing = np.radians(np.linspace(0.0, 30.0, 3001))
    normals = np.stack([np.sin(swing), np.zeros_like(swing), np.cos(swing)], axis=1)
    deadband, blend = math.radians(12.0), math.radians(3.0)
    kinked = dp.transported_rotations(normals, n_c, r_c, deadband)
    blended = dp.transported_rotations(normals, n_c, r_c, deadband, blend)
    lean = np.array([math.acos(np.clip(-(r @ TOOL_AXIS) @ n, -1, 1)) for r, n in zip(blended, normals, strict=True)])
    assert lean.max() <= deadband + 1e-9
    inside = swing <= deadband - blend
    outside = swing >= deadband + blend
    np.testing.assert_allclose(blended[inside], kinked[inside], atol=1e-12)
    np.testing.assert_allclose(blended[outside], kinked[outside], atol=1e-12)
    # the followed angle (swing - lean) is C1: its rate has no step at the deadband edge
    followed = swing - lean
    rate = np.gradient(followed, swing)
    step = np.abs(np.diff(rate)).max()
    kinked_lean = np.array([math.acos(np.clip(-(r @ TOOL_AXIS) @ n, -1, 1)) for r, n in zip(kinked, normals, strict=True)])
    kinked_step = np.abs(np.diff(np.gradient(swing - kinked_lean, swing))).max()
    assert step < 0.1 * kinked_step
    assert abs(lean[np.argmin(np.abs(swing - deadband))] - (deadband - blend / 4.0)) < 1e-4


def test_smoothed_normal_field_follows_the_surface_and_reports_deviation():
    cylinder = FakeCylinder(CENTER_ROOT, [1, 0, 0], [0, 1, 0], 0.04)
    stroke = np.array([[0.0, -0.02], [0.0, 0.02]])
    arc, normals, half = dp.smoothed_normal_field(cylinder, stroke, 1e-3)
    lookup = dp.normal_field_lookup(arc, normals, half)
    s = np.linspace(0.0, 0.04, 4001)
    smoothed = lookup(s)
    uv, _ = dp.resample_polyline_by_arclength(stroke, s)
    _, _, _, exact = cylinder.frame(uv)
    # a cylinder's normal field is already smooth: the average stays within a fraction of a degree
    assert np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", smoothed, exact), -1, 1))).max() < 0.5
    # the averaged field turns continuously (no rate steps larger than the grid resolution allows)
    turn = np.arccos(np.clip(np.einsum("ij,ij->i", smoothed[:-1], smoothed[1:]), -1, 1))
    assert np.abs(np.diff(turn)).max() < 0.02 * turn.max()
    raw = dp.normal_field_lookup(arc, normals, 0.0)(s)
    np.testing.assert_allclose(raw, exact, atol=1e-6)


def test_samples_header_declares_the_leader_arm_only_when_asked(tmp_path):
    """A follower file is byte-for-byte what it was; a leader file names its arm
    ahead of its frame so the executor resolves that arm's tool."""
    import arm_kinematics as dk
    samples = dp.assemble(0.0025, [dp.hold_rows(np.array([0.3, 0.0, 0.1]), np.eye(3), 0.01, 0.0025) + (0, None)])
    tip = np.array([0.2, -0.01, 0.0])
    right, left = tmp_path / "right.csv", tmp_path / "left.csv"
    dp.write_samples_csv(right, samples, "path", tip)
    dp.write_samples_csv(left, samples, "path", tip, arm="left")
    right_lines, left_lines = right.read_text().splitlines(), left.read_text().splitlines()
    assert "frame,right/base_link" in right_lines and not any(line.startswith("arm,") for line in right_lines)
    assert left_lines.index("arm,left") < left_lines.index("frame,left/base_link")
    assert [line for line in left_lines if not line.startswith(("arm,", "frame,"))] == \
        [line for line in right_lines if not line.startswith("frame,")]
    _, header = dp.read_samples_csv(left)
    assert header["arm"] == "left" and header["frame"] == "left/base_link"
    with pytest.raises(ValueError, match="unknown arm"):
        dp.write_samples_csv(tmp_path / "bad.csv", samples, "path", tip, arm="middle")
    assert dk.ArmModel("left").frame == header["frame"]
