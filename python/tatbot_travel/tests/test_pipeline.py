"""Fast checks of the generator's geometry and contracts (no rendering, no GPU)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from tatbot_travel import assets, ink
from tatbot_travel.camera import Intrinsics, render_plan
from tatbot_travel.expert import DOWN, tilt_limited
from tatbot_travel.lines import DrawnLine, sample_chart_curve
from tatbot_travel.motion import MotionConfig, PlacementConfig, rest_height, sample_rest_pose, sample_script
from tatbot_travel.phantom import build_phantom
from tatbot_travel.scene import SceneBuilder
from tatbot_travel.shell import canonical_shell, chart_skin
from tatbot_travel.urdf_chain import load_chain, rpy_to_matrix, rpy_to_quat


def test_blue_arm_chain_has_six_hinges_and_the_wrist_camera():
    chain = load_chain(assets.urdf_path(), "left")
    assert chain.hinge_names() == [f"left/joint_{i}" for i in range(6)]
    assert np.all(chain.limits()[:, 0] < chain.limits()[:, 1])
    assert "left/realsense_color_optical_frame" in chain.joints
    assert "left/tattoo_needle" in chain.joints


def test_rpy_quaternion_matches_matrix():
    rpy = (0.3, -1.1, 2.0)
    w, x, y, z = rpy_to_quat(rpy)
    q = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(q, rpy_to_matrix(rpy), atol=1e-9)


def test_render_plan_covers_every_real_pixel():
    intr = Intrinsics.left_wrist()
    plan = render_plan(intr)
    assert plan.map_x.shape == (intr.height, intr.width)
    assert plan.map_x.min() >= 0 and plan.map_x.max() <= plan.width - 1
    assert plan.map_y.min() >= 0 and plan.map_y.max() <= plan.height - 1


def test_phantom_is_an_arm_with_a_forearm_in_the_middle():
    phantom = build_phantom()
    length = phantom.x_range[1] - phantom.x_range[0]
    assert 0.6 < length < 0.66
    assert phantom.x_range[0] < phantom.forearm[0] < phantom.forearm[1] < phantom.x_range[1]
    assert 0.025 < float(np.median(phantom.radius_at(np.linspace(*phantom.forearm, 9)))) < 0.05


def test_closed_fist_uses_the_articulated_soma_catalog_and_retains_surface_addresses():
    import trimesh
    from tatbot_sim.inkmap.rig import load_body_rig

    opened = build_phantom(length_m=0.43)
    closed = build_phantom(length_m=0.43, hand_pose="closed")
    assert closed.x_range[1] < opened.x_range[1] - 0.02
    rig = load_body_rig()
    record = rig.catalog_record["poses"]["left-fist-reference"]
    assert record["correctives_enabled"] is True
    assert len(record["joint_rotations_euler_xyz_deg"]) == 15
    assert closed.provenance["posed_surface_sha256"] == record["surface_sha256"]
    # Cutting/capping and texture seams retain every selected canonical skin triangle.
    assert np.array_equal(closed.vertices[closed.faces], closed.body.vertices[closed.face_indices])
    transform = np.asarray(closed.provenance["phantom_from_body"])
    source = rig.posed("left-fist-reference", np.eye(4)).vertices[closed.face_indices]
    expected = source @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(closed.vertices[closed.faces], expected, atol=1e-8)
    mesh = closed.skin_mesh()
    mesh.merge_vertices()  # texture seam duplicates do not represent an open skin
    assert mesh.is_watertight and np.isfinite(mesh.vertices).all()
    assert mesh.area_faces.min() > 1e-12
    # The same closed mesh is available to rendering and the clearance code.
    assert len(trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))) == 1
    with pytest.raises(ValueError, match="hand_pose"):
        build_phantom(hand_pose="misspelled")


def test_inner_forearm_ink_lifts_to_the_upward_surface_in_a_palm_up_pose():
    from scipy.spatial.transform import Rotation

    phantom = build_phantom(hand_pose="closed")
    shell = chart_skin(phantom, theta_origin=np.pi)
    pose = sample_rest_pose(np.random.default_rng(0), phantom,
                            PlacementConfig(back_up_prob=0, palm_up_prob=1,
                                            roll_jitter_deg=0, tilt_jitter_deg=0))
    x = np.linspace(shell.x[0], shell.x[-1], 9)
    points, normals = shell.lift(x, np.zeros_like(x))
    axis = pose.apply(phantom.centre_at(x))
    assert np.all(pose.apply(points)[:, 2] > axis[:, 2])
    assert np.all(pose.rot.apply(normals)[:, 2] > 0.5)
    assert np.allclose(pose.rot.apply([0, 0, -1]), [0, 0, 1])
    # Every chart point remains on the real reference forearm, with no rotation
    # of the physical surface when its texture coordinate origin is changed.
    import trimesh

    _, distance, _ = trimesh.proximity.closest_point_naive(phantom.skin_mesh(), points)
    assert distance.max() < 1e-9
    assert pose.pos[2] == pytest.approx(rest_height(phantom, Rotation.from_euler('x', np.pi)))


def test_shell_lies_on_the_skin_evenly_charted_with_outward_normals():
    import trimesh

    phantom, shell = build_phantom(), canonical_shell()
    mesh = trimesh.Trimesh(phantom.vertices, phantom.faces, process=False)
    x = np.linspace(shell.x[0], shell.x[-1], 9)
    theta = np.linspace(-2.8, 2.8, 9)
    points, normals = shell.lift(x, theta)
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, points)
    assert distance.max() < 1e-9  # on the actual skin, wrist included
    centroids = shell.points.mean(axis=1)
    assert np.all(((shell.points - centroids[:, None]) * shell.normals).sum(axis=-1).mean(axis=1) > 0)  # outward
    perimeter = np.linalg.norm(np.diff(shell.points, axis=1), axis=-1).sum(axis=1)
    assert np.allclose(perimeter, 2 * np.pi * shell.radius, rtol=0.02)


def test_scaled_closed_pose_ink_and_render_mesh_resolve_the_same_skin_addresses():
    import trimesh

    phantom = build_phantom(length_m=0.43, hand_pose="closed").scaled(0.82, 0.76)
    shell = chart_skin(phantom, theta_origin=np.pi)
    rng = np.random.default_rng(23)
    chart = sample_chart_curve(rng, phantom, off_top_prob=0)
    skin = np.full((*ink.TEXTURE_SHAPE, 3), 200, np.uint8)
    _, coverage = ink.compose(skin, [ink.line_item(rng, chart, shell)], shell)
    graph = ink.ink_graph(coverage, shell)
    labels = graph.surface_arrays()
    assert np.allclose(phantom.body.points(labels["face_indices"], labels["barycentric"]),
                       labels["points_m"], atol=1e-12)
    rendered = trimesh.load(file_obj=io.StringIO(shell.obj_text(offset=0)), file_type="obj", process=False)
    _, distance, _ = trimesh.proximity.closest_point_naive(rendered, graph.points[::10])
    assert distance.max() < 1e-6  # sub-micrometre OBJ/proximity arithmetic
    assert np.isin(labels["face_indices"], phantom.ink_face_indices).all()
    # Targets between stored samples also resolve through the surface chart.
    for edge in graph.edges:
        point, normal, _ = edge.at(edge.length * 0.437)
        _, distance, _ = trimesh.proximity.closest_point_naive(phantom.skin_mesh(), point[None])
        assert distance.max() < 1e-9 and np.isclose(np.linalg.norm(normal), 1)


def test_saved_surface_addresses_do_not_count_as_episodes_in_stats(tmp_path, capsys):
    from argparse import Namespace

    from tatbot_travel.cli import stats
    from tatbot_travel.writer import EpisodeLabels

    labels = EpisodeLabels()
    labels.add({"mode": 0})
    labels.add({"mode": 0})
    surface = {"face_indices": np.array([17]), "barycentric": np.array([[0.2, 0.3, 0.5]])}
    labels.save(tmp_path / "labels", 0, {"body_surface": {"pose_id": "left-fist-reference"}}, surface)
    with np.load(tmp_path / "labels/episode_000000.surface.npz") as retained:
        assert np.array_equal(retained["face_indices"], surface["face_indices"])
        assert np.array_equal(retained["barycentric"], surface["barycentric"])
    assert stats(Namespace(root=tmp_path)) == 0
    assert capsys.readouterr().out.startswith("1 episodes, 2 frames:")


def test_a_painted_line_thins_to_strokes_along_it():
    rng = np.random.default_rng(3)
    phantom, shell = build_phantom(), canonical_shell()
    chart = sample_chart_curve(rng, phantom, off_top_prob=0.0)
    skin = np.full((*ink.TEXTURE_SHAPE, 3), 200, np.uint8)
    texture, coverage = ink.compose(skin, [ink.line_item(rng, chart, shell)], shell)
    graph = ink.ink_graph(coverage, shell)
    drawn, _ = shell.lift(chart[:, 0], chart[:, 1])
    assert graph.total_length == pytest.approx(np.linalg.norm(np.diff(drawn, axis=0), axis=1).sum(), rel=0.1)
    off_curve = np.min(np.linalg.norm(graph.points[:, None] - drawn[None], axis=2), axis=1)
    assert off_curve.max() < 0.002  # the traced stroke is the drawn one
    assert texture[coverage > 0.9].mean() < 120  # and it shows as ink


def _stroke(points) -> DrawnLine:
    points = np.asarray(points, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    return DrawnLine(chart=points[:, :2], points=points, normals=np.tile([0.0, 0.0, 1.0], (len(points), 1)),
                     arclength=s, width_m=0.0, colour="ink")


def test_strokes_continue_through_junctions_and_around_loops():
    loop = _stroke([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0, 0]])
    stem = _stroke([[0.1, 0, 0], [0.12, 0, 0]])
    arm_a = _stroke([[0.14, 0.02, 0], [0.12, 0, 0]])
    arm_b = _stroke([[0.12, 0, 0], [0.14, -0.02, 0]])
    graph = ink.InkGraph(edges=[loop, stem, arm_a, arm_b], ends=[(0, 0), (1, 2), (3, 2), (2, 4)])
    assert graph.continuations(0, at_far_end=True) == [(0, True)]
    assert sorted(graph.continuations(1, at_far_end=True)) == [(2, False), (3, True)]
    assert graph.continuations(1, at_far_end=False) == []
    assert graph.nearest(np.array([0.139, -0.019, 0.001]))[0] == 3
    edge, _, dist = graph.nearest_other(np.array([0.12, 0.0, 0.0]), {1, 2, 3})
    assert edge == 0 and dist == pytest.approx(np.hypot(0.11, 0.0), abs=1e-6)


def test_flash_designs_rasterise_to_ink():
    design = next(d for d in ink.design_manifest() if d.get("usage") == "artwork")
    size_mm = ink.tattoo_size_mm(np.random.default_rng(4), design)
    alpha = ink.design_raster(design["id"], size_mm)
    assert max(alpha.shape) == pytest.approx(size_mm / ink.RASTER_MM, abs=2)
    assert 0.01 < float((alpha > 0.5).mean()) < 0.9
    outline = ink.outline_version(alpha, pen_mm=1.0)
    assert 0 < float((outline > 0.5).sum()) < float((alpha > 0.5).sum()) * 1.5
    with pytest.raises(ValueError, match="physical resizing requires DBV3"):
        ink.design_raster(design["id"], size_mm * 1.2)


def test_rest_pose_lies_on_the_table_clear_of_the_base():
    rng = np.random.default_rng(0)
    phantom = build_phantom()
    for _ in range(20):
        pose = sample_rest_pose(rng, phantom, MotionConfig().placement)
        world = pose.apply(phantom.vertices)
        assert world[:, 2].min() == pytest.approx(0.0, abs=1e-9)
        assert pose.pos[2] == pytest.approx(rest_height(phantom, pose.rot))


def test_motion_script_is_continuous():
    rng = np.random.default_rng(1)
    script = sample_script(rng, build_phantom(), 60.0)
    times = np.arange(0.0, script.duration, 1.0 / 30.0)
    positions = np.array([script.pose(t).pos for t in times])
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.05  # < 1.5 m/s everywhere


def test_tilt_limit_keeps_axes_within_the_cone():
    axis = np.array([1.0, 0.0, -0.2])
    limited = tilt_limited(axis, np.radians(30.0))
    assert np.degrees(np.arccos(limited @ DOWN)) == pytest.approx(30.0, abs=1e-6)
    assert np.allclose(tilt_limited(DOWN, 0.5), DOWN)


def test_ik_puts_the_gap_point_on_a_reachable_target():
    from tatbot_travel.episode import Q_REST
    from tatbot_travel.kinematics import ArmKinematics

    model = SceneBuilder(plan=render_plan(Intrinsics.left_wrist())).compile()
    kin = ArmKinematics(model)
    target = np.array([0.33, -0.05, 0.08])
    result = kin.solve(Q_REST, target, DOWN, Q_REST, iters=80)
    assert result.ok()
    gap, axis = kin.gap_pose(result.q)
    assert np.linalg.norm(gap - target) < 0.002 and axis @ DOWN > 0.998
    assert abs(result.q[5] - Q_REST[5]) < 0.6  # the wrist roll stays about the tool's 90 deg offset


def test_chunk_scheduler_replans_early_by_the_latency_and_blends_in():
    from tatbot_travel.chunking import ChunkScheduler, History, Plan

    sync = ChunkScheduler(latency=0, horizon=32, back_to_back=False)
    assert sync.due(0)
    sync.requested()
    sync.adopt(Plan(0, np.zeros((42, 6))), 0)
    assert not sync.due(31) and sync.due(32)  # the recipe's 32 of 42

    late = ChunkScheduler(latency=10, horizon=32, blend=4, back_to_back=False)
    late.adopt(Plan(0, np.zeros((42, 6))), 0)
    assert not late.due(21) and late.due(22)  # asks 10 ticks early
    late.requested()
    late.adopt(Plan(22, np.ones((42, 6))), 32)
    ramp = [late.command(t, np.zeros(6))[0] for t in range(32, 38)]
    assert np.all(np.diff(ramp[:5]) > 0) and ramp[0] > 0 and ramp[4:] == [1.0, 1.0]  # blended, then the plan
    assert np.array_equal(late.command(22 + 42 + 4, np.full(6, 7.0)), np.full(6, 7.0))  # ran out: hold
    late.adopt(Plan(40, np.outer(np.arange(42) * 0.01, np.ones(6))), 60)
    assert np.allclose(late.command(40 + 42, np.zeros(6)), 0.42)  # a tick past its end: its last velocity

    # Played from arrival: the plan's whole motion, relative to the command it was planned from, starts at the
    # command in force when it arrives -- no step at the handover -- and the next plan is asked for at once.
    arrival = ChunkScheduler(latency=20, back_to_back=True)
    assert arrival.due(0)
    arrival.requested()
    anchor, now = np.zeros(6), np.full(6, 0.3)
    motion = np.outer(np.arange(1, 43) * 0.01, np.ones(6))  # the planner's commands: anchor + a steady motion
    plan = Plan(0, anchor + motion).rebased(20, anchor, now)
    arrival.adopt(plan, 20)
    assert arrival.due(20) and np.allclose(plan.at(20), now + 0.01) and np.allclose(plan.at(40), now + 0.21)

    # The recipe's loop: a rebased plan's first 32 commands, then hold and ask again.
    sync = ChunkScheduler(latency=60, horizon=32, sync=True, coast=0)
    sync.requested()
    sync.adopt(Plan(0, anchor + motion).rebased(60, anchor, now), 60)
    assert not sync.due(91) and sync.due(92)
    assert np.allclose(sync.command(91, now), now + 0.32) and np.array_equal(sync.command(92, now), now)

    history = History(3)
    history.push(np.zeros((2, 2, 3)), np.zeros(6), np.zeros(6))
    history.push(np.ones((2, 2, 3)), np.ones(6), np.ones(6))
    images, states, commands = history.window()
    assert images.shape == (3, 2, 2, 3) and states[:, 0].tolist() == [0.0, 0.0, 1.0]


def test_procedural_flash_is_varied_ink():
    from tatbot_travel import designs

    covered = []
    for seed in range(24):
        alpha = designs.procedural(np.random.default_rng(seed), 40.0, 0.1)
        assert alpha.dtype == np.float32 and 0 < alpha.shape[0] <= 400 and 0 < alpha.shape[1] <= 400
        covered.append(float((alpha > 0.5).mean()))
    assert min(covered) > 0.002 and max(covered) < 0.95
    assert len({round(c, 3) for c in covered}) > 20  # no two designs alike


def test_runner_gates_clamp_and_hold_before_they_abort():
    from tatbot_travel.runner import Clamp, Gates, Guard

    clamp = Clamp(np.full(6, -1.0), np.full(6, 1.0), max_step=0.1)
    step = clamp(np.zeros(6), np.array([5.0, -5.0, 0.05, 0.0, 0.0, 0.0]))
    assert np.allclose(step, [0.1, -0.1, 0.05, 0.0, 0.0, 0.0])  # rate-limited
    far = np.zeros(6)
    for _ in range(30):
        far = clamp(far, np.full(6, 5.0))
    assert np.allclose(far, 1.0)  # never past the limits

    guard = Guard(Gates(stop_mm=10.0, abort_mm=4.0, abort_s=0.5))
    assert guard.check(0.05, 0.0) == "ok" and guard.check(float("inf"), 0.1) == "ok"
    assert guard.check(0.008, 0.2) == "hold"
    assert guard.check(0.003, 0.3) == "hold"  # close, not yet for long
    assert guard.check(0.003, 0.85) == "abort"
    assert guard.check(0.02, 0.9) == "ok"  # clear again resets it


def test_hardware_reads_plugin_modules_and_the_camera_table_from_the_repo():
    from tatbot_travel import hardware

    paths = hardware.plugin("paths")
    assert paths.__name__ == "lerobot_robot_tatbot.paths" and callable(paths.driver_default)
    serial = hardware.left_wrist_serial()
    assert serial.isdigit() and len(serial) >= 8


def test_park_pose_stays_under_the_rig_ceiling_for_every_base_height():
    from tatbot_travel.episode import park_pose
    from tatbot_travel.expert import ExpertConfig
    from tatbot_travel.kinematics import ArmKinematics

    for base_z in (-0.06, 0.0, 0.04):
        model = SceneBuilder(plan=render_plan(Intrinsics.left_wrist()), base_pos=(0.0, 0.0, base_z)).compile()
        kin = ArmKinematics(model)
        q = park_pose(kin, base_z)
        kin.gap_pose(q)
        links = [i for i in range(model.nbody) if model.body(i).name.startswith("left/")]
        assert float(kin.data.xpos[links, 2].max()) - base_z < ExpertConfig().ceiling_m - 0.02


def test_a_line_as_long_as_a_small_forearm_still_fits():
    from tatbot_travel.phantom import build_phantom

    small = build_phantom().scaled(0.92, 1.0)
    for seed in range(200):  # lengths clamped to the forearm span, where rounding used to leave negative room
        chart = sample_chart_curve(np.random.default_rng(seed), small, length_range=(0.30, 0.40))
        lo, hi = small.forearm
        assert lo + 0.01 - 1e-9 <= chart[0, 0] and chart[-1, 0] <= hi - 0.01 + 1e-9


class _Planner:
    n_obs_steps = 8

    def __init__(self, q: np.ndarray):
        self.q, self.asked = q, []

    def request(self, tick: int, window) -> bool:
        self.asked.append((tick, window[2][-1].copy()))
        return True

    def take(self):
        if not self.asked:
            return []
        tick, command = self.asked.pop()
        self.asked.clear()
        commands = command + np.linspace(0.0, 0.3, 42)[:, None] * np.array([1.0, 0, 0, 0, 0, 0])  # swing the base
        return [(tick, commands, 0.5)]


class _Camera:
    def newest(self):
        import time

        return np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.float32), time.monotonic()


class _Arm:
    estopped, estop_required = False, False

    def __init__(self, q: np.ndarray):
        self.q, self.sent = q.copy(), []

    def measured(self) -> np.ndarray:
        return self.q.copy()

    def command(self, q: np.ndarray, goal_time: float) -> None:
        self.sent.append(q.copy())
        self.q = q.copy()

    def freeze(self) -> np.ndarray:
        return self.q.copy()


@pytest.mark.parametrize("drive", [True, False])
def test_a_held_arm_sees_the_plans_and_is_never_commanded(tmp_path, drive):
    import json
    import time

    from tatbot_travel.runner import Clamp, Gates, Loop, Recorder

    q = np.array([0.0, 1.0, 1.0, -0.5, 0.0, 0.0])
    planner, arm = _Planner(q), _Arm(q)
    recorder = Recorder(tmp_path, {})
    loop = Loop(planner, _Camera(), arm, q, 0, Clamp(np.full(6, -3.0), np.full(6, 3.0), 0.25 / 30), Gates(),
                recorder, drive=drive)
    for _ in range(20):
        assert loop.step(time.monotonic()) == "run"
    recorder.close(loop.summary("time"))
    commanded = [r["cmd"][0] for r in map(json.loads, (recorder.dir / "run.jsonl").read_text().splitlines())]
    assert commanded[-1] > 0.008  # the plan's intent is recorded either way (held: one capped step from the arm)
    if drive:
        assert len(arm.sent) == 20 and arm.q[0] > 0.05
    else:
        assert not arm.sent and np.allclose(arm.q, q)
        assert np.allclose(loop.sent, q)  # the policy's history holds the arm where it stands


def test_the_depth_stop_needs_a_surface_and_a_moment_not_one_bad_pixel():
    from tatbot_travel.runner import CloseFrames, Gates
    from tatbot_travel.shadow import axis_clearance

    intr = Intrinsics.left_wrist()
    lens, axis = np.array([0.016, 0.0003, 0.1668]), np.array([0.726, -0.209, 0.655])  # the laser pen, seen
    axis = axis / np.linalg.norm(axis)
    wall = np.full((480, 640), 0.2, np.float32)  # a surface 20 cm out in front of the camera
    crossing = (0.2 - lens[2]) / axis[2]
    assert abs(axis_clearance(wall, lens, axis, intr) - crossing) < 0.002
    speckle = np.full((480, 640), 0.5, np.float32)
    u, v = np.round(intr.project((lens + 0.003 * axis)[None])[0]).astype(int)
    speckle[v, u] = 0.05  # one bad depth pixel over the pen's nose
    assert axis_clearance(speckle, lens, axis, intr) == float("inf")

    close = CloseFrames(Gates(stop_mm=10.0), frames=3)
    assert not close.close(0.003, 1.0) and not close.close(0.003, 1.0)  # the same frame twice counts once
    assert not close.close(0.003, 1.03) and close.close(0.003, 1.07)  # three frames in a row
    assert not close.close(0.05, 1.1)  # clear again resets it


def test_plans_are_smoothed_and_untrained_joints_held(tmp_path):
    import json

    from tatbot_travel.chunking import condition_plan
    from tatbot_travel.runner import untrained_joints

    zigzag = np.zeros((42, 6))
    zigzag[:, 0] = np.linspace(0.0, 0.2, 42) + 0.004 * (-1.0) ** np.arange(42)  # a drift with per-tick noise
    zigzag[:, 5] = np.pi / 2 + np.linspace(0.0, 0.9, 42)  # a joint the policy never learned to move, drifting
    q_start = np.array([0.0, 0.05, 0.1, 0.05, 0.0, np.pi / 2])
    out = condition_plan(zigzag, q_start, [5], 7)
    steps = np.diff(out[5:-5, 0])
    assert np.all(steps > 0) and abs(out[20, 0] - zigzag[20, 0]) < 0.005  # the drift kept, the reversals gone
    assert np.all(out[:, 5] == np.pi / 2)

    q01, q99 = np.array([-0.03] * 5 + [0.0], np.float32), np.array([0.03] * 5 + [0.0], np.float32)
    tensors = {"action.q01": q01, "action.q99": q99}
    header, blob, offset = {}, b"", 0
    for name, value in tensors.items():
        header[name] = {"dtype": "F32", "shape": [6], "data_offsets": [offset, offset + value.nbytes]}
        blob, offset = blob + value.tobytes(), offset + value.nbytes
    head = json.dumps(header).encode()
    path = tmp_path / "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors"
    path.write_bytes(len(head).to_bytes(8, "little") + head + blob)
    assert untrained_joints(tmp_path) == [5] and untrained_joints(tmp_path / "missing") == []


def test_a_scene_view_rides_along_in_the_history_and_the_camera_resolves_from_the_fleet_table(tmp_path, monkeypatch):
    from tatbot_travel.chunking import History
    from tatbot_travel.hardware import poe_camera_source, rtsp_url

    history = History(3)
    history.push(np.zeros((2, 2, 3)), np.zeros(6), np.zeros(6), scene=np.ones((4, 4, 3)))
    history.push(np.zeros((2, 2, 3)), np.zeros(6), np.zeros(6), scene=np.full((4, 4, 3), 2.0))
    assert history.scene_window().shape == (3, 4, 4, 3) and history.scene_window()[-1, 0, 0, 0] == 2.0
    assert History(3).scene_window() is None

    address, password_env = poe_camera_source("camera5")
    assert address.count(".") == 3 and password_env == "TATBOT_CAMERA_PASSWORD_CAMERA5"
    monkeypatch.delenv(password_env, raising=False)
    env_file = tmp_path / "cameras.env"
    env_file.write_text(f"export {password_env}='a b/c'\n")
    url = rtsp_url("camera5", env_file=env_file)
    assert url.startswith("rtsp://admin:a%20b%2Fc@") and url.endswith("/cam/realmonitor?channel=1&subtype=1")


def test_the_scene_camera_lens_round_trips_and_its_render_covers_the_stream():
    import cv2

    intr = Intrinsics.scene_camera()
    assert (intr.width, intr.height) == (640, 480) and intr.fy / intr.fx == pytest.approx(1.339, abs=0.002)
    u, v = np.meshgrid(np.linspace(0, 639, 12), np.linspace(0, 479, 9))
    ux, uy = intr.undistort_normalized(u, v)
    camera = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    back, _ = cv2.projectPoints(np.stack([ux.ravel(), uy.ravel(), np.ones(ux.size)], -1), np.zeros(3), np.zeros(3),
                                camera, np.asarray(intr.distortion))
    assert np.abs(back[:, 0, :] - np.stack([u.ravel(), v.ravel()], -1)).max() < 1e-6
    plan = render_plan(intr, focal=400.0)
    assert plan.map_x.shape == (480, 640)
    assert plan.map_x.min() >= 0 and plan.map_x.max() <= plan.width - 1
    assert plan.map_y.min() >= 0 and plan.map_y.max() <= plan.height - 1


def test_the_scene_camera_sits_where_the_calibration_puts_it():
    import cv2
    import mujoco
    from tatbot_travel.camera import SCENE_CAMERA_POSE
    from tatbot_travel.episode import Q_PARK
    from tatbot_travel.kinematics import ArmKinematics
    from tatbot_travel.scene import SCENE_CAMERA

    intr, base = Intrinsics.scene_camera(), np.array([0.0, 0.0, 0.03])
    model = SceneBuilder(plan=render_plan(Intrinsics.left_wrist()), base_pos=tuple(base),
                         scene_plan=render_plan(intr, focal=400.0), scene_pose=SCENE_CAMERA_POSE).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera(SCENE_CAMERA).id
    # MuJoCo's camera looks down its -z with y up: the optical frame flipped about x.
    assert np.allclose(data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), SCENE_CAMERA_POSE[:3, :3],
                       atol=1e-5)
    assert np.allclose(data.cam_xpos[cam], base + SCENE_CAMERA_POSE[:3, 3], atol=1e-6)
    gap, _ = ArmKinematics(model).gap_pose(Q_PARK)
    local = SCENE_CAMERA_POSE[:3, :3].T @ (gap - base - SCENE_CAMERA_POSE[:3, 3])
    camera = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    (u, v), = cv2.projectPoints(local[None], np.zeros(3), np.zeros(3), camera, np.asarray(intr.distortion))[0][:, 0]
    assert 100 < u < 350 and 60 < v < 300  # the parked pen, upper left of the rig's frames


def test_the_handless_phantom_ends_at_the_wrist_in_a_rounded_stump():
    whole, stump = build_phantom(), build_phantom(handless=True)
    assert stump.forearm == whole.forearm  # the inked forearm is the same
    assert whole.x_range[1] - whole.forearm[1] > 0.1  # a hand
    assert 0.03 < stump.x_range[1] - stump.forearm[1] < 0.09  # a wrist and a dome


def test_scene_frames_arrive_a_latency_late_and_are_held():
    from tatbot_travel.episode import DelayedFrames

    frames, shown = DelayedFrames(latency=0.9), []
    for tick in range(60):
        if tick % 2 == 0:  # a 15 fps stream into a 30 Hz loop
            frames.push(tick / 30, np.full((2, 2, 3), tick, np.uint8))
        shown.append(int(frames.view(tick / 30)[0, 0, 0]))
    assert shown[:27] == [0] * 27  # the first capture stands in until anything has arrived
    assert shown[30] == 2 and shown[58] == 30  # then what was captured 0.9 s ago
    assert all(b >= a for a, b in zip(shown, shown[1:], strict=False))


def test_rig_clutter_stays_off_the_workspace_and_out_of_the_scene_cameras_sight_line():
    import re

    from tatbot_travel.camera import SCENE_CAMERA_POSE
    from tatbot_travel.world import WorldConfig, _add_rig, _clear_of_workspace, _in_front_of_workspace

    rng = np.random.default_rng(7)
    for _ in range(6):
        sb = SceneBuilder(plan=render_plan(Intrinsics.left_wrist()))
        _add_rig(sb, rng, WorldConfig(), SCENE_CAMERA_POSE)
        for xml in sb.world_xml:
            name = re.search(r'name="([a-z]+)\d+"', xml)
            if name is None or name.group(1) == "sheet":
                continue
            pos = np.array([float(x) for x in re.search(r'pos="([^"]+)"', xml).group(1).split()])
            assert _clear_of_workspace(pos[:2]), xml
            if name.group(1) == "structure":
                assert not _in_front_of_workspace(np.append(pos[:2], 0.2), SCENE_CAMERA_POSE), xml


def test_the_dataset_carries_the_scene_stream_only_when_asked():
    from tatbot_travel.writer import CAMERA_KEY, SCENE_KEY, features

    assert SCENE_KEY not in features()
    two = features(scene_hw=(480, 640))
    assert two[SCENE_KEY]["shape"] == two[CAMERA_KEY]["shape"] == (480, 640, 3)  # side by side needs one size


def _expert_on_a_line(seed: int = 5):
    from tatbot_travel.clearance import SkinDistance
    from tatbot_travel.episode import Q_PARK, Q_REST
    from tatbot_travel.expert import Expert, ExpertConfig
    from tatbot_travel.kinematics import ArmKinematics
    from tatbot_travel.scene import LENS_SITE

    rng = np.random.default_rng(seed)
    phantom, shell = build_phantom(), canonical_shell()
    skin = np.full((*ink.TEXTURE_SHAPE, 3), 200, np.uint8)
    _, coverage = ink.compose(skin, [ink.line_item(rng, sample_chart_curve(rng, phantom, off_top_prob=0.0), shell)],
                              shell)
    intr = Intrinsics.left_wrist()
    model = SceneBuilder(plan=render_plan(intr)).compile()
    expert = Expert(ExpertConfig(), ArmKinematics(model), ink.ink_graph(coverage, shell), intr,
                    np.zeros((intr.height, intr.width), bool), rng, Q_PARK.copy(), Q_REST,
                    SkinDistance(phantom, seed=0), model.site(LENS_SITE).id)
    return expert, phantom


def test_a_parked_expert_turns_to_face_ink_off_to_the_side():
    from scipy.spatial.transform import Rotation
    from tatbot_travel.episode import Q_PARK
    from tatbot_travel.expert import Mode, Observation
    from tatbot_travel.motion import Pose

    expert, phantom = _expert_on_a_line()
    bearing = np.radians(70.0)  # well out of the parked view, on the arm's left
    rot = Rotation.from_euler("z", bearing + np.pi / 2)
    pose = Pose(pos=np.array([0.38 * np.cos(bearing), 0.38 * np.sin(bearing), rest_height(phantom, rot)]), rot=rot)
    expert.reset(Q_PARK.copy(), Mode.PARK)
    q = Q_PARK.copy()
    for tick in range(150):
        q = expert.act(Observation(t=tick / 30, q=q, phantom=pose, linear_speed=0.0, angular_speed=0.0,
                                   away=False, hands=[]))
        if expert.state.mode != Mode.PARK:
            break
    assert expert.state.look_j0 == pytest.approx(expert.ink_bearing(pose))
    assert expert.state.look_j0 > 0.8 and q[0] > 0.5  # turned toward the ink


def test_the_way_back_to_park_routes_around_the_ceiling():
    from scipy.spatial.transform import Rotation
    from tatbot_travel.expert import Mode, Observation
    from tatbot_travel.motion import Pose

    expert, _ = _expert_on_a_line()
    raised = np.array([0.3, 0.9, 1.3, -1.0, 0.0, np.pi / 2])  # elbow up after tracing: straight back rises
    expert.reset(raised, Mode.PARK)
    expert.kin.gap_pose(raised)
    expert.ceiling_z = float(expert.kin.data.xpos[expert.links, 2].max()) + 1e-4  # the rig's ceiling, just above
    straight = raised + np.clip(expert.q_park - raised, -expert.cfg.park_speed / 30, expert.cfg.park_speed / 30)
    assert not expert.under_ceiling(straight)
    away = Pose(pos=np.array([2.0, 2.0, 0.0]), rot=Rotation.identity())
    q = expert._park_command(Observation(t=0.0, q=raised, phantom=away, linear_speed=0.0, angular_speed=0.0,
                                         away=True, hands=[]))
    assert not np.allclose(q, raised) and expert.under_ceiling(q)  # v6/v7's expert held still here


def test_a_real_bank_supplies_textures_and_ink_and_its_absence_is_harmless(tmp_path, monkeypatch):
    import cv2
    from tatbot_travel import realbank

    monkeypatch.delenv(realbank.ENV, raising=False)
    realbank.load.cache_clear()
    assert realbank.load() is None
    for kind in ("wrist_bg", "table"):
        (tmp_path / kind).mkdir()
        cv2.imwrite(str(tmp_path / kind / "a.jpg"), np.full((40, 60, 3), 200, np.uint8))
    (tmp_path / "ink").mkdir()
    rgba = np.zeros((30, 50, 4), np.uint8)
    rgba[10:20, 5:45, 3] = 255  # a bar of ink
    cv2.imwrite(str(tmp_path / "ink" / "i.png"), rgba)
    bank = realbank.load(str(tmp_path))
    rng = np.random.default_rng(0)
    assert bank.has("table") and not bank.has("scene_bg")
    assert bank.texture(rng, "wrist_bg", 64).shape == (64, 64, 3)
    alpha = bank.ink(rng)
    assert alpha.dtype == np.float32 and alpha.max() == 1.0 and alpha.sum() == 400
    item = ink.photo_item(rng, canonical_shell(), alpha)
    assert item.kind == "photo" and 20 <= max(item.alpha.shape) * ink.RASTER_MM <= 70
    skin = np.full((*ink.TEXTURE_SHAPE, 3), 230, np.uint8)
    texture, coverage = ink.compose(skin, [item], canonical_shell())
    assert coverage.max() > 0.9 and texture[coverage > 0.9].mean() < 200  # painted, in ballpoint colours
    realbank.load.cache_clear()
