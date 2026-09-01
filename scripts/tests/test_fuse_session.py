"""Pin the fuser: triangulation geometry, phantom rejection, and the full
chain from a synthetic sweep to a recovered robot-world transform.

    uvx --with pytest --with numpy --with scipy pytest -q scripts/tests/test_fuse_session.py

The end-to-end test is the reason to trust the whole design: it builds a
session the way the real sweep writes one (bundle json, shot dirs, .wxtl,
events.jsonl), runs fuse_session.py as a subprocess, then feeds the resulting
pose_*.json through solve_robot_world's solver and requires the true
world_from_base back to millimetres. If the corner order, the frame
conventions, the timing join, or the effort classification drift anywhere in
that chain, this fails.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "vision"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_synth as synth  # noqa: E402
import fuse_session  # noqa: E402
from solve_robot_world import solve, vector_to_rotation  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

EDGE_M = 0.056
# A real canonical wrist ID so the fixture exercises the same config as the fuser.
WRIST_TAG = 6
LINK = "right/gripper_left"


def three_cameras():
    return {
        "camera1": synth.make_camera([0.6, 0.1, 0.9], [0.1, 0.0, 0.1]),
        "camera2": synth.make_camera([-0.4, 0.5, 0.8], [0.1, 0.0, 0.1]),
        "camera3": synth.make_camera([0.1, -0.6, 1.0], [0.1, 0.0, 0.1],
                                     dist=(0.0, 0.0, 0.0, 0.0, 0.0)),
    }


def test_undistort_inverts_projection():
    camera_dict = three_cameras()["camera1"]
    camera = fuse_session.Camera("camera1", synth.bundle_json(
        three_cameras())["cameras"]["camera1"])
    point = np.array([0.2, -0.1, 0.15])
    pixel = synth.project(camera_dict, point)
    cam = camera_dict["rotation"].T @ (point - camera_dict["position"])
    assert np.allclose(camera.undistort(pixel), cam[:2] / cam[2], atol=1e-7)


def test_triangulate_and_fit_square():
    cameras = {name: fuse_session.Camera(name, cal) for name, cal in
               synth.bundle_json(three_cameras())["cameras"].items()}
    pose_true = np.eye(4)
    pose_true[:3, :3] = vector_to_rotation(np.array([0.3, -0.2, 0.9]))
    pose_true[:3, 3] = [0.15, 0.05, 0.2]
    corners_world = synth.tag_corners_world(pose_true, EDGE_M)
    observations = [
        (name, [synth.project(cam, c).tolist() for c in corners_world])
        for name, cam in three_cameras().items()]
    pose, meta = fuse_session.fuse_tag(observations, cameras, EDGE_M, 3.0, 2)
    assert pose is not None, meta
    assert np.linalg.norm(pose[:3, 3] - pose_true[:3, 3]) < 1e-4
    assert meta["rms_mm"] < 0.1
    assert abs(meta["edge_est_m"] - EDGE_M) < 1e-4


def test_follower_left_jaw_uses_seventh_driver_joint():
    chain = UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    names = chain.driver_joint_names("right")
    closed = dict(zip(names, [0.0] * 7, strict=True))
    open_jaw = dict(closed)
    open_jaw["right/left_carriage_joint"] = 0.031

    closed_pose = chain.link_pose("right/gripper_left", closed)
    open_pose = chain.link_pose("right/gripper_left", open_jaw)
    assert np.allclose(open_pose[:3, 3] - closed_pose[:3, 3], [0.0, 0.031, 0.0])
    assert np.allclose(
        chain.link_pose("right/ee_gripper_link", closed),
        chain.link_pose("right/ee_gripper_link", open_jaw),
    )


def test_single_camera_sighting_is_rejected():
    """The 16h5 phantom filter: one camera is never enough to fuse."""
    cameras = {name: fuse_session.Camera(name, cal) for name, cal in
               synth.bundle_json(three_cameras())["cameras"].items()}
    pose_true = np.eye(4)
    pose_true[:3, 3] = [0.1, 0.0, 0.2]
    corners = synth.tag_corners_world(pose_true, EDGE_M)
    observations = [("camera1",
                     [synth.project(three_cameras()["camera1"], c).tolist()
                      for c in corners])]
    pose, meta = fuse_session.fuse_tag(observations, cameras, EDGE_M, 3.0, 2)
    assert pose is None
    assert "fewer than 2 cameras" in meta["reason"]


def build_session(tmp_path, poses_count=10):
    """A synthetic sweep: wrist poses via the real URDF, one phantom shot, one
    discarded interval, plate and pad touches."""
    rng = np.random.default_rng(23)
    chain = UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    names = chain.driver_joint_names("right")

    z_true = np.eye(4)
    z_true[:3, :3] = vector_to_rotation(np.array([0.02, -0.01, 1.57]))
    z_true[:3, 3] = [0.126, 0.0, 0.0885]
    x_true = np.eye(4)
    x_true[:3, :3] = vector_to_rotation(np.array([0.1, 0.2, -0.3]))
    x_true[:3, 3] = [0.03, -0.01, 0.05]

    cameras = three_cameras()
    session = tmp_path / "session"
    session.mkdir()
    (session / "calibration.json").write_text(json.dumps(synth.bundle_json(cameras)))

    free = np.full(synth.NUM_JOINTS, 0.05)
    contact = free.copy()
    contact[2] = 2.5

    joint_vectors, efforts, kinds = [], [], []
    for _ in range(poses_count):
        q = np.zeros(synth.NUM_JOINTS)
        q[:6] = rng.uniform(-0.9, 0.9, 6)
        q[6] = rng.uniform(0.0, 0.035)
        joint_vectors.append(q)
        efforts.append(free)
        kinds.append("wrist")
    for _ in range(3):
        q = np.zeros(synth.NUM_JOINTS)
        q[:6] = rng.uniform(-0.6, 0.6, 6)
        joint_vectors.append(q)
        efforts.append(contact)
        kinds.append("touch")
    q = np.zeros(synth.NUM_JOINTS)
    q[:6] = rng.uniform(-0.6, 0.6, 6)
    joint_vectors.append(q)
    efforts.append(free)
    kinds.append("discard-me")

    centers = synth.write_wxtl(session / "teleop.wxtl", joint_vectors, efforts)
    # Current hardware reuses every wrist ID on the calibration board. The
    # production fuser therefore requires guide phase windows rather than
    # inferring physical ownership from ID exclusivity.
    (session / "guide_timeline.json").write_text(json.dumps({"entries": [
        {"phase": "wrist", "kind": "hold", "index": index + 1,
         "start_unix": center - 0.45, "end_unix": center + 0.45,
         "result": "paced"}
        for index, center in enumerate(centers[:poses_count])
    ]}))

    shot = 0
    for center, q, kind in zip(centers, joint_vectors, kinds, strict=True):
        if kind != "wrist":
            continue
        values = dict(zip(names, q, strict=True))
        world_from_tag = z_true @ chain.link_pose(LINK, values) @ x_true
        corners = synth.tag_corners_world(world_from_tag, EDGE_M)
        detections = {}
        for name, camera in cameras.items():
            pixels = [synth.project(camera, c) for c in corners]
            noisy = [(p + rng.normal(0, 0.3, 2)).tolist() for p in pixels]
            detections[name] = {"ids": [WRIST_TAG],
                                "corners": {str(WRIST_TAG): noisy}}
        synth.write_shot(session, shot, center, detections)
        shot += 1

    # A phantom id seen by a single camera during the first wrist pause: the
    # fuser must never let it reach a pose file.
    phantom = {"camera1": {"ids": [13], "corners": {"13": [[10, 10], [20, 10],
                                                          [20, 20], [10, 20]]}}}
    synth.write_shot(session, 99, centers[0], phantom)

    events = [
        {"start_unix": centers[poses_count + 1] - 0.2,
         "end_unix": centers[poses_count + 1] + 0.2,
         "kinds": ["touch"], "text": "touching the pad"},
        {"start_unix": centers[-1] + 0.5, "end_unix": centers[-1] + 1.0,
         "kinds": ["discard"], "text": "scratch that"},
    ]
    with open(session / "events.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return session, z_true, poses_count


def test_full_chain_recovers_robot_world(tmp_path):
    session, z_true, poses_count = build_session(tmp_path)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    pose_files = sorted(session.glob("pose_*.json"))
    assert len(pose_files) == poses_count, result.stdout
    first_pose = json.loads(pose_files[0].read_text())
    assert first_pose["joint_names"][-1] == "right/left_carriage_joint"
    assert len(first_pose["joints"]) == 7

    touches = json.loads((session / "touches.json").read_text())["touches"]
    assert [t["label"] for t in touches] == ["plate", "pad", "plate"]

    audit = json.loads((session / "intervals.json").read_text())["intervals"]
    assert sum(1 for i in audit if i["kind"] == "discarded") == 1
    assert not any("13" in str(i.get("kind")) for i in audit), "phantom fused"

    # Feed the fuser's output straight into the AX=ZB solver, as the driver
    # does, and require the truth back.
    chain = UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    base_from_link, world_from_tag, tag_ids = [], [], []
    for pose_file in pose_files:
        data = json.loads(pose_file.read_text())
        values = dict(zip(data["joint_names"], data["joints"], strict=True))
        for tag_id, matrix in data["world_from_tag"].items():
            base_from_link.append(chain.link_pose(LINK, values))
            world_from_tag.append(np.asarray(matrix))
            tag_ids.append(int(tag_id))
    z, _, _ = solve(base_from_link, world_from_tag, tag_ids)
    assert np.linalg.norm(z[:3, 3] - z_true[:3, 3]) * 1000 < 5.0, (
        f"world_from_base off by "
        f"{np.linalg.norm(z[:3, 3] - z_true[:3, 3]) * 1000:.2f} mm")


def test_corner_solver_downweights_one_buffered_frame(tmp_path):
    """One coherently wrong camera set must not drag the hand-eye solution."""
    session, z_true, _ = build_session(tmp_path)
    fused = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)], capture_output=True, text=True,
    )
    assert fused.returncode == 0, fused.stdout + fused.stderr
    first = session / "pose_0001.json"
    pose = json.loads(first.read_text())
    for observations in pose["corner_obs"].values():
        for observation in observations:
            observation["normalized"] = (
                np.asarray(observation["normalized"]) + [0.15, -0.12]
            ).tolist()
    first.write_text(json.dumps(pose))
    out = session / "robust.json"

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "solve_robot_world.py"),
         str(session), "--urdf", str(REPO / "urdf" / "tatbot.urdf"),
         "--calibration", str(session / "calibration.json"), "--out", str(out)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    solved = json.loads(out.read_text())
    z = np.asarray(solved["world_from_base"])
    assert solved["loss"] == "huber"
    assert solved["corner_px_median"] < 5.0
    assert np.linalg.norm(z[:3, 3] - z_true[:3, 3]) * 1000 < 8.0


def test_shared_id_session_without_guide_timeline_is_refused(tmp_path):
    session, _, _ = build_session(tmp_path)
    (session / "guide_timeline.json").unlink()
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unguided mixed-session fuse" in result.stderr


def test_wrist_phase_window_drops_buffered_arrival_frame():
    timeline = {"entries": [{
        "phase": "wrist", "start_unix": 10.0, "end_unix": 13.0,
    }]}
    windows = fuse_session.phase_windows(
        timeline, "wrist", lead_s=0.0, tail_s=0.0, settle_s=0.4
    )
    assert not fuse_session.in_any_window(10.1, windows)
    assert fuse_session.in_any_window(10.5, windows)
    assert not fuse_session.in_any_window(13.1, windows)


def test_same_id_candidates_are_rejected_per_camera(tmp_path):
    """Two physical ID-6 squares in one camera cannot be assigned to wrist."""
    session, _, poses_count = build_session(tmp_path)
    first = session / "shot_0000_sweep" / "detections.json"
    detections = json.loads(first.read_text())
    for info in detections.values():
        corners = info["corners"][str(WRIST_TAG)]
        shifted = (np.asarray(corners) + [250.0, 100.0]).tolist()
        info["candidates"] = {str(WRIST_TAG): [corners, shifted]}
        info["candidate_ids"] = [WRIST_TAG]
    first.write_text(json.dumps(detections))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(session.glob("pose_*.json"))) == poses_count - 1


def test_fuser_rerun_removes_stale_derived_outputs(tmp_path):
    session, _, poses_count = build_session(tmp_path, poses_count=4)
    command = [
        sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
        str(session),
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert len(list(session.glob("pose_*.json"))) == poses_count
    (session / "pose_9999.json").write_text("stale")
    (session / "robot_world.json").write_text("stale")
    (session / "report.json").write_text("stale")

    second = subprocess.run(command, capture_output=True, text=True)

    assert second.returncode == 0, second.stdout + second.stderr
    assert len(list(session.glob("pose_*.json"))) == poses_count
    assert not (session / "pose_9999.json").exists()
    assert not (session / "robot_world.json").exists()
    assert not (session / "report.json").exists()


def test_guide_timeline_overrides_speech_and_yields_poses(tmp_path):
    """When a guided session ran, labels come from the SCRIPT: the timeline
    says touch 3 was plate even though the transcript claims pad, and the pose
    tour's windows become named joint samples."""
    session, z_true, poses_count = build_session(tmp_path)
    from teleop_log import TeleopLog
    log = TeleopLog(session / "teleop.wxtl")
    stills = log.still_intervals()
    log.classify_contacts(stills)
    contacts = [s for s in stills if s["contact"]]
    free = [s for s in stills if not s["contact"]]
    entries = []
    for label, interval in zip(["plate", "pad", "plate"], contacts, strict=True):
        entries.append({"phase": "touch", "kind": "touch", "label": label,
                        "start_unix": interval["start_unix"] + 0.1,
                        "end_unix": interval["end_unix"] - 0.1, "result": "paced"})
    # a lying transcript event on touch 3: the timeline must win. This test
    # rewrites events.jsonl (the shared fixture's discard would legitimately
    # point at touch 3 under guided semantics).
    with open(session / "events.jsonl", "w") as f:
        f.write(json.dumps({"start_unix": contacts[2]["start_unix"] + 0.1,
                            "end_unix": contacts[2]["start_unix"] + 0.4,
                            "kinds": ["touch"], "text": "on the pad"}) + "\n")
    entries.append({"phase": "poses", "kind": "pose", "name": "palette_center",
                    "label": "centered on the palette",
                    "start_unix": free[0]["start_unix"] + 0.2,
                    "end_unix": free[0]["start_unix"] + 0.8, "result": "paced"})
    entries.append({"phase": "poses", "kind": "pose", "name": "ghost_pose",
                    "label": "nowhere", "start_unix": 1.0, "end_unix": 2.0,
                    "result": "paced"})
    wrist_entries = json.loads((session / "guide_timeline.json").read_text())["entries"]
    (session / "guide_timeline.json").write_text(json.dumps(
        {"profile": "debug", "entries": [*wrist_entries, *entries]}))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    touches = json.loads((session / "touches.json").read_text())["touches"]
    assert [t["label"] for t in touches] == ["plate", "pad", "plate"]

    poses = json.loads((session / "poses.json").read_text())
    assert "palette_center" in poses["poses"]
    assert len(poses["poses"]["palette_center"]["joints"]) == 6
    assert poses["missing"] == ["ghost_pose"]


def test_driver_publishes_named_poses(tmp_path):
    """poses.json -> config-style poses.yaml with FK'd EE positions, merged
    over what an earlier session measured."""
    session, _, _ = build_session(tmp_path)
    from teleop_log import TeleopLog
    log = TeleopLog(session / "teleop.wxtl")
    stills = log.still_intervals()
    log.classify_contacts(stills)
    free = [s for s in stills if not s["contact"]]
    (session / "guide_timeline.json").write_text(json.dumps({"entries": [
        {"phase": "poses", "kind": "pose", "name": "paper_pad_over",
         "label": "over the paper pad",
         "start_unix": free[1]["start_unix"] + 0.2,
         "end_unix": free[1]["start_unix"] + 0.8, "result": "paced"}]}))
    poses_file = tmp_path / "poses.yaml"
    poses_file.write_text(
        'tour:\n  paper_pad_over: "over the paper pad"\n'
        '  old_pose: "still wanted"\n\nposes:\n'
        '  old_pose:\n    label: "kept from before"\n'
        '    joints: [0, 0, 0, 0, 0, 0]\n'
        '  retired_pose:\n    label: "dropped from the tour"\n'
        '    joints: [0, 0, 0, 0, 0, 0]\n')

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "calibrate_session.py"),
         str(session), "--tool-id", "lutin-ballpoint-dot",
         "--no-publish", "--poses-file", str(poses_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((session / "report.json").read_text())
    assert report["poses"]["status"] == "ok"
    text = poses_file.read_text()
    assert "paper_pad_over" in text and "ee_xyz_m" in text
    assert "old_pose" in text, "tour members measured earlier must survive"
    assert "retired_pose" not in text, "a slug removed from the tour is retired"


def test_static_environment_tag_never_reaches_the_solver(tmp_path):
    """Board 3-11, wrist 3/6/7/8 and palette id 8 share a scene inventory.
    A static non-board tag seen by >=2 cameras — a stray print, or anything
    misread in the environment — must not be fused as a wrist tag: one X can
    never fit a tag that does not ride the wrist, and it would poison AX=ZB."""
    session, z_true, poses_count = build_session(tmp_path)
    static_pose = np.eye(4)
    static_pose[:3, 3] = [0.25, 0.15, 0.02]   # taped to the table, id 3
    corners = synth.tag_corners_world(static_pose, EDGE_M)
    cameras = three_cameras()
    for shot_dir in sorted(session.glob("shot_0*")):
        detections = json.loads((shot_dir / "detections.json").read_text())
        for name, camera in cameras.items():
            pixels = [synth.project(camera, c).tolist() for c in corners]
            detections.setdefault(name, {"ids": [], "corners": {}})
            detections[name]["ids"].append(3)
            detections[name]["corners"]["3"] = pixels
        (shot_dir / "detections.json").write_text(json.dumps(detections))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "never moved" in result.stdout

    for pose_file in session.glob("pose_*.json"):
        tags = json.loads(pose_file.read_text())["world_from_tag"]
        assert "3" not in tags, "static tag fused as a wrist tag"
        assert str(WRIST_TAG) in tags, "the real wrist tag must survive the guard"


def test_board_window_filter_keeps_only_board_phase_shots(tmp_path):
    """The palette shares id 8 with the board; outside the board phase a lone
    palette sighting reads as board id 8, so the board solve gets a window."""
    shots = []
    for index, unix in enumerate([100.0, 110.0, 500.0]):
        shot = synth.write_shot(tmp_path, index, unix, {"camera1": {"ids": []}})
        shots.append(shot)
    legacy = tmp_path / "shot_0099_manual"   # pose-at-a-time path: no timing
    legacy.mkdir()
    (legacy / "detections.json").write_text("{}")
    shots.append(legacy)

    kept = fuse_session.filter_shot_dirs(shots, 90.0, 120.0)
    names = [Path(p).name for p in kept]
    assert "shot_0000_sweep" in names and "shot_0001_sweep" in names
    assert "shot_0002_sweep" not in names, "arm-phase shot must be excluded"
    assert "shot_0099_manual" in names, "timing-less legacy shots stay"


def test_single_camera_corner_observations_recover_robot_world(tmp_path):
    """The rig reality (2026-08-21): a wrist tag almost never reaches two
    cameras at once. Rebuild the session so each pose is seen by exactly ONE
    camera (rotating), fuse, and solve in corner-reprojection mode — the
    truth must still come back, with no triangulated poses at all."""
    session, z_true, poses_count = build_session(tmp_path)
    shot_dirs = sorted(d for d in session.glob("shot_0*") if d.name != "shot_0099_sweep")
    camera_names = sorted(three_cameras())
    for index, shot_dir in enumerate(shot_dirs):
        detections = json.loads((shot_dir / "detections.json").read_text())
        keep = camera_names[index % len(camera_names)]
        (shot_dir / "detections.json").write_text(json.dumps(
            {keep: detections[keep]} if keep in detections else {}))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    pose_files = sorted(session.glob("pose_*.json"))
    assert len(pose_files) == poses_count, result.stdout
    for pose_file in pose_files:
        data = json.loads(pose_file.read_text())
        assert data["world_from_tag"] == {}, "nothing should triangulate"
        assert data["corner_obs"], "single sightings must be recorded"

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "solve_robot_world.py"),
         str(session), "--urdf", str(REPO / "urdf" / "tatbot.urdf"),
         "--calibration", str(session / "calibration.json"),
         "--out", str(session / "robot_world.json")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    solved = json.loads((session / "robot_world.json").read_text())
    assert solved["mode"] == "corner_reprojection"
    assert solved["observations"] == poses_count, "one sighting per pose"
    z = np.asarray(solved["world_from_base"])
    assert np.linalg.norm(z[:3, 3] - z_true[:3, 3]) * 1000 < 5.0, (
        f"single-camera solve off by "
        f"{np.linalg.norm(z[:3, 3] - z_true[:3, 3]) * 1000:.2f} mm")
    assert solved["corner_px_median"] < 3.0


def test_pivot_windows_flow_to_pivot_mode(tmp_path):
    """A pivot window over continuous motion becomes joint samples in
    touches.json, and il_touchoff switches to pivot mode — here refusing,
    since these arbitrary ramp joints violate the planted-tip constraint
    (rms gate), which is exactly what a bad pivot should produce."""
    session = tmp_path / "session"
    session.mkdir()
    q0 = np.zeros(synth.NUM_JOINTS)
    q0[:6] = [0.1, -0.4, 0.5, 0.2, 0.3, -0.2]
    q1 = q0 + 0.8
    centers = synth.write_wxtl(session / "teleop.wxtl", [q0, q1],
                               [np.full(synth.NUM_JOINTS, 0.05)] * 2,
                               still_s=0.5, move_s=6.0)
    (session / "guide_timeline.json").write_text(json.dumps({"entries": [
        {"phase": "pivot", "kind": "pivot", "index": 1,
         "start_unix": centers[0] + 0.3, "end_unix": centers[1] - 0.3,
         "result": "paced"}]}))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((session / "touches.json").read_text())
    assert len(data["pivots"]) == 1
    assert data["pivots"][0]["n"] >= 100, "20 Hz over ~6 s of rolling"
    assert data["pivots"][0]["travel_rad"] > 0.5

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"),
         str(session), "--tool-id", "lutin-ballpoint-dot",
         "--write", "--workspace", str(tmp_path / "ws.yaml")],
        capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads((session / "touchoff_report.json").read_text())
    assert report["mode"] == "pivot"
    assert report["fit"]["samples"] >= 100
    assert not (tmp_path / "ws.yaml").exists(), "refusal must write nothing"


def test_archived_palette_tip_holds_still_fuse(tmp_path):
    """ARCHIVED sessions (pre-2026-08-26): pad hovers become waypoints and
    only the palette stills are pen-tip observations. The 2026-08-26 change
    must not retroactively count those hovers as touches — that would put a
    waypoint's worth of air into the solve and report a paper_plane_z nothing
    ever touched. il_touchoff refuses here (arbitrary joints violate the
    planted constraint), which is correct for a bad capture."""
    session = tmp_path / "session"
    session.mkdir()
    rng = np.random.default_rng(9)
    joints, efforts = [], []
    for _ in range(6):
        q = np.zeros(synth.NUM_JOINTS)
        q[:6] = rng.uniform(-0.8, 0.8, 6)
        joints.append(q)
        efforts.append(np.full(synth.NUM_JOINTS, 0.05))
    centers = synth.write_wxtl(session / "teleop.wxtl", joints, efforts)
    entries = []
    for index, center in enumerate(centers):
        entries.append({"phase": "tip", "kind": "tip_hold", "index": index + 1,
                        "label": "pad" if index < 3 else "palette",
                        "start_unix": center - 0.4, "end_unix": center + 0.4,
                        "result": "paced"})
    (session / "guide_timeline.json").write_text(json.dumps({"entries": entries}))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((session / "touches.json").read_text())
    assert len(data["tip_holds"]) == 3, "three palette stills"
    poses = json.loads((session / "poses.json").read_text())
    assert set(poses["poses"]) >= {"paper_pad_over", "palette_center"}
    assert len(poses["poses"]["paper_pad_over"]["joints"]) == 6

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"),
         str(session), "--tool-id", "lutin-ballpoint-dot",
         "--write", "--workspace", str(tmp_path / "ws.yaml")],
        capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads((session / "touchoff_report.json").read_text())
    assert report["mode"] == "tip_point"
    assert report["fit"]["samples"] == 3
    assert not (tmp_path / "ws.yaml").exists(), "refusal must write nothing"
    assert report["surface"] == "palette", "archived sessions planted there"


def test_planted_pad_holds_are_all_observations(tmp_path):
    """Current tip phase: every hold is planted on ONE spot of the paper pad,
    so all of them are pen-tip observations and none is a waypoint. The
    surface reaches the touch-off, because that is what decides whether
    paper_plane_z may be called the paper the pen draws on (n_pad)."""
    session = tmp_path / "session"
    session.mkdir()
    rng = np.random.default_rng(11)
    joints, efforts = [], []
    for _ in range(9):
        q = np.zeros(synth.NUM_JOINTS)
        q[:6] = rng.uniform(-0.8, 0.8, 6)
        joints.append(q)
        efforts.append(np.full(synth.NUM_JOINTS, 0.05))
    centers = synth.write_wxtl(session / "teleop.wxtl", joints, efforts)
    entries = [{"phase": "tip", "kind": "tip_hold", "index": i + 1,
                "label": "pad_planted", "start_unix": c - 0.4,
                "end_unix": c + 0.4, "result": "paced"}
               for i, c in enumerate(centers)]
    (session / "guide_timeline.json").write_text(json.dumps({"entries": entries}))

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vision" / "fuse_session.py"),
         str(session)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((session / "touches.json").read_text())
    assert len(data["tip_holds"]) == 9, "every planted hold is an observation"
    assert data["tip_surface"] == "pad"
    assert all(h["surface"] == "pad" for h in data["tip_holds"])
    # A pose named "above the paper pad" must never be filled with one that is
    # touching it, so this phase publishes no waypoints at all.
    assert not (session / "poses.json").is_file() or not json.loads(
        (session / "poses.json").read_text())["poses"]

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"),
         str(session), "--tool-id", "lutin-ballpoint-dot",
         "--write", "--workspace", str(tmp_path / "ws2.yaml")],
        capture_output=True, text=True)
    report = json.loads((session / "touchoff_report.json").read_text())
    assert report["mode"] == "tip_point"
    assert report["fit"]["samples"] == 9
    assert report["surface"] == "pad", "must reach the touch-off"
