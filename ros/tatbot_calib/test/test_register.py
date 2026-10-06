"""The D555 registration's holds, solve and bundle, without the stack or a camera: holds point the tool down over
the table; tag sightings synthesised from forward kinematics and a known camera give that camera back, with a
bad sighting dropped; a hold the arm did not reach is named, and holds repeating one pose count as one view; a
bundle is reused only while it still describes the camera."""
from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tatbot_calib import register  # noqa: E402


def test_holds_point_the_tool_down_over_the_table_in_a_grid():
    poses = register.hold_poses((0.30, 0.0), 0.003, heading_deg=-90.0, spread_m=0.07, heights_m=(0.12, 0.17))
    assert len(poses) == 27
    # the first attitude is the arm's own heading: the tool's x axis along base -y for a heading of -90 deg
    assert np.allclose(poses[0][:3, 0], [0.0, -1.0, 0.0], atol=1e-12)
    for pose in poses:
        assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        assert pose[2, 2] < -np.cos(np.radians(25.0))            # the tool's axis within 25 deg of straight down
        assert round(pose[2, 3] - 0.003, 6) in (0.12, 0.17)
        assert abs(pose[0, 3] - 0.30) <= 0.07 + 1e-9 and abs(pose[1, 3]) <= 0.07 + 1e-9
    # consecutive grid points are neighbours: the serpentine never jumps across the grid
    points = [tuple(np.round(p[:2, 3], 3)) for p in poses[::3]]
    assert all(np.linalg.norm(np.subtract(a, b)) <= 0.07 + 1e-6 for a, b in zip(points, points[1:], strict=False))


def _synthetic(kin, arm, camera_from_base, k, dist, rng, holds=16, one_pose=False):
    from fiducials import tag_model_corners

    sightings = []
    q = np.zeros(7)
    for hold in range(holds):
        if hold == 0 or not one_pose:
            q = np.zeros(7)
            q[:6] = rng.uniform(-0.4, 0.4, 6)
        for tag in (2, 3, 4):
            points = register.corner_points(kin, arm, q, tag, 0.047)
            pixels = register._project(camera_from_base, points, k, dist) + rng.normal(0.0, 0.3, (4, 2))
            sightings.append({"hold": hold, "tag": tag, "points": points, "pixels": pixels, "tcp": kin.fk(q)})
    assert tag_model_corners(0.047).shape == (4, 3)
    return sightings


K = np.array([[322.2, 0.0, 323.4], [0.0, 321.7, 181.4], [0.0, 0.0, 1.0]])
DIST = np.array([-0.0527, 0.057, -0.0008, -0.0001, -0.0179])


def _kin():
    from tatbot_description import robot_description
    from tatbot_motion import Kinematics

    register._lib(register.Path(__file__).resolve().parents[3])
    return Kinematics(robot_description(None, arms=("right",)), "right")


def _camera():
    camera_from_base = np.eye(4)
    camera_from_base[:3, :3] = cv2.Rodrigues(np.array([2.6, 0.3, -0.2]))[0]
    camera_from_base[:3, 3] = [0.05, 0.10, 0.75]
    return camera_from_base


def test_the_solve_gives_back_a_known_camera_and_drops_a_bad_sighting():
    kin, rng, camera_from_base, k, dist = _kin(), np.random.default_rng(7), _camera(), K, DIST
    sightings = _synthetic(kin, "right", camera_from_base, k, dist, rng)
    sightings[5] = dict(sightings[5], pixels=sightings[5]["pixels"] + 25.0)   # one tag misread
    solved, kept = register.solve(sightings, k, dist)
    assert len(kept) == len(sightings) - 1 and all(s is not sightings[5] for s in kept)
    assert np.linalg.norm(solved[:3, 3] - camera_from_base[:3, 3]) < 0.002
    assert register._angle_deg(solved, camera_from_base) < 0.1
    out = register.hold_out(kept, solved, k, dist)
    assert out["views"] == 16 and out["max_mm"] < 2.0


def test_a_hold_the_arm_did_not_reach_is_named():
    """The measured tool within half a centimetre and two degrees of the hold is there; the arm left at rest by a
    move that succeeded is not (2026-09-29: 57-111 mm and 29-70 deg away, every hold of two runs)."""
    kin = _kin()
    rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0])
    q = np.array([0.2, 0.9, 0.8, -0.8, 0.1, 1.2, 0.0])
    hold = kin.fk(q)
    assert register.off_hold(kin, q, hold) is None
    nudged = hold.copy()
    nudged[:3, 3] += [0.001, -0.001, 0.0]
    assert register.off_hold(kin, q, nudged) is None
    assert "mm" in register.off_hold(kin, rest, hold)
    turned = hold.copy()
    turned[:3, :3] = hold[:3, :3] @ cv2.Rodrigues(np.array([0.0, 0.0, np.radians(3.0)]))[0]
    assert "3.0 deg" in register.off_hold(kin, q, turned)


def _bundle():
    optics = {"intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
              "distortion": {"coefficients": DIST.tolist()}}
    return {"bundle_id": "synthetic", "cameras": {register.COLOR: optics}}


def test_holds_that_repeat_one_pose_are_one_view_and_refused():
    """Twenty-seven holds that all saw the arm at one pose fit that view as closely as any, and leaving a hold out
    moves nothing: they count as one view, and the gates refuse it. Holds at random joints pass."""
    from fiducials import load_inventory

    kin, rng, camera_from_base = _kin(), np.random.default_rng(3), _camera()
    target = load_inventory(register.Path(__file__).resolve().parents[3] / "config" / "fiducials.json").target("wrist")
    stuck = register.fit("right", _bundle(), True, _synthetic(kin, "right", camera_from_base, K, DIST, rng, holds=27,
                                                              one_pose=True), target)
    assert stuck["fit"]["holds"] == 27 and stuck["fit"]["poses"] == 1 and stuck["fit"]["median_px"] < 1.0
    assert stuck["hold_out"]["max_mm"] is None
    assert any("27 holds at 1 distinct poses" in why for why in stuck["refused"])
    moving = register.fit("right", _bundle(), True, _synthetic(kin, "right", camera_from_base, K, DIST, rng, holds=20),
                          target)
    assert moving["fit"]["poses"] >= register.MIN_HOLDS and moving["refused"] == []
    assert register.pose_views([np.eye(4), np.eye(4)]) == [0, 0]


def _metadata(coefficients=(-0.0527, 0.057, -0.0008, -0.0001, -0.0179), serial="262422301129"):
    intrinsics = {"schema": "tatbot.camera-intrinsics/1", "width": 640, "height": 360, "fx": 322.2, "fy": 321.7,
                  "ppx": 323.4, "ppy": 181.4, "distortion_model": "BrownConrady",
                  "distortion_coefficients": list(coefficients)}
    return {"profile": {"stream": "color", "width": 640, "height": 360, "fps_num": 30, "fps_den": 1, "format": "yuyv"},
            "attributes": {"intrinsics": json.dumps(intrinsics), "device_serial": serial}}


def test_a_bundle_is_drafted_at_identity_and_reused_only_for_the_same_camera():
    draft = register.draft_bundle(_metadata(), None)
    assert draft["world_frame"] == register.WORLD_FRAME and draft["bundle_id"] == ""
    color, depth = draft["cameras"][register.COLOR], draft["cameras"][register.DEPTH]
    assert color["world_from_camera"] == depth["world_from_camera"]
    assert color["world_from_camera"]["rotation"] == [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    assert depth["profile"]["format"] == "z16" and color["intrinsics"] == depth["intrinsics"]
    assert color["distortion"]["model"] == "brown_conrady"
    installed = dict(draft, bundle_id="abc")
    # a warmer camera reports another focal length: the same camera, the bundle stands
    warmer = register.draft_bundle(json.loads(json.dumps(_metadata())), None)
    warmer["cameras"][register.COLOR]["intrinsics"]["fx"] = 323.2
    assert register.reusable(installed, warmer)
    assert not register.reusable(installed, register.draft_bundle(_metadata(coefficients=(0, 0, 0, 0, 0)), None))
    assert not register.reusable(installed, register.draft_bundle(_metadata(serial="other"), None))
    assert not register.reusable(dict(installed, world_frame="camera2_optical"), draft)


def test_a_bundle_binds_the_stream_the_owner_opens_not_the_bus_encoding():
    """A capture over the bus relabels colour rgb8 (it travels as JPEG) and keeps the camera's own format in
    bus_source_format; a bundle drafted from it must name the owner's yuyv, or the owner refuses it at start."""
    metadata = _metadata()
    metadata["profile"]["format"] = "rgb8"
    metadata["attributes"]["bus_source_format"] = "yuyv"
    draft = register.draft_bundle(metadata, None)
    assert draft["cameras"][register.COLOR]["profile"]["format"] == "yuyv"
    assert draft["cameras"][register.DEPTH]["profile"]["format"] == "z16"
    assert register.draft_bundle(_metadata(), None)["cameras"][register.COLOR]["profile"]["format"] == "yuyv"


def test_gates_refuse_thin_or_unsteady_fits():
    report = {"fit": {"holds": 20, "poses": 20, "median_px": 0.6, "p95_px": 1.8},
              "per_tag": {"2": {"poses": 9}, "3": {"poses": 2}}, "hold_out": {"max_mm": 1.2, "max_deg": 0.1}}
    assert register.gates(report) == ["tags ['3'] were each seen at fewer than 3 distinct poses"]
    report["per_tag"]["3"]["poses"] = 5
    assert register.gates(report) == []
    report["hold_out"]["max_mm"] = 9.0
    assert "one view moves" in register.gates(report)[0]
    report["hold_out"]["max_mm"], report["fit"]["poses"] = 1.2, 4
    assert register.gates(report) == ["20 holds at 4 distinct poses saw tags (need 12 poses, 3 cm or 15 deg apart)"]


def _target():
    from fiducials import load_inventory

    return load_inventory(register.Path(__file__).resolve().parents[3] / "config" / "fiducials.json").target("wrist")


def _overhead(at=(0.28, 0.03, 0.75)):
    """camera <- base for a camera straight over `at` looking down."""
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    base_from_camera[:3, 3] = at
    return np.linalg.inv(base_from_camera)


def test_the_chooser_takes_holds_that_separate_the_chain_over_a_cluster():
    kin, rng = _kin(), np.random.default_rng(2)
    from tatbot_calib import chain

    prior = {"k": K, "dist": DIST, "camera_from_base": _overhead()}
    clustered = [np.r_[0.1, 0.6, 0.6, -1.0, 0.0, 1.5, 0.0] + np.r_[rng.normal(0, 0.03, 6), 0.0] for _ in range(20)]
    wide = [np.r_[0.1, 0.6, 0.6, -1.0, 0.0, 1.5, 0.0] + np.r_[rng.uniform(-0.4, 0.4, 6), 0.0] for _ in range(20)]
    candidates = [(kin.fk(q), q, [2, 3, 4]) for q in clustered + wide]
    jacobians = {}
    chosen = register.choose_holds(kin, "right", prior, _target(), candidates, 8, jacobians)
    assert len(chosen) == 8 and sum(i >= len(clustered) for i in chosen) >= 6
    assert np.all(np.array(chain.predicted_sigma([jacobians[i] for i in chosen])["dq_mrad"])
                  < np.array(chain.predicted_sigma([jacobians[i] for i in range(8)])["dq_mrad"]))
    for a in chosen:   # no two chosen holds are one view
        for b in chosen:
            pa, pb = candidates[a][0], candidates[b][0]
            assert a == b or (np.linalg.norm(pa[:3, 3] - pb[:3, 3]) >= register.DISTINCT_M
                              or register._angle_deg(pa, pb) >= register.DISTINCT_DEG)


def test_a_hold_is_upright_only_near_its_pen_down_joints_and_clear_of_the_arm_itself():
    from tatbot_session import ready

    kin = _kin()
    pose = register.hold_poses((0.28, 0.03), 0.003, heading_deg=register.rest_heading_deg(kin))[0]
    posture = register.Posture(kin, ready, 0.003)
    q = ready.solve_ik_seeded(kin, pose, posture.reference(pose))
    assert posture.ok(pose, q)
    twisted = q.copy()
    twisted[4] += 1.0                                   # the wrist turned 57 deg off its pen-down joints
    assert not posture.ok(pose, twisted)
    assert not register.Posture(kin, ready, 0.003, self_gap=lambda q: 0.005).ok(pose, q)   # 5 mm off itself


def test_planned_holds_stay_upright_solved_from_the_hold_before_each():
    from tatbot_session import ready

    kin, target = _kin(), _target()
    prior = {"k": K, "dist": DIST, "camera_from_base": _overhead(), "size": (640, 360)}
    posture = register.Posture(kin, ready, 0.003)
    poses, sigma = register.planned_poses(kin, "right", prior, target, (0.28, 0.03), 0.003,
                                          heading_deg=register.rest_heading_deg(kin), spread_m=0.06,
                                          heights_m=(0.15,), count=8, ready=ready, posture=posture)
    assert len(poses) >= 6 and len(sigma["dq_mrad"]) == 4
    q = register.REST
    for pose in poses:
        q = ready.solve_ik_seeded(kin, pose, q)
        assert posture.ok(pose, q)
        assert register.off_hold(kin, q, pose) is None
