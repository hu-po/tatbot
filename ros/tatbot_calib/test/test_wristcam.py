"""The wrist hand-eye's holds (tatbot_calib.wristcam): which poses over the palette's roof tag it tries and in what
order, what hides the tag from the wrist camera, and how far the run goes past ways the checks refuse. The station
is the WXAI follower's of 2026-10-19, when 15 of the run's 30 holds were refused on their way and 6 saw no tag."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_calib import halo, program, reach, wristcam
from tatbot_description import arm_model

tatbot_motion = pytest.importorskip("tatbot_motion")

# 2026-10-19 in right/base_link: the roof tag through the station touch, the overhead camera's post, the e-stop's
# keep-out (40 mm about it); the wrist D405's colour intrinsics; the tag's 46 mm black square.
TAG = np.array([[0.5954, 0.8034, 0.0, 0.2425], [-0.8034, 0.5954, 0.0, 0.3350], [0.0, 0.0, 1.0, 0.0345],
                [0.0, 0.0, 0.0, 1.0]])
POST = reach.overhead_post([0.1012, 0.3944, 0.5771])
ESTOP = np.array([0.2268, 0.2706, 0.0321])
K = np.array([[651.08, 0.0, 650.44], [0.0, 650.32, 358.05], [0.0, 0.0, 1.0]])
MODEL = np.array([[-0.023, 0.023, 0.0], [0.023, 0.023, 0.0], [0.023, -0.023, 0.0], [-0.023, -0.023, 0.0]])


@pytest.fixture(scope="module")
def kin():
    from tatbot_description import robot_description

    return tatbot_motion.Kinematics(robot_description(None, arms=("right",)), "right")


class Cal:
    """What the hand-eye asks of program.Calibration; rows_check here refuses only the wrist on the post."""
    arm = "right"

    def __init__(self, kin):
        self.kin, self.refused = kin, 0
        self.bodies = reach.wrist_bodies(kin, "right") if kin is not None else []

    def zones(self):
        return [("palette_estop", ESTOP, 0.040, False)]

    def station_now(self):
        return np.eye(4)

    def rows_check(self, label, rows):
        if reach.wrist_clearance(self.kin, rows[-1], self.bodies, [POST]) < reach.CLEARANCE_M:
            self.refused += 1
            raise RuntimeError(f"{label}: the wrist on the post")


def hand_eye(cal) -> wristcam.HandEye:
    camera = SimpleNamespace(k=K, dist=np.zeros(5), size=(1280, 720))
    return wristcam.HandEye(cal, TAG, camera, None, MODEL, "right/realsense_color_optical_frame")


def looking_at(eye, target) -> np.ndarray:
    """camera <- base for a camera at `eye` whose optical axis meets `target`."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3], base_from_camera[:3, 3] = np.c_[x, np.cross(z, x), z], eye
    return np.linalg.inv(base_from_camera)


def test_candidates_stand_clear_of_what_rows_check_refuses_and_see_the_tag_most_varied_first(kin, monkeypatch):
    from tatbot_session import ready

    for name, value in (("HEIGHTS_M", (0.08,)), ("OFFSETS_M", (0.0, 0.05)), ("AZIMUTHS", 4),
                        ("HEADINGS_DEG", (-75, -45, -15, 15)), ("TILTS_DEG", ((0, 0), (15, 90)))):
        monkeypatch.setattr(wristcam, name, value)
    cal = Cal(kin)
    hand = hand_eye(cal)
    found = hand.candidates()
    assert found and cal.refused and np.array_equal(found, hand.candidates())   # the post refused some; same list
    cad = hand.cad_mount()
    tcp_from_mount = np.linalg.inv(kin.fk(np.zeros(7))) @ kin.frame(np.zeros(7), "right/tool_mount")
    for pose in found:   # the arm where candidates() stood it
        q = ready.solve_ik_seeded(kin, pose, reach.rest(kin))
        assert reach.wrist_clearance(kin, q, cal.bodies, [POST]) >= reach.CLEARANCE_M
        assert hand.unseen(np.linalg.inv(pose @ tcp_from_mount @ cad), None, None) is None
    headings = {round(math.degrees(math.atan2(p[1, 0], p[0, 0]))) for p in found}
    assert -75 not in headings and len(headings) >= 2   # the post stands where -75 turns the wrist
    assert wristcam.spread(found) == list(range(len(found)))


def test_spread_takes_the_farthest_next_and_breaks_ties_by_order():
    poses = [halo.tool_pose([x, 0.0, 0.1], 0.0) for x in (0.0, 0.01, 0.02, 0.03)]
    assert wristcam.spread(poses) == [0, 3, 1, 2]
    turned = [halo.tool_pose([0.0, 0.0, 0.1], 0.0), halo.tool_pose([0.015, 0.0, 0.1], 0.0),
              halo.tool_pose([0.0, 0.0, 0.1], 1.0)]   # a radian of turn counts 50 mm, over 15 mm across
    assert wristcam.spread(turned) == [0, 2, 1]
    assert wristcam.spread([]) == []


def test_the_gauge_mount_puts_the_tip_and_axis_where_the_gauge_has_them():
    mount = halo.tool_pose([0.03, -0.01, 0.05], 0.4, 0.3, 1.0)
    tip = np.array([0.0005, -0.007, 0.0745])
    gauge = wristcam.gauge_record(mount, tip, "right", "x")
    assert np.allclose(wristcam.gauge_mount(gauge, mount, tip), mount)
    off = mount @ halo.tool_pose([0.004, 0.006, -0.010], 0.05, 0.05, 0.3)   # the robot model's, a few mm and deg off
    back = wristcam.gauge_record(wristcam.gauge_mount(gauge, off, tip), tip, "right", "x")
    assert np.allclose(back["tip_cam_m"], gauge["tip_cam_m"]) and np.allclose(back["axis_cam"], gauge["axis_cam"])


def test_a_gauge_derived_from_a_moved_tip_is_the_hand_eyes_gauge_for_that_tip():
    """`calib apply --derive-gauges` moves a gauge's tip through the hand-eye its file records (gauge_derive, on the
    ROS owner): the same file a hand-eye run would have written with the new tip."""
    import sys

    from tatbot_description import repo_root

    sys.path.insert(0, str(repo_root(None) / "scripts" / "lib"))
    import gauge_derive

    mount = halo.tool_pose([0.03, -0.01, 0.05], 0.4, 0.3, 1.0)
    then, now = np.array([0.0005, -0.007, 0.0745]), np.array([0.0024, -0.0066, 0.0762])
    gauge = wristcam.gauge_record(mount, then, "xarm", "a-tool")
    derived = gauge_derive.derive(gauge, gauge_derive.tip_before(gauge, "b-tool", [0.0, 0.0, 0.0], None), now.tolist(),
                                  gauge_derive.hand_eye(gauge, Path("no-runs")), tool_id="b-tool", run="a-sweep",
                                  utc="2026-11-11T00:00:00Z")
    want = wristcam.gauge_record(mount, now, "xarm", "b-tool")
    for key in ("tip_cam_m", "cone_end_cam_m", "axis_cam", "normal_cam", "tip_mount_m", "mount_from_camera"):
        assert np.allclose(derived[key], want[key], atol=1e-12), key
    assert derived["measured_with"] == "b-tool" and derived["derived"][0]["run"] == "a-sweep"


def test_the_tag_is_hidden_by_the_image_edge_a_steep_view_the_pen_and_the_e_stops_cap():
    hand, centre = hand_eye(Cal(None)), TAG[:3, 3]
    over = looking_at(centre + [0.0, 0.0, 0.2], centre)
    assert hand.unseen(over, None, None) is None
    assert hand.unseen(looking_at(centre + [0.18, 0.0, 0.2], centre + [0.18, 0.0, 0.0]), None, None) == \
        "the image's edge"
    slant = math.radians(50)
    assert hand.unseen(looking_at(centre + 0.25 * np.array([math.sin(slant), 0.0, math.cos(slant)]), centre),
                       None, None) == "too steep"
    across = np.linspace([650.0, 358.0], [300.0, 700.0], 21)   # a pen through the tag's image
    assert hand.unseen(over, across, None) == "the pen"
    assert hand.unseen(over, across - [500.0, -300.0], None) is None
    assert hand.unseen(over, None, centre + [0.05, 0.0, 0.0]) == "the e-stop's cap"   # its image on the border
    assert hand.unseen(over, None, centre + [0.12, 0.0, 0.0]) is None


@pytest.mark.parametrize("sees", [True, False])
def test_a_refused_way_costs_a_check_not_a_hold_and_the_run_stops_after_tries(monkeypatch, sees):
    monkeypatch.setattr(program, "_preflight", lambda cal: None)
    monkeypatch.setattr(program, "_latched", lambda cal: False)
    monkeypatch.setattr(reach, "_turn", lambda kin, pose, q: 0.0)   # the holds in the pool's order
    sent = []
    rig = SimpleNamespace(goal=lambda pose, move: sent.append(pose), spin=lambda s: None, joints=lambda: np.zeros(7))

    def go_to(label, pose):   # every other way refused
        if int(label[4:]) % 2:
            raise RuntimeError(f"{label}: refused")
        rig.goal(pose, move=True)

    cal = SimpleNamespace(kin=SimpleNamespace(revolute=arm_model.load("trossen-wxai").arm_index), arm="right", rig=rig,
                          go_to=go_to, say=lambda text: None,
                          log=lambda name, row: None)
    hand = hand_eye(cal)
    hand.candidates = lambda: [halo.tool_pose([0.2, 0.3, 0.1 + 1e-3 * i], 0.0) for i in range(500)]
    hand.grab = lambda label, q: {"label": label, "q": [0.0] * 7, "frames": 5, "camera_from_tag": np.eye(4)} \
        if sees else None
    holds = hand.run()
    if sees:
        assert len(holds) == wristcam.HOLDS_WANTED and len(sent) == wristcam.HOLDS_WANTED
    else:
        assert holds == [] and len(sent) == wristcam.TRIES // 2


def test_the_fit_solves_the_tag_with_the_camera_where_the_touch_put_it_off():
    """The xArm rig's tag stood 2-3 mm and 1.2 deg from where the touch and the overhead's yaw put it (2026-11-09):
    holds seen through a true mount and tag, upright and tilted 10-15 deg as the run holds, fit both from that wrong
    tag and a mount 5 mm off."""
    import cv2
    from scipy.spatial.transform import Rotation

    k, dist = np.array([[651.0, 0.0, 640.0], [0.0, 650.0, 360.0], [0.0, 0.0, 1.0]]), np.zeros(5)
    model = np.array([[-0.023, 0.023, 0.0], [0.023, 0.023, 0.0], [0.023, -0.023, 0.0], [-0.023, -0.023, 0.0]])
    mount = wristcam._pose(Rotation.from_euler("xyz", [170.0, 5.0, 30.0], degrees=True).as_matrix(),
                           [0.06, -0.02, 0.03])
    tag = wristcam._pose(Rotation.from_euler("z", 20.0, degrees=True).as_matrix(), [0.33, 0.28, 0.0])
    touched = wristcam._pose(Rotation.from_euler("z", 21.2, degrees=True).as_matrix(), [0.332, 0.282, 0.002])
    holds = []
    for tilt, heading in ((0, 0), (15, 0), (15, 90), (15, 180), (15, 270), (0, 60), (10, 135), (10, 315)):
        turn = Rotation.from_euler("zx", [heading, tilt], degrees=True).as_matrix()
        camera = wristcam._pose(turn @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix(),
                                tag[:3, 3] + turn @ [0.0, 0.0, 0.22])
        base_from_mount = camera @ np.linalg.inv(mount)
        pts = (np.linalg.inv(camera) @ tag @ np.c_[model, np.ones(4)].T).T[:, :3]
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k, dist)
        holds.append({"label": f"hold{len(holds)}", "base_from_mount": base_from_mount,
                      "pixels": uv.reshape(-1, 2).tolist()})
    seed = mount @ wristcam._pose(np.eye(3), [0.003, -0.002, 0.004])
    fitted, median_px, per_hold, solved = wristcam.fit_mount(holds, touched, model, k, dist, seed)
    assert np.linalg.norm(fitted[:3, 3] - mount[:3, 3]) < 1e-5 and np.linalg.norm(solved[:3, 3] - tag[:3, 3]) < 1e-5
    assert median_px < 0.01 and per_hold.max() < 0.01


def test_a_sticker_turned_on_its_seat_reads_as_its_turn_and_shift():
    """The station fix reads the palette's yaw from the roof tag, so a sticker turned e on its seat turns the touch's
    palette e about the ball: the measured tag then stands turned -e about the ball from the touch's (2026-11-09:
    1.29 deg, 3.6 mm aside). The seat gives back the turn and the shift."""
    from scipy.spatial.transform import Rotation

    turn, shift = math.radians(1.29), np.array([0.00064, -0.00062])
    palette = wristcam._pose(Rotation.from_euler("z", -6.5, degrees=True).as_matrix(), [0.329, 0.280, -0.042])
    ball_p = np.array([0.0, 0.0, 0.0556])
    cad_tag = wristcam._pose(np.eye(3), [0.1145, -0.1115, -0.007])
    true_tag = cad_tag.copy()
    true_tag[:3, :3] = Rotation.from_euler("z", turn).as_matrix()
    true_tag[:2, 3] += shift
    ball = (palette @ np.append(ball_p, 1.0))[:3]
    about_ball = np.eye(4)
    about_ball[:3, :3] = Rotation.from_euler("z", turn).as_matrix()
    about_ball[:3, 3] = ball - about_ball[:3, :3] @ ball
    touched = about_ball @ palette    # the fix's yaw off by the sticker's turn, anchored at the touched ball
    seat = wristcam.tag_seat(touched, ball, touched @ cad_tag, palette @ true_tag)
    assert seat["turn_deg"] == pytest.approx(1.29, abs=0.01)
    assert seat["shift_mm"] == pytest.approx([0.64, -0.62], abs=0.05) and abs(seat["rise_mm"]) < 1e-6
    assert seat["against"]["xyz_m"] == pytest.approx(cad_tag[:3, 3]) and seat["against"]["yaw_deg"] == pytest.approx(0.0)
