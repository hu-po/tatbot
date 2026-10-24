"""The wrist hand-eye's holds (tatbot_calib.wristcam): which poses over the palette's roof tag it tries and in what
order, what hides the tag from the wrist camera, and how far the run goes past ways the checks refuse. The station
is the pink arm's of 2026-10-19, when 15 of the run's 30 holds were refused on their way and 6 saw no tag."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_calib import halo, program, reach, wristcam

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
        q = ready.solve_ik_seeded(kin, pose, reach.REST)
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
    sent = []
    rig = SimpleNamespace(goal=lambda pose, move: sent.append(pose), spin=lambda s: None, joints=lambda: np.zeros(7))

    def go_to(label, pose):   # every other way refused
        if int(label[4:]) % 2:
            raise RuntimeError(f"{label}: refused")
        rig.goal(pose, move=True)

    cal = SimpleNamespace(kin=None, arm="right", rig=rig, go_to=go_to, say=lambda text: None,
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
