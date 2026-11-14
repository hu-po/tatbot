"""The lab's views (tatbot_session.views): page points read, the page to aim by found, every hold planned and its
wrist checked before anything moves, and each view moved to, photographed by every camera and left straight up."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tatbot_motion
from tatbot_description import arm_model
from tatbot_session import client, ready, views


def test_points_are_page_millimetres_and_a_negative_x_reads():
    assert views.points_mm("-40,52; 0,52;") == [(-40.0, 52.0), (0.0, 52.0)]
    assert views.points_mm("") == []
    with pytest.raises(ValueError, match="x,y"):
        views.points_mm("12")
    with pytest.raises(ValueError, match="finite"):
        views.points_mm("nan,1")


def _run(root, workflow, name, page):
    run = root / workflow / name
    run.mkdir(parents=True)
    (run / "page.json").write_text(json.dumps(page))
    return run


def test_the_page_is_the_newest_finished_set_up_of_a_draw_or_a_touch(tmp_path):
    _run(tmp_path, "ros-draw", "20261108T100000Z-n-aaaa", {"used": np.eye(4).tolist()})
    touch = _run(tmp_path, "ros-touch", "20261108T110000Z-n-bbbb", {"used": np.eye(4).tolist()})
    _run(tmp_path, "ros-draw", "20261108T120000Z-n-cccc", {})   # a set-up that never finished
    assert views.page_run(tmp_path) == touch
    assert views.page_run(tmp_path, "20261108T100000Z-n-aaaa").name == "20261108T100000Z-n-aaaa"
    with pytest.raises(ValueError, match="no ros-draw or ros-touch run"):
        views.page_run(tmp_path, "20261108T130000Z-n-dddd")
    with pytest.raises(ValueError, match="never finished"):
        views.page_record(tmp_path / "ros-draw" / "20261108T120000Z-n-cccc")


def test_no_page_set_up_says_to_make_one(tmp_path):
    with pytest.raises(ValueError, match="tatbot ros touch"):
        views.page_run(tmp_path)


def _kin():
    model = arm_model.load("ufactory-xarm7")
    return SimpleNamespace(arm_model=model, joint_names=[f"j{i}" for i in range(7)], fk=lambda q: np.eye(4))


def test_a_hold_turning_the_wrist_out_of_its_range_refuses_the_whole_capture_before_any_motion(monkeypatch):
    kin = _kin()
    wrist = list(kin.arm_model.wrist)
    good, bad = np.zeros(7), np.zeros(7)
    good[wrist] = np.mean(np.asarray(kin.arm_model.wrist_range_rad), axis=1)
    bad[wrist[0]] = np.asarray(kin.arm_model.wrist_range_rad)[0][1] + 0.5
    answers = iter([good, bad])
    monkeypatch.setattr(ready, "solve_ik_seeded", lambda kin, target, q: next(answers))
    with pytest.raises(ValueError, match="view 1 over"):
        views.plan(kin, views.targets([np.eye(4)], [(0.0, 0.0), (0.01, 0.0)]), np.eye(4), np.zeros(7),
                   height_m=0.052)


class _Camera:
    def __init__(self, value):
        self.value = value

    def grab_both(self, n, fresh=False):
        assert fresh   # frames from where the arm stands now, none queued while it moved
        return [np.full((4, 6, 3), self.value + k, np.uint8) for k in range(n)], []


def _io(sent, fail=None):
    def execute(knots):
        sent.append(knots)
        return ("aborted", None, "stopped") if knots == fail else ("ok", None, "")
    return SimpleNamespace(measured=lambda: np.zeros(7), execute=execute)


def _motion(monkeypatch):
    monkeypatch.setattr(tatbot_motion, "plan_lift", lambda **kw: ("up" if kw["direction"][2] > 0 else "down"))
    monkeypatch.setattr(tatbot_motion, "to_knots", lambda traj, rate: traj)
    monkeypatch.setattr(ready, "joint_move", lambda names, q0, q1, **kw: ("over", float(q1[0])))


def test_each_view_is_moved_to_photographed_by_every_camera_and_left_straight_up(tmp_path, monkeypatch):
    _motion(monkeypatch)
    sent, written = [], {}
    over = [np.full(7, 0.1), np.full(7, 0.2)]
    records = views.capture(_io(sent), _kin(), {"wrist_upper": _Camera(10), "wrist_lower": _Camera(50)},
                            [np.eye(4)] * 2, over, approach_m=0.04, motion={}, frames=3, settle_s=0.0, speed_rad_s=0.3,
                            out_dir=tmp_path, write=lambda path, image: written.update({path.name: image}))
    assert sent == [("over", 0.1), "down", "up", ("over", 0.2), "down", "up"]
    assert sorted(written) == ["v00-wrist_lower.png", "v00-wrist_upper.png", "v01-wrist_lower.png",
                               "v01-wrist_upper.png"]
    assert int(written["v00-wrist_upper.png"][0, 0, 0]) == 11   # the median of 10, 11, 12
    assert [r["names"] for r in records][1] == ["v01-wrist_upper.png", "v01-wrist_lower.png"]


def test_a_view_whose_descent_fails_still_goes_back_up_and_says_which(tmp_path, monkeypatch):
    _motion(monkeypatch)
    sent = []
    with pytest.raises(RuntimeError, match="view 0 down"):
        views.capture(_io(sent, fail="down"), _kin(), {"wrist_upper": _Camera(10)}, [np.eye(4)], [np.zeros(7)],
                      approach_m=0.04, motion={}, frames=1, settle_s=0.0, speed_rad_s=0.3, out_dir=tmp_path,
                      write=lambda path, image: None)
    assert sent == [("over", 0.0), "down"]   # a failed move raises at once, as a close-up hold does: the arm holds


def test_a_rest_view_moves_nothing(tmp_path):
    sent, written = [], {}
    records = views.rest_view(_io(sent), _kin(), {"wrist_upper": _Camera(20)}, frames=2, out_dir=tmp_path,
                              write=lambda path, image: written.update({path.name: image}))
    assert sent == [] and list(written) == ["v00-wrist_upper.png"] and records[0]["view"] == 0


def _args(tmp_path, **given):
    return SimpleNamespace(**{"out": str(tmp_path / "card-p0001-run-01-before"), "points": "-40,52", "run": "",
                              "logs": str(tmp_path / "logs"), "arm": "xarm", "wrist": True, "overhead": False, "hover": None,
                              "overhead_pages": "", "sheet_at": "",
                              **given})


def test_the_client_refuses_a_label_and_a_missing_page_before_it_touches_the_arm(tmp_path, capsys):
    assert client._views(None, _args(tmp_path, out=str(tmp_path / "../x y"))) == 2
    assert "names the views" in capsys.readouterr().err
    assert client._views(None, _args(tmp_path)) == 2   # node None: it never got as far as the arm
    assert "tatbot ros touch" in capsys.readouterr().err


class _Overhead:
    """The D555 owner's capture: each call answers the newest exposure after `after_ns`."""

    def __init__(self):
        self.asked, self.stamp = [], 1000

    def capture(self, after_ns):
        self.asked.append(after_ns)
        self.stamp = max(self.stamp, after_ns) + 33
        return {"image": np.full((3, 5, 3), self.stamp % 7, np.uint8), "stamp_ns": self.stamp,
                "metadata": {"attributes": {"intrinsics": json.dumps({"schema": "tatbot.camera-intrinsics/1"})}}}


def test_the_overhead_view_is_the_median_of_exposures_each_newer_than_the_last(tmp_path):
    camera, written = _Overhead(), {}
    record = views.overhead(camera, 3, out_dir=tmp_path, write=lambda path, image: written.update({path.name: image}),
                            now_ns=lambda: 2000)
    assert camera.asked == [2000, 2034, 2068] and record["stamps_ns"] == [2033, 2067, 2101]
    assert list(written) == ["overhead.png"] and record["intrinsics"]["schema"] == "tatbot.camera-intrinsics/1"


def test_a_chosen_sheet_needs_the_overhead_to_aim_by(tmp_path, capsys):
    assert client._views(None, _args(tmp_path, sheet_at="400,-100")) == 2
    assert "add --overhead-pages" in capsys.readouterr().err


def _chooser(monkeypatch, stamps_s):
    """A node whose page topic delivers a measured Page stamped at each of `stamps_s` (wall-clock seconds), one a spin,
    and a bridge that takes the choice."""
    from tatbot_interfaces.msg import Page

    callbacks, left = [], list(stamps_s)

    def spin(node, timeout_sec):
        if left:
            page = Page(source=Page.SOURCE_MEASURED)
            page.pose.header.stamp.sec, page.pose.header.stamp.nanosec = int(left[0]), int(left.pop(0) % 1 * 1e9)
            callbacks[0](page)

    monkeypatch.setattr(client, "_call", lambda *_: SimpleNamespace(results=[SimpleNamespace(successful=True)]))
    monkeypatch.setattr(client.rclpy, "spin_once", spin)
    monkeypatch.setattr(client, "SHEET_WAIT_S", 0.5)
    return SimpleNamespace(create_subscription=lambda _type, _topic, callback, _depth: callbacks.append(callback),
                           destroy_subscription=lambda _sub: None)


def test_a_chosen_sheet_counts_once_the_overhead_measures_it_after_the_choice(monkeypatch, capsys):
    import time

    now = time.time()
    assert client._choose_sheet(_chooser(monkeypatch, [now - 5, now + 1]), "xarm", "479,-15")
    assert not client._choose_sheet(_chooser(monkeypatch, [now - 5, now - 1]), "xarm", "479,-15")
    assert "read no sheet of the print" in capsys.readouterr().err


def test_the_client_wants_some_camera(tmp_path, capsys):
    assert client._views(None, _args(tmp_path, points="", wrist=False, overhead=False)) == 2
    assert "add --overhead" in capsys.readouterr().err


def test_a_print_is_named_by_its_id_or_its_leading_hex_and_must_name_one(tmp_path):
    for name in ("stencil-981a6476aa", "stencil-981b00", "stencil-bef888"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "tracking.json").write_text("{}")
    assert views.resolve_print(tmp_path, "981a") == "stencil-981a6476aa"
    assert views.resolve_print(tmp_path, "stencil-bef888") == "stencil-bef888"
    with pytest.raises(ValueError, match="names 2"):
        views.resolve_print(tmp_path, "981")


def test_every_sheet_of_a_print_the_overhead_reads_is_a_page_and_holds_go_over_each():
    """Two mounts carrying one print: the tracker's sample and its copy are two pages; a point a page."""
    world_from_base = np.eye(4)
    first, second = np.eye(4), np.eye(4)
    first[:3, 3], second[:3, 3] = [0.4, 0.1, 0.0], [0.4, -0.1, 0.0]
    sample = {"world_from_target": first, "translation_sigma_m": 0.001, "rotation_sigma_rad": 0.01, "support": {
        "markers": {"ids": [1, 2]}, "copies": [{"world_from_target": second.tolist(), "translation_sigma_m": 0.002,
                                                "rotation_sigma_rad": 0.02, "support": {"markers": {"ids": [1]}}}]}}
    found = views.sheets(sample, world_from_base)
    assert [s["sheet"] for s in found] == [0, 1] and [s["sigma_m"] for s in found] == [0.001, 0.002]
    assert np.allclose(found[1]["world_from_page"][:3, 3], [0.4, -0.1, 0.0])   # the overhead's frame, for lab read
    assert np.allclose(found[1]["world_from_page"][:3, 2], [0.0, 0.0, -1.0])   # out of the paper, toward the camera
    aims = views.targets([s["base_from_page"] for s in found], [(0.0, 0.0)])
    assert len(aims) == 2 and not np.allclose(aims[0][0][:3, 3], aims[1][0][:3, 3])


def test_a_chosen_sheet_is_the_one_nearest_the_point_in_the_arms_base():
    first, second = np.eye(4), np.eye(4)
    first[:3, 3], second[:3, 3] = [0.4, 0.1, 0.0], [0.4, -0.1, 0.0]
    found = [{"sheet": 0, "base_from_page": first}, {"sheet": 1, "base_from_page": second}]
    assert [s["sheet"] for s in views.sheet_at(found, (0.41, -0.09), 0.05)] == [1]
    with pytest.raises(ValueError, match="at 400,100, 400,-100 mm"):
        views.sheet_at(found, (0.2, 0.0), 0.05)
