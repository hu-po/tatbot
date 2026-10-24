"""page_filter.PageFilter: the print's offset from the wrist's fits and the overhead's marker moves (ros/README.md 4.4,
stencil tracking: what track.Tracker corrects each stroke by)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_session.page_filter import CAMERAS, PageFilter
from tatbot_session.track import Tracker

GAUGE = [[-0.006, -0.006], [0.006, -0.006], [0.006, 0.006], [-0.006, 0.006], [0.0, 0.0]]
ARM = np.array([-0.0015, 0.0004])
GRAD = np.array([[-0.25, 0.0], [0.0, 0.05]])   # the forearm heart's x gradient (2026-10-23)


def arm(u):
    return ARM + GRAD @ np.asarray(u, float)


def slid(dx, dy, turn=0.0):
    m = np.eye(3)
    m[:2, :2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    m[:2, 2] = dx, dy
    return m


def residual(u, move, follow=None):
    """What a wrist reading at u on the page in use reads: the print point under the pen less u."""
    follow = np.eye(3) if follow is None else follow
    p = follow[:2, :2] @ u + follow[:2, 2]
    q = p + arm(p)
    return np.linalg.solve(move[:2, :2], q - move[:2, 2]) - u


def held(f, t0=0.0):
    for k, u in enumerate(GAUGE):
        assert f.add(t0 + 0.1 * k, u, arm(u), check=False, rms=0.0008, markers=6) == "added"
    return f


def test_the_holds_set_the_arms_error_and_the_strokes_readings_take_its_gradient_over():
    """On the forearm heart (2026-10-23) the gauge holds, read at the skin, showed no gradient across the arm where
    every hover and lift over it read -0.25 mm per mm: a hold weighs hold_m less than a stroke's reading."""
    f = PageFilter()
    for k, u in enumerate(GAUGE):
        f.add(0.1 * k, u, ARM, check=False, rms=0.0008, markers=6)          # the holds: the offset, no gradient
    assert np.linalg.norm(f.predict([[0.0, 0.0]])[0][0] - ARM) < 1e-3
    far = [[0.005, -0.003], [-0.005, 0.003]]
    assert all(np.linalg.norm(f.predict([u])[0][0] - arm(u)) > 1e-3 for u in far)   # the gradient not yet read
    for k, u in enumerate([[0.005, 0.004], [-0.006, 0.002], [0.002, -0.005], [-0.003, -0.004], [0.006, -0.001]] * 2):
        assert f.add(10.0 + 10 * k, u, arm(u), rms=0.0008, markers=5) == "added"
    for u in far:
        assert np.linalg.norm(f.predict([u])[0][0] - arm(u)) < 5e-4


def test_the_overheads_drift_moves_every_stroke_with_no_wrist_reading():
    """A forearm drifting 3 mm over a minute (2026-10-23, the heart's 4.4 mm in 6 min): the overhead's readings carry
    the correction there; the wrist only confirms it."""
    f = held(PageFilter())
    move = np.eye(3)
    for k in range(60):
        move = slid(-0.00005 * k, 0.0)
        f.overhead(1.0 + k, move, 6)
    u = np.array([0.004, 0.002])
    assert np.linalg.norm(f.predict([u])[0][0] - residual(u, move)) < 6e-4


def strokes(f, t0=10.0):
    """Stroke readings across the drawing after the holds: the filter's own once it has them."""
    for k, u in enumerate([[0.005, 0.004], [-0.006, 0.002], [0.002, -0.005], [-0.003, -0.004], [0.006, -0.001]] * 2):
        f.add(t0 + 10 * k, u, arm(u), rms=0.0008, markers=5)
    return f


def test_a_lone_misread_is_held_and_a_second_view_that_agrees_moves_the_sheet():
    f = strokes(held(PageFilter()))
    u, before = np.array([0.003, 0.003]), f.predict([[0.003, 0.003]])[0][0]
    jump = slid(0.006, 0.0)
    assert f.add(110.0, u, residual(u, jump), rms=0.0008, markers=5) == "held"
    assert np.allclose(f.predict([u])[0][0], before)                     # nothing taken from it
    assert f.add(111.0, u + 0.001, residual(u + 0.001, jump), rms=0.0008, markers=5) == "held"   # the same place again
    v = np.array([-0.004, 0.003])
    assert f.add(112.0, v, residual(v, jump), rms=0.0008, markers=5) == "moved"
    assert np.linalg.norm(f.predict([[0.0, 0.0]])[0][0] - residual(np.zeros(2), jump)) < 1e-3


@pytest.mark.parametrize("moving", [False, True])
def test_readings_that_keep_pulling_one_way_move_the_sheet_and_a_stray_one_does_not(moving):
    """A bicep moved 6.5 mm while its stroke drew (2026-10-24): each reading, on one marker, weighs little, so the filter
    took the move a little at a time over some 17 s. wrist_n of them pulling one way past pull_gate open it at once,
    read at stops or while the stroke draws (moving_m leaves the pull to the stop's own noise)."""
    f, jump = strokes(held(PageFilter())), slid(0.0065, 0.0)
    verdicts = []
    for k in range(PageFilter().wrist_n):
        u = np.array([-0.001 + 0.0008 * k, 0.001])   # along a stroke: no two apart_m apart
        verdicts.append(f.add(110.0 + 0.5 * k, u, residual(u, jump), rms=0.0006, markers=1, moving=moving))
    assert "moved" in verdicts                            # within wrist_n readings, not ~30
    assert np.linalg.norm(f.predict([[0.0, 0.0]])[0][0] - residual(np.zeros(2), jump)) < 1e-3
    g, u = strokes(held(PageFilter())), np.array([0.001, 0.001])
    g.add(110.0, u, residual(u, slid(0.0015, 0.0)), rms=0.0006, markers=1, moving=moving)
    for k in range(1, PageFilter().wrist_n):
        v = np.array([0.001 * k, 0.0])
        assert g.add(110.0 + 0.5 * k, v, arm(v), rms=0.0006, markers=1, moving=moving) == "added"


def test_a_reading_while_its_stroke_draws_weighs_moving_m_more_and_is_no_stops():
    """On the still holder of 2026-10-24 the readings taken while strokes drew scattered twice a stop's, and taken as
    stops they put the next stop's reading 0.33 mm off where stops alone put it 0.19: one moves the estimate less than
    a stop's reading alike, and with a further camera it neither joins the stop before it nor starts one."""
    u, jump = np.array([0.003, 0.002]), np.array([0.001, 0.0])
    moved = []
    for moving in (False, True):
        f = strokes(held(PageFilter()))
        assert f.add(110.0, u, arm(u) + jump, rms=0.0008, markers=5, moving=moving) == "added"
        moved.append(np.linalg.norm(f.predict([u])[0][0] - arm(u)))
    assert moved[1] < 0.5 * moved[0]
    same = strokes(held(PageFilter(moving_m=0.0)))
    same.add(110.0, u, arm(u) + jump, rms=0.0008, markers=5, moving=True)
    assert np.isclose(np.linalg.norm(same.predict([u])[0][0] - arm(u)), moved[0])   # moving_m 0: as a stop's
    two = strokes(held(PageFilter(cameras=2)))
    two.add(110.0, u, arm(u), rms=0.0008, markers=5)                                 # a stop: its error a state
    stop, error = two.stop, two.x[CAMERAS:two.views].copy()
    for k in (1, 3):                                                                 # 0.5 and 1.5 s on, along it
        v = u + np.array([0.0017 * k, 0.0])
        assert two.add(110.0 + 0.5 * k, v, arm(v), rms=0.0008, markers=5, moving=True) == "added"
        assert two.stop == stop                                                      # joined none, started none
    assert two.add(112.0, u, arm(u), rms=0.0008, markers=5) == "added"
    assert two.stop[0] == 112.0 and not np.allclose(two.stop[0], stop[0]) and error.shape == (2,)


def test_a_breath_the_overhead_reads_once_moves_the_correction_little():
    f = held(PageFilter())
    for k in range(20):
        f.overhead(1.0 + k, slid(0.0015 if k == 10 else 0.0, 0.0), 6)
    assert np.linalg.norm(f.predict([[0.0, 0.0]])[0][0] - arm([0.0, 0.0])) < 2e-4


def test_a_slide_the_overhead_holds_is_taken_whole_and_a_short_misread_is_not():
    """The 2026-10-23 move test: the mount slid ~22 mm, which the overhead read at once from 3-4 markers a second; weighed
    down as a misread, the filter had taken 74% of it by the wrist's next reading 15 s later."""
    u, jump, n = np.array([0.003, 0.002]), slid(-0.011, 0.019, 0.08), PageFilter().move[0]
    for slide in (False, True):
        f = held(PageFilter())
        for k in range(5):
            f.overhead(1.0 + k, np.eye(3), 6)
        for k in range(n if slide else n - 1):   # one reading short of move_n, then the sheet where it was
            f.overhead(6.0 + k, jump, 4)
        if not slide:
            f.overhead(6.0 + n, np.eye(3), 6)
        want = residual(u, jump) if slide else arm(u)
        assert np.linalg.norm(f.predict([u])[0][0] - want) < (5e-4 if slide else 3e-3)   # the slide: within 0.5 of 22 mm


def test_the_pages_follow_is_carried_and_a_corrected_stroke_lands_on_the_plan():
    """The page followed the sheet (ArmExecutor._followed): on the page in use the reading is the arm's again, and a
    stroke corrected by the filter inks on its planned points."""
    f = strokes(held(PageFilter()))
    move = slid(-0.003, 0.001, 0.01)
    for k in range(120):
        f.overhead(110.0 + k, move, 6)
        if k % 20 == 19:                                                 # the page followed: the readings are on it
            u = np.array([[0.004, 0.002], [-0.004, -0.002], [0.003, -0.004]][k // 40])
            f.add(110.0 + k, u, residual(u, move, move), rms=0.0008, markers=5, follow=move)
    op = {"points_m": [[0.0, 0.0], [0.004, 0.002]]}
    moved, corr = f.snapshot(move).corrected(op)
    for plan, sent in zip(op["points_m"], moved["points_m"], strict=True):
        landed = np.asarray(sent) + residual(np.asarray(sent), move, move)
        assert np.linalg.norm(landed - plan) < 5e-4
    assert np.allclose(np.asarray(op["points_m"]) - corr, moved["points_m"])


def test_a_rebase_keeps_what_the_filter_predicts_on_the_page_in_use():
    f = held(PageFilter())
    for k in range(30):
        f.overhead(1.0 + k, slid(0.002, 0.0), 6)
    follow = slid(0.001, 0.0)
    points = [[0.0, 0.0], [0.005, -0.004]]
    before = f.predict(points, follow)[0]
    f.rebased(follow)
    assert np.allclose(f.predict(points, np.eye(3))[0], before, atol=1e-6)
    assert np.allclose(f.x[:3], 0.0) and np.allclose(f.x[9:], 0.0)


def test_the_tracker_feeds_the_overheads_marker_moves_and_snapshots_on_the_page_in_use(tmp_path):
    """Each accepted reading goes into the filter after the overhead's readings up to it, and a
    stroke's snapshot carries the page's follows then."""
    move, fed = slid(-0.002, 0.0), []
    clock = SimpleNamespace(now=lambda: 100.0)

    def timed_moves(since):
        out = [(float(t), (0.002, 0.0, move, 0.0, 0.0, 6)) for t in range(1, 60) if t > since]
        fed.extend(t for t, _ in out)
        return out
    ex = SimpleNamespace(stack={}, overhead_at=0.0, clock=clock, _timed_moves=timed_moves,
                         _followed=lambda: np.eye(3), run_dir=tmp_path,
                         event=lambda kind, text, op_id="": None)
    tracker = Tracker.__new__(Tracker)
    tracker.ex, tracker.ends, tracker.pending, tracker.rigid = ex, [], [], None
    tracker.field, tracker._fed = held(PageFilter()), 0.0

    def read(tip):
        return {"fit": "markers", "tip_m": [*tip, 0.011], "cam_from_print": np.eye(4).tolist(), "inliers": 5,
                "ids": [0, 4, 7, 8, 26], "score": 1.0, "margin": 1.0, "rms_m": 0.0008, "accepted": True}
    u = np.array([0.003, 0.001])
    assert tracker._log("s0002", "s0002", u, u, 30.0, read(u + residual(u, move)), follow=np.eye(3)) == "added"
    assert tracker._fed == 30.0 and max(fed) == 59.0                     # none past the reading went in
    row = json.loads((tmp_path / "track.jsonl").read_text().splitlines()[-1])
    assert np.allclose(row["residual_m"], residual(u, move))
    snap = tracker.snapshot()
    assert tracker._fed == 59.0 and np.allclose(snap.follow, np.eye(3))
    assert np.linalg.norm(snap.predict([[0.0, 0.0]])[0][0] - residual(np.zeros(2), move)) < 5e-4


def test_the_tracker_corrects_by_the_filter_with_the_stacks_keys(tmp_path):
    """The filter is the tracker's one estimator (track.Field removed 2026-10-24), built with page.track.filter's keys
    about the drawing's centre; stack.yaml names no other."""
    from pathlib import Path

    import yaml

    stack = Path(__file__).resolve().parents[2] / "tatbot_bringup/config/stack.yaml"
    track = yaml.safe_load(stack.read_text())["page"]["track"]
    assert "filter" in track and not {"estimator", "field"} & set(track)
    ex = SimpleNamespace(stack={}, _curved=lambda: False, run_dir=tmp_path, overhead_at=5.0)
    cal = {"tip_cam_m": [0.0, 0.0, 0.15]}
    tracker = Tracker(ex, None, None, cal, {"mode": "correct", "filter": {"hold_m": 0.001}}, centre=(0.002, -0.001))
    assert isinstance(tracker.field, PageFilter) and tracker._fed == 5.0
    assert tracker.field.hold == 0.001 and np.allclose(tracker.field.c, [0.002, -0.001])


def test_a_second_camera_reads_its_own_offset_and_steadies_the_first():
    """A further wrist camera (the lower D405, its mount the URDF's until measured) reads the residual 2 mm off: the
    filter learns that offset where both cameras read and keeps the session camera's estimate on the truth, its sigma
    less than one camera's alone."""
    one, two = held(PageFilter()), held(PageFilter(cameras=2))
    offset = np.array([0.002, -0.001])
    for k, u in enumerate([[0.005, 0.004], [-0.006, 0.002], [0.002, -0.005], [-0.003, -0.004], [0.006, -0.001]] * 3):
        for f in (one, two):
            f.add(10.0 + 10 * k, u, arm(u), rms=0.0008, markers=5)
        two.add(10.2 + 10 * k, u, arm(u) + offset, rms=0.0008, markers=5, camera=1)
    assert np.allclose(two.x[two.views:two.views + 2], offset, atol=3e-4)
    for u in ([0.0, 0.0], [0.004, -0.003]):
        assert np.linalg.norm(two.predict([u])[0][0] - arm(u)) < 3e-4
    assert two.predict([[0.0, 0.0]])[1][0] < one.predict([[0.0, 0.0]])[1][0]


def test_the_filters_views_are_read_beside_the_session_camera_with_their_own_calibration(tmp_path, monkeypatch):
    """page.track.cameras: the filter reads the lower wrist D405 at every stop beside the session's camera, through the
    camera the session's preview has open, its gauge calibration carried from the session camera's through the URDF;
    a view's reading feeds the filter alone (no stroke end, no skin), and a stop's verdict is the best of its views'."""
    import wrist_cameras
    from tatbot_session import gauge

    monkeypatch.setattr(gauge, "CALIB", tmp_path)
    monkeypatch.setattr(wrist_cameras, "mounted_for_arm", lambda repo, arm: [
        {"role": "wrist_upper", "serial": "111"}, {"role": "wrist_lower", "serial": "222"}])
    monkeypatch.setattr(wrist_cameras, "optical_frames", lambda repo, arm, stream: {
        "wrist_upper": "right/realsense_color_optical_frame", "wrist_lower": "right/realsense_lower_color_optical_frame"})
    upper, lower = np.eye(4), np.diag([1.0, -1.0, -1.0, 1.0])   # the lower mount rolled pi, as the URDF has it
    upper[:3, 3] = lower[:3, 3] = [0.1, 0.0, 0.3]
    poses = {"right/realsense_color_optical_frame": upper, "right/realsense_lower_color_optical_frame": lower}
    cam = SimpleNamespace(grab_both=lambda n, fresh=False: ([np.zeros((4, 4, 3), np.uint8)], []),
                          raw_depth=[np.zeros((4, 4), np.uint16)], depth_metadata={"m": 1}, stamp_ns=None)
    seen = SimpleNamespace(**vars(cam), serial="222")
    seen.stamp_ns = 77                                 # its tap published the frame it grabbed
    cam.preview = SimpleNamespace(cameras=[seen])
    ex = SimpleNamespace(stack={}, _curved=lambda: False, run_dir=tmp_path, overhead_at=0.0, repo=tmp_path, arm="right",
                         io=SimpleNamespace(q=np.zeros(7)), kin=SimpleNamespace(frame=lambda q, f: poses[f]))
    cal = {"tip_cam_m": [0.0, 0.01, 0.15], "cone_end_cam_m": [0.0, 0.01, 0.14], "axis_cam": [0.0, 0.0, 1.0],
           "normal_cam": [0.0, 0.0, -1.0], "source": "measured"}
    tracker = Tracker(ex, cam, None, cal, {"mode": "correct", "cameras": ["wrist_lower"]})
    view = tracker.views[0]
    assert view["serial"] == "222" and tracker.field.n == 16   # + the stop's error and the view's offset
    assert np.allclose(view["cal"]["tip_cam_m"], [0.0, -0.01, -0.15]) and np.allclose(view["cal"]["axis_cam"], [0, 0, -1])
    submitted, marked = [], []
    monkeypatch.setattr(tracker, "_submit", lambda name, *a, camera=0, **kw: submitted.append((name, camera))
                        or marked.append((kw.get("stamp_ns"), kw.get("shown"))))
    tracker._grab("s0002-hover", "s0002", np.zeros(2))
    assert submitted == [("s0002-hover", 0), ("s0002-hover-wrist_lower", 1)]
    assert marked[1] == (77, seen)                     # its markers go to its own pane, at its frame's stamp
    cam.preview.cameras = []                                             # the preview not open yet: the view waits
    submitted.clear()
    tracker._grab("s0003-hover", "s0003", np.zeros(2))
    assert submitted == [("s0003-hover", 0)]
    taken = []
    monkeypatch.setattr(tracker, "_sample", lambda *a: taken.append(a[-1]) or "added")
    tracker.cam, tracker.ends, tracker.pending, tracker.print_held, tracker.printed = cam, [], [], [], []
    tracker.ex = SimpleNamespace(**vars(ex), event=lambda kind, text, op_id="": None)
    row = {"fit": "markers", "tip_m": [0.001, 0.0, 0.011], "inliers": 5, "rms_m": 0.0008, "accepted": True,
           "skin_base": [[0.0, 0.0, 0.0]]}
    u = np.zeros(2)
    assert tracker._log("s0002-wrist_lower", "s0002", u, u, 1.0, dict(row), camera=1) == "added"
    assert taken == [1] and tracker.ends == [] and tracker.skin is None
    logged = json.loads((tmp_path / "track.jsonl").read_text().splitlines()[-1])
    assert logged["camera"] == "wrist_lower"
    future = lambda: SimpleNamespace(done=lambda: True, result=lambda: {})   # noqa: E731
    verdicts = iter([None, "added"])
    monkeypatch.setattr(tracker, "_log", lambda *a, **k: next(verdicts))
    tracker.pending = [("s", "", None, u, 0.0, None, None, 0, future()), ("s-l", "", None, u, 0.0, None, None, 1, future())]
    assert tracker.drain(wait=True, best=True) == "added"


def test_a_stops_second_camera_does_not_count_that_stop_twice():
    """Two cameras reading one stop alike share its error (view_sd_m apart): the stop moves the estimate about as one
    reading does, not twice as far, as two independent readings would."""
    one, two = held(PageFilter()), held(PageFilter(cameras=2))
    u, jump = np.array([0.003, 0.002]), np.array([0.0015, 0.0])
    one.add(10.0, u, arm(u) + jump, rms=0.0008, markers=5)
    two.add(10.0, u, arm(u) + jump, rms=0.0008, markers=5)
    two.add(10.2, u, arm(u) + jump, rms=0.0008, markers=5, camera=1)
    moved = [np.linalg.norm(f.predict([u])[0][0] - arm(u)) for f in (one, two)]
    assert moved[1] < 1.2 * moved[0]


def test_a_stroke_is_read_while_it_draws_every_stroke_s_for_the_filter_alone(tmp_path):
    """page.track.stroke_s: while a stroke draws the tracker grabs a frame every stroke_s off the executor's thread, the
    joints as the frame arrived, one grab at a time; the workers read it like a stop's, but it keeps no skin and is no
    stroke end, and the stroke's readings are said in one line after its lift."""
    import threading

    now, release, events = [10.0], threading.Event(), []
    clock = SimpleNamespace(now=lambda: now[0], utc=lambda: 1e9 + now[0])

    class Cam:
        raw_depth, depth_metadata, stamp_ns, capture_timestamps = [np.zeros((4, 4), np.uint16)], {"m": 1}, None, []

        def grab_both(self, n, observe=None, fresh=False):
            release.wait(5.0)
            self.capture_timestamps = [{"joints": observe()}]
            return [np.zeros((4, 4, 3), np.uint8)], []
    ex = SimpleNamespace(stack={}, _curved=lambda: False, run_dir=tmp_path, overhead_at=0.0, clock=clock,
                         io=SimpleNamespace(q=np.arange(7.0)), event=lambda kind, text, op_id="": events.append(text),
                         _followed=lambda: np.eye(3), _timed_moves=lambda since: [])
    tracker = Tracker(ex, Cam(), None, {"tip_cam_m": [0.0, 0.0, 0.15]},
                      {"mode": "correct", "stroke_s": 0.5})
    submitted = []
    tracker._submit = lambda name, end, bgr, depth, meta, q, **kw: submitted.append((name, end, q, kw))
    op = {"id": "s0005", "points_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]}
    tracker.while_drawing(op, 0.004)
    now[0] = 10.6
    tracker.while_drawing(op, 0.006)                   # the first still out: no second
    assert len(tracker._grabs) == 1
    release.set()
    tracker._take(wait=True)
    name, end, q, kw = submitted[0]
    assert name == "s0005-draw-1" and np.allclose(end, [0.004, 0.0]) and np.allclose(q, np.arange(7.0))
    assert kw["keep"] is False and kw["op_id"] == "s0005" and kw["at"] == (10.6, 1e9 + 10.6)   # when it arrived
    tracker.while_drawing(op, 0.007)                   # 0.6 s since the last began
    now[0] = 10.7
    tracker._take(wait=True)
    tracker.while_drawing(op, 0.008)                   # 0.1 s since: none
    assert len(submitted) == 2 and not tracker._grabs
    row = {"fit": "markers", "tip_m": [0.0045, 0.0003, 0.011], "inliers": 5, "rms_m": 0.0008, "accepted": True,
           "moving": True, "utc": 1e9, "skin_base": [[0.0, 0.0, 0.0]]}   # 0.3 mm across the line, 0.5 along it
    u = np.array([0.004, 0.0])
    assert tracker._log("s0005-draw-1", "s0005", u, u, 10.6, dict(row), follow=np.eye(3)) == "added"
    assert events == [] and tracker.ends == [] and tracker.skin is None and len(tracker.field.samples) == 1
    tracker._grab = lambda *a: None
    tracker.after_lift(op)
    assert events == ["track s0005: 1 readings while it drew, 1 taken, the tip 0.30 mm off the plan's line (median)"]
    tracker._grabber.shutdown(wait=True)
