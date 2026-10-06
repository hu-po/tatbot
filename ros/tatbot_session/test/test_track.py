"""track.Field: the print's residual field from tip-on-print samples (ros/README.md 4.4, stencil tracking)."""
import numpy as np
from tatbot_session.track import Field, Tracker

GAUGE = [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02], [0.0, 0.0]]


def test_an_empty_field_corrects_nothing():
    corr, weight = Field().predict([[0.01, 0.02]])
    assert np.allclose(corr, 0) and np.allclose(weight, 0)


def test_one_sample_moves_the_whole_field():
    f = Field()
    f.add(0.0, [0.0, 0.0], [0.002, -0.001])
    near, far = f.predict([[0.0, 0.0], [0.08, 0.0]])[0]
    assert np.allclose(near, [0.002, -0.001], atol=1e-4) and np.allclose(far, [0.002, -0.001], atol=1e-4)


def test_a_newer_sample_outweighs_older_ones_and_a_strokes_correction_is_capped():
    f = Field(tau_s=60.0, cap_m=0.003)
    f.add(0.0, [0.0, 0.0], [0.001, 0.0])
    f.add(600.0, [0.0, 0.0], [0.010, 0.0], check=False)
    assert np.allclose(f.predict([[0.0, 0.0]])[0], [0.010, 0.0], atol=1e-4)      # the field is not capped
    _, corr = f.snapshot().corrected({"points_m": [[0.0, 0.0]]})
    assert np.isclose(np.linalg.norm(corr), 0.003)                             # a stroke's move is


def test_a_slide_and_a_turn_are_carried_past_the_samples():
    """Five gauge holds of a sheet slid 13 mm and turned 7 deg give the field 25 mm beyond them."""
    c, s = np.cos(np.radians(7.0)), np.sin(np.radians(7.0))
    rot, slide = np.array([[c, -s], [s, c]]), np.array([0.013, 0.002])

    def truth(u):
        return (rot - np.eye(2)) @ u + slide
    f = Field()
    for u in GAUGE:
        f.add(0.0, u, truth(np.asarray(u)), check=False)                 # the goal's gauge holds
    assert f.add(1.0, [0.01, 0.03], truth(np.array([0.01, 0.03]))) == "added"   # a lift on the turned sheet
    for p in ([0.0, 0.035], [-0.02, 0.035], [0.02, -0.035], [0.0, -0.045]):
        assert np.linalg.norm(f.predict([p])[0][0] - truth(np.asarray(p))) < 2e-4


def test_the_arms_local_error_is_followed_near_its_samples():
    f = Field(sigma_m=0.01)
    for u in GAUGE:
        f.add(0.0, u, [0.001, 0.0], check=False)
    f.add(1.0, [0.03, 0.03], [0.003, 0.0])                                  # a local bump the plane misses
    assert f.predict([[0.03, 0.03]])[0][0][0] > 0.0025
    assert abs(f.predict([[-0.02, -0.02]])[0][0][0] - 0.001) < 3e-4


def test_a_jump_is_held_until_a_second_sample_agrees_then_the_field_starts_again():
    f = Field()
    f.add(0.0, [0.0, 0.0], [0.001, 0.0])
    assert f.add(1.0, [0.001, 0.0], [0.006, 0.0]) == "held"          # one far sample: a misfit or a move
    assert np.allclose(f.predict([[0.0, 0.0]])[0], [0.001, 0.0], atol=1e-4)
    assert f.add(2.0, [0.002, 0.0], [0.0062, 0.0]) == "moved"        # a second agrees: the sheet moved
    assert np.allclose(f.predict([[0.0, 0.0]])[0], [0.0061, 0.0], atol=3e-4)
    assert f.add(3.0, [0.0, 0.0], [0.0061, 0.0]) == "added"


def test_a_lone_misfit_is_dropped_when_the_next_sample_agrees_with_the_field():
    f = Field()
    f.add(0.0, [0.0, 0.0], [0.001, 0.0])
    assert f.add(1.0, [0.0, 0.0], [-0.0055, 0.0]) == "held"
    assert f.add(2.0, [0.0, 0.0], [0.0011, 0.0]) == "added"
    assert f.held is None and len(f.samples) == 2



def _turned(deg, slide, centre=(0.0, 0.0)):
    """The residual of a sheet turned deg about centre and slid: where the pen inks less where it was meant to."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    rot = np.array([[c, -s], [s, c]])
    return lambda u: (rot - np.eye(2)) @ (np.asarray(u) - centre) + slide


def test_a_turned_sheets_move_is_confirmed_by_a_sample_across_the_page():
    """Two samples of one turn disagree by the turn times their distance, and still make one rigid move."""
    f = Field()
    for u in GAUGE:
        f.add(0.0, u, [0.0, 0.0], check=False)
    truth = _turned(15.0, [0.006, -0.004], centre=(0.01, 0.0))
    assert f.add(1.0, [0.0, -0.01], truth([0.0, -0.01])) == "held"
    assert f.add(2.0, [0.018, 0.0], truth([0.018, 0.0])) == "moved"
    assert np.allclose(f.predict([[0.0, -0.01]])[0][0], truth([0.0, -0.01]), atol=2e-4)


def test_a_lattice_misfit_does_not_confirm_a_move():
    f = Field()
    for u in GAUGE:
        f.add(0.0, u, [0.0, 0.0], check=False)
    assert f.add(1.0, [0.0, 0.0], [0.004, 0.0]) == "held"                 # a real 4 mm slide
    assert f.add(2.0, [0.005, 0.0], [0.004, 0.0065]) == "held"            # a second view a lattice row off
    assert len(f.samples) == 5


def test_two_samples_across_a_turned_sheet_carry_its_turn():
    f = Field()
    truth = _turned(10.0, [0.002, 0.001])
    f.add(0.0, [0.0, 0.0], truth([0.0, 0.0]), check=False)
    f.add(0.0, [0.02, 0.0], truth([0.02, 0.0]), check=False)
    assert np.allclose(f.predict([[0.0, 0.02]])[0][0], truth([0.0, 0.02]), atol=1e-4)

def test_a_sample_that_agrees_with_an_uncapped_field_is_not_held():
    f = Field(cap_m=0.005)
    f.add(0.0, [0.0, 0.0], [0.019, 0.0])
    assert f.add(1.0, [0.001, 0.0], [0.0192, 0.0]) == "added"


def test_a_snapshot_is_not_moved_by_later_samples_and_moves_an_op_by_minus_r():
    f = Field()
    f.add(0.0, [0.0, 0.0], [0.001, 0.002])
    snap = f.snapshot()
    f.add(1.0, [0.0, 0.0], [0.0012, 0.0021])
    op = {"id": "s0", "points_m": [[0.0, 0.0], [0.001, 0.0]]}
    moved, corr = snap.corrected(op)
    assert np.allclose(np.asarray(moved["points_m"]), np.asarray(op["points_m"]) - corr)
    assert len(snap.samples) == 1 and op["points_m"][0] == [0.0, 0.0]


def test_the_tip_offset_turns_with_the_camera_not_the_sheet():
    """The offset rides with the pen: a sheet turned under the camera sees it turned the other way."""
    tracker = Tracker.__new__(Tracker)
    tracker.offsets = [(0.011, np.array([0.003, 0.0])), (0.0032, np.array([0.004, 0.0]))]

    def cam_from_print(yaw):                       # the camera's x axis at yaw in the print
        print_from_cam = np.eye(4)
        print_from_cam[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        return np.linalg.inv(print_from_cam)
    assert np.allclose(tracker._offset(0.011, cam_from_print(0.0)), [0.003, 0.0])
    assert np.allclose(tracker._offset(0.011, cam_from_print(np.pi / 2)), [0.0, 0.003])
    assert np.allclose(tracker._offset(0.0032, cam_from_print(np.pi)), [-0.004, 0.0])


def test_a_fit_the_runs_page_cannot_find_is_tried_from_the_overheads(monkeypatch):
    """A sheet slid 50 mm is past the wrist fit's reach from the run's page; the overhead saw where it went."""
    import sys
    from types import SimpleNamespace

    from tatbot_session import track

    def measure(bgr, depth, meta, base_from_camera, base_from_page, lay, tip_cam):
        found = base_from_page[1, 3] > 0.01
        return {"fit": True, "margin": 120 if found else 16, "accepted": found}
    monkeypatch.setitem(sys.modules, "stencil_tip", SimpleNamespace(measure=measure))
    moved = np.eye(4)
    moved[1, 3] = 0.05
    job = {"bgr": None, "depth": None, "meta": {}, "q": None, "used": np.eye(4), "tip_cam": None,
           "base_from_camera": np.eye(4)}
    assert not track._measure({**job, "overhead": None})["accepted"]
    out = track._measure({**job, "overhead": moved})
    assert out["accepted"] and out["prior"] == "overhead"


def test_the_overheads_prior_is_the_runs_page_moved_as_the_sheet_moved():
    from types import SimpleNamespace

    tracker = Tracker.__new__(Tracker)
    used, newest = np.eye(4), np.eye(4)
    used[:3, 3] = [0.3, 0.0, 0.002]
    newest[:3, 3] = [0.0, 0.047, 0.0]
    tracker.ex = SimpleNamespace(arm="right", camera=np.eye(4), stack={"page": {"max_lost_s": 5.0}},
                                 node=SimpleNamespace(camera_page=lambda arm: (newest, {"source": "measured"})))
    np.testing.assert_allclose(tracker._overhead(used)[:3, 3], [0.3, 0.047, 0.002])
    newest[:3, 3] = [0.0, 0.001, 0.0]                                    # under 2 mm: the same prior
    assert tracker._overhead(used) is None
    newest[:3, 3] = [0.0, 0.047, 0.0]
    tracker.ex.node.camera_page = lambda arm: (newest, {"source": "lost", "measured_age_s": 0.4})
    assert tracker._overhead(used) is not None                           # hidden by the arm a moment ago: still fresh
