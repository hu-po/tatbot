"""Page plane from touches, trim, drift and re-plan decisions, arc bookkeeping."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_session import geometry as g


def test_fit_plane_exact_and_oriented():
    pts = [[0.3, -0.2, 0.03], [0.34, -0.2, 0.031], [0.32, -0.16, 0.029]]
    n, d = g.fit_plane(pts, [0, 0, 1])
    assert n[2] > 0 and abs(np.linalg.norm(n) - 1) < 1e-12
    for p in pts:
        assert abs(n @ p + d) < 1e-12
    n2, d2 = g.fit_plane(pts, [0, 0, -1])
    assert np.allclose(n2, -n) and math.isclose(d2, -d)
    with pytest.raises(ValueError):
        g.fit_plane(pts[:2], [0, 0, 1])


def test_touched_page_takes_xy_yaw_from_camera_and_height_tilt_from_touches():
    camera = g.rpy_matrix([0.371, -0.231, 0.045], [0.0, 0.0, 0.3])  # camera says 45 mm, level
    tilt = g.rpy_matrix([0, 0, 0], [0.02, -0.01, 0.0])
    normal = tilt[:3, 2]
    d = -normal @ np.array([0.371, -0.231, 0.0334])  # the touched paper passes 33.4 mm under the centre
    page = g.touched_page(camera, normal, d)
    assert np.allclose(page[:3, 2], normal)
    assert abs(normal @ page[:3, 3] + d) < 1e-12  # origin on the touched plane
    assert np.allclose(page[:2, 3], camera[:2, 3], atol=1e-3)  # x, y from the camera (projection along n)
    assert np.allclose(page[:3, :3].T @ page[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(page[:3, :3]) > 0
    # yaw: page x is the camera x projected onto the plane
    x_proj = camera[:3, 0] - (camera[:3, 0] @ normal) * normal
    assert np.allclose(page[:3, 0], x_proj / np.linalg.norm(x_proj))


def test_page_used_applies_correction_then_trim_in_page_frame():
    camera = g.rpy_matrix([0.37, -0.23, 0.04], [0, 0, math.pi / 2])
    touched = camera.copy()
    touched[2, 3] = 0.0334
    correction = np.linalg.inv(camera) @ touched
    used = g.page_used(camera, correction, [0.001, -0.002])
    # trim is in the page frame: page x is base +y after the 90 degree yaw
    assert np.allclose(used[:3, 3], [0.37 + 0.002, -0.23 + 0.001, 0.0334])
    # the correction rides with the page: a moved page keeps the touched height offset
    moved = g.rpy_matrix([0.38, -0.23, 0.04], [0, 0, math.pi / 2])
    assert np.allclose(g.page_used(moved, correction, [0, 0])[:3, 3], [0.38, -0.23, 0.0334])


def test_page_moved_thresholds():
    a = g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0])
    assert not g.page_moved(a, g.rpy_matrix([0.3709, -0.23, 0.03], [0, 0, 0]), 0.001, 0.0087)
    assert g.page_moved(a, g.rpy_matrix([0.3711, -0.23, 0.03], [0, 0, 0]), 0.001, 0.0087)
    assert not g.page_moved(a, g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0.0086]), 0.001, 0.0087)
    assert g.page_moved(a, g.rpy_matrix([0.37, -0.23, 0.03], [0, 0, 0.0088]), 0.001, 0.0087)


def test_touch_points_are_a_triangle_about_the_page_centre():
    pts = g.touch_points(0.012)
    assert np.allclose(pts, [[-0.012, -0.012], [0.012, -0.012], [0.0, 0.012]])
    assert np.all(np.abs(pts) <= 0.031 - 0.005)  # well inside the 62 x 112 mm clear centre


def test_arc_at_latch_uses_the_measured_tip_and_falls_back_to_the_start():
    arc = np.array([np.nan, np.nan, 0.010, 0.011, 0.012, 0.013, np.nan])
    tip = np.array([[0, 0, 0.01], [0, 0, 0.005], [0.010, 0, 0], [0.011, 0, 0], [0.012, 0, 0], [0.013, 0, 0],
                    [0.013, 0, 0.01]])
    traj = SimpleNamespace(arc_m=arc, tip=tip)
    assert g.arc_at(traj, 1) == 0.010  # nothing drawn yet: resume where this plan starts
    assert g.arc_at(traj, 4) == 0.012
    assert g.arc_at(traj, 5, tip_measured=[0.0111, 0, 0]) == 0.011  # measured lags the command
    assert g.arc_at(traj, 6) == 0.013  # latched in the lift: the whole stroke was drawn


def test_stroke_length_closes_closed_strokes():
    pts = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01]]
    assert g.stroke_length(pts) == pytest.approx(0.02)
    assert g.stroke_length(pts, closed=True) == pytest.approx(0.02 + math.hypot(0.01, 0.01))
    assert g.stroke_length([[0.0, 0.0]]) == 0.0


def test_tool_down_pose_points_the_tool_into_the_page_without_wrist_spin():
    page = g.rpy_matrix([0.37, -0.23, 0.03], [0.05, 0.0, 0.4])
    current = g.rpy_matrix([0.3, 0, 0.1], [math.pi, 0, 0.2])
    pose = g.tool_down_pose(page, [0.01, 0.02], 0.01, current)
    assert np.allclose(pose[:3, 2], -page[:3, 2])
    assert np.allclose(pose[:3, 3], (page @ [0.01, 0.02, 0.01, 1])[:3])
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    assert pose[:3, 0] @ current[:3, 0] > 0.99


def test_first_contact_extrapolates_the_force_ramp_to_its_start():
    # A hold and fast leg at -7 N, then a 0.5 mm/s descent from 6 mm above the page to 1.7 mm into a
    # 1.5 N/mm give with the in-air force wandering around -1 N (bench 2026-09-26).
    rng = np.random.default_rng(1)
    hold_h, hold_f = np.full(200, 0.020), np.full(200, -7.0)
    h = np.linspace(0.006, -0.0017, 800)
    f = -1.0 + 0.3 * np.sin(h * 900.0) * (h > 0) + 1500.0 * np.clip(-h, 0, None) + rng.normal(0, 0.05, h.size)
    fit = g.first_contact(np.concatenate([hold_h, h]), np.concatenate([hold_f, f]))
    assert fit is not None and fit["method"] == "fit" and abs(fit["offset_m"] - 0.0017) < 0.0002
    assert abs(fit["stiffness_n_m"] - 1500.0) < 150.0 and abs(fit["base_n"] + 1.0) < 0.3
    # A noisy ramp whose fit is implausible falls back to the give model: rise / 1500 N/m.
    wobbly = f + 1.5 * np.sin(h * 4000.0) * (h < 0)
    model = g.first_contact(h, wobbly)
    assert model is not None and abs(model["offset_m"] - model["rise_n"] / 1500.0) < 1e-12 or model["method"] == "fit"
    assert g.first_contact(h[:30], f[:30]) is None                  # too few samples
    assert g.first_contact(h, np.full(h.size, -1.0)) is None        # no rise: the trip pose stands


def test_fiducial_contact_time_finds_where_the_gap_opens():
    # 16.5 Hz fiducials over 20 s; the EE stops at t = 12 s while the encoders descend 0.2 mm/s.
    rng = np.random.default_rng(5)
    t = np.arange(0, 20, 1 / 16.5)
    gap = 0.004 + 0.0002 * np.clip(t - 12.0, 0, None) + rng.normal(0, 0.0003, t.size)
    fit = g.fiducial_contact_time(t, gap)
    assert fit is not None and abs(fit["t_c"] - 12.0) < 1.0 and abs(fit["rate_m_s"] - 0.0002) < 0.00006
    assert g.fiducial_contact_time(t, 0.004 + rng.normal(0, 0.0003, t.size)) is None   # no contact: flat
    assert g.fiducial_contact_time(t[:16], gap[:16]) is None   # twice min_after: no hinge to try, no crash


def test_a_lost_page_is_stale_past_max_lost_s_or_when_never_measured():
    assert g.measured_age(None, 100.0) is None
    assert g.measured_age(90.0, 100.0) == 10.0
    assert g.stale_page({"source": "measured", "measured_age_s": 0.0}, 5.0) is None
    assert g.stale_page({"source": "fixed", "measured_age_s": None}, 5.0) is None
    assert g.stale_page({"source": "lost", "measured_age_s": 1.2}, 5.0) is None   # a flicker stands
    assert "lost for 3600 s" in g.stale_page({"source": "lost", "measured_age_s": 3600.0}, 5.0)
    assert "since the stack started" in g.stale_page({"source": "lost", "measured_age_s": None}, 5.0)


def test_touches_sit_about_the_drawing_and_inside_a_small_one():
    program = {"ops": [{"op": "stroke", "points_m": [[0.002, 0.029], [0.028, 0.055]]}]}   # 26 mm, centre (15, 42)
    pts = g.touch_layout(program, 0.012)
    assert np.allclose(pts.mean(axis=0), [0.015, 0.042 - 0.0104 / 3])   # spread 0.8 x the 13 mm half-size
    assert np.all(pts >= [0.002, 0.029]) and np.all(pts <= [0.028, 0.055])
    tiny = {"ops": [{"op": "stroke", "points_m": [[0.0, 0.0], [0.010, 0.010]]}]}
    assert np.ptp(g.touch_layout(tiny, 0.012)[:, 0]) == pytest.approx(0.008)   # 0.8 of the 5 mm half-size
    line = {"ops": [{"op": "stroke", "points_m": [[-0.01, 0.0], [0.01, 0.0]]}]}
    assert np.ptp(g.touch_layout(line, 0.012)[:, 1]) == pytest.approx(0.008)   # a line: the 4 mm floor
    assert np.allclose(g.touch_layout({"ops": []}, 0.012), g.touch_points(0.012))


def test_a_drawing_must_stay_inside_the_prints_clear_centre():
    """The session refuses, before the arm moves, a program whose lines (their width included) leave the clear
    centre of the print it draws on, however the program's own page was sized."""
    half = {"size_m": [0.083, 0.127], "clear_m": [0.0432, 0.0878], "geometry": "print stencil-half"}

    def program(half_width_m, line_width_m=0.0005):
        square = [[-half_width_m, -half_width_m], [half_width_m, -half_width_m], [half_width_m, half_width_m]]
        return {"resources": [{"id": "pen", "tool": {"line_width_m": line_width_m}}],
                "ops": [{"op": "tool_change", "id": "t", "resource_id": "pen"},
                        {"op": "stroke", "id": "s", "resource_id": "pen", "points_m": square,
                         "generation_width_m": 0.0003}]}

    assert g.off_print(program(0.020), half) is None
    assert g.off_print({"ops": []}, half) is None
    why = g.off_print(program(0.025), half)
    assert why and "x -25.2 to 25.2" in why and "x -21.6 to 21.6" in why and "83 x 127 mm" in why
    assert g.off_print(program(0.0213, line_width_m=0.0004), half) is None
    assert g.off_print(program(0.0213, line_width_m=0.0008), half)   # the pen's own line reaches the border
    assert g.off_print(program(0.025), {"size_m": [0.100, 0.150], "clear_m": [0.062, 0.112]}) is None


def test_the_gauge_holds_keep_the_prints_own_inset():
    """geometry.hold_box: the clear centre less the print's inset_m, HOLD_INSET_M (5 mm) without one."""
    assert g.hold_box([0.032, 0.050]) == pytest.approx((0.011, 0.020))
    assert g.hold_box([0.032, 0.050], 0.0025) == pytest.approx((0.0135, 0.0225))
    holds = g.gauge_layout(None, 0.02, [0.032, 0.050], 0.0025)
    assert pytest.approx(np.abs(holds).max(axis=0).tolist()) == [0.0135, 0.02]
    assert np.allclose(g.gauge_layout(None, 0.02, [0.032, 0.050]), g.gauge_layout(None, 0.02, [0.032, 0.050], None))


def _forearm_tile():
    """The forearm print's 3 x 4 grid of 12 mm markers (16.5 mm pitch), id -> TL/TR/BR/BL page metres."""
    ids = (8, 22, 12, 7, 15, 23, 4, 0, 20, 26, 27, 21)
    return {i: np.array([[x, y], [x + 0.012, y], [x + 0.012, y - 0.012], [x, y - 0.012]])
            for i, (x, y) in zip(ids, [(-0.0225 + 0.0165 * c, 0.03075 - 0.0165 * r) for r in range(4) for c in range(3)],
                                  strict=True)}


SEEN = g.rpy_matrix([0, 0, 0.67], [np.radians(218.0), 0, 0])   # the overhead's frame from _support's page


def _support(printed, ids, radius, along, *, du=0.0, dv=0.0, turn_deg=0.0, relift=None, lift=0.0):
    """A Page's support JSON: the print wrapped on a cylinder of `radius` round print axis `along` (u: the limb runs
    across the print; v: along it), seen 38 deg off the camera's axis at 0.67 m, slid (du, dv) on the skin,
    turned about its centre and lifted along the page's normal; `relift` scales each anchor point along its ray (a
    plane's lift, not a move)."""
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    points = []
    for i in ids:
        u, v = (printed[i] @ np.array([[c, -s], [s, c]]).T + [du, dv]).T
        arc, flat = (v, u) if along == "u" else (u, v)
        local = np.c_[flat, radius * np.sin(arc / radius), radius * (np.cos(arc / radius) - 1) + lift]
        if along == "v":
            local = local[:, [1, 0, 2]]
        world = (SEEN @ np.c_[local, np.ones(4)].T).T[:, :3]
        points.append(world * (1.0 if relift is None else relift(len(world))[:, None]))
    return json.dumps({"markers": {"ids": list(ids)}, "anchor_points_m": np.concatenate(points).tolist()})


# Pairwise cases cover every radius/direction, radius/marker-set and direction/marker-set combination.
@pytest.mark.parametrize("radius, along, ids", [
    (0.040, "u", (8, 22)), (0.168, "v", (8, 22)),
    (0.040, "v", (7, 8, 22)), (0.168, "u", (7, 8, 22)),
    (0.040, "u", (12, 22, 23)), (0.168, "v", (12, 22, 23))])
def test_marks_moved_reads_a_curved_skin_by_its_markers_and_no_depth(radius, along, ids):
    """A forearm (r 40 mm) and the silicone mount (r 168 mm), the limb across or along the print, the two or three
    markers the overhead reads past the arm: a lift of every corner along its ray (a plane over the sheet tilting
    14 deg) reads no move; a 6 mm slide on the skin reads 6 mm, a 5 deg turn 5 deg."""
    printed, rng = _forearm_tile(), np.random.default_rng(0)
    ref = g.page_marks(_support(printed, ids, radius, along))
    lifted = g.page_marks(_support(printed, ids, radius, along, relift=lambda n: rng.uniform(0.99, 1.01, n)))
    slide, turn, move, _, bend, _ = g.marks_moved(ref, lifted, printed)
    assert np.allclose([slide, turn], 0.0, atol=1e-9) and np.allclose(move, np.eye(3), atol=1e-9)
    assert np.isnan(bend) if len(ids) < 3 else bend < 1e-9   # under three markers no bend is read
    for du, dv in ((0.006, 0.0), (0.0, 0.006)):
        slide, _, move, *_ = g.marks_moved(ref, g.page_marks(_support(printed, ids, radius, along, du=du, dv=dv)), printed)
        assert slide == pytest.approx(0.006, abs=0.0003)
        c = np.mean([printed[i].mean(axis=0) for i in ids], axis=0)   # the move itself at the markers, print axes
        assert np.allclose(move[:2, :2] @ c + move[:2, 2] - c, [du, dv], atol=0.0005)
    _, turn, *_ = g.marks_moved(ref, g.page_marks(_support(printed, ids, radius, along, turn_deg=5.0)), printed)
    assert 4.3 < np.degrees(turn) < 5.3   # a marker sliding round a 40 mm limb tilts toward the camera: reads ~10% low


def test_marks_moved_reads_the_rise_along_the_pages_normal_and_a_slide_as_none():
    """A forearm lifted 30 mm along the page's normal reads +30 mm of rise along it, and 0.4-0.8 of the lift as a slide
    across the rays; a 15 mm slide along the limb changes the corners' ranges several mm on the overhead's oblique view
    (4-6.5 mm replayed on the rig), but reads no rise along the normal. Without the normal the rise is the ranges' (none for a record kept with rays
    alone). Straight toward the overhead 60 mm moves no ray: the ranges read it."""
    printed, up = _forearm_tile(), SEEN[:3, 2]
    ref = g.page_marks(_support(printed, (7, 8, 22), 0.040, "v"))
    slide, _, _, rise, *_ = g.marks_moved(ref, g.page_marks(_support(printed, (7, 8, 22), 0.040, "v", lift=0.030)),
                                         printed, up)
    assert rise == pytest.approx(0.030, abs=1e-9) and 0.012 < slide < 0.024
    slid = g.page_marks(_support(printed, (7, 8, 22), 0.040, "v", dv=0.015))
    assert abs(g.marks_moved(ref, slid, printed, up)[3]) < 1e-9 and abs(g.marks_moved(ref, slid, printed)[3]) > 0.004
    assert g.marks_moved({i: c[:, :2] for i, c in ref.items()}, ref, printed, up)[3] == 0.0
    support = json.loads(_support(printed, (7, 8, 22), 0.040, "v"))
    points = np.asarray(support["anchor_points_m"], float)
    support["anchor_points_m"] = (points * (1.0 - 0.060 / np.linalg.norm(points, axis=1))[:, None]).tolist()
    slide, _, _, rise, *_ = g.marks_moved(ref, g.page_marks(json.dumps(support)), printed)
    assert slide < 1e-9 and rise == pytest.approx(0.060)


def test_page_marks_takes_rays_and_leaves_out_what_it_cannot_pair():
    printed = _forearm_tile()
    marks = g.page_marks(_support(printed, (8, 22), 0.168, "u"))
    assert sorted(marks) == [8, 22] and marks[8].shape == (4, 3)
    twice = json.loads(_support(printed, (8, 22), 0.168, "u"))
    twice["markers"]["ids"] = [22, 22]
    assert g.page_marks(json.dumps(twice)) == {}
    assert g.page_marks("") == {} and g.page_marks('{"markers": {"ids": [1]}}') == {}
    one = g.page_marks(_support(printed, (22, 23), 0.168, "u"))
    assert g.marks_moved(marks, one, printed) is None   # one marker in common is no reading


# --- the overhead's watch of a person's skin by its markers (geometry.sheet_watch) ----------------------------------
DRAWING = np.array([[-0.013, -0.012], [0.013, -0.012], [0.013, 0.012], [-0.013, 0.012], [0.0, 0.0]])   # 26 x 24 mm
TOTAL = (0.010, 0.05)


def _move(dx, dy=0.0, rise=0.0, bend=np.nan):
    """A marks_moved reading of a sheet slid (dx, dy) from its setup."""
    m = np.eye(3)
    m[:2, 2] = [dx, dy]
    return float(np.hypot(dx, dy)), 0.0, m, rise, bend


def _person(seed, n=300, sway=0.0006, drift=0.0015 / 60):
    """A forearm read at 1 Hz as on 2026-10-19/20: a slow sway (per axis, ~20 s to forget), breathing 0.3 mm at
    0.3 Hz and a drift one way. [(t, reading)]."""
    rng, keep = np.random.default_rng(seed), np.exp(-1 / 20.0)
    x = np.zeros((n, 2))
    for k in range(1, n):
        x[k] = keep * x[k - 1] + np.sqrt(1 - keep ** 2) * sway * rng.standard_normal(2)
    x[:, 0] += drift * np.arange(n) + 0.0003 * np.sin(2 * np.pi * 0.3 * np.arange(n))
    return [(float(k), _move(*x[k], rise=0.0002 * rng.standard_normal())) for k in range(n)]


def _first_fire(timed, asked):
    """When sheet_watch first fires as the stroke loop asks it (at times `asked`, on its node's two-minute history),
    the page following the follow_s median past follow_m whenever it reads no move (ArmExecutor._follow_marks), and
    why; (None, verdicts seen)."""
    seen, followed = set(), np.eye(3)
    for t in asked:
        history = [r for r in timed if t - 120 < r[0] <= t]
        verdict, why = g.sheet_watch(history, DRAWING, followed, TOTAL)
        if verdict == "fire":
            return t, why
        seen.add(verdict)
        now = g.median_move(g.window(history, g.WATCH["follow_s"]))
        if verdict is None and now is not None and g.apart(now[0], followed, DRAWING) > g.WATCH["follow_m"]:
            followed = now[0]
    return None, seen


def test_the_watch_never_sets_a_persons_sway_breath_and_drift_up_again():
    """0.6 mm of sway per axis (the right forearm's), breathing and 1.5 mm/min of drift for five minutes, asked
    before every stroke (each 30 s), the page following: nothing fires; the drift is the follow's."""
    assert _first_fire(_person(0), range(30, 300, 30))[0] is None


def test_a_step_that_stays_fires_once_held_and_one_that_comes_back_does_not():
    """An 8 mm step (past off_m, 7 mm since the owner doubled every stop) that stays fires once the 10 s median has
    held it 10 s (16-18 s replayed on the rig's readings); one that comes back within 8 s is pending at most; a 5 mm
    one is followed."""
    still = [(float(t), _move(0.0)) for t in range(120)]
    stays = still + [(float(t), _move(0.008)) for t in range(120, 200)]
    t, why = _first_fire(stays, range(120, 150))
    assert 120 + 10 < t <= 120 + 20 and why == "8.0 mm off the page, held 10 s"
    assert _first_fire(still + [(float(t), _move(0.005)) for t in range(120, 200)], range(120, 200, 5))[0] is None
    back = still + [(float(t), _move(0.008 if t < 128 else 0.0)) for t in range(120, 200)]
    assert _first_fire(back, range(120, 160)) == (None, {None, "pending"})


def test_a_steady_ramp_is_followed_and_fires_only_past_the_total():
    """4 mm/min from setup, followed before each stroke: never off the page; the 20 s median passes 10 mm at ~160 s
    and fires there."""
    ramp = [(float(t), _move(0.004 / 60 * t)) for t in range(300)]
    t, why = _first_fire(ramp, range(30, 300, 5))
    assert 155 <= t <= 170 and "from its setup" in why


def test_a_move_that_lasts_stays_fired_however_long_ago_and_one_that_went_back_is_none():
    """The node keeps two minutes of readings: an 8 mm step a person made while no stroke asked (a pause) still fires
    ten minutes on, never followed; the page that followed it is no move. A step held 15 s and gone back is none:
    the watch judges the newest readings. A gap holds the state: a step read for 3 s before the arm hid the print is
    pending, and fires once read again."""
    step = [(float(t), _move(0.008 if t >= 60 else 0.0)) for t in range(600)]
    for t in (150, 300, 600):
        verdict, why = g.sheet_watch(step[t - 120:t], DRAWING, np.eye(3), TOTAL)
        assert verdict == "fire" and why == f"8.0 mm off the page, held {min(t - 1 - 65, 117)} s"
    assert g.sheet_watch(step[480:600], DRAWING, _move(0.008)[2], TOTAL) == (None, "")
    back = [(float(t), _move(0.008 if 60 <= t < 75 else 0.0)) for t in range(120)]
    assert g.sheet_watch(back, DRAWING, np.eye(3), TOTAL) == (None, "")
    still = [(float(t), _move(0.0)) for t in range(60)]
    brief = still + [(float(t), _move(0.008)) for t in range(60, 68)]
    assert g.sheet_watch(brief, DRAWING, np.eye(3), TOTAL)[0] == "pending"
    assert g.sheet_watch(brief + [(float(t), _move(0.008)) for t in range(130, 134)], DRAWING, np.eye(3), TOTAL)[0] == "fire"
    assert len(g.window(brief + [(130.0, _move(0.0))], 20.0)) == 21   # the newest 20 s holding three: before the gap


def test_the_watch_fires_on_a_rise_or_a_bend_that_lasts_and_reads_the_setup_as_its_reference():
    still = [(float(t), _move(0.0, bend=0.0002)) for t in range(60)]
    risen = still + [(float(t), _move(0.0, rise=0.003)) for t in range(60, 80)]   # rise_m stays 2.5 mm: depth
    assert g.sheet_watch(risen, DRAWING, np.eye(3), TOTAL) == ("fire", "+3.0 mm along its normal, held 14 s")
    assert g.sheet_watch(still + [(float(t), _move(0.0, rise=0.002)) for t in range(60, 80)], DRAWING, np.eye(3),
                         TOTAL) == (None, "")
    bent = still + [(float(t), _move(0.0, bend=0.004)) for t in range(60, 80)]
    assert g.sheet_watch(bent, DRAWING, np.eye(3), TOTAL) == ("fire", "bent 4.0 mm, held 14 s")
    early = [(float(t), _move(0.0075)) for t in range(5, 25)]       # 7.5 mm off the setup's reference 5 s after it
    assert g.sheet_watch(early, DRAWING, np.eye(3), TOTAL)[0] == "fire"
    assert g.sheet_watch(early, DRAWING, _move(0.0075)[2], TOTAL) == (None, "")   # the page followed it
