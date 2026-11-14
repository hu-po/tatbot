"""The dose model: pokes along a stroke from the planner's time law, its stopping ends' dwell, and what flying ends
and blended corners change."""
import numpy as np
import pytest
import tatbot_motion as tm
from tatbot_motion import dose
from tatbot_motion import plan as pl
from tatbot_motion import timelaw as tl

SPEED = 0.0035
LINE = [[0.0, 0.0], [0.03, 0.0]]
ELL = [[0.0, 0.0], [0.02, 0.0], [0.02, 0.02]]


@pytest.fixture(scope="module")
def motion():
    return tm.load_motion()


def flying(motion, **fly):
    return {**motion, "draw": {**motion["draw"], "ends": "fly", "fly": {**motion["draw"]["fly"], **fly}}}


def test_the_rate_is_the_recorded_one_and_a_stroke_takes_rate_times_time(motion):
    assert dose.rate_hz(motion) == motion["machine"]["rate_hz"] == 51.0
    d = dose.stroke(LINE, SPEED, motion)
    t, _, _, _ = pl.drawing_time_law(np.array(LINE), SPEED, motion)
    assert d.pokes.sum() == pytest.approx(51.0 * len(t) / motion["control_rate_hz"])
    assert d.drawing_s == pytest.approx(len(t) / motion["control_rate_hz"])
    assert d.steady_per_mm == pytest.approx(14.571, abs=1e-3)
    assert dose.stroke(LINE, SPEED, motion, hz=79.0).pokes.sum() == pytest.approx(d.pokes.sum() * 79.0 / 51.0)
    assert dose.stroke(LINE, SPEED / 2, motion).summary()["middle_per_mm"] == pytest.approx(2 * 14.571, abs=0.1)


def test_a_stopping_stroke_dwells_at_its_ends_and_eases_over_its_first_millimetres(motion):
    record = dose.stroke(LINE, SPEED, motion).summary()
    assert record["middle_per_mm"] == pytest.approx(14.6, abs=0.1)
    assert record["start_per_mm"] == pytest.approx(27.3, abs=1.0) and record["end_per_mm"] == pytest.approx(27.3, abs=1.0)
    # the last 1.5 mm of the descent and the first of the lift, at the touchdown budget: about a second each
    down_s, up_s = dose.dwell_s(motion)
    assert 0.95 < down_s < 1.15 and 0.95 < up_s < 1.15
    assert record["touchdown_pokes"] == pytest.approx(51.0 * down_s, abs=0.1)
    assert record["liftoff_pokes"] == pytest.approx(51.0 * up_s, abs=0.1)
    settled = dose.stroke(LINE, SPEED, motion, settle_s=1.0)
    assert settled.touchdown == pytest.approx(51.0 * (down_s + 1.0))
    # a 6.8 mm stroke is eased over its first and last 0.9 mm: a quarter more than the steady dose along it, and
    # with its dwells over twice the steady dose end to end
    short = dose.stroke([[0.0, 0.0], [0.0068, 0.0]], SPEED, motion)
    assert short.pokes.sum() / (6.8 * short.steady_per_mm) == pytest.approx(1.26, abs=0.02)
    assert short.total / (6.8 * short.steady_per_mm) > 2.0


def test_a_corner_is_the_densest_place_between_a_strokes_ends(motion):
    record = dose.stroke(ELL, SPEED, motion).summary()
    assert record["corner_peak_per_mm"] == pytest.approx(25.7, abs=1.5)   # 1.7 mm/s at the turn
    assert dose.stroke(LINE, SPEED, motion).summary()["corner_peak_per_mm"] is None


def test_a_dot_and_a_joined_end(motion):
    dot = dose.stroke([[0.0, 0.0]], SPEED, motion)
    down_s, up_s = dose.dwell_s(motion)
    assert dot.length_m == 0.0 and dot.summary()["start_per_mm"] is None
    assert dot.total == pytest.approx(51.0 * (down_s + up_s + 2 * motion["draw"]["min_ease_s"]))
    joined = dose.stroke(LINE, SPEED, motion, joined=True, lift_at_end=False)
    assert (joined.start, joined.end, joined.touchdown, joined.liftoff) == ("join", "join", 0.0, 0.0)
    assert joined.summary()["start_per_mm"] == pytest.approx(27.3, abs=1.0)   # still eased from rest on the skin


def test_flying_ends_take_the_steady_dose_within_five_per_cent(motion):
    fly = flying(motion)
    for points in (LINE, [[0.0, 0.0], [0.0068, 0.0]], [[0.0, 0.0], [0.003, 0.0]]):
        d = dose.stroke(points, SPEED, fly)
        record = d.summary()
        assert (d.start, d.end) == ("fly", "fly") and record["touchdown_pokes"] == record["liftoff_pokes"] == 0.0
        for key in ("start_per_mm", "middle_per_mm", "end_per_mm"):
            assert record[key] == pytest.approx(d.steady_per_mm, rel=0.05)
        assert record["over_mm"] == 0.0
        # the marking band reaches a little outside the stroke: the glide's part under band_m
        assert -0.0006 < d.edges_m[0] < 0.0 and d.length_m < d.edges_m[-1] < d.length_m + 0.0006
    short = dose.stroke([[0.0, 0.0], [0.0029, 0.0]], SPEED, fly)   # under its two insets: it would never reach its path
    assert (short.start, short.end) == ("stop", "stop") and short.touchdown > 50
    mixed = dose.stroke(LINE, SPEED, fly, lift_at_end=False)
    assert (mixed.start, mixed.end) == ("fly", "join")
    assert mixed.summary()["start_per_mm"] == pytest.approx(14.6, abs=0.3) and mixed.summary()["end_per_mm"] > 25
    held = dose.stroke(LINE, SPEED, fly, clear_m=[0.061, 0.1])   # the run-out would leave the clear area
    assert (held.start, held.end) == ("fly", "stop") and held.liftoff > 50


def test_the_flying_law_reaches_the_draw_speed_before_the_band_and_the_path_inside_the_stroke(motion):
    fly = pl.stroke_ends(flying(motion))
    assert pl.stroke_ends(motion) is None and fly.height_m == pytest.approx(0.002)
    dt = 1.0 / motion["control_rate_hz"]
    for points in (LINE, [[0.0, 0.0], [0.003, 0.0]], ELL):
        length = tl.polyline_length(np.array(points))
        path, s, lift, s0 = pl.flying_law(np.array(points), SPEED, flying(motion), fly, True, True)
        run, inset = pl.fly_run_m(fly, SPEED), fly.inset_m
        assert run == pytest.approx(0.5 * SPEED * fly.ease_s + fly.glide_m - fly.inset_m)
        assert s0 == pytest.approx(run) and tl.polyline_length(path) == pytest.approx(length + 2 * run)
        speed = np.diff(np.r_[0.0, s]) / dt
        near = lift < fly.height_m - 1e-6                       # anywhere under the height the eases happen at
        straight = np.abs(s - s0 - 0.02) > 0.002 if points is ELL else np.ones(len(s), bool)
        assert speed[near & straight].min() > 0.995 * SPEED     # at the draw speed all the way through the band
        assert lift[0] == pytest.approx(fly.height_m) and lift[-1] == pytest.approx(fly.height_m)
        on_path = (s >= s0 + inset - 1e-9) & (s <= s0 + length - inset + 1e-9)
        assert lift.min() >= 0.0 and lift.min() < 1e-6 and (not on_path.any() or np.abs(lift[on_path]).max() < 1e-9)
        down = lift[s < s0 + inset]
        assert np.all(np.diff(down) <= 1e-12)                   # down only, then level, then up only
        # the pen is over the band wherever it is more than half a millimetre outside the stroke
        outside = (s < s0 - 0.0005) | (s > s0 + length + 0.0005)
        assert lift[outside].min() >= fly.band_m
    one_end, s, lift, s0 = pl.flying_law(np.array(LINE), SPEED, flying(motion), fly, True, False)
    assert s0 > 0 and lift[-1] == 0.0 and np.allclose(one_end[-1], LINE[-1])


def test_the_uneven_time_law_is_the_even_one_when_its_eases_agree():
    even = tl.time_law(0.02, 8.0, 1.5, 0.0025)
    for ease_out in (None, 1.5):
        again = tl.time_law(0.02, 8.0, 1.5, 0.0025, ease_out)
        assert all(np.array_equal(a, b) for a, b in zip(even, again, strict=True))
    t, s, sdot = tl.time_law(0.02, 8.0, 1.0, 0.0025, 2.0)
    cruise = 0.02 / (8.0 - 1.5)
    assert s[-1] == pytest.approx(0.02) and np.all(np.diff(s) >= 0.0) and sdot.max() == pytest.approx(cruise)
    assert s[np.searchsorted(t, 1.0)] == pytest.approx(0.5 * cruise * 1.0, rel=1e-3)        # half the cruise distance
    assert 0.02 - s[np.searchsorted(t, 6.0)] == pytest.approx(0.5 * cruise * 2.0, rel=1e-2)
    assert sdot[0] < 0.01 * cruise and sdot[-1] == 0.0
    with pytest.raises(ValueError, match="eases"):
        tl.time_law(0.02, 2.0, 1.0, 0.0025, 1.5)


def test_a_blended_corner_stays_within_its_radius_and_takes_a_quarter_off_the_corners_dose(motion):
    points = np.array(ELL)
    rounded, at = pl.blend_corners(points, 0.00015)
    assert np.array_equal(rounded[0], points[0]) and np.array_equal(rounded[-1], points[-1])
    assert np.all(np.diff(at) > 0.0) and at[-1] == pytest.approx(0.04)
    gap = np.linalg.norm(rounded - points[1], axis=1).min()
    assert 0.0 < gap <= 0.00015 * (np.sqrt(2.0) - 1.0) + 1e-9      # the arc's bulge off a right angle
    seg = np.diff(points, axis=0)
    for q in rounded:   # never outside the corner it cuts: within the bulge of one of the two chords
        off = min(np.linalg.norm(q - (a + np.clip((q - a) @ d / (d @ d), 0, 1) * d)) for a, d in zip(points[:-1], seg, strict=True))
        assert off <= 0.00015
    hairpin, _ = pl.blend_corners(np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.0002]]), 0.0005)
    assert np.linalg.norm(hairpin - [0.01, 0.0], axis=1).min() <= 0.0005
    fine = np.c_[np.cos(np.linspace(0, 1, 60)), np.sin(np.linspace(0, 1, 60))] * 0.01   # 1 deg a step: left alone
    assert np.array_equal(pl.blend_corners(fine, 0.0005)[0], fine)
    op = {"points_m": ELL, "generation_width_m": 0.0003}
    assert pl.corner_blend_m(op, motion) == 0.0                                           # today's sharp corner
    blended = {**motion, "corners": {**motion["corners"], "blend_radius_m": 0.001}}
    assert pl.corner_blend_m(op, blended) == pytest.approx(0.00015)                       # half the mark at most
    assert pl.corner_blend_m({"points_m": ELL}, blended) == 0.0
    sharp = dose.stroke(ELL, SPEED, motion).summary()["corner_peak_per_mm"]
    eased = dose.stroke(pl.blend_corners(points, 0.00015)[0], SPEED, motion).summary()["corner_peak_per_mm"]
    assert eased < 0.75 * sharp


def test_motion_yaml_refuses_a_bad_ends_block(tmp_path, motion):
    import yaml

    def load(**draw):
        path = tmp_path / "motion.yaml"
        path.write_text(yaml.safe_dump({**motion, "draw": {**motion["draw"], **draw}}))
        return tm.load_motion(path)

    assert load()["draw"]["ends"] == "stop"
    assert load(ends="fly")["draw"]["ends"] == "fly"
    with pytest.raises(ValueError, match="draw.ends"):
        load(ends="whip")
    with pytest.raises(ValueError, match="draw.fly"):
        load(fly={**motion["draw"]["fly"], "inset_m": 0.01})
    with pytest.raises(ValueError, match="draw.fly"):
        load(fly={**motion["draw"]["fly"], "band_m": 0.02})
