"""A program's dose: pokes per millimetre of each stroke from the planner's time law and the machine's rate, the
pokes each spot of the page takes, and both in the compile's stats and its preview."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tatbot_ink
from ink_designs import REPO, artwork, element, program, write
from tatbot_ink import dose
from tatbot_motion import load_motion

W = 0.0003
PROGRAMS = Path(__file__).parent / "dose_programs"   # private: the strokes of two programs drawn on the rig


@pytest.fixture(scope="module")
def motion():
    return load_motion()


def flying(motion):
    return {**motion, "draw": {**motion["draw"], "ends": "fly"}}


def _program(strokes, speed=0.0035, continues=()):
    ops = [{"op": "stroke", "id": f"s{k:04d}", "resource_id": "ink", "closed": False, "continues": k in continues,
            "generation_width_m": W, "points_m": points} for k, points in enumerate(strokes)]
    return {"draw_speed_m_s": speed, "page": {"kind": "stencil", "size_m": [0.1, 0.1], "clear_m": [0.1, 0.1]},
            "resources": [{"id": "ink", "pen_mode": "ride", "tool": {"id": "tool", "line_width_m": None}}], "ops": ops}


def test_a_long_stroke_takes_the_steady_dose_in_its_middle_and_twice_it_at_its_stopping_ends(motion):
    record = dose.summary(_program([[[0.0, 0.0], [0.03, 0.0]]]), motion)
    steady = motion["machine"]["rate_hz"] / 3.5
    assert record["model"] == "tatbot-dose/1" and record["ends"] == "stop" and record["ends_drawn"] == {"stop": 2}
    assert record["steady_per_mm"] == pytest.approx(steady, abs=0.01)
    start, middle, end, corner, over, down, up = record["strokes"]["s0000"]
    assert (down, up) == (record["touchdown_pokes"], record["liftoff_pokes"])
    assert middle == pytest.approx(steady, abs=0.1) and corner is None
    assert 1.7 < start / steady < 2.0 and 1.7 < end / steady < 2.0   # the 0.5 s ease over its first 0.9 mm
    assert 50 < record["touchdown_pokes"] < 58 and 50 < record["liftoff_pokes"] < 58   # ~1 s in the last 1.5 mm
    assert 0.0 < over < 0.02 and record["over_2x"]["mm"] == pytest.approx(over * 30, abs=0.06)


def test_a_cell_takes_the_pokes_laid_while_the_mark_covered_it(motion):
    once = dose.summary(_program([[[0.0, 0.0], [0.03, 0.0]]]), motion)["cells"]
    assert 53 < once["max_pokes"] < 110 and once["max_passes"] > 10   # the descent's dwell under the first point
    pokes, passes, origin, cell = dose.grid(_program([[[0.0, 0.0], [0.03, 0.0]]]), motion)
    row, col = np.floor((np.array([0.0, 0.015]) - origin[::-1]) / cell).astype(int)
    assert passes[row, col] == pytest.approx(1.0, abs=0.2)           # the middle of the line: one steady pass
    assert pokes[row, col] == pytest.approx(motion["machine"]["rate_hz"] / 3.5 * 0.3, rel=0.2)
    thrice = dose.grid(_program([[[0.0, 0.0], [0.03, 0.0]]] * 3), motion)[1]
    assert thrice[row, col] == pytest.approx(3.0, abs=0.6)
    assert dose.cells(None)["hotspots"] == [] and dose.cells(None)["max_pokes"] == 0.0


def test_a_continuing_chunk_is_eased_on_the_skin_without_a_dwell(motion):
    chunks = _program([[[0.0, 0.0], [0.03, 0.0]], [[0.03, 0.0], [0.06, 0.0]]], continues={1})
    record = dose.summary(chunks, motion)
    assert record["ends_drawn"] == {"join": 2, "stop": 2}
    doses = dose.strokes(chunks, motion)
    assert doses["s0000"].touchdown > 50 and doses["s0000"].liftoff == 0.0
    assert doses["s0001"].touchdown == 0.0 and doses["s0001"].liftoff > 50
    assert record["strokes"]["s0000"][2] > 25   # the join still stops: its ease is on the skin


def test_flying_ends_take_the_steady_dose_and_no_dwell(motion):
    record = dose.summary(_program([[[0.0, 0.0], [0.03, 0.0]], [[0.0, 0.01], [0.004, 0.01]]]), flying(motion))
    steady = record["steady_per_mm"]
    assert record["ends"] == "fly" and record["ends_drawn"] == {"fly": 4}
    assert record["touchdown_pokes"] == 0.0 and record["liftoff_pokes"] == 0.0 and record["over_2x"]["mm"] == 0.0
    for start, middle, end, *_ in record["strokes"].values():
        assert max(start, middle, end) < 1.05 * steady and min(start, middle, end) > 0.95 * steady
    assert record["cells"]["max_passes"] < 1.5
    at_edge = dose.summary(_program([[[0.0, 0.0], [0.049, 0.0]]]), flying(motion))   # 1 mm inside the clear area
    assert at_edge["ends_drawn"] == {"fly": 1, "stop": 1} and at_edge["liftoff_pokes"] > 50


def test_the_compile_and_its_preview_carry_the_dose(tmp_path, motion):
    lines = [element([[.002, .01], [.018, .01]], width=W) for _ in range(3)]
    result = tatbot_ink.compile(write(tmp_path, artwork(program([("pen-1", lines)]))), repo=REPO)
    record = result["stats"]["dose"]
    assert record["rate_hz"] == motion["machine"]["rate_hz"] and set(record["strokes"]) == {
        op["id"] for op in result["ops"] if op["op"] == "stroke"}
    assert record["cells"]["mm2_at_least"]["3"] > 4 and record["cells"]["max_passes"] > 30
    drawing = sum(d.drawing_s for d in dose.strokes(result, motion).values())
    assert result["stats"]["time_estimate"]["components_s"]["drawing"] == pytest.approx(drawing, abs=1e-3)
    svg = tatbot_ink.write_preview(result, tmp_path / "preview.svg").read_text()
    assert "dose at 51 Hz, ends stop: 14.6 pokes/mm steady" in svg and '<g id="dose"' in svg
    without = copy.deepcopy(result)
    del without["stats"]["dose"]
    assert '<g id="dose"' not in tatbot_ink.preview.preview_svg(without)


# The two programs of 2026-11-10 (a 3RL's black outline, then a 7RL's red fill): what the dose model says of them.
# Flying, the strokes under 3 mm keep their stopping ends: 30 of the black's 129 and 52 of the red's 148.
@pytest.mark.parametrize("name, stop, fly", [
    ("black", {"pokes": 57875, "mean": 20.5, "start": 27.3, "end": 22.3, "over": 0.011, "doubled": 54, "spot": 261,
               "ends": {"stop": 258}},
     {"pokes": 45979, "mean": 16.3, "start": 14.6, "end": 14.6, "over": 0.002, "doubled": 30, "spot": 147,
      "ends": {"fly": 198, "stop": 60}}),
    ("red", {"pokes": 61923, "mean": 21.0, "start": 27.3, "end": 21.8, "over": 0.011, "doubled": 81, "spot": 264,
             "ends": {"stop": 296}},
     {"pokes": 50415, "mean": 17.1, "start": 14.6, "end": 14.6, "over": 0.003, "doubled": 52, "spot": 153,
      "ends": {"fly": 192, "stop": 104}}),
])
def test_the_rig_programs_dose(motion, name, stop, fly):
    path = PROGRAMS / f"{name}.json"
    if not path.is_file():
        pytest.skip("the rig programs' strokes are not in this checkout")
    drawn = json.loads(path.read_text())
    for mode, want in ((motion, stop), (flying(motion), fly)):
        record = dose.summary(drawn, mode)
        assert len(record["strokes"]) == drawn["stats"]["strokes"] and record["ends_drawn"] == want["ends"]
        assert record["pokes"] == pytest.approx(want["pokes"], rel=0.01)
        assert record["mean_per_mm"] == pytest.approx(want["mean"], abs=0.3)
        assert record["start_per_mm"]["median"] == pytest.approx(want["start"], abs=1.5)
        assert record["end_per_mm"]["median"] == pytest.approx(want["end"], abs=1.5)
        assert record["middle_per_mm"]["median"] == pytest.approx(14.6, abs=0.1)
        assert record["over_2x"]["share"] == pytest.approx(want["over"], abs=0.005)
        assert record["over_2x"]["strokes_end_to_end"] == want["doubled"]
        assert record["cells"]["max_pokes"] == pytest.approx(want["spot"], rel=0.15)   # the grid's phase moves it
