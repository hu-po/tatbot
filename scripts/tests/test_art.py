"""python/tatbot_art: colour maths, ink evidence, the achievable set and the predicted render."""
import json
import subprocess

import numpy as np
import pytest
from tatbot_art import acquire, cli
from tatbot_art.acquire import canvas_program
from tatbot_art.color import delta_e00, from_8bit, oklab, oklab_to_srgb
from tatbot_art.gamut import achievable, mix
from tatbot_art.inks import InkLook, load_catalog, look
from tatbot_art.raster import add_stroke
from tatbot_art.render import minutes, render_program
from tatbot_art.swatch import hatch_lines

# Sharma, Wu and Dalal (2005), test pairs 1, 2, 7, 17 and 25.
SHARMA = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
          ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
          ((50, 0, 0), (50, -1, 2), 2.3669),
          ((50, 2.5, 0), (73, 25, -18), 27.1492),
          ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA)
def test_ciede2000_matches_published_pairs(lab1, lab2, expected):
    assert float(delta_e00(lab1, lab2)) == pytest.approx(expected, abs=1e-4)


def test_oklab_round_trips_and_white_is_neutral():
    rgb = np.array([[0.1, 0.5, 0.9], [1.0, 0.0, 0.0]])
    assert np.allclose(oklab_to_srgb(oklab(rgb)), rgb, atol=1e-6)
    assert np.allclose(oklab([1.0, 1.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-6)


def test_look_takes_the_strongest_source_for_the_substrate():
    catalog = {"red": {"rgb": [255, 0, 0], "appearance": [
        {"srgb": [250, 80, 50], "source": "vendor-photo", "substrate": "any"},
        {"srgb": [200, 40, 40], "source": "swatch", "substrate": "paper"}]}}
    assert look(catalog, "red", "paper").source == "swatch"
    assert look(catalog, "red", "silicone").srgb == pytest.approx(tuple(from_8bit([250, 80, 50])))
    assert look({"red": {"rgb": [255, 0, 0]}}, "red").source == "nominal"
    with pytest.raises(ValueError, match="unknown ink"):
        look(catalog, "blue")


def test_catalogue_lists_the_owned_set_with_vendor_photo_colours():
    catalog = load_catalog()
    for ink in ("rose_pink", "bright_orange", "golden_yellow", "lemon_yellow", "silver", "light_green",
                "light_purple", "brown", "bright_red", "lime_green", "snow_white"):
        assert look(catalog, ink).source == "vendor-photo", ink
    assert look(catalog, "true_black").source == "nominal"   # another maker's bottle than the set's photo shows


def _look(ink_id, rgb, opaque=False):
    return InkLook(ink_id, ink_id, tuple(from_8bit(rgb)), "nominal", opaque)


def test_achievable_set_prices_line_and_keeps_distinct_colours():
    looks = [_look("yellow", [253, 201, 73]), _look("purple", [106, 81, 159]), _look("black", [20, 20, 20])]
    swatches = achievable(looks, (0.95, 0.95, 0.93), width_mm=0.6)
    ids = [sw.id for sw in swatches]
    assert len(ids) == len(set(ids)) and len(swatches) > 3 * 3
    for sw in swatches:
        for lay in sw.layers:
            assert sw.line_mm_per_cm2[lay.ink_id] == pytest.approx(100.0 / lay.pitch_mm)
            assert lay.pitch_mm == pytest.approx(0.6 / lay.coverage)
    labs = np.array([oklab(sw.srgb) for sw in swatches])
    assert labs[:, 0].min() < 0.35 < 0.9 < labs[:, 0].max()


def test_hatched_raster_coverage_matches_the_area_model():
    """At n = 1 the area model is the hatched raster's mean, so the rasterizer's coverage is honest."""
    size, px_per_mm, width_mm, cov = 240, 20.0, 0.6, 0.5
    coverage = np.zeros((size, size))
    for line in hatch_lines(size, 45.0, width_mm / cov * px_per_mm):
        add_stroke(coverage, line, width_mm * px_per_mm)
    assert coverage.mean() == pytest.approx(cov, abs=0.03)
    ink = _look("red", [220, 40, 40])
    assert mix((1.0, 1.0, 1.0), [(ink, coverage.mean())], n=1.0) == pytest.approx(
        mix((1.0, 1.0, 1.0), [(ink, cov)], n=1.0), abs=0.03)


def _program(mm_per_dip=60.0):
    tool = {"id": "liner", "ink": "dip", "line_width_m": 0.0005}
    stroke = [[-0.02, 0.0], [0.02, 0.0]]  # 40 mm
    ops = [{"op": "tool_change", "id": "t0", "resource_id": "red", "initial": True,
            "activation": {"method": "rinse"}},
           {"op": "dip", "id": "d1", "resource_id": "red"}]
    ops += [{"op": "stroke", "id": f"s{k}", "resource_id": "red", "points_m": [[x, y + 0.005 * k] for x, y in stroke]}
            for k in range(2)]
    return {"page": {"size_m": [0.05, 0.03]}, "stats": {"time_estimate": {"modeled_s": 60.0}},
            "resources": [{"id": "red", "ink_id": "bright_red", "tool": tool, "dip": {"mm_per_dip": mm_per_dip}}],
            "ops": ops}


def test_render_draws_measured_width_and_fades_toward_the_next_dip():
    catalog = {"bright_red": {"rgb": [255, 0, 0]}}
    full = render_program(_program(), catalog, "paper", px_per_mm=10.0, fade=False)
    assert full.srgb.shape == (300, 500, 3)
    assert full.report["inks"]["red"]["area_mm2"] == pytest.approx(2 * 40 * 0.5, rel=0.08)
    faded = render_program(_program(), catalog, "paper", px_per_mm=10.0)
    assert faded.coverage["red"].sum() < full.coverage["red"].sum()
    first, second = faded.coverage["red"][150 - 2:150 + 3], faded.coverage["red"][100 - 2:100 + 3]
    assert first.sum() > second.sum()  # the second 40 mm runs past 45 mm of a 60 mm dip
    assert faded.report["weakest_source"] == "nominal"


def test_render_darkens_by_the_programs_dose_where_it_has_one():
    """A compiled program's stats.dose: a stroke's first and last millimetre, its middle and the spots where the pen
    came down and lifted darken as that many steady passes; without the record a stroke is its passes, as before."""
    catalog = {"bright_red": {"rgb": [255, 0, 0]}}
    program = _program()
    for op in program["ops"][2:]:
        op["deposition"] = 0.4   # a light pass, so more pokes can still darken it
    plain = render_program(program, catalog, "paper", px_per_mm=10.0, fade=False)
    assert plain.report["dose"] is None
    # s0: 4x the steady dose over its first millimetre, steady in its middle and at its end, 3 passes' pokes where
    # it came down; s1 has no row and is drawn by its passes
    program["stats"]["dose"] = {"model": "tatbot-dose/1", "rate_hz": 51.0, "ends": "stop", "steady_per_mm": 14.57,
                                "strokes": {"s0": [58.3, 14.57, 14.57, None, 0.03, 21.9, 0.0]}}
    dosed = render_program(program, catalog, "paper", px_per_mm=10.0, fade=False)
    assert dosed.report["dose"] == {"model": "tatbot-dose/1", "rate_hz": 51.0, "ends": "stop", "steady_per_mm": 14.57}
    row = slice(148, 153)                                             # s0, along y = 0
    start, middle, end = slice(51, 59), slice(200, 300), slice(441, 449)
    assert dosed.coverage["red"][row, middle].mean() == pytest.approx(plain.coverage["red"][row, middle].mean(), rel=0.01)
    assert dosed.coverage["red"][row, end].mean() == pytest.approx(plain.coverage["red"][row, end].mean(), rel=0.01)
    assert dosed.coverage["red"][row, start].mean() > 2.0 * plain.coverage["red"][row, start].mean()   # 1 - 0.6^4
    assert dosed.coverage["red"][98:103].sum() == pytest.approx(plain.coverage["red"][98:103].sum(), rel=1e-6)   # s1
    program["stats"]["dose"]["strokes"]["s0"] = [14.57, 14.57, 14.57, None, 0.0, 0.0, 0.0]
    steady = render_program(program, catalog, "paper", px_per_mm=10.0, fade=False)
    assert steady.coverage["red"].sum() == pytest.approx(plain.coverage["red"].sum(), rel=0.01)   # one pass is one pass


def test_render_scales_ink_by_deposition_and_ramps_it_along_a_stroke():
    catalog = {"bright_red": {"rgb": [255, 0, 0]}}
    full = render_program(_program(), catalog, "paper", px_per_mm=10.0, fade=False)
    assert full.report["inks"]["red"]["partial_mm"] == 0.0
    program = _program()
    for op in program["ops"][2:]:
        op["deposition"] = 0.5
    half = render_program(program, catalog, "paper", px_per_mm=10.0, fade=False)
    assert half.coverage["red"].sum() == pytest.approx(0.5 * full.coverage["red"].sum(), rel=0.01)
    program = _program()
    program["ops"][2]["deposition"] = [1.0, 0.0]  # a whip: full at its start, none at its end
    whip = render_program(program, catalog, "paper", px_per_mm=10.0, fade=False)
    row = whip.coverage["red"][148:153]
    assert row[:, 60:80].mean() > 0.8 > 0.1 > row[:, 420:440].mean()
    assert whip.report["inks"]["red"]["partial_mm"] == pytest.approx(40.0)
    program["ops"][2]["deposition"] = [1.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="3 deposition values for 2 points"):
        render_program(program, catalog, "paper")


def test_drawn_canvas_keeps_each_path_deposition(tmp_path):
    art = tmp_path / "art.json"
    art.write_text(json.dumps({"program": {"layers": [{"elements": [
        {"points_m": [[0.0, 0.0], [0.01, 0.0]], "width_m": 0.0005, "deposition": 0.4},
        {"points_m": [[0.0, 0.01], [0.01, 0.01]], "width_m": 0.0005}]}]}}))
    lay = type("Layer", (), {"ink_id": "red"})()
    ops = canvas_program([(lay, art)], (20.0, 20.0))["ops"]
    assert [op["deposition"] for op in ops] == [0.4, 1.0]


def test_a_stroke_coming_back_over_its_own_line_is_another_pass_as_the_compiler_counts_it():
    """tatbot_ink.passes counts a spot again when the pen comes back after going more than its width along the line:
    a retrace or a pendulum's next swing darkens like a second stroke, its joints and its turn stay one pass."""
    from tatbot_art.raster import stroke_coverage

    def mid(points, shape=(40, 200)):
        return stroke_coverage(shape, points, 6.0, 0.5)[1]

    line, back = mid([[20, 20], [180, 20]]), mid([[20, 20], [180, 20], [20, 20.0001]])
    assert line[line.shape[0] // 2, 80] == pytest.approx(0.5)
    assert back[back.shape[0] // 2, 80] == pytest.approx(1 - 0.5 ** 2)   # out and back: two passes
    assert back[back.shape[0] // 2, -6] == pytest.approx(0.5)            # at the turn, within a width: one
    assert mid([[20, 20], [180, 20], [180, 180]], (200, 200)).max() == pytest.approx(0.5)   # a corner: one
    zig = [[20.0 + 160 * (k % 2), 20.0 + 3 * k] for k in range(8)]       # swings 3 px apart, a 6 px pen
    separate = np.zeros((60, 200))
    for a, b in zip(zig, zig[1:], strict=False):
        add_stroke(separate, [a, b], 6.0, 0.5)
    joined = np.zeros((60, 200))
    add_stroke(joined, zig, 6.0, 0.5)   # the same swings as one pendulum stroke: the same ink
    assert joined[25:35, 60:140].mean() == pytest.approx(separate[25:35, 60:140].mean(), rel=0.05)
    assert joined[25:35, 60:140].mean() > 0.7   # each spot under two swings, not one


def test_minutes_charge_the_dip_but_not_an_initial_selection_rinse():
    t = minutes(_program())
    assert (t["dips"], t["rinses"], t["swaps"]) == (1, 0, 0)
    assert t["minutes"] == pytest.approx((60 + 48) / 60, abs=0.05)


def test_cli_swatch_and_render_write_their_outputs(tmp_path):
    assert cli.main(["swatch", "--inks", "golden_yellow,light_purple", "--px-per-mm", "4",
                     "--out", str(tmp_path / "sw")]) == 0
    report = json.loads((tmp_path / "sw" / "swatches.json").read_text())
    assert report["count"] == len(report["swatches"]) > 6
    assert (tmp_path / "sw" / "swatches.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    program = tmp_path / "program.json"
    program.write_text(json.dumps(_program()))
    assert cli.main(["render", str(program), "--out", str(tmp_path / "r")]) == 0
    assert json.loads((tmp_path / "r" / "render.json").read_text())["time"]["dips"] == 1
    assert cli.main(["swatch", "--inks", "no_such_ink", "--out", str(tmp_path / "x")]) == 2


def test_a_refused_tone_wedge_reports_dbv3s_reason(tmp_path, monkeypatch, capsys):
    refused = subprocess.CompletedProcess([], 3, stdout="drawingbot: no drawable paths\n", stderr="")
    monkeypatch.setattr(acquire.subprocess, "run", lambda *a, **k: refused)
    assert cli.main(["tone", "--app", "/nonexistent", "--out", str(tmp_path)]) == 1
    assert "drawingbot: no drawable paths" in capsys.readouterr().err
