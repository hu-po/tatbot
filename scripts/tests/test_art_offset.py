"""python/tatbot_art offset fill: an ink's regions as passes one mark apart along their edges, a band too narrow for two
drawn once along its middle, frozen as acquired artwork; and separations that bind each ink to its own cartridge."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from tatbot_art import acquire, cli, pipeline
from tatbot_art.color import from_8bit
from tatbot_art.inks import load_catalog
from tatbot_art.offset import ADAPTER, Offset, artwork, generate, mark_mm
from tatbot_art.png import write_png
from tatbot_art.separate import Layer
from tatbot_contracts.artwork import require_acquired_artwork

PX = 10.0


def _band(width_mm, length_mm=30.0):
    region = np.zeros((200, 400), bool)
    w = int(round(width_mm * PX))
    region[100 - w // 2:100 - w // 2 + w, 50:50 + int(length_mm * PX)] = True
    return region


def _line(p):
    pts = np.vstack([p["points_mm"], p["points_mm"][:1]]) if p["closed"] else p["points_mm"]
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def test_a_band_one_mark_wide_is_one_pass_along_its_middle():
    paths, info = generate(_band(1.0), PX, Offset(pitch_mm=0.9))
    assert len(paths) == 1 and not paths[0]["closed"] and paths[0]["kind"] == "middle"
    assert _line(paths[0]) == pytest.approx(30.0, abs=1.5)
    assert np.allclose(paths[0]["points_mm"][:, 1], 10.0, atol=0.15)   # down its middle, y up from the lower left
    assert info["covered"] > 0.98


def test_a_band_two_marks_wide_is_one_ring_of_two_passes():
    paths, info = generate(_band(1.9), PX, Offset(pitch_mm=0.9))
    assert [p["kind"] for p in paths] == ["ring"] and paths[0]["closed"]
    assert info["line_mm"] == pytest.approx(60.0, abs=3.0) and info["covered"] > 0.98


def test_a_disc_takes_about_one_line_per_pitch_and_is_covered():
    y, x = np.mgrid[:300, :300]
    paths, info = generate((x - 150) ** 2 + (y - 150) ** 2 < 100 ** 2, PX, Offset(pitch_mm=1.15))
    assert all(p["kind"] == "ring" for p in paths)
    assert info["line_per_mm2"] == pytest.approx(1 / 1.15, rel=0.1) and info["covered"] > 0.99


def test_a_curved_band_is_one_path_however_its_skeleton_steps():
    """A ring 1 mm wide: its skeleton is a pixel staircase all the way round, and stays one closed path."""
    y, x = np.mgrid[:400, :400]
    r = np.hypot(x - 200, y - 200)
    paths, info = generate((r >= 150) & (r < 160), PX, Offset(pitch_mm=0.9))
    assert len(paths) == 1 and paths[0]["kind"] == "middle" and paths[0]["closed"]
    assert _line(paths[0]) == pytest.approx(2 * math.pi * 15.5, rel=0.03)
    assert info["covered"] > 0.97


def test_a_crossing_runs_straight_through():
    region = np.zeros((300, 300), bool)
    region[145:155, 30:270] = True
    region[30:270, 145:155] = True
    paths, _ = generate(region, PX, Offset(pitch_mm=0.9))
    long = [p for p in paths if _line(p) > 20]
    assert len(long) == 2
    for p in long:   # each runs on through the crossing, along x or along y
        span = np.ptp(p["points_mm"], axis=0)
        assert max(span) > 22 and min(span) < 1.0


def test_an_inset_under_half_a_pitch_reaches_past_the_edge():
    region = _band(1.9)
    inside = generate(region, PX, Offset(pitch_mm=0.9))[1]
    under = generate(region, PX, Offset(pitch_mm=0.9, inset_mm=0.2))[1]
    assert under["inset_mm"] == 0.2 and under["outside_mm2"] > inside["outside_mm2"] + 5


def test_a_magnum_lays_no_offset_fill():
    assert mark_mm("lutin-11rl-wormhole") == pytest.approx(1.16, abs=0.05)
    with pytest.raises(ValueError, match="magnum"):
        mark_mm("lutin-9m-ex")


def test_an_offset_artwork_is_admitted_and_compiles_to_its_passes(tmp_path):
    import tatbot_ink

    region = np.zeros((150, 300), bool)
    region[40:60, 30:270] = True    # a 2 mm bar: one ring
    region[100:110, 30:270] = True  # a 1 mm bar: one middle
    paths, info = generate(region, PX, Offset(pitch_mm=0.9))
    art = artwork(paths, (30.0, 15.0), "true-black", (0.2, 0.2, 0.2), info, name="bars", source_sha256="0" * 64,
                  identifier="bars")
    assert art["conversion"]["adapter"] == ADAPTER and require_acquired_artwork(art)
    (tmp_path / "artwork.json").write_text(json.dumps(art))
    result = tatbot_ink.compile(tmp_path / "artwork.json", repo=Path(__file__).resolve().parents[2],
                                tool_id="lutin-3rl-bugpin")
    strokes = [op for op in result["ops"] if op["op"] == "stroke"]
    assert len(strokes) == 2


def _image(tmp_path):
    img = np.ones((200, 200, 3))
    img[40:160, 40:150] = from_8bit((20, 20, 22))      # black, with a yellow letter body inside it
    img[50:150, 50:140] = from_8bit((253, 201, 73))
    write_png(tmp_path / "piece.png", np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
    return tmp_path / "piece.png"


@pytest.fixture
def fitted_3rl(monkeypatch):
    """The drawing arm's fitted cartridge, pinned to one whose dip card is read. The rig's own is whatever
    config/workspace.yaml records on the arm stack.yaml draws with, and the compile refuses a design on a cartridge
    whose dip values are borrowed (the 7RL's)."""
    monkeypatch.setattr(acquire, "fitted_tool", lambda arm=None: "lutin-3rl-bugpin")


def test_the_offset_fill_separates_without_dbv3_and_draws_each_ink_with_its_cartridge(tmp_path, fitted_3rl):
    out = tmp_path / "sep"
    assert cli.main(["separate", str(_image(tmp_path)), "--inks", "golden_yellow,true_black", "--width-mm", "20",
                     "--style", "linocut", "--no-mixes", "--fill", "offset", "--tool", "golden_yellow=lutin-11rl-wormhole",
                     "--overlap-mm", "0.4", "--out", str(out)]) == 0
    report = json.loads((out / "separation.json").read_text())
    assert [(lay["ink"], lay["fill"]) for lay in report["layers"]] == [("golden_yellow", "solid:offset"),
                                                                        ("true_black", "solid:offset")]
    assert report["tools"] == {"golden_yellow": "lutin-11rl-wormhole", "true_black": acquire.fitted_tool()}
    assert report["pitch_mm"]["golden_yellow"] == pytest.approx(1.16, abs=0.05)
    assert not list((out / "jobs").glob("*/job.json")) and len(list((out / "jobs").glob("*/offset.json"))) == 2
    inks = {r["id"]: r for r in yaml.safe_load((out / "inks.yaml").read_text())["resources"]}
    assert inks["golden_yellow"]["tool_id"] == "lutin-11rl-wormhole"
    assert {r["activation"]["method"] for r in inks.values()} == {"manual"}
    assert report["compile"]["tool_changes"] == 2 and report["time"]["swaps"] == 1
    sep, *_ = pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "flush",
                            width_mm=20.0, style="linocut", px_per_mm=PX, line_width_mm=0.6, mixes=False, seed=1,
                            tone_path=None, solid_fill="offset", tools=report["tools"])
    body = sep.layers[0].source < 255
    rows, cols = np.nonzero(body)
    edge_mm = 2 * (np.ptp(rows) + np.ptp(cols)) / PX
    grown = report["layers"][0]["area_mm2"] - body.sum() / PX ** 2
    assert grown == pytest.approx(0.4 * edge_mm, rel=0.15)   # the body reaches 0.4 mm under the black round it


def test_an_ink_after_another_on_its_cartridge_is_rinsed_in_one_on_another_swapped_in(tmp_path):
    arts = []
    for ink in ("golden_yellow", "bright_red", "true_black"):
        art = tmp_path / f"{ink}.json"
        art.write_text(json.dumps({"content_sha256": ink[0] * 64}))
        arts.append((Layer(ink, "solid:offset", None, 0.0, 0.0), art))
    tools = {"golden_yellow": "lutin-11rl-wormhole", "bright_red": "lutin-11rl-wormhole", "true_black": "lutin-3rl-bugpin"}
    doc = yaml.safe_load(acquire.inks_file(tmp_path / "inks.yaml", arts, {}, tools).read_text())
    method = {r["id"]: r["activation"]["method"] for r in doc["resources"]}
    assert method == {"golden_yellow": "manual", "bright_red": "rinse", "true_black": "manual"}
    one = yaml.safe_load(acquire.inks_file(tmp_path / "one.yaml", arts, {}, "lutin-3rl-bugpin").read_text())
    assert {r["activation"]["method"] for r in one["resources"]} == {"rinse"}


def test_offset_settings_are_refused_without_the_offset_fill(tmp_path):
    with pytest.raises(ValueError, match="offset fill"):
        pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "out", width_mm=20.0,
                      style="flash", px_per_mm=PX, line_width_mm=0.6, mixes=False, seed=1, tone_path=None,
                      pitch_mm={"true_black": 0.9})
    with pytest.raises(ValueError, match="does not hold"):
        pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "out", width_mm=20.0,
                      style="flash", px_per_mm=PX, line_width_mm=0.6, mixes=False, seed=1, tone_path=None,
                      tools={"bright_red": "lutin-11rl-wormhole"})


def test_a_negative_overlap_separates_for_the_dark_ink_first_with_a_gap_and_compiles_in_that_order(tmp_path, fitted_3rl):
    """Dark first, the order the chicano lettering was drawn in (2026-11-10): the black's job comes first and the
    yellow keeps 0.4 mm clear of it, so the compile's ink-order check finds their marks apart."""
    out = tmp_path / "sep"
    assert cli.main(["separate", str(_image(tmp_path)), "--inks", "golden_yellow,true_black", "--width-mm", "20",
                     "--style", "linocut", "--no-mixes", "--fill", "offset", "--tool", "golden_yellow=lutin-11rl-wormhole",
                     "--overlap-mm", "-0.4", "--out", str(out)]) == 0
    report = json.loads((out / "separation.json").read_text())
    assert [lay["ink"] for lay in report["layers"]] == ["true_black", "golden_yellow"] and report["overlap_mm"] == -0.4
    order = report["compile"]["ink_order"]
    assert order["order"] == ["true_black", "golden_yellow"]
    assert [(p["darker"], p["lighter"], p["verdict"]) for p in order["pairs"]] == [("true_black", "golden_yellow",
                                                                                   "clear")]
    kw = {"width_mm": 20.0, "style": "linocut", "px_per_mm": PX, "line_width_mm": 0.6, "mixes": False, "seed": 1,
          "tone_path": None, "solid_fill": "offset", "tools": report["tools"]}
    flush, *_ = pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "flush", **kw)
    apart, *_ = pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "apart",
                              overlap_mm=-0.4, **kw)
    assert [lay.ink_id for lay in apart.layers] == ["true_black", "golden_yellow"]
    black, yellow = (lay.source < 255 for lay in apart.layers)
    whole = next(lay.source < 255 for lay in flush.layers if lay.ink_id == "golden_yellow")
    assert (black == next(lay.source < 255 for lay in flush.layers if lay.ink_id == "true_black")).all()
    assert yellow.sum() < whole.sum() and not (yellow & ~whole).any()
    from scipy import ndimage as ndi

    gap_px = ndi.distance_transform_edt(~black)[yellow].min()
    assert 0.4 * PX <= gap_px <= 0.4 * PX + 1.5
    with pytest.raises(ValueError, match="nothing of golden_yellow stays"):
        pipeline.plan(_image(tmp_path), ["golden_yellow", "true_black"], load_catalog(), tmp_path / "none",
                      overlap_mm=-8.0, **kw)
