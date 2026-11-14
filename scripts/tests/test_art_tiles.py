"""python/tatbot_art at a stencil tile's size: rounds composed for the tile, fitted to its print, and what the size
costs in detail."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tatbot_art import acquire, cli, image, pipeline, pyramid, tiles
from tatbot_art.color import from_8bit
from tatbot_art.inks import InkLook, load_catalog
from tatbot_art.loop import TILE_INKS, propose, tile_round
from tatbot_art.separate import detail, separate
from tatbot_art.tone import load as load_tone

YELLOW, BLACK = (253, 201, 73), (20, 20, 22)
LOOKS = [InkLook("golden_yellow", "golden_yellow", tuple(from_8bit(YELLOW)), "nominal", False),
         InkLook("true_black", "true_black", tuple(from_8bit(BLACK)), "nominal", False)]


def _print(root: Path, page_mm=(40, 62)) -> Path:
    """A generated print's manifest, as much of it as the page check reads."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "tracking.json").write_text(json.dumps(
        {"pattern_id": "stencil-tile", "page_mm": list(page_mm), "clear_center_uv": [0, 0, 1, 1]}))
    return root


def _art(path: Path, size=(300, 200)) -> Path:
    """Flat yellow and black on white, the art `size` (w, h) px inside a 20 px white margin."""
    w, h = size
    img = np.full((h + 40, w + 40, 3), 255, np.uint8)
    img[20:20 + h, 20:20 + w // 2] = YELLOW
    img[20:20 + h, 20 + w // 2:20 + w] = BLACK
    Image.fromarray(img).save(path)
    return path


@pytest.mark.parametrize("size", [(300, 200), (200, 600)])
def test_the_fitted_width_is_the_widest_the_room_holds(tmp_path, size):
    src, room = _art(tmp_path / "a.png", size), (40.0, 62.0)
    width = image.fit_width(src, room, 10.0)
    h, w = image.prepare(src, width, 10.0).shape[:2]
    assert w / 10 <= room[0] and h / 10 <= room[1]
    wider = image.prepare(src, width + 0.1, 10.0).shape[:2]
    assert wider[1] / 10 > room[0] or wider[0] / 10 > room[1]


def test_the_room_is_the_print_less_the_offset_of_the_centre(tmp_path):
    tile = _print(tmp_path / "print")
    assert acquire.page_room(tile) == (40.0, 62.0)
    assert acquire.page_room(tile, "5,-1") == (30.0, 60.0)
    assert acquire.page_room() == (90.0, 134.0)
    with pytest.raises(ValueError, match="off the page"):
        acquire.page_room(tile, "25,0")


def _sep(img):
    return separate(img, LOOKS, style="flash", tone=load_tone(), px_per_mm=10.0, line_width_mm=0.6, mixes=False)


def test_detail_counts_colour_under_2_mm_what_cleaning_dropped_and_narrow_gaps():
    img = np.ones((120, 220, 3))
    img[10:110, 10:60] = from_8bit(YELLOW)       # a 5 mm yellow block
    img[10:110, 80:95] = from_8bit(YELLOW)       # a 1.5 mm yellow bar: under 2 mm
    img[10:110, 110:113] = from_8bit(YELLOW)     # a 0.3 mm sliver, thinner than a line: cleaned away
    img[10:110, 130:160] = from_8bit(BLACK)
    img[10:110, 168:198] = from_8bit(BLACK)      # 0.8 mm of paper between two black blocks, 0.4 once each widens
    d = detail(_sep(img), "true_black")
    yellow = d["inks"]["golden_yellow"]
    assert yellow["area_mm2"] == pytest.approx(65, abs=2) and yellow["under_mm2"] == pytest.approx(16, abs=2)
    assert yellow["dropped_mm2"] == pytest.approx(3, abs=1)
    assert d["gaps_mm2"] == pytest.approx(4, abs=1.5) and d["darkest"] == "true_black"


def test_a_plain_design_has_no_narrow_detail_and_no_gaps():
    img = np.ones((120, 120, 3))
    img[10:110, 10:60] = from_8bit(YELLOW)
    img[10:110, 60:110] = from_8bit(BLACK)
    d = detail(_sep(img), "true_black")
    assert d["inks"]["golden_yellow"]["share"] < 0.05 and d["gaps_mm2"] < 0.5


def test_separate_fits_a_design_to_the_print(tmp_path, capsys):
    tile, src = _print(tmp_path / "print"), _art(tmp_path / "a.png", (200, 600))
    out = tmp_path / "sep"
    assert cli.main(["separate", str(src), "--inks", "golden_yellow,true_black", "--fit", "--stencil", str(tile),
                     "--out", str(out)]) == 0
    report = json.loads((out / "separation.json").read_text())
    assert report["fit"] == {"room_mm": [40.0, 62.0]} and report["page"]["fits"]
    assert report["canvas_mm"][1] == pytest.approx(62, abs=0.15) and report["detail"]["darkest"] == "true_black"
    assert "colour under 2 mm" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--width-mm", "30"], []])
def test_fit_and_a_width_are_one_choice(tmp_path, capsys, extra):
    src = _art(tmp_path / "a.png")
    args = ["separate", str(src), "--inks", "true_black", "--out", str(tmp_path / "sep"), *extra]
    assert cli.main(args + (["--fit"] if extra else [])) == 2
    assert "one of --width-mm or --fit" in capsys.readouterr().err


def test_a_tile_round_composes_for_the_tile():
    tile = tile_round("forearm18")
    assert tile["tile_mm"] == [40.0, 62.0] and tile["aspect"] == "9:16" and tile["max_inks"] == TILE_INKS
    assert "upright 40 by 62 mm" in tile["sentence"] and "no more than a third" in tile["sentence"]
    assert tile["minutes_budget"] == 51 and "dotwork" in tile["avoid_styles"]   # 25 min at 35 mm, by area
    with pytest.raises(ValueError, match="unknown tile"):
        tile_round("thigh")


def _design(style, inks):
    return {"genes": {"style": style, "subject": "a koi fish", "palette": "+".join(inks), "variant": "base"},
            "inks": [{"id": i} for i in inks]}


def test_propose_for_a_tile_keeps_to_what_the_tile_can_draw_and_says_its_size():
    designs = {"a": _design("american-traditional", ["golden_yellow", "bright_red", "light_green", "blue_sky", "true_black"]),
               "d": _design("dotwork", ["bright_red", "true_black"])}
    tile = tile_round("forearm18")
    asks = {"q": {"kind": "near", "anchors": ["a"], "n": 2, "at": "1"}, "r": {"kind": "near", "anchors": ["d"], "n": 1, "at": "2"}}
    prompts = propose(designs, {"u": {"v": {"a": 1, "d": 3}}}, load_catalog(), round_no=6, n=12, explore=0.5, seed=1,
                      asks=asks, tile=tile)
    assert len(prompts) == 12 and all(len(p["inks"]) <= TILE_INKS for p in prompts)
    assert all(p["text"].endswith(tile["sentence"]) for p in prompts)
    assert not any(p["style"] == "dotwork" for p in prompts)   # however well dotwork scores
    assert not any(p.get("ask") for p in prompts)   # five inks, or dots, cannot be drawn on the tile; both asks wait


def test_a_design_over_the_tile_budget_is_flagged_not_dropped(tmp_path):
    sep = tmp_path / "separations" / "r6-01-koi-1"
    sep.mkdir(parents=True)
    report = {"image_sha256": "a" * 64, "canvas_mm": [40, 60], "page": {"fits": True}, "inks": {"true_black": {}},
              "layers": [{"ink": "true_black"}], "detail": {}, "line_mm": {"true_black": 900},
              "time": {"minutes": 82.0, "dips": 40, "rinses": 1}, "fidelity": {"vs_target": {"mean": 9.0}}}
    (sep / "separation.json").write_text(json.dumps(report))
    design = tiles.Design("r6-01-koi-1", tmp_path / "koi.jpg", ["true_black"], "flash", "packed", "r6-01-koi")
    assert tiles.row(design, sep, 51)["over_budget"] and not tiles.row(design, sep, 90)["over_budget"]


def _tile_round(tmp_path: Path, tile="forearm18") -> tuple[Path, Path]:
    """A corpus holding one round's Flow images and sidecars, and its plan."""
    corpus = tmp_path / "corpus"
    prompts = []
    for k, size in enumerate([(180, 300), (300, 300)], 1):   # one taller than the tile, one square
        pid = f"r6-0{k}-comic-a-koi"
        folder = corpus / "flow" / pid
        folder.mkdir(parents=True)
        for j in (1, 2):
            _art(folder / f"{pid}-{j}.jpg", size)
            (folder / f"{pid}-{j}.json").write_text(json.dumps(
                {"schema": "tatbot.art-generation/1", "prompt_id": pid, "prompt": "a koi", "model": "test"}))
        prompts.append({"id": pid, "inks": ["golden_yellow", "true_black"], "separation": "flash", "fill": "packed"})
    plan = corpus / "flow" / "prompts-r6.json"
    plan.write_text(json.dumps({"round": 6, "tile": tile_round(tile) if tile else None, "prompts": prompts}))
    return corpus, plan


def test_tiles_separates_a_tile_round_into_the_corpus_at_the_tile(tmp_path, capsys):
    corpus, plan = _tile_round(tmp_path)
    out = corpus / "tiles-r6.json"
    assert cli.main(["tiles", "--plan", str(plan), "--corpus", str(corpus), "--stencil", str(_print(tmp_path / "p")),
                     "--out", str(out)]) == 0
    index = json.loads(out.read_text())
    assert index["tile"] == "forearm18" and index["stencil"]["page_mm"] == [40, 62] and not index["acquired"]
    assert index["minutes_budget"] == 51
    assert [r["id"] for r in index["tiles"]] == ["r6-01-comic-a-koi-1", "r6-01-comic-a-koi-2", "r6-02-comic-a-koi-1",
                                                  "r6-02-comic-a-koi-2"]
    for r in index["tiles"]:
        report = json.loads((corpus / "separations" / r["id"] / "separation.json").read_text())
        assert report["fit"] == {"room_mm": [40.0, 62.0]} and r["fits"] and r["canvas_mm"] == report["canvas_mm"]
        assert report["generation"]["prompt_id"] == r["prompt_id"]   # the sidecar travels with the separation
    assert max(r["canvas_mm"][1] for r in index["tiles"]) == pytest.approx(62, abs=0.15)
    assert max(r["canvas_mm"][0] for r in index["tiles"]) == pytest.approx(40, abs=0.15)
    assert "4 designs on forearm18 print" in capsys.readouterr().out


def test_tiles_takes_named_designs_and_lists_one_that_fails(tmp_path, capsys):
    corpus, plan = _tile_round(tmp_path)
    Image.new("RGB", (64, 64), "white").save(corpus / "flow" / "r6-02-comic-a-koi" / "r6-02-comic-a-koi-1.jpg")
    out = tmp_path / "tiles.json"
    assert cli.main(["tiles", "r6-02-comic-a-koi-1", "r6-02-comic-a-koi-2", "--plan", str(plan), "--corpus",
                     str(corpus), "--stencil", str(_print(tmp_path / "p")), "--out", str(out)]) == 1
    index = json.loads(out.read_text())
    assert [r["id"] for r in index["tiles"]] == ["r6-02-comic-a-koi-2"]
    assert index["failed"][0]["id"] == "r6-02-comic-a-koi-1" and "no artwork" in index["failed"][0]["reason"]


@pytest.mark.parametrize("tile, page, message", [(None, (40, 62), "not composed for a tile"),
                                                 ("forearm18", (45, 63), "composed for the forearm18 tile")])
def test_tiles_never_fits_a_design_to_another_size(tmp_path, capsys, tile, page, message):
    corpus, plan = _tile_round(tmp_path, tile)
    assert cli.main(["tiles", "--plan", str(plan), "--corpus", str(corpus), "--stencil",
                     str(_print(tmp_path / "p", page)), "--out", str(tmp_path / "tiles.json")]) == 2
    assert message in capsys.readouterr().err and not (corpus / "separations").exists()


def test_promote_draws_a_design_at_the_size_and_on_the_print_it_was_sketched(tmp_path):
    sep = tmp_path / "separations" / "r6-01-comic-a-koi-1"
    assert pyramid.sketched_at(sep) == (pyramid.WIDTH_MM, None)
    sep.mkdir(parents=True)
    (sep / "separation.json").write_text(json.dumps({"canvas_mm": [38.7, 62.0], "page": {"stencil": "/prints/f18"}}))
    assert pyramid.sketched_at(sep) == (38.7, Path("/prints/f18"))


def test_the_summary_line_names_colour_mostly_under_the_fine_width():
    report = {"inks": {"bright_red": {}, "true_black": {}}, "colours": [], "layers": [{"ink": "bright_red"}],
              "weakest_source": "nominal",
              "detail": {"fine_mm": 2.0, "gap_mm": 1.0, "darkest": "true_black", "gaps_mm2": 3.5,
                         "inks": {"bright_red": {"area_mm2": 40.0, "under_mm2": 30.0, "share": 0.75, "dropped_mm2": 2.0},
                                  "true_black": {"area_mm2": 90.0, "under_mm2": 80.0, "share": 0.89, "dropped_mm2": 0}}}}
    line = pipeline.summary_line(report)
    assert "colour under 2 mm 75% (bright_red 75%)" in line and "3.5 mm² of paper gaps under 1 mm" in line
    assert "nothing drawn in true_black" in line
