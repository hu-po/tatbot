"""python/tatbot_art brief: `tatbot art separate --brief` separates an Inkmap placement's image at its placed width."""
import base64
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tatbot_art import acquire, brief, cli, pipeline
from tatbot_art.image import art_box, flatten_paper, load
from tatbot_contracts.art import validate_render
from tatbot_contracts.canonical import canonical_digest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = json.loads((REPO / "config/inkmap/examples/forearm-placement-v7.json").read_text())


def _rehash(value: dict) -> dict:
    value["content_sha256"] = canonical_digest(value)
    return value


def _brief(tmp_path: Path, *, places=None, art=None, mirror=False) -> Path:
    """What Inkmap's Export brief writes for the example placement (web/inkmap/src/core/brief.ts makeBrief)."""
    record = art or EXAMPLE["designs"]["ring"]
    p = EXAMPLE["placements"][0]
    rows = places or [{"id": p["id"], "art_id": "ring", "width_mm": 30.0, "height_mm": 30.0, "rotation_rad": 0.5,
                       "mirror": mirror, "anchor": p["anchor"], "site": None, "sentence": "on the left forearm"}]
    doc = _rehash({"schema": "tatbot.inkmap-brief/1", "content_sha256": "", "name": "Forearm ring",
                   "body": EXAMPLE["body"], "art": {"ring": record}, "placements": rows})
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(doc))
    return path


def test_the_placement_image_is_written_with_its_width_inks_and_style(tmp_path):
    given = brief.source(_brief(tmp_path), None, tmp_path / "out")
    assert given["image"] == tmp_path / "out" / "source" / "ring.png"
    assert given["image"].read_bytes() == brief.image_bytes(EXAMPLE["designs"]["ring"])
    assert (given["width_mm"], given["inks"], given["style"]) == (30.0, ["true_black"], "geometric")
    assert given["report"]["placement"] == "p-forearm-spiral" and given["report"]["rotation_rad"] == 0.5
    assert not given["image"].with_suffix(".json").exists(), "stock art has no generation sidecar"


def test_generated_art_keeps_its_prompt_and_a_mirrored_placement_is_separated_mirrored(tmp_path):
    art = copy.deepcopy(EXAMPLE["designs"]["ring"])
    art["source"] = {"kind": "generated", "identifier": None, "license": None, "attribution": None,
                     "generation": {"prompt": "a ring of ink", "model": "test-model", "generator": "flow-ui"}}
    given = brief.source(_brief(tmp_path, art=_rehash(art), mirror=True), None, tmp_path / "out")
    assert given["image"].name == "ring-mirrored.png"
    original = np.asarray(Image.open(tmp_path / "out" / "source" / "ring-mirrored.png").convert("RGB"))
    source = np.asarray(Image.open(io.BytesIO(brief.image_bytes(art))).convert("RGB"))
    assert np.array_equal(original, source[:, ::-1])
    side = json.loads(given["image"].with_suffix(".json").read_text())
    assert side == {"schema": "tatbot.art-generation/1", "prompt": "a ring of ink", "model": "test-model",
                    "generator": "flow-ui", "subject": "Ring"}


def test_a_crop_that_disagrees_with_the_brief_is_refused(tmp_path):
    art = copy.deepcopy(EXAMPLE["designs"]["ring"])
    art["art_px"] = [0, 0, 64, 64]
    with pytest.raises(ValueError, match="crops the image to"):
        brief.source(_brief(tmp_path, art=_rehash(art)), None, tmp_path / "out")


def test_several_placements_need_one_named(tmp_path):
    row = {"art_id": "ring", "width_mm": 30.0, "height_mm": 30.0, "rotation_rad": 0, "mirror": False,
           "anchor": EXAMPLE["placements"][0]["anchor"], "site": None, "sentence": None}
    path = _brief(tmp_path, places=[{**row, "id": "a"}, {**row, "id": "b", "width_mm": 41.5, "height_mm": 41.5}])
    with pytest.raises(ValueError, match="places 2 designs"):
        brief.source(path, None, tmp_path / "out")
    with pytest.raises(ValueError, match="the brief places a, b"):
        brief.source(path, "c", tmp_path / "out")
    assert brief.source(path, "b", tmp_path / "out")["width_mm"] == 41.5


def test_a_tampered_brief_is_refused(tmp_path):
    path = _brief(tmp_path)
    doc = json.loads(path.read_text())
    doc["placements"][0]["width_mm"] = 60.0
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="content digest"):
        brief.source(path, None, tmp_path / "out")


def test_separate_plans_the_brief_at_its_width(tmp_path, capsys):
    out = tmp_path / "sep"
    assert cli.main(["separate", "--brief", str(_brief(tmp_path)), "--out", str(out)]) == 0
    report = json.loads((out / "separation.json").read_text())
    assert report["canvas_mm"][0] == 30.0 and report["style"] == "geometric" and list(report["inks"]) == ["true_black"]
    assert report["brief"]["placement"] == "p-forearm-spiral" and report["brief"]["width_mm"] == 30.0
    assert "planned only" in capsys.readouterr().out
    assert cli.main(["separate", "--brief", str(_brief(tmp_path)), "--width-mm", "40", "--out", str(out)]) == 2
    assert "Inkmap is where they change" in capsys.readouterr().err
    assert cli.main(["separate", "--out", str(out)]) == 2


def _two_placements(tmp_path: Path, **second) -> Path:
    row = {"art_id": "ring", "width_mm": 30.0, "height_mm": 30.0, "rotation_rad": 0, "mirror": False,
           "anchor": EXAMPLE["placements"][0]["anchor"], "site": None, "sentence": None}
    return _brief(tmp_path, places=[{**row, "id": "a"}, {**row, "id": "b", **second}])


def test_a_brief_with_several_placements_separates_each_in_drawing_order(tmp_path, capsys):
    out = tmp_path / "sep"
    assert cli.main(["separate", "--brief", str(_two_placements(tmp_path, width_mm=41.5, height_mm=41.5)),
                     "--out", str(out)]) == 0
    first, second = (json.loads((out / folder / "separation.json").read_text()) for folder in ("01-ring", "02-ring"))
    assert first["brief"]["placement"] == "a" and second["brief"]["placement"] == "b"
    assert second["canvas_mm"][0] == 41.5
    assert "2 of 2 placements separated" in capsys.readouterr().out




def test_one_failing_placement_does_not_stop_the_rest(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(acquire, "generate", lambda *a: pytest.fail("a design past its page must not reach DBV3"))
    monkeypatch.setattr(pipeline, "build", _build_only_checking_the_page)
    monkeypatch.setattr(pipeline, "ink_on_white", lambda *_: np.ones((8, 8, 3)))
    out = tmp_path / "sep"
    assert cli.main(["separate", "--brief", str(_two_placements(tmp_path, width_mm=100.0, height_mm=100.0)),
                     "--out", str(out), "--app", "/nonexistent"]) == 1
    run = capsys.readouterr()
    assert "02-ring: the design is 100 x 100 mm" in run.err and "1 of 2 placements separated" in run.out
    assert (out / "01-ring" / "separation.json").exists()
    assert "failed" in json.loads((out / "02-ring" / "separation.json").read_text())


def _build_only_checking_the_page(sep, catalog, jobs, canvas_mm, report, out, *, app, substrate, stencil, at,
                                  allow_marker_loss=False):
    if refusal := pipeline.check_page(report, canvas_mm, stencil, at):
        raise ValueError(refusal)
    return {**report, "line_mm": {}, "time": {"minutes": 0, "dips": 0, "rinses": 0},
            "fidelity": {"vs_target": {"mean": 0}}}


def test_a_design_past_the_page_is_refused_before_dbv3_and_leaves_its_report(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(acquire, "generate", lambda *a: pytest.fail("a design past its page must not reach DBV3"))
    row = {"id": "big", "art_id": "ring", "width_mm": 100.0, "height_mm": 100.0, "rotation_rad": 0, "mirror": False,
           "anchor": EXAMPLE["placements"][0]["anchor"], "site": None, "sentence": None}
    out = tmp_path / "sep"
    assert cli.main(["separate", "--brief", str(_brief(tmp_path, places=[row])), "--out", str(out),
                     "--app", "/nonexistent"]) == 2
    assert "the design is 100 x 100 mm, and the nominal page holds 90 x 134 mm" in capsys.readouterr().err
    report = json.loads((out / "separation.json").read_text())
    assert report["page"]["fits"] is False and report["page"]["drawable_mm"] == [90.0, 134.0]
    assert "100 x 100 mm" in report["failed"]


def test_the_entry_point_reaches_the_page_check(tmp_path):
    """scripts/art.py's own import roots, not this suite's: -I drops everything conftest added."""
    run = subprocess.run([sys.executable, "-I", str(REPO / "scripts/art.py"), "separate", "--brief",
                          str(_brief(tmp_path)), "--out", str(tmp_path / "sep")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads((tmp_path / "sep" / "separation.json").read_text())["page"]["fits"] is True


def test_planning_past_the_page_only_says_so(tmp_path, capsys):
    row = {"id": "big", "art_id": "ring", "width_mm": 100.0, "height_mm": 100.0, "rotation_rad": 0, "mirror": False,
           "anchor": EXAMPLE["placements"][0]["anchor"], "site": None, "sentence": None}
    assert cli.main(["separate", "--brief", str(_brief(tmp_path, places=[row])), "--out", str(tmp_path / "sep")]) == 0
    assert "holds 90 x 134 mm" in capsys.readouterr().out


def test_the_page_check_reads_a_named_print_and_the_canvas_centre(tmp_path):
    (tmp_path / "tracking.json").write_text(json.dumps(
        {"pattern_id": "tile", "page_mm": [45, 63], "clear_center_uv": [0, 0, 1, 1]}))
    assert acquire.page_fit((45, 63), tmp_path)["fits"]
    assert not acquire.page_fit((45.5, 63), tmp_path)["fits"]
    assert acquire.page_fit((60, 60))["fits"]
    assert not acquire.page_fit((60, 60), at="20,0")["fits"]


def test_the_summary_names_an_ink_that_drew_nothing():
    report = {"inks": {"light_green": {}, "true_black": {}}, "colours": [], "layers": [{"ink": "true_black"}],
              "weakest_source": "nominal"}
    assert pipeline.summary_line(report).endswith("nothing drawn in light_green")
    drawn = {**report, "layers": [{"ink": "true_black"}, {"ink": "light_green"}], "line_mm": {"true_black": 40},
             "time": {"minutes": 1, "dips": 1, "rinses": 0}, "fidelity": {"vs_target": {"mean": 5}}}
    assert pipeline.summary_line(drawn).endswith("nothing drawn in light_green")
    assert "nothing drawn" not in pipeline.summary_line({**drawn, "line_mm": {"true_black": 40, "light_green": 9}})


FIXTURES = REPO / "web/inkmap/tests/fixtures"


def test_a_transparent_image_is_on_white_paper_and_a_photo_is_upright_as_inkmap_reads_them():
    """web/inkmap/tests/e2e/art.mjs holds the browser to these same rectangles."""
    assert art_box(flatten_paper(load(FIXTURES / "art-alpha.png"))) == (54, 64, 93, 93)
    photo = load(FIXTURES / "art-exif6.jpg")
    assert photo.shape[:2] == (200, 100)
    assert art_box(flatten_paper(photo)) == (23, 14, 53, 53)


def test_a_mirrored_transparent_design_keeps_its_paper(tmp_path):
    data = (FIXTURES / "art-alpha.png").read_bytes()
    art = copy.deepcopy(EXAMPLE["designs"]["ring"])
    art["image"] = {"media_type": "image/png", "sha256": hashlib.sha256(data).hexdigest(), "width_px": 200,
                    "height_px": 200, "data_base64": base64.b64encode(data).decode()}
    art["art_px"] = [54, 64, 93, 93]
    given = brief.source(_brief(tmp_path, art=_rehash(art), mirror=True), None, tmp_path / "out")
    pixels = np.asarray(Image.open(given["image"]).convert("RGB"))
    assert (pixels[0, 0] == 255).all(), "a transparent corner is white paper, not black ink"


def test_the_render_inkmap_loads_is_the_placements_ink_on_white_and_nothing_else(tmp_path):
    """web/inkmap/tests/render-record.test.ts reads this module's fixture: one record, two readers."""
    assert validate_render(json.loads((FIXTURES / "render.json").read_text()))["lost_inks"] == ["light_green"]
    report = {"weakest_source": "vendor-photo", "time": {"minutes": 12.5, "dips": 7, "rinses": 1},
              "brief": {"art_sha256": "a" * 64, "width_mm": 6.0, "height_mm": 4.0, "mirror": True}}
    picture = np.ones((40, 60, 3))
    picture[10:20, 20:30] = 0.2
    record = json.loads(brief.write_render(tmp_path, report, picture, ["light_green"]).read_text())
    assert validate_render(record) == record
    assert (record["width_mm"], record["mirror"], record["dips"]) == (6.0, True, 7)
    assert (record["image"]["width_px"], record["image"]["height_px"]) == (60, 40)
    assert "strokes" not in json.dumps({k: v for k, v in record.items() if k != "image"}), "a picture, never paths"
    with pytest.raises(ValueError, match="content digest"):
        validate_render({**record, "width_mm": 7.0})


def test_ink_on_white_is_the_design_canvas_wherever_it_sits_on_the_page(tmp_path):
    (tmp_path / "program").mkdir()
    stroke = {"op": "stroke", "resource_id": "k", "points_m": [[0.008, 0.0], [0.012, 0.0]], "generation_width_m": 0.0006}
    (tmp_path / "program" / "program.json").write_text(json.dumps(
        {"page": {"size_m": [0.05, 0.04]}, "resources": [{"id": "k", "ink_id": "true_black", "tool": {}}], "ops": [stroke]}))
    picture = pipeline.ink_on_white(tmp_path, {"true_black": {"rgb": [0, 0, 0]}}, (10.0, 6.0), 10.0, [10.0, 0.0])
    assert picture.shape[:2] == (60, 100), "the canvas, not the page"
    assert picture[30, 50].max() < 0.6, "a stroke through the canvas's centre, where --at put it"
    assert (picture[0, 0] > 0.99).all(), "white round it, for a viewer to multiply over skin"


def test_a_stock_brief_is_the_collections_art_at_a_width():
    entry = json.loads((REPO / "web/inkmap/public/designs/manifest.json").read_text())["designs"][0]
    record = brief.stock_art(entry, (REPO / "web/inkmap/public" / entry["image"]).read_bytes())
    assert record["palette"] == entry["palette"] and record["art_px"] == entry["art_px"]
    doc = brief.stock_brief(entry["id"], record, 45.0)
    place = doc["placements"][0]
    assert (place["art_id"], place["width_mm"], place["mirror"]) == (entry["id"], 45.0, False)
    assert place["height_mm"] == round(45 * entry["art_px"][3] / entry["art_px"][2], 3)


def test_stock_renders_writes_each_width_and_lists_it_in_the_collection(tmp_path, monkeypatch):
    designs = tmp_path / "public" / "designs"
    designs.mkdir(parents=True)
    source = json.loads((REPO / "web/inkmap/public/designs/manifest.json").read_text())
    entry = {k: v for k, v in source["designs"][0].items() if k != "renders"}
    (designs / "manifest.json").write_text(json.dumps({**source, "designs": [entry]}))
    (designs / Path(entry["image"]).name).write_bytes((REPO / "web/inkmap/public" / entry["image"]).read_bytes())
    real_main = cli.main

    def separate(argv):
        if argv[0] != "separate":
            return real_main(argv)
        (Path(argv[argv.index("--out") + 1]) / "inkmap-render.json").write_text("{}")
        return 0
    monkeypatch.setattr(cli, "main", separate)
    assert real_main(["stock-renders", "--app", "/nonexistent", "--widths", "30,45.5", "--designs", str(designs),
                      "--out", str(tmp_path / "work")]) == 0
    written = json.loads((designs / "manifest.json").read_text())["designs"][0]["renders"]
    assert written == [f"designs/renders/{entry['id']}-30.json", f"designs/renders/{entry['id']}-45.5.json"]
    assert (designs / "renders" / f"{entry['id']}-45.5.json").exists()


def test_stock_renders_separate_with_the_catalogue_it_was_given(tmp_path, monkeypatch):
    designs = tmp_path / "public" / "designs"
    designs.mkdir(parents=True)
    source = json.loads((REPO / "web/inkmap/public/designs/manifest.json").read_text())
    entry = {k: v for k, v in source["designs"][0].items() if k != "renders"}
    (designs / "manifest.json").write_text(json.dumps({**source, "designs": [entry]}))
    (designs / Path(entry["image"]).name).write_bytes((REPO / "web/inkmap/public" / entry["image"]).read_bytes())
    real_main, seen = cli.main, []

    def separate(argv):
        seen.append(argv)
        (Path(argv[argv.index("--out") + 1]) / "inkmap-render.json").write_text("{}")
        return 0
    monkeypatch.setattr(cli, "main", separate)
    catalog = REPO / "config" / "inks.yaml"
    assert real_main(["--catalog", str(catalog), "stock-renders", "--app", "/nonexistent", "--widths", "30",
                      "--designs", str(designs), "--out", str(tmp_path / "work")]) == 0
    assert seen[0][:2] == ["--catalog", str(catalog)]
