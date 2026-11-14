"""`tatbot art separate --fit` on a fiducial-tile print: the widest a design keeps the print's markers at."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from tatbot_art import cli, fit, pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def forearm18(tmp_path_factory):
    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    import stencil_tile

    out = tmp_path_factory.mktemp("forearm18")
    stencil_tile.artwork("forearm18", 0, out)
    return out


def _art(path: Path) -> Path:
    """A solid black block, 200 x 300 px, inside a white margin: a design that fills whatever it is drawn on."""
    img = np.full((340, 240, 3), 255, np.uint8)
    img[20:320, 20:220] = (20, 20, 20)
    Image.fromarray(img).save(path)
    return path


def _separate(tmp_path, forearm18, *extra):
    out = tmp_path / "sep"
    code = cli.main(["separate", str(_art(tmp_path / "block.png")), "--inks", "true_black", "--fit", "--stencil",
                     str(forearm18), "--out", str(out), *extra])
    return code, out


def test_a_design_filling_the_tile_is_fitted_down_to_what_its_markers_carry(tmp_path, forearm18, capsys):
    code, out = _separate(tmp_path, forearm18)
    assert code == 0
    report = json.loads((out / "separation.json").read_text())
    record = report["fit"]["markers"]
    assert report["fit"]["room_mm"] == [40.0, 62.0] and record["page_mm"] == pytest.approx(40, abs=0.15)
    steps = record["steps"]
    assert len(steps) > 1 and not steps[0]["within"] and steps[-1]["within"]
    assert [s["within"] for s in steps[:-1]] == [False] * (len(steps) - 1)        # the first size that holds
    assert all(b["width_mm"] < a["width_mm"] for a, b in zip(steps, steps[1:], strict=False))
    assert steps[0]["overhead"] < 4 <= steps[-1]["overhead"] and steps[-1]["wrist"] >= 3
    assert record["carried_mm"] == steps[-1]["width_mm"] == report["canvas_mm"][0] < record["page_mm"]
    assert {s["by"] for s in steps} == {"separation"}
    assert f"the print's markers carry {record['carried_mm']:g} mm" in capsys.readouterr().out


def test_allow_marker_loss_keeps_the_widest_the_page_holds(tmp_path, forearm18):
    code, out = _separate(tmp_path, forearm18, "--allow-marker-loss")
    report = json.loads((out / "separation.json").read_text())
    assert code == 0 and report["fit"] == {"room_mm": [40.0, 62.0]}
    assert report["canvas_mm"][0] == pytest.approx(40, abs=0.15)


def test_the_compiled_programs_count_has_the_last_word(tmp_path, forearm18, monkeypatch):
    """With DBV3, each size is compiled and the fit goes on down while the program's own paths are under the
    budget: the separation's regions and the drawn paths are two counts of one design."""
    built = []

    def build(sep, catalog, jobs, canvas_mm, report, out, **kwargs):
        built.append(canvas_mm[0])
        under = len(built) < 3   # the first two sizes' paths cover too much
        ink = {"resource": "true_black", "after": "s0009", "overhead": 3 if under else 5, "anchors": 0, "lost": [1]}
        counted = {"markers": 6, "clean": {"overhead": 6}, "anchors": 0, "inks": [ink],
                   "wrist": {"min": 4, "at": "s0002", "under": 0, "lost": [], "curve": []},
                   "budget": {"overhead_min": 4, "anchors_min": 2, "wrist_min": 3}, "within_budget": not under}
        return {**report, "compile": {"markers": counted}, "time": {"minutes": 1, "dips": 1, "rinses": 0},
                "fidelity": {"vs_target": {"mean": 1.0}}, "line_mm": {"true_black": 1}}

    monkeypatch.setattr(pipeline, "build", build)
    code, out = _separate(tmp_path, forearm18, "--app", "/nowhere")
    record = json.loads((out / "separation.json").read_text())["fit"]["markers"]
    program = [s for s in record["steps"] if s["by"] == "program"]
    assert code == 0 and [s["within"] for s in program] == [False, False, True]
    assert record["carried_mm"] == program[-1]["width_mm"] == pytest.approx(built[-1], abs=0.01)
    assert built[0] > built[1] > built[2]


def test_a_design_no_size_of_which_keeps_the_markers_is_refused(tmp_path, forearm18, monkeypatch, capsys):
    under = {"clean": {"overhead": 6}, "anchors": 0, "inks": [{"resource": "true_black", "after": "true_black",
             "overhead": 0, "anchors": 0, "lost": [1]}], "markers": 6, "wrist": {"min": 6, "at": None},
             "budget": {"overhead_min": 4, "anchors_min": 2, "wrist_min": 3}}
    monkeypatch.setattr(fit, "counts", lambda *args: under)
    code, _ = _separate(tmp_path, forearm18)
    assert code == 2 and "no size of this design down to 6 mm leaves the print the markers" in capsys.readouterr().err


def test_a_print_without_markers_is_fitted_to_its_page_alone(tmp_path):
    (tmp_path / "print").mkdir()
    (tmp_path / "print" / "tracking.json").write_text(json.dumps(
        {"pattern_id": "stencil-plain", "page_mm": [40, 62], "clear_center_uv": [0, 0, 1, 1]}))
    code, out = _separate(tmp_path, tmp_path / "print")
    assert code == 0 and json.loads((out / "separation.json").read_text())["fit"] == {"room_mm": [40.0, 62.0]}
