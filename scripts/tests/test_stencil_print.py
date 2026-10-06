"""Stencil print sheets: rulers, label and exact physical sizes."""

import json
import re

import pytest
from cli_runner import REPO, tatbot
from PIL import Image
from stencil_print import FOOT_MM, LABEL_MM, RULER_MM, ruler_marks, sheet_layout, write_sheets

ARTWORK = REPO/"docs/assets/stencil-frames/tatbot-42"


def test_sheet_wraps_the_page_with_rulers_and_a_label_at_exact_size(tmp_path):
    manifest = write_sheets(ARTWORK, tmp_path, dpi=300)
    assert manifest["sheet_mm"] == [100+RULER_MM, LABEL_MM+150+FOOT_MM]
    assert manifest["page_offset_mm"] == [RULER_MM, LABEL_MM]
    pattern = json.loads((ARTWORK/"tracking.json").read_text())["pattern_id"]
    assert manifest["label"] == f"pattern {pattern[8:20]}   100 x 150 mm"
    assert "150 mm down the side and 100 mm along the bottom" in manifest["paper"]
    sheet = Image.open(tmp_path/"stencil-app.png")
    assert sheet.mode == "1" and round(sheet.info["dpi"][0]) == 300
    assert abs(sheet.width/300*25.4-108) < .2 and abs(sheet.height/300*25.4-160) < .2
    # The page is pasted unchanged at its offset.
    page = Image.open(ARTWORK/"stencil.png").convert("1")
    offset = (round(RULER_MM*300/25.4), round(LABEL_MM*300/25.4))
    assert list(sheet.crop((*offset, offset[0]+page.width, offset[1]+page.height)).getdata()) == list(page.getdata())
    for paper, box in (("a4", (595.2, 841.92)), ("letter", (612., 792.))):
        found = re.search(rb"MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)", (tmp_path/f"paper-{paper}.pdf").read_bytes())
        assert abs(float(found[1])-box[0]) < 1 and abs(float(found[2])-box[1]) < 1
    assert json.loads((tmp_path/"sheet.json").read_text())["files"].keys() == {
        "stencil-app.png", "sheet.svg", "paper-a4.pdf", "paper-letter.pdf"}


def test_ruler_marks_every_millimetre_and_numbers_every_ten():
    marks = ruler_marks(150)
    assert len(marks) == 151 and [m[2] for m in marks if m[2]] == [str(v) for v in range(0, 151, 10)]
    assert {m[1] for m in marks if m[0] % 5 == 0 and m[0] % 10} == {2.0}


def test_a_half_skin_sheet_fits_half_a_skin_with_both_rulers_and_a_centred_label():
    sheet, lines, texts = sheet_layout(83, 127, "pattern 0123456789ab   83 x 127 mm")
    assert sheet == (91, 137) and sheet[0] <= 185/2 and sheet[1] <= 140
    ink = [point for line in lines for point in line] + [(x, y) for x, y, *_ in texts]
    assert all(0 <= x <= sheet[0] and 0 <= y <= sheet[1] for x, y in ink)
    numbers = [(x, y, text) for x, y, text, _, _ in texts if text.isdigit()]
    down = [text for x, _, text in numbers if x < RULER_MM]
    along = [text for _, y, text in numbers if y > LABEL_MM+127]
    assert down == [str(v) for v in range(0, 121, 10)] and along == [str(v) for v in range(0, 81, 10)]
    assert texts[0][0] == RULER_MM+83/2 and texts[0][4] == "ma"   # centred over the page


def test_a_fit_area_smaller_than_the_sheet_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot hold the 108 x 160 mm sheet"):
        write_sheets(ARTWORK, tmp_path, fit_area_mm=(105.0, 200.0))
    assert not any(tmp_path.iterdir())


def test_cli_plans_print_sheets():
    result = tatbot("vision", "stencil", "print", "--dry-run", "--seed", "101", "--output", "/tmp/x")
    assert result.returncode == 0, result.stderr
    assert "stencil_print.py" in result.stdout and "--seed 101" in result.stdout
    refused = tatbot("vision", "stencil", "print", "--dry-run", "--artwork", "x", "--set", "a=b", "--output", "/tmp/x")
    assert refused.returncode != 0
