"""The marker budget: the predictor against what the cameras recorded, the budget's refusals, the compile."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from ink_designs import REPO, artwork, element, program, write

pytest.importorskip("cv2")
pytest.importorskip("PIL")

import tatbot_ink  # noqa: E402
from tatbot_ink import CompileError, markers  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
# The wrist cameras' decoded markers at each stroke's hover over the chicano TATBOT's black (2026-11-10, 129
# strokes on the silicone tile, seed 0), the session camera's and the second's: 9-10 a reading fell to 1-2.
RECORDED = {
    "wrist_2": [15, 14, 15, 17, 12, 7, 7, 9, 7, 9, 8, 6, 9, 8, 9, 8, 8, 12, 12, 9, 7, 5, 7, 6, 6, 13, 12, 11, 14, 12,
                10, 11, 7, 6, 4, 6, 6, 6, 13, 11, 13, 9, 8, 5, 5, 7, 7, 6, 4, 7, 4, 5, 4, 4, 4, 4, 5, 4, 6, 8, 8, 10,
                11, 7, 9, 9, 10, 9, 10, 7, 7, 8, 6, 4, 6, 5, 4, 7, 5, 4, 5, 6, 6, 5, 6, 6, 5, 4, 5, 4, 3, 2, 2, 2, 3,
                2, 2, 1, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 1, 2, 2, 3, 4, 4, 4, 2, 3, 3, 2, 2, 1, 1, 2, 2, 2, 1, 3, 1, 1],
    "wrist_1": [4, 5, 6, 6, 9, 8, 9, 9, 9, 9, 11, 6, 13, 12, 13, 14, 10, 9, 13, 9, 11, 13, 8, 12, 8, 8, 9, 9, 14, 8,
                12, 10, 11, 10, 11, 11, 7, 9, 7, 8, 11, 6, 8, 10, 11, 12, 11, 10, 9, 12, 12, 10, 10, 12, 11, 9, 13,
                10, 12, 11, 11, 12, 10, 8, 9, 8, 3, 10, 8, 9, 9, 10, 9, 10, 8, 10, 9, 10, 9, 9, 8, 7, 8, 5, 5, 4, 4,
                2, 4, 4, 2, 3, 3, 4, 3, 2, 2, 1, 2, 1, 0, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 6, 6, 3, 4, 5, 6,
                3, 4, 2, 4, 5, 5, 4, 5],
}
OVERHEAD_RECORDED = (20, 1)   # the overhead's markers before the black and after it, read from its frames


@pytest.fixture(scope="module")
def prints(tmp_path_factory):
    """Generated tile prints by layout, seed 0 (the silicone one is the rig's installed print)."""
    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    import stencil_tile

    made = {}

    def get(layout):
        if layout not in made:
            made[layout] = tmp_path_factory.mktemp(layout)
            stencil_tile.artwork(layout, 0, made[layout])
        return made[layout]
    return get


def chicano(name):
    return json.loads((FIXTURES / f"chicano-{name}.json").read_text())


def strokes(lines, width=0.0009):
    """What markers.predict reads of a program: one black resource drawing `lines` (page metres)."""
    return {"resources": [{"id": "black", "rgb": [60, 60, 63], "tool": {"id": "tool", "line_width_m": None}}],
            "ops": [{"op": "stroke", "id": f"s{n:04d}", "resource_id": "black", "closed": False,
                     "generation_width_m": width, "points_m": line} for n, line in enumerate(lines)]}


def circle(radius_m, at=(0.0, 0.0)):
    return [[at[0] + radius_m * np.cos(a), at[1] + radius_m * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 60)]


def test_the_model_reproduces_what_the_cameras_read_of_the_chicano_black(prints):
    """The overhead's 20 markers of 23 fell to 1; the wrist's readings fell from 9-10 to 1-2 over the strokes."""
    report = markers.predict(chicano("black"), prints("silicone"), REPO)
    assert report["markers"] == 23 and report["anchors"] == 4
    assert report["clean"]["overhead"] >= OVERHEAD_RECORDED[0]
    assert report["inks"][0]["overhead"] <= OVERHEAD_RECORDED[1] + 1
    curve = report["wrist"]["curve"]
    predicted = {}
    for (start, _, near), following in zip(curve, [*curve[1:], [len(RECORDED["wrist_2"]), 0, 0]], strict=True):
        predicted.update(dict.fromkeys(range(start, following[0]), near))
    seen = np.mean([RECORDED["wrist_2"], RECORDED["wrist_1"]], axis=0)
    error = np.array([predicted[n] for n in range(len(seen))]) - seen
    assert np.abs(error).mean() < 1.8                 # markers, over the run
    assert abs(error[80:].mean()) < 1.0               # and unbiased where the readings ran out
    under = sum(predicted[n] < 3 for n in predicted)
    assert 15 <= under <= 45                          # 34 strokes' hovers read under 3 on one camera or the other


def test_the_chicano_programs_are_refused_naming_the_stage_and_the_markers(prints):
    black = markers.predict(chicano("black"), prints("silicone"), REPO)
    lines = markers.refusals(black)
    assert len(lines) == 2
    assert re.match(r"after true_black \(stroke s0157\) the overhead would read [012] of the print's 23 markers", lines[0])
    assert "the budget is 4, 2 of them anchors" in lines[0] and "lost 0, 1, 5," in lines[0]
    assert "a wrist view would hold" in lines[1] and "the budget is 3" in lines[1]
    red = markers.predict(chicano("red"), prints("silicone"), REPO)
    assert markers.refusals(red) and red["inks"][0]["anchors"] < 2
    # the red as it was drawn, over the black: the print it started on was already past the overhead's reading
    over = markers.predict(chicano("red"), prints("silicone"), REPO, drawn=[chicano("black")])
    assert over["clean"]["overhead"] <= 2 and over["wrist"]["min"] <= 1
    assert re.match(r"markers: the overhead reads 23 of 23 before and [012] after", markers.summary(black))


def test_a_small_design_on_a_forearm18_tile_is_within_the_budget(prints):
    report = markers.predict(strokes([circle(0.006)]), prints("forearm18"), REPO)
    assert report["markers"] == 6 and report["anchors"] == 0 and report["clean"] == {"overhead": 6, "wrist": 6}
    assert report["inks"][0]["overhead"] >= 4 and report["wrist"]["min"] >= 3
    assert markers.refusals(report) == []
    # the same tile hatched across its top half: two markers left of six
    hatch = [[[-0.018, y], [0.018, y]] for y in np.arange(0.002, 0.029, 0.002)]
    lines = markers.refusals(markers.predict(strokes(hatch), prints("forearm18"), REPO))
    assert lines and "the budget is 4)" in lines[0]   # a layout without anchors asks for none


def test_each_ink_is_a_stage_and_an_empty_program_keeps_every_marker(prints):
    two = strokes([[[-0.018, 0.022], [0.018, 0.022]], [[-0.018, -0.022], [0.018, -0.022]]])
    two["resources"].append({"id": "red", "rgb": [255, 0, 0], "tool": {"id": "tool", "line_width_m": None}})
    two["ops"][1]["resource_id"] = "red"
    report = markers.predict(two, prints("forearm18"), REPO)
    assert [(ink["resource"], ink["after"]) for ink in report["inks"]] == [("black", "s0000"), ("red", "s0001")]
    assert report["inks"][0]["overhead"] >= report["inks"][1]["overhead"]
    assert set(report["inks"][0]["lost"]) <= set(report["inks"][1]["lost"])
    empty = markers.predict(strokes([]), prints("forearm18"), REPO)
    assert empty["inks"] == [] and empty["wrist"]["min"] == 6 and markers.refusals(empty) == []


def test_a_print_without_markers_has_no_budget(tmp_path):
    (tmp_path / "tracking.json").write_text(json.dumps({"pattern_id": "stencil-plain", "page_mm": [83, 127]}))
    assert markers.predict(strokes([circle(0.01)]), tmp_path, REPO) is None
    assert markers.refusals(None) == [] and "no marker budget" in markers.summary(None)


def design(tmp_path, name, lines, canvas=(0.036, 0.058)):
    """Acquired artwork of `lines` (canvas metres, origin its top-left as acquired), centred on the page."""
    return write(tmp_path, artwork(program([("pen-1", [element(line, width=0.0009) for line in lines])], canvas=canvas)),
                 f"{name}.json")


def test_the_compile_writes_the_counts_and_says_when_a_design_is_under_the_budget(tmp_path, prints, monkeypatch):
    small = design(tmp_path, "small", [np.add(circle(0.006), [0.018, 0.029]).tolist()])
    result = tatbot_ink.compile(small, repo=REPO, stencil=prints("forearm18"))
    counts = result["stats"]["markers"]
    assert counts["within_budget"] and counts["clean"] == {"overhead": 6, "wrist": 6}
    assert counts["budget"] == {"overhead_min": 4, "anchors_min": 2, "wrist_min": 3}
    assert "allow_marker_loss" not in result["preparation"]
    assert "markers" not in tatbot_ink.compile(small, repo=REPO)["stats"]   # no print named: nothing to count

    hatch = design(tmp_path, "hatch", [[[0.001, y], [0.035, y]] for y in np.arange(0.002, 0.056, 0.002)])
    noted = tatbot_ink.compile(hatch, repo=REPO, stencil=prints("forearm18"))   # stack.yaml: not enforced
    assert not noted["stats"]["markers"]["within_budget"]
    assert any(note.startswith("marker budget (not enforced") for note in noted["stats"]["notes"])

    enforced = {**markers.settings(REPO), "enforce": True}
    monkeypatch.setattr(markers, "settings", lambda repo: enforced)
    with pytest.raises(CompileError, match="covers the markers its print is tracked by: after .* the overhead would read"):
        tatbot_ink.compile(hatch, repo=REPO, stencil=prints("forearm18"))
    allowed = tatbot_ink.compile(hatch, repo=REPO, stencil=prints("forearm18"), allow_marker_loss=True)
    assert allowed["preparation"]["allow_marker_loss"] is True and not allowed["stats"]["markers"]["within_budget"]
    assert tatbot_ink.compile(small, repo=REPO, stencil=prints("forearm18"))["stats"]["markers"]["within_budget"]


def test_a_count_that_cannot_be_made_is_a_note_until_the_budget_is_enforced(tmp_path, prints, monkeypatch):
    """A layout record the detector cannot load: the program is prepared, and refused only where it must be counted."""
    broken = tmp_path / "print"
    broken.mkdir()
    (broken / "tracking.json").write_text((prints("forearm18") / "tracking.json").read_text())
    (broken / "tile.json").write_text(json.dumps({"layout": "forearm18", "tile_mm": [40, 62], "markers": []}))
    small = design(tmp_path, "small", [np.add(circle(0.006), [0.018, 0.029]).tolist()])
    noted = tatbot_ink.compile(small, repo=REPO, stencil=broken)
    assert "markers" not in noted["stats"] and noted["stats"]["notes"][-1].startswith("marker budget not counted: the print's")
    enforced = {**markers.settings(REPO), "enforce": True}
    monkeypatch.setattr(markers, "settings", lambda repo: enforced)
    with pytest.raises(CompileError, match="could not be counted"):
        tatbot_ink.compile(small, repo=REPO, stencil=broken)
    assert "markers" not in tatbot_ink.compile(small, repo=REPO, stencil=broken, allow_marker_loss=True)["stats"]


def test_ink_already_on_the_print_counts_against_the_markers(tmp_path, prints, monkeypatch):
    """--after: a second program on a print the first covered is under the budget though its own ink is not."""
    hatch = design(tmp_path, "hatch", [[[0.001, y], [0.035, y]] for y in np.arange(0.002, 0.056, 0.002)])
    first = tatbot_ink.compile(hatch, repo=REPO, stencil=prints("forearm18"))
    path = tatbot_ink.write_program(first, tmp_path / "first" / "program.json")
    small = design(tmp_path, "small", [np.add(circle(0.006), [0.018, 0.029]).tolist()])
    second = tatbot_ink.compile(small, repo=REPO, stencil=prints("forearm18"), after=[path])
    assert not second["stats"]["markers"]["within_budget"] and len(second["preparation"]["drawn_after"]) == 1
    wiped = tatbot_ink.compile(small, repo=REPO, stencil=prints("forearm18"), after=[path], wiped=True)
    assert not wiped["stats"]["markers"]["within_budget"]   # a wipe takes off wet ink, not the lines over the markers
    assert second["stats"]["markers"]["clean"]["overhead"] < 4


def test_spared_order_covers_markers_late_and_keeps_each_inks_paths_together(tmp_path, prints, monkeypatch):
    """page.markers.order spare: the path over bare skin first, then of those reaching one new marker the nearest;
    source order unless the setting asks."""
    over = [[0.002, 0.008], [0.016, 0.008]]          # across the bottom-left marker (canvas metres, y up)
    other = [[0.020, 0.030], [0.034, 0.030]]         # across the middle-right marker
    again = [[0.002, 0.012], [0.016, 0.012]]         # the bottom-left marker again
    bare = [[0.018, 0.020], [0.018, 0.030]]          # up the gap between the columns: no marker
    path = design(tmp_path, "order", [over, other, again, bare])
    plain = tatbot_ink.compile(path, repo=REPO, stencil=prints("forearm18"), allow_marker_loss=True)
    assert "stroke_order" not in plain["preparation"]
    source = [op["src"]["path"] for op in plain["ops"] if op["op"] == "stroke"]
    assert source == ["path-0-0", "path-0-1", "path-0-2", "path-0-3"]

    spared = {**markers.settings(REPO), "order": "spare"}
    monkeypatch.setattr(markers, "settings", lambda repo: spared)
    program = tatbot_ink.compile(path, repo=REPO, stencil=prints("forearm18"), allow_marker_loss=True)
    assert program["preparation"]["stroke_order"] == "spare"
    drawn = [op["src"]["path"] for op in program["ops"] if op["op"] == "stroke"]
    # bare skin; the nearest of the three that each reach a new marker; the nearer of the two over the bottom-left
    # marker; and the other of them, which by then reaches nothing new
    assert drawn == ["path-0-3", "path-0-1", "path-0-2", "path-0-0"]
    assert program["stats"]["contact_m"] == plain["stats"]["contact_m"]
    assert tatbot_ink.compile(path, repo=REPO)["preparation"].get("stroke_order") is None   # no print: source order

    spared["order"] = "farthest"
    with pytest.raises(CompileError, match="source or spare"):
        tatbot_ink.compile(path, repo=REPO, stencil=prints("forearm18"))


def test_sparing_the_markers_keeps_more_of_them_in_the_wrists_view_over_the_chicano_black(prints):
    from tatbot_ink.place import Stroke

    black = chicano("black")
    placed = [Stroke(np.asarray(op["points_m"], float), False, op["generation_width_m"], {"id": op["id"]})
              for op in black["ops"]]
    ordered = markers.spare_order(placed, lambda s: "true_black", lambda s: s.generation_width_m, prints("silicone"), REPO)
    assert sorted(s.src["id"] for s in ordered) == sorted(op["id"] for op in black["ops"])
    by_id = {op["id"]: op for op in black["ops"]}
    spared = markers.predict({**black, "ops": [by_id[s.src["id"]] for s in ordered]}, prints("silicone"), REPO)
    source = markers.predict(black, prints("silicone"), REPO)
    assert spared["wrist"]["under"] < source["wrist"]["under"] / 2
    assert spared["inks"][0]["overhead"] == source["inks"][0]["overhead"]   # the same ink in the end
