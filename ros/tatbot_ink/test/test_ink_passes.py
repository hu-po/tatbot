"""Pen passes: a line once is one pass, a retrace or another stroke over it another; a drawn line's encoder jitter
is not; dot-sized paths on their own ink's lines are dropped at preparation, and dots on bare skin or on another
ink's line stay."""
from __future__ import annotations

import numpy as np
import pytest
import tatbot_ink
from ink_designs import REPO, artwork, element, program, write
from tatbot_ink import passes

W = 0.0003   # the 3RL's generation width


def test_one_line_is_one_pass_of_its_footprint():
    record = passes.summary([([[0, 0], [0.01, 0]], W)])
    assert record["max"] == record["p95"] == 1 and record["mm2_at_least"]["2"] == 0 and record["hotspots"] == []
    assert record["inked_mm2"] == pytest.approx(10 * 0.3 + np.pi * 0.15 ** 2, rel=0.05)


def test_retrace_and_overlay_pass_again_and_crossings_only_where_they_cross():
    out_and_back = passes.summary([([[0, 0], [0.01, 0], [0, 0]], W)])
    assert out_and_back["max"] == 2
    assert out_and_back["mm2_at_least"]["2"] == pytest.approx(10 * 0.3 - 0.3 * 0.3, rel=0.1)  # not at the turn
    assert passes.summary([([[0, 0], [0.01, 0]], W)] * 3)["max"] == 3
    cross = passes.summary([([[0, 0], [0.01, 0]], W), ([[0.005, -0.005], [0.005, 0.005]], W)])
    assert cross["max"] == 2 and cross["mm2_at_least"]["2"] == pytest.approx(0.09, rel=0.3)


def test_hotspots_are_the_densest_squares_apart():
    lines = [([[0, y], [0.004, y]], W) for y in np.arange(0, 0.001, 0.0001)] * 2 + [([[0.008, 0], [0.012, 0]], W)] * 9
    hot = passes.summary(lines)["hotspots"]
    assert len(hot) >= 2 and all(h["mean_passes"] > 1 for h in hot)
    assert hot[0]["mean_passes"] >= hot[1]["mean_passes"]
    assert all(np.hypot(*np.subtract(a["at_mm"], b["at_mm"])) >= 2 for a in hot for b in hot if a is not b)


def test_a_hotspot_on_the_page_axes_reads_zero_not_negative_zero():
    # a design centred on the page's origin (2026-10-23's heart) put one at -0.0 mm, which the draw refuses
    at = [v for dy in np.linspace(-1e-4, 1e-4, 21) for dx in np.linspace(-1e-4, 1e-4, 5)
          for h in passes.summary([([[dx - 0.002, dy], [dx + 0.002, dy]], W)] * 3)["hotspots"] for v in h["at_mm"]]
    zeros = [v for v in at if v == 0]
    assert zeros and all(np.copysign(1.0, v) > 0 for v in zeros)


def test_drawn_lines_are_smoothed_before_counting():
    s = np.linspace(0, 0.01, 58)   # 20 Hz at 3.5 mm/s, each sample 0.15 mm off the line like FK of the encoders
    jitter = np.c_[s, np.zeros_like(s)] + np.random.default_rng(1).normal(0, 0.00015, (58, 2))
    plan = {"resources": [{"id": "r", "tool": {"line_width_m": None}}],
            "ops": [{"op": "stroke", "id": "s1", "resource_id": "r", "generation_width_m": W, "closed": False,
                     "points_m": [[0, 0], [0.01, 0]]}]}
    assert passes.summary([(jitter, W)])["max"] >= 2
    assert passes.summary(passes.drawn_lines(plan, {"s1": [jitter.tolist()], "unknown": [[[0, 0]]]}))["max"] == 1
    assert passes.summary([]) == passes.summary(passes.drawn_lines(plan, {}))


def test_the_program_and_its_preview_carry_the_passes(tmp_path):
    lines = [element([[.002, .01], [.018, .01]], width=W) for _ in range(3)]
    result = tatbot_ink.compile(write(tmp_path, artwork(program([("pen-1", lines)]))), repo=REPO)
    assert result["stats"]["passes"]["max"] == 3 and result["stats"]["passes"]["mm2_at_least"]["3"] > 4
    svg = tatbot_ink.write_preview(result, tmp_path / "preview.svg").read_text()
    assert "marks: max 3" in svg and '<g id="passes"' in svg and 'fill="#f59f00"' in svg


def test_dots_on_other_ink_are_dropped_and_dots_on_bare_skin_stay(tmp_path):
    line = element([[.002, .01], [.018, .01]], width=W)
    on_it = element([[.0100, .01], [.0104, .01001]], width=W)          # 0.4 mm along the line
    squiggle = element([[.006, .0101], [.0068, .0101], [.0062, .0099], [.0069, .0099]], width=W)  # 2 mm in 0.9
    alone = element([[.01, .016], [.0103, .016]], width=W)             # 6 mm off any other line
    dash = element([[.012, .01], [.0135, .01]], width=W)               # 1.5 mm: no dot
    path = write(tmp_path, artwork(program([("pen-1", [line, on_it, squiggle, alone, dash])])))
    result = tatbot_ink.compile(path, repo=REPO)
    kept = [op["src"]["path"] for op in result["ops"] if op["op"] == "stroke"]
    assert kept == ["path-0-0", "path-0-3", "path-0-4"]
    assert [d["path"] for d in result["preparation"]["dropped_dots"]] == ["path-0-1", "path-0-2"]
    assert result["stats"]["dots"] == {"dropped": 2, "dropped_contact_m": pytest.approx(.0004 + .00213, abs=2e-5),
                                       "kept_alone": 1}
    whole = tatbot_ink.compile(path, repo=REPO, min_stroke_m=0)
    assert whole["stats"]["paths"] == 5 and whole["stats"]["dots"]["dropped"] == 0
    assert whole["stats"]["est_s"] > result["stats"]["est_s"]
    assert result["stats"]["passes"]["max"] >= 2 and " marks: max" in tatbot_ink.write_preview(result, tmp_path / "p.svg").read_text()
    with pytest.raises(tatbot_ink.CompileError, match="min_stroke_m"):
        tatbot_ink.compile(path, repo=REPO, min_stroke_m=float("nan"))


def test_a_dot_on_another_inks_line_stays_and_one_on_its_own_inks_line_goes(tmp_path):
    import shutil

    from resource_fixtures import fixture_repo, manifest, resource
    from tatbot_ink.input import read_input

    black, red = [[.002, .01], [.018, .01]], [[.002, .005], [.018, .005]]
    art = artwork(program([("black", [element(black, width=W)]),
                           ("red", [element(red, width=W), element([[.008, .005], [.0084, .005]], width=W),
                                    element([[.010, .01], [.0104, .01001]], width=W)]),
                           ("black-2", [element([[.012, .01], [.0124, .01]], width=W)])],
                          inks={"black": [0, 0, 0], "red": [1, 0, 0], "black-2": [0, 0, 0]}))
    path = write(tmp_path, art)
    repo = fixture_repo(tmp_path / "repo")
    (repo / "ros/tatbot_motion/config").mkdir(parents=True)
    shutil.copyfile(REPO / "ros/tatbot_motion/config/motion.yaml", repo / "ros/tatbot_motion/config/motion.yaml")
    inks = manifest(tmp_path / "inks.yaml", read_input(art), [resource("B"), resource("R", ink="red")],
                    [("black", "B"), ("red", "R"), ("black-2", "B")])
    result = tatbot_ink.compile(path, repo=repo, inks_path=inks)
    kept = [(op["resource_id"], op["src"]["path"]) for op in result["ops"] if op["op"] == "stroke"]
    # the red dot on the black line is red's only mark there; the black-2 dot is the black line's own ink
    assert kept == [("B", "path-0-0"), ("R", "path-1-0"), ("R", "path-1-2")]
    assert [d["path"] for d in result["preparation"]["dropped_dots"]] == ["path-1-1", "path-2-0"]
    assert result["stats"]["dots"]["dropped"] == 2 and result["stats"]["dots"]["kept_alone"] == 1


def test_passes_are_counted_at_the_generation_width_or_a_given_mark_and_the_stats_say_which(tmp_path):
    from resource_fixtures import manifest, resource
    from tatbot_ink.input import read_input

    # two 0.3 mm paths 0.5 mm apart: apart at the 3RL's path width (no line width: its mark on skin is not
    # measured), one twice-passed band at a given 0.8 mm
    art = artwork(program([("pen-1", [element([[.002, .01], [.018, .01]], width=W),
                                      element([[.002, .0105], [.018, .0105]], width=W)])]))
    inks = manifest(tmp_path / "inks.yaml", read_input(art),
                    [resource("black", tool="lutin-3rl-bugpin", ink="nighthawk_black", mode="ride", slot="inkcap_medium_1")],
                    [("pen-1", "black")], substrate="silicon_skin")
    path = write(tmp_path, art)
    sheet = tatbot_ink.compile(path, repo=REPO, inks_path=inks)["stats"]["passes"]
    assert sheet["marks"] == {"black": {"width_mm": None, "from": "generation width"}}
    assert sheet["max"] == 1 and sheet["cell_mm"] == pytest.approx(W * 1000 / 6, abs=1e-4)
    given = tatbot_ink.compile(path, repo=REPO, inks_path=inks, mark_width_m=0.0008)["stats"]["passes"]
    assert given["marks"] == {"black": {"width_mm": 0.8, "from": "given"}}
    assert given["max"] == 2 and given["mm2_at_least"]["2"] > 4 and given["cell_mm"] == pytest.approx(0.8 / 6, abs=1e-4)
    # a tool with a line width is counted at it: the ballpoint's 0.5 mm
    assert passes.mark({"id": "lutin-ballpoint-dot", "line_width_m": 0.0005}) == {
        "width_mm": 0.5, "from": "lutin-ballpoint-dot line.width_mm"}
    # a program prepared before marks is counted at its line width, else the generation width, and says so
    old = {"resources": [{"id": "black", "tool": {"line_width_m": None}}], "ops": [], "stats": {}}
    assert passes.marks(old) == {"black": {"width_mm": None, "from": "generation width"}}
    with pytest.raises(tatbot_ink.CompileError, match="mark_width_m"):
        tatbot_ink.compile(path, repo=REPO, inks_path=inks, mark_width_m=float("inf"))
