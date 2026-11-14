"""python/tatbot_lab: studies, trials, cells on a dense tag print, test patterns and plans compiled into programs.

The print here is the silicone tile's real layout (stencil_tile.places) written as a generated print's tile.json and
tracking.json; the compile is tatbot_ink's own, against the repo's tool datasheets and palette."""
import json
from pathlib import Path

import numpy as np
import pytest
import stencil_tile
import yaml
from tatbot_contracts import lab
from tatbot_lab import patterns, plan, propose, space, store, tile

PATTERN = "stencil-" + "ab" * 32


def _print(directory, layout="silicone", pattern=PATTERN):
    """A generated print's records for `layout`: its markers where stencil_tile.build puts them."""
    spec = stencil_tile.LAYOUTS[layout]
    marks = stencil_tile.places(spec)
    cell, pitch = spec["cell_mm"], spec["pitch_mm"]
    tw, th = spec["tile_mm"]
    ox, oy = (tw - ((spec["cols"] - 1) * pitch + 6 * cell)) / 2, (th - ((spec["rows"] - 1) * pitch + 6 * cell)) / 2
    markers = [{"id": k, "role": role, "row": r, "col": c, "side_mm": side,
                "corners_mm": [[ox + x, oy + y], [ox + x + side, oy + y], [ox + x + side, oy + y + side],
                               [ox + x, oy + y + side]]}
               for k, (role, r, c, x, y, side) in enumerate(marks)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tile.json").write_text(json.dumps({"layout": layout, "tile_mm": [tw, th], "pitch_mm": pitch,
                                                     "markers": markers}))
    (directory / "tracking.json").write_text(json.dumps({"pattern_id": pattern, "page_mm": [tw, th],
                                                         "clear_center_uv": [0, 0, 1, 1]}))
    return directory


@pytest.fixture()
def lab_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_LAB_ROOT", str(tmp_path / "lab"))
    return tmp_path / "lab"


@pytest.fixture()
def study(tmp_path, lab_root):
    record = {"schema": lab.STUDY, "id": "card", "created": "2026-11-08T00:00:00Z", "question": "What does it lay?",
              "arm": "right", "stencil": {"pattern_id": PATTERN, "layout": "silicone", "tile_mm": [111.0, 157.5]},
              "cell_mm": 16.0, "constants": {"battery_v": 6.0, "machine_hz": 51.0, "stroke_mm": 4.0},
              "cartridges": ["lutin-3rl-bugpin"], "inks": ["true_black", "bright_red"],
              "palette": {"true_black": "inkcap_large_2", "bright_red": "inkcap_medium_1", "water": "inkcap_large_1"},
              "techniques": list(space.TECHNIQUES), "bounds": {"trim_mm": [-2.5, -1.0], "speed_mm_s": [2.5, 6.0]}}
    store.create(record)
    return record


def _catalog():
    return yaml.safe_load((Path(__file__).resolve().parents[2] / "config/inks.yaml").read_text())


def _proposal(technique="line", ink="true_black", **settings):
    return {"technique": technique, "cartridge": "lutin-3rl-bugpin", "inks": [ink],
            "settings": {"trim_mm": -2.0, "speed_mm_s": 4.0, **settings}}


def test_a_study_names_a_cap_for_every_ink_and_for_water(study):
    bad = {**study, "id": "other", "palette": {"true_black": "inkcap_large_2", "bright_red": "inkcap_medium_1"}}
    with pytest.raises(ValueError, match="water"):
        lab.study(bad)
    with pytest.raises(ValueError, match="two liquids in one cap"):
        lab.study({**study, "palette": {**study["palette"], "bright_red": "inkcap_large_2"}})


def test_a_trial_stays_inside_its_study(study):
    with pytest.raises(ValueError, match="outside the study's bounds"):
        propose.add(study, [_proposal(trim_mm=-3.0)])
    with pytest.raises(ValueError, match="holds pitch_mm fixed"):
        propose.add(study, [_proposal(pitch_mm=1.0)])
    with pytest.raises(ValueError, match="not the study's"):
        propose.add(study, [_proposal(ink="golden_yellow")])
    assert store.trials("card") == {}   # nothing of a refused batch is written


def test_depth_and_speed_are_never_defaulted():
    with pytest.raises(ValueError, match="never defaulted"):
        space.resolve("line", {"trim_mm": -2.0})
    with pytest.raises(ValueError, match="does not use deposition"):
        space.resolve("line", {"trim_mm": -2.0, "speed_mm_s": 4.0, "deposition": 0.5})
    assert space.resolve("whip", {"trim_mm": -2.0, "speed_mm_s": 4.0})["deposition"] == 0.0


def test_one_cell_sits_on_each_grid_marker_and_costs_only_it(tmp_path):
    p = tile.load(_print(tmp_path / "print"))
    cells = tile.cells(p, 16.0)
    grid = [i for i, role in p.roles.items() if role == "marker"]
    assert len(cells) == len(grid) == 19
    for cell in cells:
        assert tile.covered(p, cell.box_mm) == [int(cell.id[1:])]
    with pytest.raises(ValueError, match="overlaps its neighbours"):
        tile.cells(p, 23.0)


def test_the_fill_order_starts_at_the_edge_and_spreads(tmp_path):
    p = tile.load(_print(tmp_path / "print"))
    cells = tile.cells(p, 16.0)
    order = tile.fill_order(cells)
    centres = np.array([c.centre for c in order])
    assert np.linalg.norm(centres[0]) == pytest.approx(max(np.linalg.norm(c.centre) for c in cells))
    # the first four drawn are farther apart than any four neighbours
    first = centres[:4]
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(first) for b in first[i + 1:]]
    assert min(gaps) > 2 * p.pitch_mm
    assert [c.id for c in tile.fill_order(cells, {order[0].id})] == [c.id for c in order[1:]]


def test_patterns_fit_their_cell_or_are_refused():
    box = (-8.0, -8.0, 8.0, 8.0)
    line = patterns.place("line", {"trim_mm": -2, "speed_mm_s": 4, "passes": 2}, box)
    assert len(line) == 10 and all(dep is None for _, dep in line)
    np.testing.assert_allclose(line[0][0], line[1][0])   # a pass repeats its stroke in place
    whip = patterns.place("whip", {"trim_mm": -2, "speed_mm_s": 4, "deposition": 0.2}, box)
    assert whip[0][1][0] == 1.0 and whip[0][1][-1] == pytest.approx(0.2)
    carry = patterns.place("carry", {"trim_mm": -2, "speed_mm_s": 4}, box)
    assert carry[0][0][0, 0] < carry[0][0][-1, 0] and carry[1][0][0, 0] > carry[1][0][-1, 0]   # serpentine
    with pytest.raises(ValueError, match="more than"):
        patterns.place("line", {"trim_mm": -2, "speed_mm_s": 4, "length_mm": 20.0}, box)


def test_the_card_fills_one_silicone_sheet_in_few_runs(study):
    records = propose.add_card(study, "lutin-3rl-bugpin", ["true_black", "bright_red"])
    assert len(records) == 19
    assert len({space.run_key(r) for r in records}) == 6
    repeats = [r for r in records if r.get("repeat_of")]
    assert len(repeats) == 2
    by_id = {r["id"]: r for r in records}
    for r in repeats:
        original = by_id[r["repeat_of"]]
        assert original["settings"] == r["settings"] and original["inks"] == r["inks"]


def test_a_plan_draws_light_to_dark_one_cell_a_trial(tmp_path, study):
    p = tile.load(_print(tmp_path / "print"))
    propose.add_card(study, "lutin-3rl-bugpin", ["true_black", "bright_red"])
    result = plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())
    runs = result["runs"]
    assert [r["trim_mm"] for r in runs] == sorted((r["trim_mm"] for r in runs), reverse=True)   # shallow first
    assert all(r["inks"] in (["bright_red", "true_black"], ["true_black"]) for r in runs)      # black last
    assert sum(len(r["cells"]) for r in runs) == 19 and not result["unplaced"]
    trials = store.trials("card")
    assert {t["status"] for t in trials.values()} == {"planned"}
    run_dir = store.plans_dir("card") / result["plan"] / "run-01"
    cells = lab.cells(json.loads((run_dir / "cells.json").read_text()))
    program = json.loads((run_dir / "program" / "program.json").read_text())
    # every trial is its own resource, starting on a fresh dip; inks lightest first
    changes = [op["resource_id"] for op in program["ops"] if op["op"] == "tool_change"]
    assert changes == runs[0]["trials"]
    inks = {r["id"]: r["ink_id"] for r in program["resources"]}
    assert list(dict.fromkeys(inks[c] for c in changes)) == runs[0]["inks"]
    assert all(c["dips_mm"][0] == 0.0 for c in cells["cells"])
    strokes = np.vstack([np.array(op["points_m"])[:, :2] * 1000 for op in program["ops"] if op["op"] == "stroke"])
    boxes = np.array([c["box_mm"] for c in cells["cells"]])
    inside = [((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])).any()
              for x, y in strokes]
    assert all(inside)   # every point the program draws lies in a cell of this run


def test_a_trial_dips_at_its_own_cadence_or_the_datasheets(tmp_path, study, lab_root):
    """Two 168 mm carry cells in red: one at the cadence the 3RL's datasheet gives red, one at its own 400 mm, a
    single dip; its resource carries its own dwell and depth in the cap too."""
    record = {**study, "id": "dips", "bounds": {**study["bounds"], "mm_per_dip": [40.0, 400.0],
                                                "dip_dwell_s": [0.5, 4.0], "dip_above_ink_mm": [0.0, 3.0]}}
    store.create(record)
    p = tile.load(_print(tmp_path / "print"))
    propose.add(record, [_proposal("carry", ink="bright_red"),
                         _proposal("carry", ink="bright_red", mm_per_dip=400.0, dip_dwell_s=2.0, dip_above_ink_mm=1.5)])
    result = plan.plan(record, store.trials("dips"), p, "sheet-a", _catalog())
    run_dir = store.plans_dir("dips") / result["plan"] / "run-01"
    cells = {c["trial"]: c for c in json.loads((run_dir / "cells.json").read_text())["cells"]}
    tool = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/tools/lutin-3rl-bugpin.yaml").read_text())
    per_dip = tool["dip"]["mm_per_dip_by_ink"].get("bright_red", tool["dip"]["mm_per_dip"])
    assert cells["t0001"]["dips_mm"] == pytest.approx(list(np.arange(0.0, 14 * 12.0, per_dip)), abs=0.01)
    assert cells["t0002"]["dips_mm"] == [0.0]
    program = json.loads((run_dir / "program" / "program.json").read_text())
    dips = {r["id"]: r["dip"] for r in program["resources"]}
    assert dips["t0001"]["mm_per_dip"] == per_dip and dips["t0001"]["dwell_s"] == tool["dip"]["dwell_s"]
    assert (dips["t0002"]["mm_per_dip"], dips["t0002"]["dwell_s"], dips["t0002"]["above_ink_m"]) == (400.0, 2.0,
                                                                                                    0.0015)
    assert not {"drain_s", "drain_above_rim_m"} & (set(dips["t0001"]) | set(dips["t0002"]))


def test_a_trial_drains_its_dips_over_the_cap(tmp_path, study, lab_root):
    """The drain is a resource setting: one cell at the datasheet's dip, which does not drain, one holding 1.5 s
    over the rim at the datasheet's height, one at its own height."""
    record = {**study, "id": "drain", "bounds": {**study["bounds"], "dip_drain_s": [0.0, 3.0],
                                                 "dip_drain_above_rim_mm": [1.0, 5.0]}}
    store.create(record)
    propose.add(record, [_proposal("line", ink="bright_red"), _proposal("line", ink="bright_red", dip_drain_s=1.5),
                         _proposal("line", ink="bright_red", dip_drain_s=3.0, dip_drain_above_rim_mm=5.0)])
    result = plan.plan(record, store.trials("drain"), tile.load(_print(tmp_path / "print")), "sheet-a", _catalog())
    program = json.loads((store.plans_dir("drain") / result["plan"] / "run-01/program/program.json").read_text())
    dips = {r["id"]: r["dip"] for r in program["resources"]}
    assert "drain_s" not in dips["t0001"]
    assert (dips["t0002"]["drain_s"], dips["t0002"]["drain_above_rim_m"]) == (1.5, 0.003)
    assert (dips["t0003"]["drain_s"], dips["t0003"]["drain_above_rim_m"]) == (3.0, 0.005)
    assert program["stats"]["time_estimate"]["components_s"]["drain"] == 4.5


def test_the_cards_carry_cells_dip_once_when_the_study_bounds_the_cadence(study, lab_root):
    record = {**study, "id": "once", "bounds": {**study["bounds"], "mm_per_dip": [40.0, 400.0]}}
    store.create(record)
    records = propose.add_card(record, "lutin-3rl-bugpin", ["true_black", "bright_red"])
    assert {r["settings"].get("mm_per_dip") for r in records if r["technique"] == "carry"} == {400.0}
    assert all("mm_per_dip" not in r["settings"] for r in records if r["technique"] != "carry")


def test_the_dip_card_is_one_run_of_the_mark_the_carry_and_a_ladder_of_each_dip_setting(tmp_path, study, lab_root):
    """The 1007RL's card in red: its datasheet's dip is borrowed, which the lab may still draw with."""
    bounds = {**study["bounds"], "mm_per_dip": [40.0, 300.0], "dip_dwell_s": [0.25, 1.5],
              "dip_above_ink_mm": [0.5, 3.0], "dip_drain_s": [0.0, 3.0]}
    record = {**study, "id": "dip-7rl", "cartridges": ["lutin-7rl"], "bounds": bounds}
    store.create(record)
    records = propose.add_dip_card(record, "lutin-7rl", "bright_red")
    assert [r["technique"] for r in records] == ["line", "line", "carry"] + ["line"] * 8
    assert {(r["settings"]["trim_mm"], r["settings"]["speed_mm_s"]) for r in records} == {(-2.0, 4.25)}
    assert records[1]["repeat_of"] == records[0]["id"] and set(records[0]["settings"]) == {"trim_mm", "speed_mm_s"}
    assert records[2]["settings"]["mm_per_dip"] == 300.0

    def ladder(name):
        return [r["settings"][name] for r in records if name in r["settings"]]
    assert ladder("dip_dwell_s") == [0.25, 0.875, 1.5] and ladder("dip_above_ink_mm") == [0.5, 1.75, 3.0]
    assert ladder("dip_drain_s") == [1.5, 3.0]   # no drain is the first two cells' dip
    result = plan.plan(record, store.trials("dip-7rl"), tile.load(_print(tmp_path / "print")), "sheet-a", _catalog())
    assert len(result["runs"]) == 1 and not result["unplaced"]
    run_dir = store.plans_dir("dip-7rl") / result["plan"] / "run-01"
    program = json.loads((run_dir / "program" / "program.json").read_text())
    assert {r["dip_card"] for r in program["resources"]} == {"borrowed"}
    cells = json.loads((run_dir / "cells.json").read_text())["cells"]
    assert all(c["dips_mm"] == [0.0] for c in cells)   # every cell on one fresh dip, the carry too
    with pytest.raises(ValueError, match="bound mm_per_dip or any of"):
        propose.dip_card(study, "lutin-3rl-bugpin", "true_black")


def test_a_sheet_once_spent_sends_trials_to_the_next(tmp_path, study):
    p = tile.load(_print(tmp_path / "print"))
    propose.add_card(study, "lutin-3rl-bugpin", ["true_black", "bright_red"])
    plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())
    propose.add(study, [_proposal()])
    with pytest.raises(ValueError, match="no free cell"):
        plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())
    result = plan.plan(study, store.trials("card"), p, "sheet-b", _catalog())
    assert result["runs"][0]["cells"] == [tile.fill_order(tile.cells(p, 16.0))[0].id]


def test_a_sheet_is_one_skin_whichever_study_drew_on_it(tmp_path, study):
    """A study made for new bounds on the same print starts where the other left the sheet; a dropped trial frees its
    cell; another print's study shares nothing."""
    p = tile.load(_print(tmp_path / "print"))
    propose.add(study, [_proposal(), _proposal(trim_mm=-1.5)])
    first = plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())
    taken = {c for r in first["runs"] for c in r["cells"]}
    deep = {**study, "id": "deep", "bounds": {"trim_mm": [-4.0, -1.75], "speed_mm_s": [2.5, 6.0]}}
    store.create(deep)
    propose.add(deep, [_proposal(trim_mm=-3.0)])
    second = plan.plan(deep, store.trials("deep"), p, "sheet-a", _catalog())
    assert not taken & {c for r in second["runs"] for c in r["cells"]}
    dropped = [dict(t, status="dropped") for t in store.trials("card").values()]
    store.append("card", dropped)
    assert plan.spent(deep, store.trials("deep"), "sheet-a") == {c for r in second["runs"] for c in r["cells"]}
    store.create({**study, "id": "other-print", "stencil": {**study["stencil"], "pattern_id": "stencil-" + "cd" * 32}})
    assert plan.spent(store.load("other-print"), {}, "sheet-a") == set()


def test_a_plan_that_fails_leaves_nothing(tmp_path, study):
    p = tile.load(_print(tmp_path / "print"))
    propose.add(study, [_proposal(), _proposal(trim_mm=-1.0)])

    def fails(*args):
        raise ValueError("no compile today")
    with pytest.raises(ValueError, match="no compile today"):
        plan.plan(study, store.trials("card"), p, "sheet-a", _catalog(), compile_program=fails)
    assert not list(store.plans_dir("card").iterdir())
    assert {t["status"] for t in store.trials("card").values()} == {"proposed"}


def test_a_plan_refuses_another_print(tmp_path, study):
    p = tile.load(_print(tmp_path / "print", pattern="stencil-" + "cd" * 32))
    propose.add(study, [_proposal()])
    with pytest.raises(ValueError, match="the study's is"):
        plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())


def _planned(tmp_path, study):
    p = tile.load(_print(tmp_path / "print"))
    propose.add(study, [_proposal(), _proposal(ink="bright_red")])
    return plan.plan(study, store.trials("card"), p, "sheet-a", _catalog())


def test_capture_photographs_every_cell_and_after_aims_by_the_before_page(tmp_path, study, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from tatbot_lab import cli

    result = _planned(tmp_path, study)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    args = SimpleNamespace(study="card", plan=result["plan"], run="run-01", rest_view=False, overhead=True,
                           overhead_only=False, sheet_at=None)
    assert cli.cmd_capture(SimpleNamespace(**vars(args), side="before")) == 0
    before = calls[0]
    assert before[1:3] == ["ros", "views"] and "--hold" in before and "--overhead" in before
    points = next(a for a in before if a.startswith("--points="))[len("--points="):].split(";")
    assert len(points) == 2
    with pytest.raises(ValueError, match="no held before views"):
        cli.cmd_capture(SimpleNamespace(**vars(args), side="after"))
    seen = store.plans_dir("card") / result["plan"] / "run-01" / "views" / "before"
    seen.mkdir(parents=True)
    (seen / "views.json").write_text(json.dumps({"page_run": "20261108T100000Z-ros-aaaa"}))
    assert cli.cmd_capture(SimpleNamespace(**vars(args), side="after")) == 0
    after = calls[-1]
    assert after[after.index("--run") + 1] == "20261108T100000Z-ros-aaaa" and "--hold" not in after
    assert {t["status"] for t in store.trials("card").values()} == {"drawn"}


def test_a_capture_at_a_chosen_sheet_aims_by_the_overhead_and_after_keeps_that_sheet(tmp_path, study, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from tatbot_lab import cli

    result = _planned(tmp_path, study)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    args = {"study": "card", "plan": result["plan"], "run": "run-01", "rest_view": False, "overhead": True,
            "overhead_only": False}
    assert cli.cmd_capture(SimpleNamespace(**args, side="before", sheet_at="400,-100")) == 0
    pattern = json.loads((store.plans_dir("card") / result["plan"] / "run-01" / "cells.json").read_text())["pattern_id"]
    assert {f"--overhead-pages={pattern}", "--sheet-at=400,-100", "--hover-mm=12", "--seconds=20"} <= set(calls[0])
    assert "--run" not in calls[0]
    seen = store.plans_dir("card") / result["plan"] / "run-01" / "views" / "before"
    seen.mkdir(parents=True)
    (seen / "views.json").write_text(json.dumps({"page_run": None, "sheet_at": "400,-100"}))
    assert cli.cmd_capture(SimpleNamespace(**args, side="after", sheet_at=None)) == 0
    assert "--sheet-at=400,-100" in calls[-1] and "--run" not in calls[-1]


def test_a_rest_view_capture_sends_no_points(tmp_path, study, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from tatbot_lab import cli

    result = _planned(tmp_path, study)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    assert cli.cmd_capture(SimpleNamespace(study="card", plan=result["plan"], run="run-01", side="after",
                                           rest_view=True, overhead=False, overhead_only=False, sheet_at=None)) == 0
    assert not any(a.startswith("--points") or a in ("--run", "--overhead") for a in calls[0])


def test_an_overhead_only_capture_touches_no_arm_and_a_dry_one_changes_no_trial(tmp_path, study, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from tatbot_lab import cli

    result = _planned(tmp_path, study)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    assert cli.cmd_capture(SimpleNamespace(study="card", plan=result["plan"], run="run-01", side="dry",
                                           rest_view=False, overhead=True, overhead_only=True, sheet_at=None)) == 0
    assert "--no-wrist" in calls[0] and "--overhead" in calls[0]
    assert not any(a.startswith("--points") or a in ("--run", "--hold") for a in calls[0])
    assert {t["status"] for t in store.trials("card").values()} == {"planned"}


def test_an_overhead_read_takes_the_chosen_sheets_region_unless_given_one(tmp_path):
    from types import SimpleNamespace

    pytest.importorskip("cv2")   # the region projects through tatbot_lab.read, which reads with OpenCV
    from tatbot_lab import cli

    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = [0.1, 0.0, 0.9]
    optics = {"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0, "width": 1280, "height": 720,
              "distortion_coefficients": [0.0] * 5}
    (tmp_path / "views.json").write_text(json.dumps({"sheet_at": "400,-100", "overhead": {"intrinsics": optics},
                                                     "sheets": [{"world_from_page": pose.tolist()}]}))
    p = SimpleNamespace(tile_mm=(40.0, 62.0))
    roi, how = cli._region(SimpleNamespace(roi=None, instrument="overhead"), tmp_path, p)
    assert how == "sheet" and roi[0] < 640 + 0.1 * 640 / 0.9 < roi[2] and roi[1] < 360 < roi[3]
    assert cli._region(SimpleNamespace(roi=None, instrument="wrist"), tmp_path, p) == (None, None)
    assert cli._region(SimpleNamespace(roi="1,2,3,4", instrument="overhead"), tmp_path, p) == ([1.0, 2.0, 3.0, 4.0],
                                                                                              "given")
    assert cli._region(SimpleNamespace(roi=None, instrument="overhead"), tmp_path / "none", p) == (None, None)


def _planned_run(tmp_path, study):
    result = _planned(tmp_path, study)
    return result, result["runs"][0]


def test_a_lab_run_photographs_draws_with_its_cartridge_stated_photographs_again_and_reads(tmp_path, study,
                                                                                          monkeypatch):
    from types import SimpleNamespace

    from tatbot_lab import cli

    result, run = _planned_run(tmp_path, study)
    steps = []
    monkeypatch.setattr(cli, "cmd_capture", lambda ns: steps.append(("capture", ns.side, ns.sheet_at)) or 0)
    monkeypatch.setattr(cli, "cmd_read", lambda ns: steps.append(("read", ns.instrument, ns.run_id)) or 0)

    class Draw:
        def __init__(self, argv, **kw):
            steps.append(("draw", argv))
            self.stdout, self.returncode = iter(['{"ok": true, "run_id": "20261109T010000Z-ros-abcd"}\n']), 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("subprocess.Popen", Draw)
    ns = SimpleNamespace(study="card", plan=result["plan"], run=run["run"], stencil="/print", sheet_at="400,-100",
                         start="before")
    assert cli.cmd_run(ns) == 0
    assert [s[:2] for s in steps] == [("capture", "before"), ("draw", steps[1][1]), ("capture", "after"),
                                      ("read", "wrist"), ("read", "overhead")]
    argv = steps[1][1]
    assert argv[1:3] == ["--ee-tool", "lutin-3rl-bugpin"] and "--hold" not in argv and "--no-inspect" in argv
    assert f"--trim-mm={run['trim_mm']:g}" in argv and "--sheet-at=400,-100" in argv
    assert steps[0][2] == "400,-100" and steps[-1][2] == "20261109T010000Z-ros-abcd"
    record = json.loads((store.plans_dir("card") / result["plan"] / run["run"] / "lab-run.json").read_text())
    assert [record["steps"][s]["exit"] for s in cli.RUN_STEPS] == [0, 0, 0, 0]
    assert record["steps"]["draw"] == {"at": record["steps"]["draw"]["at"], "exit": 0,
                                       "ros_run_id": "20261109T010000Z-ros-abcd"} and "--ends=fly" not in argv
    # the same run with its stroke ends drawn moving: the draw is told, and the run's record keeps it
    assert cli.cmd_run(SimpleNamespace(**{**vars(ns), "start": "draw", "ends": "fly"})) == 0
    assert "--ends=fly" in steps[-4][1]
    record = json.loads((store.plans_dir("card") / result["plan"] / run["run"] / "lab-run.json").read_text())
    assert record["steps"]["draw"]["ends"] == "fly"


def test_a_failed_step_stops_the_run_and_from_resumes_it_with_what_came_before(tmp_path, study, monkeypatch, capsys):
    from types import SimpleNamespace

    from tatbot_lab import cli

    result, run = _planned_run(tmp_path, study)
    after = iter([5, 0])
    steps = []
    monkeypatch.setattr(cli, "cmd_capture", lambda ns: steps.append(ns.side) or (0 if ns.side == "before" else next(after)))
    monkeypatch.setattr(cli, "_draw", lambda study, summary, sheet_at, ends=None: (steps.append("draw") or 0,
                                                                               {"ros_run_id": "r1"}))
    monkeypatch.setattr(cli, "cmd_read", lambda ns: steps.append(f"read:{ns.run_id}") or 0)
    ns = SimpleNamespace(study="card", plan=result["plan"], run=run["run"], stencil="/print", sheet_at=None,
                         start="before")
    assert cli.cmd_run(ns) == 5 and steps == ["before", "draw", "after"]
    assert "--from after" in capsys.readouterr().out
    steps.clear()
    assert cli.cmd_run(SimpleNamespace(**{**vars(ns), "start": "after"})) == 0
    assert steps == ["after", "read:r1", "read:r1"]   # the draw's run id kept from the first attempt


def test_a_trial_is_judged_by_its_wrist_read_whatever_the_overhead_read_after():
    from tatbot_lab import cli

    wrist, overhead = {"dL_mean": 5.1, "ink_width_mm": [0.4]}, {"dL_mean": 8.1}
    fresh = {"reads": []}
    assert cli._trial_metrics(fresh, "overhead", overhead) == overhead   # nothing better yet
    read = {"reads": [{"instrument": "wrist"}], "metrics": wrist}
    assert cli._trial_metrics(read, "overhead", overhead) == wrist       # lab run reads the overhead second
    assert cli._trial_metrics({"reads": [{"instrument": "overhead"}], "metrics": overhead}, "wrist", wrist) == wrist


def test_a_study_may_bound_the_dose_in_place_of_the_speed(study):
    """Needle pokes per millimetre in a stroke's middle are the machine's rate over the speed: a dose bound is
    stored as the speed bound that gives it, the most pokes the slowest, and every trial's dose reads back."""
    assert space.speed_bounds([10.2, 25.5], 51.0) == [2.0, 5.0]
    assert space.dose_per_mm(study, {"speed_mm_s": 3.5}) == pytest.approx(14.571, abs=1e-3)
    with pytest.raises(ValueError, match="positive"):
        space.speed_bounds([0.0, 20.0], 51.0)
    ladder = [p for p in propose.card(study, "lutin-3rl-bugpin", ["true_black"]) if "speed ladder" in p["hypothesis"]]
    assert ladder and all(f"{51.0 / p['settings']['speed_mm_s']:.1f} pokes/mm" in p["hypothesis"] for p in ladder)
