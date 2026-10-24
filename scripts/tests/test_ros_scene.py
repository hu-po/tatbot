"""The ROS Rerun bridge's pure scene models (tatbot_contracts.ros_scene) on a program and ledger shaped from a real draw
(fixtures/ros_scene: run ids and hosts scrubbed, the ledger's floats rounded, cut to 7 ops) and a made-up palette, ink
catalog, palette load, station fix and touch, so the suite needs none of the deployment's own files."""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tatbot_contracts import ros_drawn
from tatbot_contracts import ros_scene as scene
from tatbot_description.transforms import rpy_matrix

FIXTURES = Path(__file__).parent / "fixtures" / "ros_scene"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "ros-draw" / "20261018T164526Z-node-0000"
    run.mkdir(parents=True)
    for name in ("program.json", "ledger.jsonl"):
        shutil.copy(FIXTURES / name, run / name)
    return run


def test_ledger_tail_reads_whole_lines_by_offset(tmp_path):
    path = tmp_path / "ledger.jsonl"
    tail = scene.LedgerTail(path)
    assert tail.poll() == []                                  # no ledger yet
    path.write_text('{"event": "sent", "op": "s1"}\n{"event": "do')
    assert [row["op"] for row in tail.poll()] == ["s1"]
    with path.open("a") as stream:
        stream.write('ne", "op": "s1"}\n{"torn\n{"event": "sent", "op": "s2"}\n')
    assert [(row["event"], row["op"]) for row in tail.poll()] == [("done", "s1"), ("sent", "s2")]
    assert tail.poll() == [] and tail.offset == path.stat().st_size


def test_draw_state_from_a_real_ledger(run_dir):
    state = scene.DrawState.load(run_dir)
    assert state.arm == "right" and sorted(state.planned) == ["s0002", "s0003", "s0004", "s0005", "s0006"]
    first = json.loads((run_dir / "ledger.jsonl").read_text().splitlines()[0])
    sent = next(json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()
                if '"s0003"' in line and '"sent"' in line)
    assert state.poll(now_s=first["t"]) == {"page", "planned", "done"} and state.lines == {}
    assert state.poll(now_s=sent["t"]) == {"palette", "planned", "done", "design"}   # s0002's correction moves both
    assert sorted(state.lines) == ["s0002"] and state.dips == Counter({"inkcap_large_2": 1})
    assert state.next_op() == 3 and state.ink(3) == {"ink": "nighthawk_black", "slot": "inkcap_large_2"}
    assert state.dip_slot(0) == "inkcap_large_2" and state.dip_slot(2) is None
    assert {"done"} <= state.poll() and sorted(state.lines) == ["s0002", "s0003", "s0004", "s0005", "s0006"]
    assert state.next_op() is None and state.page["source"] == "measured"
    assert state.colour("s0002") == (76, 76, 76)              # black ink, lifted to read on a dark view
    summary = state.summary()                                  # no meta.json: the start from the run id, no end
    assert summary == {"id": run_dir.name, "t": state.t, "index": None, "total": 7, "done": 7, "status": None,
                       "started_s": scene.stamp_s("20261018T164526Z"), "ended_s": None}


def test_the_trim_is_the_cartridges_newest_pen_trim_row_of_the_runs_motion_yaml_limit(run_dir):
    state = scene.DrawState.load(run_dir)
    state.poll()
    assert state.trim(3) == {"m": 0.0, "limit_m": None, "used_m": 0.0} and state.trim(None) is None   # no row yet
    shutil.copy(REPO / "ros" / "tatbot_motion" / "config" / "motion.yaml", run_dir / "motion.yaml")
    with (run_dir / "ledger.jsonl").open("a") as stream:
        for trim_m in (-0.0001, -0.0002):
            stream.write(json.dumps({"arm": "right", "event": "pen_trim", "resource": "nighthawk_black",
                                     "t": state.t, "trim_m": trim_m}) + "\n")
    assert state.poll() == set() and state.trim(3) == {"m": -0.0002, "limit_m": 0.004, "used_m": -0.0002}


def test_the_trim_a_needle_stroke_takes_is_its_sent_rows_cap_while_it_is_the_op_drawn(run_dir):
    """A needle cartridge's stroke takes the trim no deeper than its sent row's trim_cap_m (the skin read along it):
    keys pressed past it show held there while that stroke is drawn, and as they are once the next op comes."""
    state = scene.DrawState.load(run_dir)
    state.poll()
    op = state.ops[3]["id"]
    with (run_dir / "ledger.jsonl").open("a") as stream:
        stream.write("\n")                                       # the fixture's last row is torn
        for row in ({"event": "sent", "op": op, "index": 3, "arc_m": 0.0, "trim_cap_m": -0.00265},
                    {"event": "pen_trim", "resource": "nighthawk_black", "trim_m": -0.004}):
            stream.write(json.dumps({"arm": "right", "t": state.t, **row}) + "\n")
    state.poll()
    assert state.trim(3)["m"] == -0.004 and state.trim(3)["used_m"] == pytest.approx(-0.00265)
    assert state.trim(4)["used_m"] == -0.004                    # another op: no cap known for it


def test_summary_reads_the_run_logs_end(run_dir):
    state = scene.DrawState.load(run_dir)
    meta = {"status": "running", "started_at": "2026-10-18T16:45:26.500Z", "ended_at": None}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    state.poll(now_s=0.0)
    assert state.summary()["ended_s"] is None and state.summary()["started_s"] == 1792341926.0
    assert state.summary()["index"] == 0 and state.summary()["done"] == 0
    (run_dir / "meta.json").write_text(json.dumps({**meta, "status": "interrupted", "ended_at": "2026-10-18T17:00:00Z"}))
    os.utime(run_dir / "meta.json", ns=(1, 1))                 # a new mtime, as the run log's finalize gives it
    state.poll()
    assert state.summary()["status"] == "interrupted" and state.summary()["ended_s"] == 1792342800.0
    assert scene.stamp_s("latest") is None and scene.stamp_s(None) is None


def test_current_stroke_is_cut_at_its_arc_on_the_print_and_moved_onto_the_page_by_the_newest_correction(run_dir):
    state = scene.DrawState.load(run_dir)
    state.poll()
    planned = state.planned["s0003"]
    newest = [row for row in state.rows if row["event"] == "sent" and row.get("op") in state.planned][-1]
    current = state.current("s0003", 0.0005)
    np.testing.assert_allclose(current[0], planned[0])
    np.testing.assert_allclose(state.shift, newest["correction_m"])
    np.testing.assert_allclose(state.on_page(current)[0], planned[0] - newest["correction_m"])
    assert np.isclose(np.linalg.norm(np.diff(current, axis=0), axis=1).sum(), 0.0005)
    assert state.current("d0001", 0.001) is None


def _off_plan_mm(line, planned) -> float:
    return 1e3 * float(np.mean([np.linalg.norm(planned - p, axis=1).min() for p in np.asarray(line)]))


def test_a_corrected_strokes_line_is_on_the_print_where_the_design_has_it(run_dir):
    """Stencil tracking drew each stroke at its points less its correction, on the page in use: each measured line,
    plus its own stroke's correction, lies on its planned stroke (0.14-0.27 mm off it, 1.3-2.0 mm before), so the
    design map's done strokes overlap its design. A page set up again puts the page in use back on the print."""
    state = scene.DrawState.load(run_dir)
    state.poll()
    rows = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines() if line.endswith("}")]
    raw = {row["op"]: row["line"] for row in rows if row["event"] == "done" and row.get("line")}
    for op, lines in state.lines.items():
        assert _off_plan_mm(lines[0], state.planned[op]) < 0.3 < 1.2 < _off_plan_mm(raw[op], state.planned[op])
        np.testing.assert_allclose(state.on_page(lines[0]), np.asarray(raw[op]) + state.corrections[op] - state.shift)
    page = next(row for row in rows if row["event"] == "page")
    state._apply({**page, "t": state.t})
    assert not state.shift.any()


def test_cut_open_and_closed():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    np.testing.assert_allclose(scene.cut(square, 1.5), [[0, 0], [1, 0], [1, 0.5]])
    np.testing.assert_allclose(scene.cut(square, 3.5, closed=True)[-1], [0, 0.5])
    assert len(scene.cut(square, 9.0)) == 4 and len(scene.cut(square, 9.0, closed=True)) == 5


def test_design_map_matches_a_real_drawn_svg():
    """A real run's drawn.svg (111 x 157.5 mm print): plan-s0002 starts `M 72.490 65.779`, drawn-s0002-0
    `M 71.858 66.199`, from its program's points_m[0] and its ledger's line[0]."""
    size = (0.111, 0.1575)
    points = [[0.016989742187499997, 0.012971088541666665], [0.016357548745670497, 0.01255092175211836]]
    np.testing.assert_allclose(scene.design_mm(points, size), [[72.490, 65.779], [71.858, 66.199]], atol=5e-4)
    np.testing.assert_allclose(scene.design_mm(points, size), [ros_drawn._xy(p, size) for p in points])


def test_fitted_page_lifts_like_page_surface(run_dir):
    state = scene.DrawState.load(run_dir)
    state.poll()
    assert scene.page_surface(state.page, fitted=False) is None
    surface = scene.page_surface(state.page, fitted=True)
    uv = state.planned["s0005"]
    base = scene.transform(state.page["used"], scene.page_local(uv, surface))
    np.testing.assert_allclose(base, surface.lift(uv)[0], atol=1e-9)
    lifted = scene.page_local(uv, surface)[:, 2]
    assert 0.0005 < np.ptp(scene.page_local(state.planned["s0002"], surface)[:, 2]) + abs(lifted).max() < 0.02
    flat = scene.page_local(uv)
    assert np.all(flat[:, 2] == 0.0) and np.allclose(flat[:, :2], uv)
    grid = scene.page_grid((0.111, 0.1575), surface)
    assert len(grid) == 12 + 17 and all(line.shape[1] == 3 for line in grid)
    assert len(scene.table_grid()) == 15 + 19


def test_page_layer_holds_the_fields_edge_past_its_extent(run_dir):
    """The fixture's field covers a corner of the print: extrapolated, its polynomial puts the far corner 38 mm under
    the page; the page layer holds the edge's height there, and its edges follow the surface between corners."""
    state = scene.DrawState.load(run_dir)
    state.poll()
    surface = scene.page_surface(state.page, fitted=True)
    x0, y0, x1, y1 = surface.height.extent
    inside = np.array([[u, v] for u in np.linspace(x0, x1, 30) for v in np.linspace(y0, y1, 30)])
    low, high = surface.height(inside).min(), surface.height(inside).max()
    assert surface.height([[0.0555, 0.07875]])[0] < -0.03                    # the bare polynomial at that corner
    lines = scene.page_lines((0.111, 0.1575), (0.08, 0.12), surface)
    heights = np.concatenate([line[:, 2] for line in lines])
    assert len(lines) == 2 + 12 + 17 and low - 5e-4 < heights.min() and heights.max() < high + 5e-4
    assert lines[0].shape == (97, 3) and np.allclose(lines[0][0], lines[0][-1])
    uv = state.planned["s0005"]                                              # a stroke keeps the arm's own lift
    np.testing.assert_allclose(scene.page_local(uv, surface), scene.page_local(uv, surface, within_fit=False))


def test_urdf_visuals_resolve_meshes_and_batch_cylinders(tmp_path):
    (tmp_path / "pkg" / "meshes").mkdir(parents=True)
    (tmp_path / "pkg" / "meshes" / "a.stl").write_bytes(b"solid a\nendsolid a\n")
    (tmp_path / "local.stl").write_bytes(b"solid b\nendsolid b\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f"""<robot name="r"><material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
      <link name="world"/>
      <link name="right/base_link">
        <visual><origin xyz="0 0 0.1" rpy="0 0 1.5707963"/><geometry><mesh filename="package://pkg/meshes/a.stl"/>
        </geometry><material name="grey"/></visual>
        <visual><geometry><mesh filename="local.stl" scale="0.001 0.001 0.001"/></geometry></visual>
        <visual><geometry><mesh filename="file://{tmp_path}/missing.stl"/></geometry></visual>
        <collision><geometry><mesh filename="local.stl"/></geometry></collision></link>
      <link name="right/pen">
        <visual><origin xyz="0 0 0.01"/><geometry><cylinder radius="0.01" length="0.02"/></geometry></visual>
        <visual><origin xyz="0 0 0.03"/><geometry><cylinder radius="0.008" length="0.02"/></geometry>
        <material name="tip"><color rgba="1 0 0 1"/></material></visual></link></robot>""")
    visuals = scene.urdf_visuals(urdf, packages=lambda name: tmp_path / name if name == "pkg" else None)
    meshes = [v for v in visuals if v["kind"] == "mesh"]
    assert [(v["link"], v["name"], v["path"].name, v["rgba"]) for v in meshes] == [
        ("right/base_link", "visual_0", "a.stl", (128, 128, 128, 255)), ("right/base_link", "visual_1", "local.stl", None)]
    assert meshes[1]["scale"] == [0.001] * 3 and np.isclose(meshes[0]["link_from_visual"][2, 3], 0.1)
    batch = scene.cylinders(visuals)["right/pen"]
    assert batch["radii"] == [0.01, 0.008] and batch["colors"][1] == (255, 0, 0, 255)
    np.testing.assert_allclose(batch["centers"], [[0, 0, 0.01], [0, 0, 0.03]])
    palette = scene.urdf_visuals(REPO / "urdf" / "palette.urdf")
    assert palette[0]["link"] == "palette_root" and palette[0]["path"].is_file()


def test_newest_runs_and_station_fix(tmp_path, run_dir):
    station = tmp_path / "ros-station"
    for name in ("20261018T131405Z-node-0001", "20261018T132147Z-node-0002"):
        (station / name).mkdir(parents=True)
        shutil.copy(FIXTURES / "station.json", station / name / "station.json")
    (station / "latest").symlink_to(station / "20261018T131405Z-node-0001")
    assert scene.newest_run(station).name == "20261018T132147Z-node-0002"
    assert scene.palette_fix_path(tmp_path, run_dir).parent.name == "20261018T132147Z-node-0002"
    for name in ("d0028-20261018T165711Z", "d0001-20261018T164921Z"):
        (run_dir / "station" / name).mkdir(parents=True)
        shutil.copy(FIXTURES / "station.json", run_dir / "station" / name / "station.json")
    assert scene.palette_fix_path(tmp_path, run_dir).parent.name == "d0028-20261018T165711Z"
    (station / "20261018T190000Z-node-0003").mkdir()                        # a station measured after the draw wins
    shutil.copy(FIXTURES / "station.json", station / "20261018T190000Z-node-0003" / "station.json")
    assert scene.palette_fix_path(tmp_path, run_dir).parent.name == "20261018T190000Z-node-0003"
    assert scene.palette_fix_path(tmp_path / "missing") is None and scene.newest_run(tmp_path / "missing") is None
    # a bag replayed on its own clock: no fix (nor run) from after the node's now
    before = scene.stamp_s("20261018T170000Z")
    assert scene.palette_fix_path(tmp_path, run_dir, now_s=before).parent.name == "d0028-20261018T165711Z"
    assert scene.palette_fix_path(tmp_path, None, now_s=before).parent.name == "20261018T132147Z-node-0002"
    assert scene.newest_run(station, now_s=before).name == "20261018T132147Z-node-0002"
    assert scene.palette_fix_path(tmp_path, run_dir, now_s=scene.stamp_s("20261018T130000Z")) is None


def _deployed(tmp_path) -> Path:
    """A deployed repo: the fixtures' palette, inks and palette load, the example fiducials, the arms and palette URDF."""
    (tmp_path / "config").mkdir(parents=True)
    for name in ("palette.yaml", "inks.yaml", "palette_load.yaml"):
        shutil.copy(FIXTURES / name, tmp_path / "config" / name)
    shutil.copy(REPO / "config" / "examples" / "fiducials.json", tmp_path / "config" / "fiducials.json")
    shutil.copy(REPO / "config" / "arms.json", tmp_path / "config" / "arms.json")
    shutil.copytree(REPO / "urdf", tmp_path / "urdf", ignore=lambda _, names: [n for n in names if n != "palette.urdf"])
    return tmp_path


def test_palette_from_the_touch_and_its_caps(tmp_path):
    repo = _deployed(tmp_path / "repo")
    pose = scene.palette_pose(repo, "right", FIXTURES / "station.json", FIXTURES / "station-touch-right.json")
    fix = json.loads((FIXTURES / "station.json").read_text())
    assert pose["how"] == "touched" and pose["measured_s"] == 1767323045.0
    assert np.linalg.norm(pose["base_from_palette"][:3, 3] - np.asarray(fix["base_from_palette"])[:3, 3]) < 0.02
    stale = scene.palette_pose(repo, "right", FIXTURES / "station.json", tmp_path / "no-touch.json")
    assert stale["how"].startswith("fix only: ") and np.allclose(stale["base_from_palette"], fix["base_from_palette"])
    caps = {cap["slot"]: cap for cap in scene.caps(repo, pose["base_from_palette"], Counter({"inkcap_large_2": 3}))}
    black, absent = caps["inkcap_large_2"], caps["inkcap_small_1"]
    assert black["state"] == "loaded" and black["rgb"] == (76, 76, 76) and black["ink"] == "example_black"
    assert black["text"] == "L2 black ≥11.5 mm 85% · 3 dips" and black["brief"] == "L2 black ≥85%"
    assert np.isclose(black["level"], 0.0115) and np.isclose(black["fill"], 0.0115 / 0.0135)
    assert np.isclose(black["floor"][2], black["rim"][2] - 0.0135)              # the inside floor, under the wall
    assert caps["inkcap_medium_1"]["text"] == "M1 white ≥7.5 mm 79%"
    assert absent["state"] == "absent" and absent["height"] == 0.0 and absent["text"] == "S1 no cap"
    assert np.isclose(absent["rim"][2], black["rim"][2])                         # its ring where its rim would be
    water = caps["inkcap_large_1"]                                               # not in this catalog
    assert water["state"] == "unknown" and water["text"] == "L1 water? not an ink" and water["rgb"] is None


def test_a_cap_is_loaded_only_as_a_dip_admits_it(tmp_path):
    """Loaded: the cap there, a catalog ink and its level (tatbot_session.cap's admission); anything else is not."""
    repo = _deployed(tmp_path / "repo")
    slots = {"inkcap_large_1": ("example_black", "true", 0.0115),
             "inkcap_medium_1": ("none", "null", None),           # declared unknown
             "inkcap_small_1": ("none", "false", None),           # no cap
             "inkcap_small_2": ("example_red", "true", None),     # an ink, no level
             "inkcap_medium_2": ("none", "true", None)}           # empty; large_2 never declared
    (repo / "config" / "palette_load.yaml").write_text("schema_version: 1\nslots:\n" + "".join(
        f"  {slot}:\n    ink: {ink}\n    cap_present: {present}\n"
        + (f"    level_lower_bound_m: {level}\n" if level else "") for slot, (ink, present, level) in slots.items()))
    caps = scene.caps(repo, np.eye(4), Counter({"inkcap_large_1": 1}))
    assert [(c["alias"], c["state"], c["text"]) for c in caps] == [
        ("L1", "loaded", "L1 black ≥11.5 mm 85% · 1 dip"), ("M1", "unknown", "M1 unknown"),
        ("S1", "absent", "S1 no cap"), ("S2", "unknown", "S2 red, level?"), ("M2", "empty", "M2 empty"),
        ("L2", "unknown", "L2 unknown")]
    assert [c["level"] is None for c in caps] == [False] + [True] * 5


def _stl(path) -> np.ndarray:
    """A binary STL's triangles, (N, 3, 3)."""
    import struct

    data = Path(path).read_bytes()
    rows = np.frombuffer(data, np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                         struct.unpack_from("<I", data, 80)[0], 84)
    return rows["v"].reshape(-1, 3, 3).astype(float)


def _surface(triangles, xy, window: float = 0.012) -> float:
    """The highest triangle crossing the z line through each xy point within +-window of z = 0 (the carrier's outer
    face under a tag), its maximum over the points."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    den = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    top = -np.inf
    for p in xy:
        d = p - a[:, :2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = (d[:, 0] * e2[:, 1] - e2[:, 0] * d[:, 1]) / den, (e1[:, 0] * d[:, 1] - d[:, 0] * e1[:, 1]) / den
        z = (a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]))[(u >= 0) & (v >= 0) & (u + v <= 1)]
        top = max(top, z[np.abs(z) < window].max(initial=-np.inf))
    return top


def test_each_wrist_tag_quad_stands_clear_of_its_carrier():
    """The right arm's measured wrist tags lie up to 2.4 mm inside its nominal printed carrier (2026-10-19), so the
    URDF's quads, 0.2 mm off their tag, showed only a sliver: each now stands TAG_LIFT_M further out, clear of the face.
    The ROS stack's URDF holds the right arm alone."""
    import xml.etree.ElementTree as ET

    arm = "right"
    urdf = REPO / "urdf" / "tatbot.urdf"
    joints = {j.find("child").get("link"): (j.find("parent").get("link"), rpy_matrix(
        *([float(v) for v in j.find("origin").get(k).split()] for k in ("xyz", "rpy")))) for j in ET.parse(urdf).iter("joint")}
    visuals = scene.urdf_visuals(urdf)
    cube = next(v for v in visuals if v["link"] == f"{arm}/ee_mount" and "fiducial_cube" in v["path"].name)
    parent, parent_from_mount = joints[f"{arm}/ee_mount"]
    carrier = scene.transform(parent_from_mount @ cube["link_from_visual"] @ np.diag(cube["scale"] + [1]),
                              _stl(cube["path"]).reshape(-1, 3)).reshape(-1, 3, 3)
    quads = [v for v in visuals if v["kind"] == "mesh" and scene.is_tag_quad(v["path"]) and v["link"].startswith(arm)]
    assert len(quads) == 3
    grid = np.stack(np.meshgrid(*[np.linspace(-0.0235, 0.0235, 9)] * 2), -1).reshape(-1, 2)   # the black square
    for quad in quads:
        tag_parent, parent_from_tag = joints[quad["link"]]
        assert tag_parent == parent
        np.testing.assert_allclose(quad["link_from_visual"][:3, 3], [0, 0, 0.0002 + scene.TAG_LIFT_M])
        face = parent_from_tag @ quad["link_from_visual"]
        local = scene.transform(np.linalg.inv(face), carrier.reshape(-1, 3)).reshape(-1, 3, 3)
        assert -scene.TAG_LIFT_M < _surface(local, grid) < -0.0003, quad["link"]   # sunk before, clear by 0.3 mm now


def test_the_palette_takes_the_arms_black_and_shows_its_tag(tmp_path):
    repo = _deployed(tmp_path / "repo")
    for part in ("meshes/palette/calibration_palette_v11.stl", "meshes/tags/36h11_000_46mm/tag.glb"):
        (repo / "urdf" / part).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "urdf" / part, repo / "urdf" / part)
    robot = REPO / "urdf" / "tatbot.urdf"
    assert scene.material_rgba(robot) == (26, 26, 26, 255) and scene.material_rgba(robot, "none") is None
    print_, tag = scene.palette_visuals(repo, robot)
    assert print_["link"] == tag["link"] == "palette_root" and print_["rgba"] == (26, 26, 26, 255)
    assert tag["name"] == "tag_0" and scene.is_tag_quad(tag["path"]) and tag["rgba"] is None   # its texture alone
    from tatbot_motion import station

    seat = station.palette_from_tag(repo / "urdf" / "palette.urdf")
    np.testing.assert_allclose(tag["link_from_visual"][:3, :3], seat[:3, :3])
    np.testing.assert_allclose(tag["link_from_visual"][:3, 3], seat[:3, 3] + seat[:3, 2] * scene.TAG_LIFT_M)
    (repo / "urdf" / "meshes/tags/36h11_000_46mm/tag.glb").unlink()
    assert [v["name"] for v in scene.palette_visuals(repo, robot)] == ["visual_0"]   # no asset: the print alone


def test_tag_assets_name_the_generated_quads():
    assert scene.tag_asset("apriltag_36h11", 2, 0.047) == "meshes/tags/36h11_002_47mm/tag.glb"
    assert scene.tag_asset("apriltag_36h11", 0, 0.046) == "meshes/tags/36h11_000_46mm/tag.glb"
    assert scene.is_tag_quad(REPO / "urdf" / scene.tag_asset("apriltag_36h11", 0, 0.046))
    assert not scene.is_tag_quad(REPO / "urdf/meshes/palette/calibration_palette_v11.glb")


def test_page_outline_in_world():
    world_from_page = rpy_matrix([0.3, -0.2, 0.05], [0.0, 0.0, 0.0])
    page = scene.transform(world_from_page, scene.page_local(scene.rectangle(0.100, 0.150)[:, :2]))
    assert page.shape == (5, 3) and np.allclose(page[0], page[-1])
    assert np.allclose(page.min(0), [0.25, -0.275, 0.05]) and np.allclose(page.max(0), [0.35, -0.125, 0.05])


def test_a_pose_moved_past_noise():
    pose = rpy_matrix([0.3, -0.2, 0.05], [0.0, 0.0, 0.1])
    assert scene.moved(None, pose) and not scene.moved(pose, pose)
    assert not scene.moved(pose, pose + 3e-4) and scene.moved(pose, rpy_matrix([0.3, -0.2, 0.051], [0.0, 0.0, 0.1]))
    assert scene.moved(pose, rpy_matrix([0.3, -0.2, 0.05], [0.0, 0.0, 0.101]))


def test_tip_path_steps_and_chunks():
    path = scene.TipPath(min_step_m=0.001, chunk=3)
    assert path.add([0, 0, 0]) and not path.add([0.0005, 0, 0])
    for x in (0.002, 0.004):
        assert path.add([x, 0, 0])
    assert path.index == 0 and path.array().shape == (3, 3)
    assert path.add([0.006, 0, 0])               # a full chunk closes; the next starts at its last point
    assert path.index == 1 and np.allclose(path.array()[:, 0], [0.004, 0.006])
    path.clear()
    assert path.index == 0 and path.array().shape == (0, 3)


RIG_PALETTE = rpy_matrix([0.306, 0.299, 0.026], [0.0, 0.0, np.pi + 0.148])      # the rig's palette, turned about z


def _boxes(layers, pose, view=scene.DEMO_VIEW) -> np.ndarray:
    """Each label's box in the Scene pane's pixels as Rerun draws it (centred on its point, its top just under it):
    rows of (left, top, right, bottom)."""
    labels, layout = layers["labels"][1], scene.CAP_LABELS
    (x, y), half = scene.to_screen(scene.transform(pose, labels["positions"]), view)[0].T, np.array(
        [scene.label_px(text) / 2 for text in labels["labels"]])
    top = y + layout["top_px"]
    return np.column_stack([x - half, top, x + half, top + layout["box_px"]])


def _overlaps(boxes) -> list:
    return [(a, b) for a in range(len(boxes)) for b in range(a + 1, len(boxes))
            if (boxes[a, :2] < boxes[b, 2:]).all() and (boxes[b, :2] < boxes[a, 2:]).all()]


def test_palette_layers_mark_each_cap_by_state_and_label_them_in_a_column(tmp_path):
    repo = _deployed(tmp_path / "repo")
    pose = RIG_PALETTE
    caps = scene.caps(repo, pose, Counter({"inkcap_large_2": 2}))
    layers = scene.palette_layers(caps, pose, highlight="inkcap_large_2")
    assert {name: kind for name, (kind, _) in layers.items()} == {
        "caps": "Cylinders3D", "ink": "Cylinders3D", "marks": "Points3D", "labels": "Points3D", "leaders": "LineStrips3D"}
    rings, ink, marks, labels, leaders = (args for _, args in layers.values())
    assert rings["colors"] == [scene.CAP_RGB["unknown"], scene.CAP_RGB["loaded"], scene.CAP_RGB["absent"],
                               scene.CAP_RGB["absent"], scene.CAP_RGB["loaded"], scene.NEXT_RGB]
    assert rings["line_radii"] == [-0.6, -1.0, -0.6, -0.6, -1.0, -1.5]          # UI points: they hold at any zoom
    assert rings["lengths"][2] == 0.0002 and rings["fill_mode"] == "majorwireframe"
    np.testing.assert_allclose(rings["centers"][5][2], caps[5]["rim"][2] - 0.007)
    assert ink["fill_mode"] == "solid" and ink["colors"] == [caps[1]["rgb"], caps[4]["rgb"], caps[5]["rgb"]]
    np.testing.assert_allclose(ink["lengths"], [0.0075, 0.0075, 0.0115])         # floor to the level at load
    np.testing.assert_allclose([c[2] - h / 2 for c, h in zip(ink["centers"], ink["lengths"], strict=True)],
                               [caps[k]["floor"][2] for k in (1, 4, 5)])
    # each cap's mark over its rim, clear of the D555 cloud on the caps' tops, by state
    lift = scene.CAP_LABELS["lift_m"]
    np.testing.assert_allclose(marks["positions"], [c["rim"] + [0.0, 0.0, lift] for c in caps])
    assert marks["colors"] == rings["colors"] and marks["radii"] == [-3.0, -3.5, -2.0, -2.0, -3.5, -4.0]
    # a row per loaded or next cap, as the eye sees their marks, the rest grouped by what they say, the footnote
    assert labels["labels"] == ["M1 white ≥7.5 mm 79%", "M2 red ≥7.5 mm 79%", "next dip: L2 black ≥11.5 mm 85% · 2 dips",
                                "L1 water? not an ink", "S1 S2 no cap", scene.CAP_FOOTNOTE]
    assert labels["colors"] == [scene.CAP_RGB["loaded"], scene.CAP_RGB["loaded"], scene.NEXT_RGB,
                                scene.CAP_RGB["unknown"], scene.CAP_RGB["absent"], scene.CAP_RGB["empty"]]
    assert [len(strip) for strip in leaders["strips"]] == [2, 4, 2, 2, 4, 4]       # a stem each; loaded ones lead on
    assert leaders["colors"][5] == (*scene.NEXT_RGB, 160)

    # the column at the Demo eye, in its pixels
    layout, (width, height) = scene.CAP_LABELS, scene.DEMO_VIEW["pane_px"]
    boxes = _boxes(layers, pose)
    mark_px = scene.to_screen(scene.transform(pose, marks["positions"]))[0]
    np.testing.assert_allclose((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[0, 0] + boxes[0, 2]) / 2)   # centred
    edge = boxes[:, 2].max()
    np.testing.assert_allclose(edge, mark_px[:, 0].min() - layout["gap_px"])     # the widest clear of the caps
    np.testing.assert_allclose(np.diff(boxes[:, 1]), layout["pitch_px"])         # rows a pitch apart, a box high
    assert layout["pitch_px"] > layout["box_px"] and not _overlaps(boxes)
    assert (boxes[:, :2] >= 0).all() and (boxes[:, 2] <= width).all() and (boxes[:, 3] <= height).all()
    middle = (boxes[:, 1] + boxes[:, 3]) / 2
    # the led rows centred on their marks would stand the last row past the pane's foot: the column moves up into it
    np.testing.assert_allclose(boxes[-1, 3], height - layout["margin_px"])
    assert middle[:3].mean() < mark_px[[1, 4, 5], 1].mean()
    for row, k in zip((0, 1, 2), (1, 4, 5), strict=True):
        strip = scene.transform(pose, leaders["strips"][k])
        px = scene.to_screen(strip)[0]
        np.testing.assert_allclose(px[1], mark_px[k], atol=1e-6)                  # from the cap's mark
        np.testing.assert_allclose(px[2:, 1], middle[row])                        # level along its row
        np.testing.assert_allclose(px[3, 0], boxes[row, 2] + layout["end_px"])     # ending clear of its own label
        np.testing.assert_allclose(px[2, 0], edge + layout["end_px"] + layout["stub_px"])
        assert (px[:3, 0] > edge).all()                    # its slant right of every label, its level run in its row
        assert (strip[1:, 2] > scene.transform(pose, [caps[k]["rim"]])[0, 2] + 0.01).all()   # and over the cloud


def test_caps_not_loaded_share_a_row_by_what_they_say(tmp_path):
    """The forearm's load, one black cap and the rest empty, reads in three rows; with nothing loaded, one row."""
    repo = _deployed(tmp_path / "repo")
    slots = ["inkcap_large_1", "inkcap_medium_1", "inkcap_small_1", "inkcap_small_2", "inkcap_medium_2", "inkcap_large_2"]
    def load(black):
        (repo / "config" / "palette_load.yaml").write_text("schema_version: 1\nslots:\n" + "".join(
            f"  {slot}:\n    ink: {'example_black' if slot == black else 'none'}\n    cap_present: true\n"
            + ("    level_lower_bound_m: 0.0115\n" if slot == black else "") for slot in slots))
        return scene.palette_layers(scene.caps(repo, RIG_PALETTE, Counter()), RIG_PALETTE)

    forearm = load("inkcap_large_1")
    assert forearm["labels"][1]["labels"] == ["L1 black ≥11.5 mm 85%", "M1 S1 S2 M2 L2 empty", scene.CAP_FOOTNOTE]
    assert [len(strip) for strip in forearm["leaders"][1]["strips"]] == [4, 2, 2, 2, 2, 2]
    middle = _boxes(forearm, RIG_PALETTE)[0, 1::2].mean()
    mark = scene.to_screen(scene.transform(RIG_PALETTE, forearm["marks"][1]["positions"][:1]))[0][0]
    np.testing.assert_allclose(middle, mark[1])                                   # its leader runs level
    assert load(None)["labels"][1]["labels"] == ["L1 M1 S1 S2 M2 L2 empty"]       # no footnote to no level


def test_a_column_past_the_pane_moves_into_it():
    """Caps high in the pane would stand their rows above its top: the column moves down into it, its top first."""
    marks = scene.from_screen([[400.0, 20.0], [410.0, 30.0]], 0.7)
    rows, points, leaders = scene.label_column(marks, [("M1 white", 1), ("L1 black", 0)], ["S1 S2 empty"], "note")
    assert [text for text, _ in rows] == ["L1 black", "M1 white", "S1 S2 empty", "note"] and set(leaders) == {0, 1}
    tops = scene.to_screen(points)[0][:, 1] + scene.CAP_LABELS["top_px"]
    np.testing.assert_allclose(tops[0], scene.CAP_LABELS["margin_px"])
    np.testing.assert_allclose(np.diff(tops), scene.CAP_LABELS["pitch_px"])


def _turned(az_deg: float, el_deg: float) -> tuple:
    """The Demo eye turned about its look-at by az_deg about z and el_deg up, at its distance."""
    look_at = np.asarray(scene.DEMO_VIEW["look_at"])
    offset = np.asarray(scene.DEMO_VIEW["eye"]) - look_at
    az, el = np.arctan2(offset[1], offset[0]) + np.radians(az_deg), np.arcsin(
        offset[2] / np.linalg.norm(offset)) + np.radians(el_deg)
    return tuple(look_at + np.linalg.norm(offset) * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                                                              np.sin(el)]))


@pytest.mark.parametrize("eye", [(0.68, 0.51, 0.52),                            # a touch turned the station's (10-19)
                                 *[_turned(az, el) for az in (-30, 30) for el in (-15, 15)]])
def test_the_rows_stay_apart_and_off_the_caps_while_a_touch_turns_the_eye(tmp_path, eye):
    """Laid out for the default eye, the column holds for an eye turned 30° about the scene's look-at or tilted 15°
    at the same zoom: no row over another nor over a cap's mark, and the loaded and next caps' rows in the pane (the
    rest may pass its foot; `tatbot viewer reset` restores the default)."""
    repo = _deployed(tmp_path / "repo")
    caps = scene.caps(repo, RIG_PALETTE, Counter())
    layers = scene.palette_layers(caps, RIG_PALETTE, highlight="inkcap_large_2")
    view = {**scene.DEMO_VIEW, "eye": eye}
    boxes = _boxes(layers, RIG_PALETTE, view)
    marks = scene.to_screen(scene.transform(RIG_PALETTE, layers["marks"][1]["positions"]), view)[0]
    assert not _overlaps(boxes)
    assert not [(m, b) for m in marks for b in boxes if (m > b[:2]).all() and (m < b[2:]).all()]
    assert (boxes[:3, :2] >= 0).all() and (boxes[:3, 2:] <= scene.DEMO_VIEW["pane_px"]).all()


def _frame_set(frames) -> bytes:
    """A tatbot-vision-frame-set v1 blob by the wire's layout (4-byte header length, JSON header, payloads)."""
    header = {"magic": "tatbot-vision-frame-set", "version": 1, "frames": [
        {"metadata": metadata, "payload": {kind: {**shape, "bytes": len(data)}}} for metadata, kind, shape, data in frames]}
    text = json.dumps(header).encode()
    return len(text).to_bytes(4, "big") + text + b"".join(data for *_, data in frames)


def test_wrist_depth_from_the_tap_wire():
    from frame_set_wire import unpack

    depth = (np.arange(640 * 480, dtype="<u2") % 4000).reshape(480, 640)
    intrinsics = {"fx": 380.0, "fy": 381.0, "ppx": 320.5, "ppy": 240.5, "width": 640, "height": 480}
    stamp = {"normalized_unix_ns": 1792342631962000000}
    blob = _frame_set([
        ({"sensor_name": "realsense2_color", "timestamps": stamp, "attributes": {}}, "Encoded", {"format": "Jpeg"}, b"\xff\xd8"),
        ({"sensor_name": "realsense2_depth", "timestamps": stamp, "attributes": {
            "intrinsics": json.dumps(intrinsics), "depth_units_m": "0.0001", "aligned_to": "realsense2_color",
            "ros_frame": "right/realsense_depth_optical_frame", "physical_arm": "right"}},
         "Depth", {"width": 640, "height": 480}, depth.tobytes())])
    frame = scene.wrist_depth(unpack(blob)[1])
    assert frame["stamp_ns"] == 1792342631962000000 and frame["ros_frame"] == "right/realsense_depth_optical_frame"
    assert frame["depth"].shape == (240, 320) and frame["depth"][1, 1] == depth[2, 2] and frame["meter"] == 10000.0
    kept = depth[::2, ::2]                                     # 0.07-0.50 m kept; nearer, farther and 0 are 0
    assert np.array_equal(frame["depth"], np.where((kept >= 700) & (kept <= 5000), kept, 0))
    assert frame["depth"][0, 0] == 0 and frame["depth"].max() == 3998 and frame["depth"][frame["depth"] > 0].min() == 700
    far = scene.wrist_depth(unpack(blob)[1], range_m=(0.07, 0.2))["depth"]
    assert far.max() == 2000
    np.testing.assert_allclose(frame["focal"] + frame["principal"], [190.0, 190.5, 160.25, 120.25])
    assert scene.wrist_depth({}) is None


def _wrist_blob(depth, color=b"\xff\xd8", intrinsics=None):
    intrinsics = intrinsics or {"fx": 380.0, "fy": 381.0, "ppx": 320.5, "ppy": 240.5, "width": 640, "height": 480}
    stamp = {"normalized_unix_ns": 1792342631962000000}
    height, width = depth.shape
    return _frame_set([
        ({"sensor_name": "realsense2_color", "timestamps": stamp, "attributes": {}}, "Encoded", {"format": "Jpeg"}, color),
        ({"sensor_name": "realsense2_depth", "timestamps": stamp, "attributes": {
            "intrinsics": json.dumps(intrinsics), "depth_units_m": "0.0001", "aligned_to": "realsense2_color",
            "ros_frame": "right/realsense_depth_optical_frame", "physical_arm": "right"}},
         "Depth", {"width": width, "height": height}, depth.astype("<u2").tobytes())])


def test_the_wrist_cloud_is_the_depths_rays_coloured_from_the_tap_and_tinted():
    from frame_set_wire import unpack

    depth = np.zeros((480, 640), "<u2")
    depth[100:140, 200:260] = 2000                      # a 0.2 m patch; the rest is no return
    blob = _wrist_blob(depth)
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :320] = (255, 0, 0)                         # the patch's columns are in the red half
    cloud = scene.wrist_cloud(unpack(blob)[1], decode=lambda data: image)
    frame = scene.wrist_depth(unpack(blob)[1])
    assert cloud["depth"].shape == frame["depth"].shape and cloud["ros_frame"] == frame["ros_frame"]
    points, colors = cloud["points"], cloud["colors"]
    assert points.dtype == np.float32 and colors.dtype == np.uint8 and len(points) == len(colors) == 20 * 30
    np.testing.assert_allclose(points[:, 2], 0.2)
    # Rerun's pixel-centre ray under the same pinhole: column 100 of the 320-wide plane, principal 160.25.
    (fx, _), (px, _) = frame["focal"], frame["principal"]
    np.testing.assert_allclose(points[0, 0], (100 + 0.5 - px) / fx * 0.2, rtol=1e-6)
    style = scene.WRIST_CLOUD
    tinted = np.round(np.array([255, 0, 0]) * (1 - style["tint_share"]) + np.array(style["tint"]) * style["tint_share"])
    assert (colors == tinted.astype(np.uint8)).all()
    # Rows within the band budget, a fixed count of them, the later ones empty.
    assert len(cloud["bands"]) == -(-style["max_points"] // style["band"])
    assert sum(len(p) for p, _, _ in cloud["bands"]) == len(points)
    assert all(len(p) <= style["band"] and len(p) == len(c) for p, c, _ in cloud["bands"])
    assert len(cloud["bands"][-1][0]) == 0 and cloud["bands"][-1][2] == 0.0
    # Every pixel of the patch is a point, so neighbours are one pixel apart: 0.2 m / fx; a point is wider than half that.
    np.testing.assert_allclose(cloud["bands"][0][2], style["radius_share"] * 0.2 * 2 / sum(frame["focal"]), rtol=1e-6)
    assert 2 * cloud["bands"][0][2] > 0.2 / fx


def test_the_wrist_cloud_keeps_its_budget_and_takes_the_tint_without_colour():
    from frame_set_wire import unpack

    depth = np.full((480, 640), 3000, "<u2")
    blob = _wrist_blob(depth)
    for decode in (None, lambda data: None, lambda data: 1 / 0, lambda data: np.zeros((4, 4), np.uint8)):
        cloud = scene.wrist_cloud(unpack(blob)[1], decode=decode)
        assert 0 < len(cloud["points"]) <= scene.WRIST_CLOUD["max_points"]
        assert (cloud["colors"] == np.array(scene.WRIST_CLOUD["tint"], np.uint8)).all()
    assert scene.wrist_cloud({}) is None


def test_a_tag_glb_loses_its_samplers_and_nothing_else():
    """Rerun 0.36 warns once per primitive about a glTF sampler it ignores (15 notifications on the station,
    2026-10-19); the wrist-tag meshes ask for nearest and clamp-to-edge."""
    import struct

    from tatbot_contracts.ros_scene import mesh_asset
    tags = sorted((Path(__file__).resolve().parents[2] / "urdf/meshes/tags").glob("*/tag.glb"))
    if not tags:
        pytest.skip("no wrist-tag meshes in this tree")
    before = tags[0].read_bytes()
    after, media = mesh_asset(tags[0])
    assert media == "model/gltf-binary" and after[:4] == b"glTF"
    assert struct.unpack_from("<I", after, 8)[0] == len(after) and struct.unpack_from("<I", after, 12)[0] % 4 == 0
    size = struct.unpack_from("<I", after, 12)[0]
    doc, old = json.loads(after[20:20 + size]), json.loads(before[20:20 + struct.unpack_from("<I", before, 12)[0]])
    assert "samplers" in old and "samplers" not in doc and all("sampler" not in t for t in doc["textures"])
    assert doc["images"] == old["images"] and doc["meshes"] == old["meshes"]
    assert after[20 + size:] == before[20 + struct.unpack_from("<I", before, 12)[0]:]   # the BIN chunk untouched
    stl = next((Path(__file__).resolve().parents[2] / "urdf/meshes").rglob("*.stl"), None)
    if stl is not None:
        assert mesh_asset(stl) == (stl.read_bytes(), None)   # any other mesh goes as its file


def test_the_design_frame_centres_the_design_and_the_tip_shows_only_inside_it():
    from tatbot_contracts.ros_scene import design_frame, within
    strokes = [np.array([[30.0, 80.0], [50.0, 100.0]]), np.array([[40.0, 70.0], [45.0, 110.0]])]
    frame = design_frame(strokes)   # a design low-right on a 83 x 127 mm print
    lo, hi = frame[:4].min(0), frame[:4].max(0)
    assert np.allclose(frame[0], frame[-1]) and np.allclose((lo + hi) / 2, [40.0, 90.0])   # centred on the strokes
    assert np.all(lo < [30.0, 70.0]) and np.all(hi > [50.0, 110.0])                          # with a margin
    assert within(frame, [40.0, 90.0]) and not within(frame, [-120.0, 300.0])               # a resting tip is out
    assert design_frame([]) is None and design_frame([np.zeros((0, 2))]) is None


def test_the_print_lies_under_the_design_map_where_printed_and_inside_the_frame():
    gray = np.full((1260, 900), 255, np.uint8)   # a 45 x 63 mm print at 20 px/mm, one 12 mm marker at (16.5, 24) mm
    gray[480:720, 330:570] = 0
    frame = scene.design_frame([np.array([[14.0, 22.0], [31.0, 40.0]])])
    lo, hi = frame[:4].min(0), frame[:4].max(0)
    for size_m, shift in (([0.045, 0.063], 0.0), ([0.065, 0.083], 10.0)):   # the stack's page larger: the print centred
        under = scene.stencil_underlay(gray, [45.0, 63.0], size_m, frame + shift, max_px=200)
        rgba, (tx, ty, _), (sx, sy, _) = under["rgba"], under["translation"], under["scale"]
        assert max(rgba.shape[:2]) <= 200 and np.all(rgba[..., :3] == scene.STENCIL_RGBA[:3])
        corner, far = np.array([tx, ty]), np.array([tx, ty]) + np.array(rgba.shape[1::-1]) * (sx, sy)
        assert np.all(corner >= lo + shift - 1e-9) and np.all(far <= hi + shift + 1e-9)   # the fit stays the frame's
        ink = rgba[..., 3] > scene.STENCIL_RGBA[3] / 2   # over the marker (by area, half in at its edges)
        cols, rows = np.flatnonzero(ink.any(0)), np.flatnonzero(ink.any(1))
        assert np.allclose(tx + np.array([cols[0], cols[-1] + 1]) * sx - shift, [16.5, 28.5], atol=sx)
        assert np.allclose(ty + np.array([rows[0], rows[-1] + 1]) * sy - shift, [24.0, 36.0], atol=sy)
        assert rgba[..., 3].max() == scene.STENCIL_RGBA[3] and rgba[0, 0, 3] == 0   # ink faint, paper clear
    assert scene.stencil_underlay(gray, [45.0, 63.0], [0.045, 0.063], None) is None
    assert scene.stencil_underlay(gray, [45.0, 63.0], [0.045, 0.063], frame + 100.0) is None   # off the print


def test_the_print_a_run_draws_on_is_its_run_logs(run_dir):
    state = scene.DrawState.load(run_dir)
    assert state.pattern_id is None
    (run_dir / "meta.json").write_text(json.dumps({"arms": {"right": {"pattern_id": "stencil-fc77"}}}))
    state.poll()
    assert state.pattern_id == "stencil-fc77"
