"""Pin the ink registry and the charge model.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_ink_spec.py

What the design depends on: the three tools have the three modes; a laser is
refused for any task that needs ink; the ballpoint rehearses on dry caps with
the same plan a real needle would make on full ones; the planner's dips are
exactly what the charge arithmetic predicts; the ledger replays
deterministically and rehearsal/sim events never move real stock; the two
script-owned YAML files survive a write/read round-trip through the strict
parser.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import ink_spec  # noqa: E402
import tool_spec  # noqa: E402


@pytest.fixture
def palette():
    return ink_spec.load_palette(REPO)


@pytest.fixture
def dry(palette):
    return {s: ink_spec.SlotLoad(s, None) for s in palette}


@pytest.fixture
def full(palette):
    return {s: ink_spec.SlotLoad(s, "nighthawk_black", palette[s].size.capacity_ul * 0.7) for s in palette}


def pol(tool_id):
    return ink_spec.policy_for(tool_spec.load_tool(tool_id, REPO))


# --- registries ------------------------------------------------------------------

def test_registries_load_and_agree(palette):
    inks = ink_spec.load_inks(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    assert set(load) == set(palette)
    assert "nighthawk_black" in inks and inks["true_blue"].rgb == (0, 0, 255)
    # every URDF inkcap frame is a slot, and every slot is a URDF frame
    urdf = (REPO / "urdf" / "tatbot.urdf").read_text()
    for slot_id in palette:
        assert f'<link name="{slot_id}"' in urdf, slot_id
    assert urdf.count('<link name="inkcap_') == len(palette)


def test_palette_root_in_base():
    right_pos = ink_spec.palette_root_in_base(REPO, arm="right")
    left_pos = ink_spec.palette_root_in_base(REPO, arm="left")

    assert len(right_pos) == 3 and len(left_pos) == 3
    assert right_pos == pytest.approx((0.126, 0.2675, 0.085))
    assert left_pos == pytest.approx((0.126, -0.2675, 0.085))
    # Left and right arm base links are symmetric along the Y-axis
    assert right_pos[0] == pytest.approx(left_pos[0])
    assert right_pos[1] == pytest.approx(-left_pos[1])
    assert right_pos[2] == pytest.approx(left_pos[2])

    with pytest.raises(ValueError, match="no joint has child"):
        ink_spec.palette_root_in_base(REPO, arm="invalid_arm")


def test_cap_geometry_is_self_consistent(palette):
    large = palette["inkcap_right_large"].size
    assert large.surface_depth_m(0) == pytest.approx(large.depth_m)
    assert large.surface_depth_m(large.capacity_ul) == pytest.approx(0.0, abs=1e-6)
    assert 0 < large.surface_depth_m(large.capacity_ul / 2) < large.depth_m


def test_palette_layout_from_urdf():
    layout = ink_spec.palette_layout_from_urdf(REPO)
    assert len(layout) == 10
    for slot_id, xyz in layout.items():
        assert slot_id.startswith("inkcap_")
        assert len(xyz) == 3
        assert all(isinstance(v, float) for v in xyz)
    assert layout["inkcap_right_large"] == (0.0, -0.055, 0.0)
    assert layout["inkcap_left_large"] == (0.0, 0.055, 0.0)


def test_palette_urdf_error_handling(tmp_path):
    # test palette_layout_from_urdf with no inkcap joints
    urdf_dir = tmp_path / "urdf"
    urdf_dir.mkdir()
    (urdf_dir / "tatbot.urdf").write_text("<robot name=\"test\"><link name=\"root\"/></robot>")
    with pytest.raises(ValueError, match=r"no inkcap_\* joints"):
        ink_spec.palette_layout_from_urdf(tmp_path)

    # test palette_root_in_base with missing joints
    with pytest.raises(ValueError, match="no joint has child 'palette_root'"):
        ink_spec.palette_root_in_base(tmp_path, arm="right")


def test_three_tools_three_modes():
    assert pol("lutin-3rl-bugpin").mode == "real"
    assert pol("lutin-ballpoint-dot").mode == "rehearsal"
    assert pol("picosecond-laser-pen").mode == "none"
    assert not pol("picosecond-laser-pen").dips
    # rehearsal mirrors real so it plans the same dips
    r, b = pol("lutin-3rl-bugpin"), pol("lutin-ballpoint-dot")
    assert (r.uptake_ul, r.deposit_ul_per_mm, r.bleed_ul_per_s) == (b.uptake_ul, b.deposit_ul_per_mm, b.bleed_ul_per_s)


def test_bad_policy_refused():
    class Fake:
        tool_id = "fake"
        raw = {"ink": {"mode": "real", "charge_capacity_ul": 1.0, "uptake_ul": 2.0, "dip_depth_m": 0.003}}
    with pytest.raises(ValueError, match="uptake_ul exceeds"):
        ink_spec.policy_for(Fake())
    Fake.raw = {"ink": {"mode": "wet"}}
    with pytest.raises(ValueError, match="ink.mode"):
        ink_spec.policy_for(Fake())


# --- supply validation ---------------------------------------------------------------

def test_laser_refused_for_ink_tasks(palette, full):
    with pytest.raises(ink_spec.InkSupplyError, match="ink.mode none"):
        ink_spec.require_supply(pol("picosecond-laser-pen"), palette, full, needs_ink=True)
    ink_spec.require_supply(pol("picosecond-laser-pen"), palette, full, needs_ink=False)  # erase is fine


def test_real_refuses_dry_and_low_caps(palette, dry, full):
    real = pol("lutin-3rl-bugpin")
    with pytest.raises(ink_spec.InkSupplyError, match="no usable right-arm cap"):
        ink_spec.require_supply(real, palette, dry, needs_ink=True)
    ink_spec.require_supply(real, palette, full, needs_ink=True)
    low = {s: ink_spec.SlotLoad(s, "nighthawk_black", 1.0) for s in palette}  # 1 uL: below min_fill_frac
    with pytest.raises(ink_spec.InkSupplyError):
        ink_spec.require_supply(real, palette, low, needs_ink=True)
    # wrong colour is no supply either
    with pytest.raises(ink_spec.InkSupplyError, match="of true_blue"):
        ink_spec.require_supply(real, palette, full, needs_ink=True, ink_id="true_blue")


def test_rehearsal_accepts_dry_caps(palette, dry):
    ink_spec.require_supply(pol("lutin-ballpoint-dot"), palette, dry, needs_ink=True)
    assert [s.slot_id for s in ink_spec.usable_slots(pol("lutin-ballpoint-dot"), palette, dry)] == [
        s for s in palette if palette[s].arm == "right"]


def test_left_arm_slots_are_not_used_by_the_right(palette, full):
    for s in ink_spec.usable_slots(pol("lutin-3rl-bugpin"), palette, full, arm="right"):
        assert palette[s.slot_id].arm == "right"


# --- charge model and planner ------------------------------------------------------------

def test_stroke_cost_counts_time_on_skin():
    p = pol("lutin-3rl-bugpin")
    moving = p.stroke_ul(100.0, 10.0)
    parked = p.stroke_ul(0.0, 10.0)
    assert parked > 0, "a parked needle still bleeds"
    assert moving == pytest.approx(100 * p.deposit_ul_per_mm + 10 * p.bleed_ul_per_s)


def test_planner_matches_the_arithmetic(palette, full):
    p = pol("lutin-3rl-bugpin")
    per = p.stroke_ul(60.0, 5.0)
    strokes = [ink_spec.StrokeNeed(60.0, 5.0)] * 20
    plans = ink_spec.plan_dips(strokes, p, palette, full)
    assert plans[0].before_stroke == 0 and plans[0].reason == "session_start"
    # after a dip the needle holds uptake_ul; it lasts floor(uptake/per) strokes
    # before the next stroke's need exceeds what is left
    stride = int(p.uptake_ul // per) if per > 0 else len(strokes)
    expect = [0]
    while expect[-1] + stride < len(strokes):
        expect.append(expect[-1] + stride)
    assert [d.before_stroke for d in plans] == expect
    assert all(d.reason == "low_charge" for d in plans[1:])
    assert all(d.charge_after_ul <= p.charge_capacity_ul for d in plans)


def test_colour_change_forces_a_dip(palette):
    p = pol("lutin-3rl-bugpin")
    load = {s: ink_spec.SlotLoad(s, None) for s in palette}
    load["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "nighthawk_black", 1000.0)
    load["inkcap_right_medium_0"] = ink_spec.SlotLoad("inkcap_right_medium_0", "true_blue", 500.0)
    strokes = [ink_spec.StrokeNeed(5, 0.5, "nighthawk_black"), ink_spec.StrokeNeed(5, 0.5, "true_blue"),
               ink_spec.StrokeNeed(5, 0.5, "true_blue")]
    plans = ink_spec.plan_dips(strokes, p, palette, load)
    assert [(d.before_stroke, d.reason, d.slot_id) for d in plans] == [
        (0, "session_start", "inkcap_right_large"),
        (1, "color_change", "inkcap_right_medium_0"),
    ]
    assert plans[1].charge_before_ul == 0.0, "a colour change starts from a wiped needle"


def test_none_never_dips_and_rehearsal_equals_real(palette, dry, full):
    strokes = [ink_spec.StrokeNeed(40, 4)] * 12
    assert ink_spec.plan_dips(strokes, pol("picosecond-laser-pen"), palette, full) == []
    real = ink_spec.plan_dips(strokes, pol("lutin-3rl-bugpin"), palette, full)
    reh = ink_spec.plan_dips(strokes, pol("lutin-ballpoint-dot"), palette, dry)
    assert [(d.before_stroke, d.reason) for d in real] == [(d.before_stroke, d.reason) for d in reh]


def test_real_planner_drains_the_cap(palette):
    p = pol("lutin-3rl-bugpin")
    load = {s: ink_spec.SlotLoad(s, None) for s in palette}
    # a small cap sitting 1.5 dips above the min_fill_frac floor: the first
    # dip fits, the second lands on the floor, the third has no usable cap
    size = palette["inkcap_right_small_0"].size
    floor = p.min_fill_frac * size.capacity_ul * size.usable_frac
    load["inkcap_right_small_0"] = ink_spec.SlotLoad("inkcap_right_small_0", "nighthawk_black", floor + 1.5 * p.uptake_ul)
    strokes = [ink_spec.StrokeNeed(1000, 100)] * 3
    with pytest.raises(ink_spec.InkSupplyError, match="stroke 2"):
        ink_spec.plan_dips(strokes, p, palette, load)


def test_dip_depth_follows_fill(palette):
    p = pol("lutin-3rl-bugpin")
    slot = palette["inkcap_right_large"]
    assert ink_spec.dip_plunge_m(p, slot, 0.0) == pytest.approx(p.dip_depth_m)
    half = ink_spec.dip_plunge_m(p, slot, slot.size.capacity_ul / 2)
    assert half > ink_spec.dip_plunge_m(p, slot, slot.size.capacity_ul * 0.7)
    assert ink_spec.dip_plunge_m(p, slot, 1.0) <= slot.size.depth_m


# --- ledger -------------------------------------------------------------------------------

def test_ledger_replay_ignores_rehearsal_and_sim(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ink_spec.append_event("cap.fill", "real", path=path, slot="inkcap_right_large", ink_id="nighthawk_black", ul=500, bottle="b1")
    ink_spec.append_event("dip", "real", path=path, slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("dip", "rehearsal", path=path, slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("dip", "sim", path=path, slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("stroke", "real", path=path, contact_mm=30, contact_s=3, ul=0.15)
    ink_spec.append_event("stroke", "rehearsal", path=path, contact_mm=30, contact_s=3, ul=0.15)
    r = ink_spec.replay(ink_spec.read_events(path))
    assert r.cap_fill_ul["inkcap_right_large"] == pytest.approx(498.5)
    assert r.dips == 1 and r.stroke_ul == pytest.approx(0.15)
    assert r.bottle_used_ul["b1"] == 500
    assert r.ignored == {"rehearsal": 2, "sim": 1}
    with pytest.raises(ValueError):
        ink_spec.append_event("dip", "wet", path=path)
    # every line is one JSON object with the common keys
    for line in path.read_text().splitlines():
        ev = json.loads(line)
        assert {"utc", "node", "kind", "mode"} <= set(ev)


# --- script-owned files round-trip ------------------------------------------------------

def _copy_repo_configs(tmp_path):
    (tmp_path / "config" / "tools").mkdir(parents=True)
    for rel in (ink_spec.INKS_RELPATH, ink_spec.PALETTE_RELPATH, ink_spec.LOAD_RELPATH, ink_spec.INVENTORY_RELPATH):
        shutil.copy(REPO / rel, tmp_path / rel)
    return tmp_path


def test_palette_load_round_trip(tmp_path, palette):
    repo = _copy_repo_configs(tmp_path)
    load = ink_spec.load_palette_load(repo, palette)
    load["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "nighthawk_black", 1200.0, "kuro_sumi_black_01", "2026-08-28T00:00:00Z")
    ink_spec.write_palette_load(load, repo, note="test")
    back = ink_spec.load_palette_load(repo, palette)
    assert back["inkcap_right_large"] == load["inkcap_right_large"]
    assert back["inkcap_right_small_1"].dry
    text = (repo / ink_spec.LOAD_RELPATH).read_text()
    assert text.startswith("# palette_load.yaml"), "header comment preserved"
    assert "note: test" in text


def test_inventory_round_trip(tmp_path):
    repo = _copy_repo_configs(tmp_path)
    inv = ink_spec.load_inventory(repo)
    inv["cartridges"]["quelle_1003rl_box01"]["count"] = 37
    inv["bottles"]["kuro_sumi_black_01"]["remaining_ml"] = 300.5
    ink_spec.write_inventory(inv, repo)
    back = ink_spec.load_inventory(repo)
    assert back["cartridges"]["quelle_1003rl_box01"]["count"] == 37
    assert back["bottles"]["kuro_sumi_black_01"]["remaining_ml"] == 300.5
    assert back["cartridges"]["quelle_1003rl_box01"]["opened"] == "2025-02-25", "dates stay strings"
    assert back["caps"] == inv["caps"]


def test_dataset_metadata_is_self_contained():
    meta = ink_spec.dataset_ink_metadata(tool_spec.load_tool("lutin-ballpoint-dot", REPO), REPO)
    assert meta["policy"]["mode"] == "rehearsal"
    assert set(meta["slots"]) == {s for s, p in ink_spec.load_palette(REPO).items() if p.arm == "right"}
    json.dumps(meta)  # serialisable as written


# --- cap selection and mise en place -----------------------------------------------------

def test_select_slot_prefers_the_inks_cap_size_and_says_why(palette):
    real = pol("lutin-3rl-bugpin")
    inks = ink_spec.load_inks(REPO)
    load = {s: ink_spec.SlotLoad(s, "nighthawk_black", 200.0) for s in palette}
    load["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "nighthawk_black", 1500.0)
    c = ink_spec.select_slot(real, palette, load, "right", "nighthawk_black", inks)
    assert palette[c.slot_id].size.size_id == "medium", c   # small caps are narrow; the needle misses
    assert "lining prefers medium" in c.reason
    # a session need larger than any small cap holds moves it to a cap that covers it
    c = ink_spec.select_slot(real, palette, load, "right", "nighthawk_black", inks, need_ul=900.0)
    assert c.slot_id == "inkcap_right_large" and "covers the remaining 900.0" in c.reason
    # colour ink prefers medium
    load2 = {s: ink_spec.SlotLoad(s, "true_blue", 200.0) for s in palette}
    c = ink_spec.select_slot(real, palette, load2, "right", "true_blue", inks)
    assert palette[c.slot_id].size.size_id == "medium"
    # rehearsal: first usable cap, no ink reasoning
    c = ink_spec.select_slot(pol("lutin-ballpoint-dot"), palette,
                             {s: ink_spec.SlotLoad(s, None) for s in palette}, "right", None, inks)
    assert c.reason.startswith("rehearsal")


def test_mise_en_place_fills_dry_caps_and_flags_stock(palette):
    real = pol("lutin-3rl-bugpin")
    inks = ink_spec.load_inks(REPO)
    inv = ink_spec.load_inventory(REPO)
    dry = {s: ink_spec.SlotLoad(s, None) for s in palette}
    items = ink_spec.mise_en_place(real, palette, dry, inks, inv, {"nighthawk_black": 300.0},
                                   tool_id="lutin-3rl-bugpin")
    kinds = [i.kind for i in items]
    assert "fill" in kinds and "cartridge" in kinds and kinds[-1] == "weigh"
    fills = [i for i in items if i.kind == "fill"]
    assert all(i.ink_id == "nighthawk_black" and "nighthawk_black_01" in i.text for i in fills)
    assert sum(i.ul for i in fills) >= 300.0 * 1.15
    # a cap below the floor is a refill, not a new fill; an unused ink is noted
    low = dict(dry)
    low["inkcap_right_small_0"] = ink_spec.SlotLoad("inkcap_right_small_0", "nighthawk_black", 5.0)
    low["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "true_blue", 900.0)
    items = ink_spec.mise_en_place(real, palette, low, inks, inv, {"nighthawk_black": 100.0},
                                   tool_id="lutin-3rl-bugpin")
    assert any(i.kind == "refill" and i.slot_id == "inkcap_right_small_0" for i in items)
    assert any(i.kind == "info" and i.slot_id == "inkcap_right_large" for i in items)
    # rehearsal wants dry caps; a wet one is a dump
    items = ink_spec.mise_en_place(pol("lutin-ballpoint-dot"), palette, low, inks, inv, {},
                                   tool_id="lutin-ballpoint-dot")
    assert any(i.kind == "dump" for i in items)
    assert ink_spec.mise_en_place(pol("picosecond-laser-pen"), palette, dry, inks, inv, {})[0].kind == "ok"


def test_need_from_a_program_matches_the_stroke_arithmetic():
    p = pol("lutin-3rl-bugpin")
    square = [[0, 0], [0.04, 0], [0.04, 0.04], [0, 0.04], [0, 0]]   # 160 mm
    need = ink_spec.need_from_polylines([square], p, 0.03, settle_s=0.2)
    assert need == pytest.approx(p.stroke_ul(160.0, 0.16 / 0.03 + 0.2))
    # every shape a program comes in
    run_meta = {"episodes": [{"strokes_canvas_m": [square]}, {"path_canvas_m": square}]}
    assert ink_spec.program_polylines(run_meta) == [square, square]
    assert ink_spec.program_polylines({"strokes": [square]}) == [square]
    assert ink_spec.program_polylines([square]) == [square]
    assert ink_spec.program_polylines(square) == [square]


def test_dip_plan_carries_the_draining_cap_fill(palette):
    p = pol("lutin-3rl-bugpin")
    load = {s: ink_spec.SlotLoad(s, None) for s in palette}
    load["inkcap_right_medium_0"] = ink_spec.SlotLoad("inkcap_right_medium_0", "nighthawk_black", 400.0)
    plans = ink_spec.plan_dips([ink_spec.StrokeNeed(400, 40)] * 4, p, palette, load)
    assert len(plans) >= 2 and plans[0].cap_fill_ul == 400.0
    assert plans[1].cap_fill_ul == pytest.approx(400.0 - p.uptake_ul)
    assert plans[0].why_slot


# --- 2026-08-29: supply loads, per-ink dips, synced ledgers -----------------------------

def test_supply_loads_are_not_the_bench(palette):
    wet = ink_spec.supply_load("wet", palette, "nighthawk_black")
    assert all(not wet[s].dry for s in palette if palette[s].arm == "right")
    assert all(wet[s].dry for s in palette if palette[s].arm != "right")
    assert all(wet[s].fill_ul == pytest.approx(palette[s].size.capacity_ul * palette[s].size.usable_frac)
               for s in palette if palette[s].arm == "right")
    dry = ink_spec.supply_load("dry", palette)
    assert all(sl.dry for sl in dry.values())
    bench = ink_spec.supply_load("bench", palette)
    assert set(bench) == set(palette)
    with pytest.raises(ValueError, match="needs an ink_id"):
        ink_spec.supply_load("wet", palette)
    with pytest.raises(ValueError, match="not one of"):
        ink_spec.supply_load("damp", palette)


def test_per_ink_dip_overrides_refine_the_datasheet():
    base = pol("lutin-3rl-bugpin")
    thick = ink_spec.Ink("x", "X", (0, 0, 0), "with x", viscosity="opaque",
                         dip={"uptake_ul": 99.0, "dip_dwell_s": 1.2, "dip_depth_m": 0.005})
    got = ink_spec.policy_with_ink(base, thick)
    assert got.dip_dwell_s == 1.2 and got.dip_depth_m == 0.005
    assert got.uptake_ul == base.charge_capacity_ul, "uptake is capped at the tool's capacity"
    assert got.deposit_ul_per_mm == base.deposit_ul_per_mm, "untouched keys stay the datasheet's"
    assert ink_spec.policy_with_ink(base, None) is base
    with pytest.raises(ValueError, match="dip keys"):
        ink_spec._dip_overrides("x", {"colour": 1})
    with pytest.raises(ValueError, match=">= 0"):
        ink_spec._dip_overrides("x", {"uptake_ul": -1})


def test_synced_ledgers_are_read_once(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(ledger))
    a = ink_spec.append_event("dip", "rehearsal", slot="inkcap_right_medium_0")
    b = ink_spec.append_event("dip", "rehearsal", slot="inkcap_right_medium_1")
    assert a["id"] != b["id"]
    remote = ink_spec.remote_ledger_dir()
    remote.mkdir()
    # nodea's copy holds one of ours (synced back) and one of its own
    (remote / "nodea.jsonl").write_text(
        json.dumps(a, sort_keys=True) + "\n" +
        json.dumps({**b, "id": "ffff", "node": "nodea", "utc": "2026-01-01T00:00:00Z"}, sort_keys=True) + "\n")
    evs = ink_spec.read_events()
    assert [e["id"] for e in evs] == ["ffff", a["id"], b["id"]], "deduped by id, oldest first"
    assert len(ink_spec.read_events(include_remote=False)) == 2


def test_tag8_sits_on_the_rack_root():
    off = ink_spec.tag8_in_palette_root()
    assert len(off) == 3 and abs(off[2]) < 0.02 and off[0] == 0 and off[1] == 0


# --- scripts/ink.py coverage -------------------------------------------------------------

import ink  # noqa: E402


@pytest.fixture
def ink_repo(tmp_path, monkeypatch):
    """Isolated REPO fixture for ink.py testing."""
    (tmp_path / "config" / "tools").mkdir(parents=True)
    for rel in (ink_spec.INKS_RELPATH, ink_spec.PALETTE_RELPATH, ink_spec.LOAD_RELPATH, ink_spec.INVENTORY_RELPATH):
        shutil.copy(REPO / rel, tmp_path / rel)
    if (REPO / "config" / "workspace.yaml").exists():
        shutil.copy(REPO / "config" / "workspace.yaml", tmp_path / "config" / "workspace.yaml")
    for f in (REPO / "config" / "tools").glob("*.yaml"):
        shutil.copy(f, tmp_path / "config" / "tools" / f.name)

    monkeypatch.setattr(ink, "REPO", tmp_path)
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("TATBOT_INK_SESSION", str(tmp_path / "session.json"))
    return tmp_path


def test_ink_helpers_and_die(capsys):
    assert ink._die("error msg") == 2
    captured = capsys.readouterr()
    assert "ink: error msg" in captured.err

    # _pairs
    weighs = [
        {"slot": "inkcap_right_large", "when": "before", "grams": 10.0},
        {"slot": "inkcap_right_large", "when": "after", "grams": 9.5},
        {"bottle_id": "b1", "when": "after", "grams": 100.0},  # no before
        {"bottle_id": "b2", "when": "before", "grams": 200.0},
    ]
    pairs = ink._pairs(weighs)
    assert len(pairs) == 1
    assert pairs[0][0]["grams"] == 10.0 and pairs[0][1]["grams"] == 9.5

    # _require_tool
    class DummyArgs:
        tool_id = None
    assert ink._require_tool(DummyArgs(), "cmd") == "cmd needs --ee-tool <id> (or TATBOT_EE_TOOL): name the tool in the gripper"
    DummyArgs.tool_id = "lutin-ballpoint-dot"
    assert ink._require_tool(DummyArgs(), "cmd") is None


def test_ink_cmd_status(ink_repo, capsys):
    assert ink.main(["status"]) == 0
    captured = capsys.readouterr()
    assert "palette (" in captured.out
    assert "bottles:" in captured.out
    assert "cartridges:" in captured.out
    assert "blank caps:" in captured.out
    assert "ledger " in captured.out

    # active tool unknown case
    (ink_repo / "config" / "workspace.yaml").unlink()
    assert ink.main(["status"]) == 0
    captured_no_ws = capsys.readouterr()
    assert "fitted tool: unknown" in captured_no_ws.out


def test_ink_cmd_load_and_dump(ink_repo, capsys):
    # Error cases
    assert ink.main(["load", "bad_slot", "nighthawk_black", "--ul", "100"]) == 2
    assert ink.main(["load", "inkcap_right_large", "bad_ink", "--ul", "100"]) == 2
    assert ink.main(["load", "inkcap_right_large", "nighthawk_black", "--ul", "999999"]) == 2
    assert ink.main(["load", "inkcap_right_large", "nighthawk_black", "--ul", "100", "--bottle", "bad_bottle"]) == 2
    assert ink.main(["load", "inkcap_right_large", "nighthawk_black", "--ul", "100", "--cap-stock", "bad_cap"]) == 2

    # Valid load
    rc = ink.main([
        "load", "inkcap_right_large", "nighthawk_black", "--ul", "500",
        "--bottle", "nighthawk_black_01", "--cap-stock", "emalla_15mm"
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "inkcap_right_large: nighthawk_black 500 uL" in captured.out

    # Conflict load (slot holds different ink)
    assert ink.main(["load", "inkcap_right_large", "true_blue", "--ul", "100"]) == 2

    # Dump
    assert ink.main(["dump", "bad_slot"]) == 2
    assert ink.main(["dump", "inkcap_right_large"]) == 0
    captured_dump = capsys.readouterr()
    assert "inkcap_right_large: dumped 500 uL of nighthawk_black" in captured_dump.out


def test_ink_cmd_bottle(ink_repo, capsys):
    # Error cases
    assert ink.main(["bottle", "add", "b_existing", "--ink", "nighthawk_black", "--ml", "100"]) == 0
    assert ink.main(["bottle", "add", "b_existing", "--ink", "nighthawk_black", "--ml", "100"]) == 2
    assert ink.main(["bottle", "add", "b_bad_ink", "--ink", "unknown_ink", "--ml", "100"]) == 2
    assert ink.main(["bottle", "open", "unknown_bottle"]) == 2

    # Open bottle (with remaining_ml None check)
    inv = ink_spec.load_inventory(ink_repo)
    inv["bottles"]["b_no_rem"] = {"ink": "nighthawk_black", "ml": 50, "purchased": None, "opened": None, "remaining_ml": None, "retired": False}
    ink_spec.write_inventory(inv, ink_repo)
    assert ink.main(["bottle", "open", "b_no_rem"]) == 0
    inv_after = ink_spec.load_inventory(ink_repo)
    assert inv_after["bottles"]["b_no_rem"]["remaining_ml"] == 50

    # Retire bottle
    assert ink.main(["bottle", "retire", "b_existing"]) == 0
    capsys.readouterr()


def test_ink_cmd_cartridge(ink_repo, capsys):
    # Add error cases
    assert ink.main(["cartridge", "add", "c_new", "--spec", "spec", "--needle-code", "1003RL", "--count", "10"]) == 0
    assert ink.main(["cartridge", "add", "c_new", "--spec", "spec", "--needle-code", "1003RL", "--count", "10"]) == 2
    assert ink.main(["cartridge", "add", "c_bad_fit", "--spec", "spec", "--needle-code", "1003RL", "--count", "10", "--fits", "bad_tool"]) == 2

    # Fit, count, retire errors & valid
    assert ink.main(["cartridge", "fit", "bad_c"]) == 2
    assert ink.main(["cartridge", "fit", "c_new", "2"]) == 0
    inv = ink_spec.load_inventory(ink_repo)
    assert inv["cartridges"]["c_new"]["count"] == 8

    assert ink.main(["cartridge", "count", "c_new", "15"]) == 0
    inv = ink_spec.load_inventory(ink_repo)
    assert inv["cartridges"]["c_new"]["count"] == 15

    assert ink.main(["cartridge", "retire", "c_new"]) == 0
    inv = ink_spec.load_inventory(ink_repo)
    assert inv["cartridges"]["c_new"]["retired"] is True
    capsys.readouterr()


def test_ink_cmd_caps(ink_repo, capsys):
    assert ink.main(["caps", "count", "bad_cap", "50"]) == 2
    assert ink.main(["caps", "count", "emalla_15mm", "80"]) == 0
    inv = ink_spec.load_inventory(ink_repo)
    assert inv["caps"]["emalla_15mm"]["count"] == 80
    capsys.readouterr()


def test_ink_cmd_weigh_and_fit(ink_repo, capsys):
    # Weigh errors & valid
    assert ink.main(["weigh", "bad_target", "10.0", "--when", "before"]) == 2
    assert ink.main(["weigh", "inkcap_right_large", "15.0", "--when", "before"]) == 0
    assert ink.main(["weigh", "kuro_sumi_black_01", "300.0", "--when", "after"]) == 0

    # cmd_fit with no pairs / bracketed dips
    assert ink.main(["fit"]) == 2
    captured = capsys.readouterr()
    assert "no before/after weigh pairs" in captured.err

    # Create a weigh pair bracketing dips
    ink_spec.append_event("weigh", "real", slot="inkcap_right_large", grams=15.0, when="before")
    ink_spec.append_event("dip", "real", slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("stroke", "real", contact_mm=100.0, contact_s=5.0, ul=0.5)
    ink_spec.append_event("dip", "real", slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("stroke", "real", contact_mm=200.0, contact_s=10.0, ul=1.0)
    ink_spec.append_event("dip", "real", slot="inkcap_right_large", uptake_ul=1.5)
    ink_spec.append_event("weigh", "real", slot="inkcap_right_large", grams=12.0, when="after")

    assert ink.main(["fit"]) == 0
    captured_fit = capsys.readouterr()
    assert "uptake_ul:" in captured_fit.out
    assert "deposit_ul_per_mm:" in captured_fit.out or "collinear" in captured_fit.out


def test_ink_cmd_ledger_and_reconcile(ink_repo, capsys):
    ink_spec.append_event("cap.fill", "real", slot="inkcap_right_large", ink_id="nighthawk_black", ul=500.0, bottle="nighthawk_black_01")
    assert ink.main(["ledger", "-n", "1"]) == 0
    captured = capsys.readouterr()
    assert "cap.fill" in captured.out

    # Reconcile without write
    assert ink.main(["reconcile"]) == 0
    captured_rec = capsys.readouterr()
    assert "caps (palette_load vs ledger replay):" in captured_rec.out

    # Reconcile with write when drift exists
    load = ink_spec.load_palette_load(ink_repo, ink_spec.load_palette(ink_repo))
    load["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "nighthawk_black", 100.0)
    ink_spec.write_palette_load(load, ink_repo)
    assert ink.main(["reconcile", "--write"]) == 0
    captured_write = capsys.readouterr()
    assert "wrote config/palette_load.yaml" in captured_write.out


def test_ink_cmd_mise(ink_repo, capsys, tmp_path):
    # Argument validation
    assert ink.main(["mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--strokes-mm", "100"]) == 2
    assert ink.main(["mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--program", "dummy.json"]) == 2

    # Valid need spec
    assert ink.main([
        "mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--need", "nighthawk_black=300"
    ]) == 0
    captured = capsys.readouterr()
    assert "mise en place — lutin-ballpoint-dot" in captured.out

    prog_file = tmp_path / "prog.json"
    prog_file.write_text(json.dumps([[[0, 0], [0.01, 0.01]]]))
    assert ink.main([
        "mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--ink", "nighthawk_black",
        "--program", str(prog_file)
    ]) == 0
    captured_prog = capsys.readouterr()
    assert "mise en place — lutin-ballpoint-dot" in captured_prog.out

    # Empty program
    empty_prog = tmp_path / "empty_prog.json"
    empty_prog.write_text(json.dumps({}))
    assert ink.main([
        "mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--ink", "nighthawk_black",
        "--program", str(empty_prog)
    ]) == 2


@pytest.mark.xfail(reason="Bug in scripts/ink.py cmd_mise: 'if args.strokes_mm and not needs' triggers before 'if args.ink and args.strokes_mm' populates needs", strict=True)
def test_ink_cmd_mise_strokes_mm_with_ink_bug(ink_repo):
    # This combination should work according to ink.py docstring/argparse, but bugs out due to order of checks
    rc = ink.main([
        "mise-en-place", "--ee-tool", "lutin-ballpoint-dot", "--ink", "nighthawk_black",
        "--strokes-mm", "100"
    ])
    assert rc == 0


def test_ink_cmd_plan(ink_repo, capsys):
    assert ink.main(["plan", "--ee-tool", "lutin-ballpoint-dot", "--strokes", "100,5", "200,10,nighthawk_black"]) == 0
    captured = capsys.readouterr()
    assert "lutin-ballpoint-dot (rehearsal):" in captured.out

    # Supply error on real tool with dry caps
    assert ink.main(["plan", "--ee-tool", "lutin-3rl-bugpin", "--strokes", "100,5"]) == 2


def test_ink_cmd_session(ink_repo, capsys, tmp_path):
    assert ink.main(["session", "status"]) == 0
    with pytest.raises(SystemExit) as exc:
        ink.main(["session", "invalid_verb"])
    assert exc.value.code == 2

    # start requiring tool
    assert ink.main(["session", "start"]) == 2

    # start session with program
    prog_file = tmp_path / "prog.json"
    prog_file.write_text(json.dumps([[[0, 0], [0.01, 0.01]]]))
    assert ink.main(["session", "start", "--ee-tool", "lutin-ballpoint-dot", "--program", str(prog_file)]) == 0
    assert ink.main(["session", "status"]) == 0

    # end session
    assert ink.main(["session", "end"]) == 0
    assert ink.main(["session", "end"]) == 2  # no open session

    # rebuild session
    assert ink.main(["session", "rebuild"]) == 2  # missing id
    evs = ink_spec.read_events()
    sess_id = next(e["session_id"] for e in evs if e.get("kind") == "session.start")
    assert ink.main(["session", "rebuild", sess_id, "--write"]) == 0
    assert ink.main(["session", "rebuild", "non_existent_id"]) == 2


def test_ink_cmd_sync(ink_repo, capsys, monkeypatch):
    import subprocess
    # Mock subprocess.run for scp
    def mock_run(cmd, check=False, capture_output=True, text=True):
        class Res:
            returncode = 0 if "fail_host" not in cmd[6] else 1
            stderr = "connection refused" if "fail_host" in cmd[6] else ""
        return Res()

    monkeypatch.setattr(subprocess, "run", mock_run)

    assert ink.main(["sync", "node1", "fail_host"]) == 1
    captured = capsys.readouterr()
    assert "fail_host: connection refused" in captured.err


def test_ink_arg_parsing_validation():
    assert ink.main(["bottle", "add", "b1"]) == 2  # missing --ink and --ml
    assert ink.main(["cartridge", "add", "c1"]) == 2  # missing args
