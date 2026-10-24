"""The ROS Rerun bridge's status (tatbot_contracts.ros_status): each item's text and state from the bridge's samples,
the board's layout on its fixed canvas and the markdown card."""
from tatbot_contracts import ros_status as status

SAFETY = {"t": 99.5, "estop_source": 2, "estop_ok": True, "estop_age_s": 0.004, "probe_triggered": False,
          "latched": False, "latch": "none", "machine_powered": True, "machine_reply_age_s": 0.03}
RUN = {"id": "20261018T201223Z-node-4704", "t": 97.0, "index": 36, "total": 112, "done": 36, "started_s": 40.0,
       "ended_s": None, "status": "running"}
FEEDBACK = {"t": 99.9, "index": 36, "total": 112, "phase": "DRAW", "arc_m": 0.0123}


def states(items: dict) -> dict:
    return {key: state for key, (_, state) in items.items()}


def test_nothing_known_reads_unknown_or_off():
    items = status.items(100.0)
    assert list(items) == ["run", "op", "phase", "trim", "ink", "estop", "probe", "machine", "latch", "wrist"]
    assert states(items) == {"run": "off", "op": "off", "phase": "off", "trim": "off", "ink": "off", "estop": "warn",
                             "probe": "off", "machine": "off", "latch": "warn", "wrist": "off"}
    assert items["estop"] == ("e-stop: no state", "warn") and items["probe"] == ("probe disabled", "off")
    assert items["wrist"] == ("wrist: no frame yet (opens in a draw)", "off")


def test_a_live_draw_is_green_and_its_ages_stay_off_the_board():
    items = status.items(100.0, run=RUN, feedback=FEEDBACK, safety=SAFETY, wrist={"t": 99.2, "latest": False},
                         ink={"ink": "nighthawk_black", "slot": "inkcap_large_2"}, palette_s=-80.0,
                         trim={"m": -0.0015, "limit_m": 0.004}, probe_enabled=True, machine_relay=True)
    assert set(states(items).values()) == {"ok"}
    assert [text for text, _ in items.values()] == [
        "run 20:12Z · 60 s", "op 37/112", "phase DRAW · arc 12.3 mm", "trim -1.5 of ±4.0 mm",
        "ink nighthawk_black @ large_2 · palette fix 3 min", "e-stop OK", "probe clear", "machine on", "latch clear",
        "wrist live"]
    later = status.items(100.4, run=RUN, feedback={**FEEDBACK, "t": 100.3}, safety={**SAFETY, "t": 100.35})
    assert later["estop"] == items["estop"] and later["op"] == items["op"]     # unchanged: nothing new to log


def test_faults_and_warnings():
    latched = {**SAFETY, "estop_ok": False, "latched": True, "latch": "estop", "probe_triggered": True,
               "machine_reply_age_s": -1.0}
    items = status.items(100.0, run=RUN, feedback={**FEEDBACK, "phase": "LATCHED"}, safety=latched,
                         probe_enabled=True, machine_relay=True)
    assert items["estop"] == ("e-stop STOP", "fault") and items["latch"] == ("LATCHED: estop", "fault")
    assert items["phase"] == ("phase LATCHED", "fault") and items["probe"] == ("probe triggered", "warn")
    assert items["machine"] == ("machine: no reply", "warn") and items["trim"] == ("trim -", "off")
    silent = status.items(130.0, safety=SAFETY, probe_enabled=True, machine_relay=True)   # a SafetyState 30 s old
    assert silent["estop"] == ("e-stop OK · 30 s old", "warn") and silent["latch"] == ("latch clear · 30 s old", "warn")
    assert silent["machine"] == ("machine on · 31 s old", "warn") and silent["probe"] == ("probe · 30 s old", "warn")
    off = status.items(100.0, safety={**SAFETY, "machine_powered": False, "estop_source": 0}, machine_relay=True)
    assert off["machine"] == ("machine off", "off") and off["estop"] == ("e-stop none", "off")
    assert status.items(100.0, safety={**SAFETY, "machine_reply_age_s": None}, machine_relay=True)["machine"] == (
        "machine: no state", "warn")


def test_between_goals_the_ledger_speaks():
    stale = status.items(100.0, run=RUN, feedback={**FEEDBACK, "t": 90.0}, ink={"ink": "black", "slot": None})
    assert stale["op"] == ("op 37/112 · 10 s ago", "warn") and stale["phase"] == ("phase DRAW · 10 s ago", "warn")
    assert stale["ink"] == ("ink black · no palette fix", "warn") and stale["run"] == ("run 20:12Z · 60 s", "ok")
    ledger = status.items(100.0, run=RUN)
    assert ledger["op"] == ("op 37/112 (ledger)", "warn") and ledger["phase"] == ("phase: between goals", "warn")
    quiet = status.items(400.0, run=RUN, wrist={"t": 200.0, "latest": False})
    assert quiet["run"] == ("run 20:12Z · idle 5 min", "warn") and quiet["wrist"] == ("wrist 3 min ago", "off")
    drawing = status.items(400.0, run=RUN, feedback={**FEEDBACK, "t": 399.0}, wrist={"t": 390.0, "latest": True})
    assert drawing["wrist"] == ("wrist 10 s ago", "warn")
    assert status.items(400.0, wrist={"t": 399.0, "latest": True})["wrist"] == ("wrist live (latest tf)", "warn")


def test_an_ended_run_says_done_and_when():
    ended = {**RUN, "index": 7, "done": 7, "total": 115, "status": "interrupted", "ended_s": 100.0 - 3 * 3600}
    items = status.items(100.0, run=ended, ink={"ink": "black", "slot": "inkcap_large_2"})
    assert items["run"] == ("run 20:12Z interrupted 3.0 h ago", "off") and items["op"] == ("done 7/115", "off")
    assert items["phase"] == ("phase -", "off") and items["ink"] == ("ink -", "off")
    finished = status.items(100.0, run={**RUN, "index": None, "done": 112, "status": "ok", "ended_s": 40.0})
    assert finished["run"] == ("run 20:12Z ended 60 s ago", "off") and finished["op"] == ("done 112/112", "ok")
    # a bag replayed on its own clock: the run's end is still ahead of it
    replay = status.items(30.0, run={**RUN, "t": 29.0, "started_s": 10.0, "status": "ok", "ended_s": 40.0})
    assert replay["run"] == ("run 20:12Z · 20 s", "ok")
    assert status.short_id("not-a-run-id") == "not-a-run-id"


def test_the_trim_cell_shows_the_cartridges_trim_and_its_limit_and_turns_yellow_at_it():
    """The operator's Fn+W/Fn+S trim (motion.yaml pen.trim): its value of its limit, yellow at the limit, grey with no
    run going. Each text keeps to two lines of a 150 px cell (~13 characters a line at the station's 17 px em), as
    the phase cell's with the arc does."""
    def trim(m, limit=0.004, run=RUN):
        return status.items(100.0, run=run, trim={"m": m, "limit_m": limit})["trim"]

    assert trim(0.0) == ("trim +0.0 of ±4.0 mm", "ok") and trim(0.0021) == ("trim +2.1 of ±4.0 mm", "ok")
    assert trim(-0.004) == ("trim -4.0 mm: its limit", "warn") and trim(0.004)[1] == "warn"
    assert trim(-0.0015, limit=None) == ("trim -1.5 mm", "ok")                 # the run's motion.yaml not read yet
    assert trim(-0.0015, run={**RUN, "status": "ok", "ended_s": 90.0}) == ("trim -", "off")
    long = status.items(100.0, run=RUN, feedback={**FEEDBACK, "arc_m": 0.1234}, trim={"m": -0.0039, "limit_m": 0.004})
    assert long["phase"][0] == "phase DRAW · arc 123.4 mm"
    assert all(len(text) <= 26 for text in (long["phase"][0], long["trim"][0], trim(-0.004)[0]))
    held = status.items(100.0, run=RUN, trim={"m": -0.004, "limit_m": 0.004, "used_m": -0.00265})["trim"]
    assert held == ("trim -4.0 · held -2.65 mm", "warn") and len(held[0]) <= 26   # a press the skin read holds back
    assert status.items(100.0, run=RUN, trim={"m": -0.002, "limit_m": 0.004, "used_m": -0.002})["trim"][1] == "ok"


def test_board_cells_tile_the_canvas_and_the_card_names_states():
    items = status.items(100.0, run=RUN, feedback=FEEDBACK, safety={**SAFETY, "latched": True, "latch": "guard"})
    boxes = {key: status.box(key, item) for key, item in items.items()}
    for row in status.ROWS:
        assert sum(width for _, width in row) == status.CANVAS[0]
        right = 0.0
        for key, width in row:
            (x, y), (hx, hy) = boxes[key]["centers"][0], boxes[key]["half_sizes"][0]
            assert right < x - hx and x + hx < right + width and y - hy > 0 and y + hy < status.CANVAS[1]
            right += width
    assert boxes["latch"]["colors"] == [status.RGB["fault"]] == [(0xE7, 0x4C, 0x3C)]
    assert boxes["op"]["colors"] == [(0x2E, 0xCC, 0x71)] and boxes["op"]["labels"] == ["op 37/112"]
    assert boxes["probe"]["colors"] == [(0x7F, 0x8C, 0x8D)] and boxes["estop"]["show_labels"] is True
    card = status.card(items, RUN["id"])
    assert card.startswith("**run** 20261018T201223Z-node-4704\n\n") and "LATCHED: guard [fault]" in card
    assert "op 37/112 · phase DRAW" in card and "probe disabled [off]" in card


def test_outside_an_ink_op_the_ink_cell_lists_the_loaded_caps():
    caps = [{"state": "loaded", "brief": "L1 black ≥85%", "alias": "L1", "fill": 0.85},
            {"state": "empty", "brief": "M1 empty", "alias": "M1", "fill": None},
            {"state": "absent", "brief": "S1 no cap", "alias": "S1", "fill": None},
            {"state": "loaded", "brief": "L2 water ≥44%", "alias": "L2", "fill": 0.44}]
    assert status.items(100.0, caps=caps)["ink"] == ("loaded L1 black ≥85% · L2 water ≥44%", "ok")
    assert status.items(100.0, caps=caps[1:3])["ink"] == ("no ink cap loaded", "off")
    assert status.items(100.0)["ink"] == ("ink -", "off")                       # no palette yet
    ended = {**RUN, "status": "ok", "ended_s": 90.0}
    assert status.items(100.0, run=ended, ink={"ink": "black", "slot": None}, caps=caps)["ink"][0].startswith("loaded")
    drawing = status.items(100.0, run=RUN, ink={"ink": "black", "slot": "inkcap_large_1"}, caps=caps, palette_s=40.0)
    assert drawing["ink"] == ("ink black @ large_1 · palette fix 60 s", "ok")   # an ink op names its own


def test_the_ink_cell_keeps_to_two_lines_and_the_scene_says_the_rest():
    """A third line of the ink cell runs into the wrist bar under it: more caps say less of each."""
    def cap(alias, word, fill):
        return {"state": "loaded", "brief": f"{alias} {word} ≥{fill:.0%}", "alias": alias, "fill": fill}

    three = [cap("L1", "black", 0.85), cap("S2", "red", 0.42), cap("L2", "water", 0.44)]
    assert status.items(100.0, caps=three)["ink"][0] == "loaded L1 ≥85% · S2 ≥42% · L2 ≥44%"
    six = [cap(a, "black", 0.85) for a in ("L1", "M1", "S1", "S2", "M2", "L2")]
    assert status.items(100.0, caps=six)["ink"][0] == "loaded L1 M1 S1 S2 M2 L2"
    assert all(len(status.items(100.0, caps=c)["ink"][0]) <= status.INK_CELL_CHARS for c in (three, six, six[:2]))


def test_age():
    assert [status.age(s) for s in (None, 0.0042, 12.4, 600, 3 * 3600)] == ["", "4 ms", "12 s", "10 min", "3.0 h"]
