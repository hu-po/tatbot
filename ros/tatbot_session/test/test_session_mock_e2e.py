"""End to end on ros2_control mock hardware: compile a square -> Draw -> every op done in the ledger ->
MCAP bag -> drawn.svg; unsupported legacy pauses refuse; a crash-left `sent` op waits for SKIP.
Mock hardware never latches, so the e-stop, touch and land paths run in test_session_fake_e2e."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import session_graph as sg

pytestmark = pytest.mark.skipif(not sg.ZENOHD.exists(), reason="needs ROS 2 Jazzy with rmw_zenoh_cpp")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    g = sg.Graph(tmp_path_factory.mktemp("mock_e2e"))
    stack = sg.base_stack(hardware="mock")
    stack["page"]["fixed"]["right"]["xyz"][2] = 0.040   # synthetic raised skin, reachable looking straight down
    stack["touch"]["enabled"] = True
    stack["sensing"] = {"wrist": "real"}   # fixture_session supplies the deterministic sensor port
    stack["page"].update(carry_max_s=0.0, track={"mode": "off"})
    stack["page"]["locate"].update(wrist=True, poses=1, frames=1, settle_s=0.0, holds=False)
    stack["page"]["surface"].update(frames=5, pool=1)
    try:
        yield g.up(sg.mock_urdf(), stack, motion=sg.fast_motion(mock=True), sensing=True)
    finally:
        g.stop()


def test_status_and_idle_decide(graph):
    right = sg.last_json(graph.client("status", "--json").stdout)["safety"]["right"]
    assert right["estop_ok"] is True and right["latched"] is False and right["estop_source"] == 0
    # the graph's sim switch answers at once: off, with a fresh reply (a stack with no switch reports age -1)
    assert right["machine_off"] is True and right["machine_powered"] is False
    assert 0.0 <= right["machine_reply_age_s"] < 1.0
    decide = sg.last_json(graph.client("decide", "right", "continue").stdout)
    assert decide["accepted"] is False and "nothing waits" in decide["reason"]


def test_draw_square_to_completion(graph):
    program = sg.square_program(graph.tmp, name="square")
    document = json.loads(program.read_text())
    document['research'] = {'study_sha256': 'a'*64, 'trial': '0000', 'side': 'a', 'page': 'synthetic-mock-sheet',
                            'slot': '0:0', 'slot_m': [.026, .026]}
    program.write_text(json.dumps(document))
    proc = graph.client("draw", str(program), "--arm", "right", timeout=150)
    result = sg.last_json(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ops = json.loads(program.read_text())["ops"]
    assert result["done"] == len(ops) and result["uncertain"] == []
    rows = sg.ledger(result["run_dir"])
    progress = [row for row in rows if row['event'] in ('sent', 'done')]
    assert progress[0]['event'] == 'done' and progress[0]['op'] == 't0000'
    assert [row['event'] for row in progress[1:]] == ['sent', 'done']*(len(ops)-1)
    assert sum(row['event'] == 'page' for row in rows) == 1
    run = Path(result["run_dir"])
    assert list((run / "bag").glob("*.mcap")) and (run / "bag" / "metadata.yaml").is_file()
    info = graph.run(["ros2", "bag", "info", str(run / "bag")], timeout=30).stdout
    for topic in ("/joint_states", "/tatbot/events", "/tatbot/goals", "/right_arm_controller/controller_state"):
        assert topic in info, info
    svg = (run / "drawn.svg").read_text()
    assert 'id="drawn-s0001-0"' in svg and 'id="plan-s0001"' in svg
    meta = json.loads((run / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["hardware"] == "mock" and meta["arms"]["right"]["tool"]
    page = json.loads((run / "page.json").read_text())
    assert page["pattern_id"] == "fixed_right"
    # the session camera alone (no further camera here) reads it from over the page centre at 50 and 100 mm
    assert page["locate"]["wrist"]["poses"] == 2 and page["locate"]["fused"]
    assert [view.get("over_m") for view in page["locate"]["views"]] == [0.05, 0.1]
    assert all(view["fit"] for view in page["locate"]["views"])   # no top-down view after them
    holds = [r for r in page["gauge"] if not r.get("check")]
    assert len(holds) >= 5 and all(t["method"] == "wrist gauge" for t in page["touches"])
    assert page["surface"]["check_m"] and max(map(abs, page["surface"]["check_m"])) <= 0.0009
    import numpy as np

    loo = np.asarray(page["surface"]["report"]["loo_m"], float)
    assert np.isfinite(loo).sum() >= 3 and max(abs(loo[np.isfinite(loo)])) <= 0.0009
    receipt = graph.evidence('synthetic-mock-sheet', '0:0')
    assert receipt.returncode == 0, receipt.stderr
    evidence = json.loads(receipt.stdout)
    assert evidence['runtime_live'] and evidence['meta']['runtime'] == evidence['runtime']
    native = evidence['runtime']['controller']
    assert native['complete'], native
    libraries = {Path(item['path']).name for item in native['files']}
    assert {'ros2_control_node', 'libjoint_trajectory_controller.so', 'libmock_components.so'} <= libraries
    assert evidence['claim']['run_id'] == result['run_id'] and evidence['ledger'] == rows
    assert evidence['timing']['complete'] and evidence['timing']['duration_s'] > 0
    assert len(evidence['timing']['attempts']) == 1 and meta['draw_timing'] == evidence['timing']
    duplicate = graph.client('draw', str(program), '--arm', 'right', timeout=30)
    assert duplicate.returncode != 0 and 'occupied' in duplicate.stdout
    assert sg.ledger(result['run_dir']) == rows  # refusal appended no robot operations
    graph.last_run = result["run_id"]


def test_legacy_pause_refuses_without_robot_operations(graph):
    program = sg.square_program(graph.tmp, name='pause', sides=(.010, .006), pause=True)
    proc = graph.client('draw', str(program))
    assert proc.returncode != 0 and 'only tool_change, dip and stroke' in proc.stdout


@pytest.mark.slow
def test_resume_crash_left_sent_waits_for_skip(graph):
    # the cancel lands in the first square (26 s); the resume skips it and draws the second (10 s) inside its minute
    program = sg.square_program(graph.tmp, name='crash', sides=(.006, .002), speed_m_s=.001)
    original = graph.client_bg('draw', str(program), log='crash-client.log')
    try:
        assert graph.wait_for('crash-client.log', ' draw', 90)
        assert graph.client('cancel').returncode == 0
        original.wait(timeout=60)
    finally:
        graph.end(original)
    result = sg.last_json((graph.tmp/'crash-client.log').read_text())
    run_id, run_dir = result['run_id'], Path(result['run_dir'])
    op_id, last = (op['id'] for op in json.loads(program.read_text())['ops'] if op['op'] == 'stroke')
    with (run_dir / "ledger.jsonl").open("a") as stream:  # a crash right after `sent`
        stream.write(json.dumps({"t": time.time(), "event": "sent", "arm": "right", "op": op_id, "index": 1,
                                 "arc_m": 0.0}) + "\n")
    proc = graph.client_bg("draw", "", "--run-id", run_id, log="resume-client.log")
    try:
        assert graph.wait_for("resume-client.log", " uncertain "+op_id, 60)
        assert sg.last_json(graph.client("decide", "right", "skip").stdout)["accepted"] is True
        assert proc.wait(timeout=60) == 0
    finally:
        graph.end(proc)
    rows = sg.ledger(run_dir)
    progress = [row for row in rows if row.get('op') == op_id and row['event'] in ('skipped', 'decision')]
    assert progress[-1]['event'] == 'skipped' and progress[-2]['decision'] == 'skip'
    assert [row['event'] for row in rows if row.get('op') == last][-1] == 'done'
    assert (run_dir / "bag-2").is_dir()
    meta = json.loads((run_dir/'meta.json').read_text())
    measured = meta['draw_timing']
    assert measured['complete'] and len(measured['attempts']) == 2
    assert measured['duration_s'] == sum(a['duration_s'] for a in measured['attempts'])
    assert measured['duration_s'] > meta['duration_s']
