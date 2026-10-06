"""The stroke ledger: resume, aborted arc, crash-left `sent`, decisions."""
import json

from tatbot_session import ledger

OPS = [{"id": f"s{i:04d}", "op": "stroke"} for i in range(4)]


def test_events_are_fixed():
    assert ledger.EVENTS == ("sent", "done", "aborted", "skipped", "decision", "page", "dip_phase", "pen_trim")


def test_page_and_dip_rows_never_stand_for_an_ops_progress(tmp_path):
    book = ledger.Ledger(tmp_path / "ledger.jsonl")
    book.append("sent", "right", op="s0000", arc_m=0)
    book.append("page", "right", page={}, workspace={})
    book.append("dip_phase", "right", op="s0000", phase="dwell", status="ok")
    assert ledger.uncertain(OPS, book.rows(), "right") == ["s0000"] and ledger.next_index(OPS, book.rows(), "right") == 0


def test_fresh_run_starts_at_zero(tmp_path):
    book = ledger.Ledger(tmp_path / "ledger.jsonl")
    assert ledger.next_index(OPS, book.rows(), "right") == 0
    assert ledger.uncertain(OPS, book.rows(), "right") == []


def test_resume_skips_done_and_resumes_aborted_at_its_arc(tmp_path):
    book = ledger.Ledger(tmp_path / "ledger.jsonl")
    book.append("sent", "right", op="s0000", index=0, arc_m=0.0)
    book.append("done", "right", op="s0000", index=0, arc_m=0.04)
    book.append("sent", "right", op="s0001", index=1, arc_m=0.0)
    book.append("aborted", "right", op="s0001", index=1, arc_m=0.0133, reason="latched: estop")
    rows = book.rows()
    assert ledger.next_index(OPS, rows, "right") == 1
    assert ledger.resume_arc(rows, "right", "s0001") == 0.0133
    assert ledger.resume_arc(rows, "right", "s0002") == 0.0
    assert ledger.uncertain(OPS, rows, "right") == []
    # the resumed attempt is sent from its arc, then done
    book.append("sent", "right", op="s0001", index=1, arc_m=0.0133)
    book.append("done", "right", op="s0001", index=1, arc_m=0.05)
    assert ledger.next_index(OPS, book.rows(), "right") == 2
    assert ledger.resume_arc(book.rows(), "right", "s0001") == 0.0


def test_crash_left_sent_is_uncertain_until_redraw_or_skip(tmp_path):
    book = ledger.Ledger(tmp_path / "ledger.jsonl")
    book.append("sent", "right", op="s0000", index=0, arc_m=0.0)
    book.append("done", "right", op="s0000", index=0, arc_m=0.04)
    book.append("sent", "right", op="s0001", index=1, arc_m=0.002)
    rows = book.rows()
    assert ledger.uncertain(OPS, rows, "right") == ["s0001"]
    assert ledger.next_index(OPS, rows, "right") == 1
    assert ledger.op_status(rows, "right", "s0001") == "sent"
    # REDRAW: an aborted row at arc 0 clears the uncertainty and restarts the op from its start
    book.append("decision", "right", op="s0001", index=1, decision="redraw")
    book.append("aborted", "right", op="s0001", index=1, arc_m=0.0, reason="redraw")
    rows = book.rows()
    assert ledger.uncertain(OPS, rows, "right") == []
    assert ledger.resume_arc(rows, "right", "s0001") == 0.0
    # SKIP moves past the op
    book.append("sent", "right", op="s0001", index=1, arc_m=0.0)
    book.append("decision", "right", op="s0001", index=1, decision="skip")
    book.append("skipped", "right", op="s0001", index=1)
    rows = book.rows()
    assert ledger.next_index(OPS, rows, "right") == 2
    assert ledger.counts(rows) == {"done": 1, "skipped": 1}


def test_from_op_and_arms_are_independent(tmp_path):
    book = ledger.Ledger(tmp_path / "ledger.jsonl")
    book.append("done", "left", op="s0000", index=0, arc_m=0.04)
    rows = book.rows()
    assert ledger.next_index(OPS, rows, "right") == 0
    assert ledger.next_index(OPS, rows, "left") == 1
    assert ledger.next_index(OPS, rows, "right", start=2) == 2
    for op in OPS:
        book.append("done", "right", op=op["id"], arc_m=0.0)
    assert ledger.next_index(OPS, book.rows(), "right") is None


def test_lines_are_json_and_a_torn_tail_is_dropped(tmp_path):
    path = tmp_path / "ledger.jsonl"
    book = ledger.Ledger(path)
    row = book.append("aborted", "right", op="s0002", index=2, arc_m=0.0123456789, reason="cancelled")
    assert row["arc_m"] == 0.012346 and row["reason"] == "cancelled"
    with path.open("a") as stream:
        stream.write('{"t": 1, "event": "do')
    rows = ledger.read(path)
    assert len(rows) == 1 and json.loads(path.read_text().splitlines()[0])["event"] == "aborted"


def test_a_row_after_a_torn_line_keeps_its_own_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.Ledger(path).append("sent", "right", op="s0000", index=0, arc_m=0.0)
    with path.open("a") as stream:
        stream.write('{"t": 1, "event": "done", "arm": "ri')  # the crash
    resumed = ledger.Ledger(path)
    resumed.append("decision", "right", op="s0000", index=0, decision="redraw")
    resumed.append("sent", "right", op="s0000", index=0, arc_m=0.0)
    assert [r["event"] for r in ledger.read(path)] == ["sent", "decision", "sent"]
