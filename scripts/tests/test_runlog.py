#!/usr/bin/env python3
"""Pin the run-log layout and banner format to what the docs promise.

    uvx --with pytest pytest -q scripts/tests/test_runlog.py

AGENTS.md tells agents to `grep tatbot-run` and describes the run directory;
docs/run_logs.md literalincludes the LAYOUT block. Those are claims made TO
agents, so they are the ones worth a test — if the writer drifts from them, an
agent follows an instruction that no longer works and falls back to asking a
human for terminal output, which is the whole failure this system exists to
end.

Runs from scripts/githooks/pre-commit when the log system is touched.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import tatbot_runlog as rl  # noqa: E402

SRC = Path(rl.__file__).read_text()


def _layout_block() -> str:
    start = SRC.index("# BEGIN LAYOUT")
    end = SRC.index("# END LAYOUT")
    return SRC[start:end]


def test_layout_markers_exist():
    """docs/run_logs.md literalincludes between these; losing them empties the doc."""
    assert "# BEGIN LAYOUT" in SRC
    assert "# END LAYOUT" in SRC
    assert len(_layout_block().splitlines()) > 5


def test_run_dir_matches_documented_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    run = rl.init("selftest", prune_first=False, emit_banner=False)
    (run.dir / "flight-test.csv").write_text("t_mono\n0.0\n")
    run.artifact(run.path("flight-test.csv"))
    run.finalize(0)

    produced = {p.name for p in run.dir.iterdir()}
    documented = set(re.findall(r"^#   (\S+?)\s{2,}", _layout_block(), re.M))
    # Everything the run actually wrote must be named in the doc block (the
    # block also names optional files, which is fine).
    undocumented = {n for n in produced if not n.startswith("flight-")} - documented
    assert not undocumented, f"run dir has undocumented entries: {undocumented}"

    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "ok"
    assert meta["exit_code"] == 0
    assert meta["schema_version"] == rl.SCHEMA_VERSION
    assert meta["run_id"] == run.run_id


def test_run_id_is_self_locating():
    """AGENTS.md promises the node can be read out of a run id."""
    run_id = rl._mint_run_id("nodea")
    assert rl.RUN_ID_RE.match(run_id), run_id
    assert rl._node_of(run_id) == "nodea"
    # Sorting run ids must sort them by time; that is why they are UTC.
    ids = sorted([rl._mint_run_id("nodea"), "20200101T000000Z-nodea-0000"])
    assert ids[0] == "20200101T000000Z-nodea-0000"


def test_banner_format_matches_agents_md():
    """AGENTS.md tells agents to grep this token; keep it greppable."""
    line = rl.banner("start", "20260821T164233Z-nodea-a3f1", "pid=1 log=/tmp/x")
    m = rl.BANNER_RE.match(line)
    assert m, line
    assert m.group("run_id") == "20260821T164233Z-nodea-a3f1"
    assert rl.BANNER_TOKEN in line

    agents = Path(__file__).resolve().parents[2] / "AGENTS.md"
    if agents.is_file():
        assert rl.BANNER_TOKEN in agents.read_text(), (
            "AGENTS.md no longer mentions the banner token agents are told to grep")


def test_prune_refuses_everything_it_does_not_recognise(tmp_path):
    """The gate in front of ~300 GB of camera evidence. Refuse by default."""
    cfg = rl.load_config()
    cfg["log_root"] = str(tmp_path)
    wf = tmp_path / "vision"
    wf.mkdir(parents=True)

    def mk(name, meta=None, keep=False, write_meta=True):
        d = wf / name
        d.mkdir(parents=True, exist_ok=True)
        if write_meta:
            (d / "meta.json").write_text(json.dumps(
                meta or {"status": "ok", "ended_at": "2020-01-01T00:00:00.000Z"}))
        if keep:
            (d / "KEEP").touch()
        return d

    assert rl.why_not_deletable(mk("20200101T000000Z-nodeb-0001"), tmp_path, "vision", cfg) is None
    assert rl.why_not_deletable(mk("20200101T000000Z-nodeb-0002", keep=True), tmp_path, "vision", cfg) == "KEEP marker"
    assert rl.why_not_deletable(mk("20200101T000000Z-nodeb-0003", write_meta=False), tmp_path, "vision", cfg) == "meta.json unreadable"
    # Pre-runlog evidence is named like this.
    assert rl.why_not_deletable(mk("session-20260818_151612-poe"), tmp_path, "vision", cfg) == "name is not a run id"
    alive = mk("20200101T000000Z-nodeb-0004", {"status": "running", "started_at": "2020-01-01T00:00:00.000Z",
                                             "node": {"hostname": rl._node(), "pid": os.getpid()}})
    assert rl.why_not_deletable(alive, tmp_path, "vision", cfg) == "still running here"
    outside = tmp_path / "elsewhere" / "20200101T000000Z-nodeb-0005"
    outside.mkdir(parents=True)
    (outside / "meta.json").write_text("{}")
    assert rl.why_not_deletable(outside, tmp_path, "vision", cfg) == "outside the workflow directory"


def test_legacy_workflow_is_disabled_by_default():
    cfg = rl.load_config()
    assert rl.retention_for("legacy", cfg).get("enabled") is False


def test_attach_returns_none_without_a_run(monkeypatch):
    """Absence of a run dir is normal — bench runs must keep working."""
    monkeypatch.delenv("TATBOT_RUN_DIR", raising=False)
    rl._CURRENT = None
    assert rl.attach() is None


def test_native_estop_lines_populate_run_counter(tmp_path, monkeypatch):
    """C++ cannot emit structured Python events, so preserve its evidence."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    run = rl.init("teleop", prune_first=False, emit_banner=False)
    (run.dir / "console.log").write_text(
        "starting\nE-STOP: pressed -- holding both arms\n"
        "E-STOP: heartbeat fault -- holding both arms\n"
    )
    run.finalize(0)

    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["counters"]["estop"] == 2


def test_finalize_is_idempotent_across_process_instances(tmp_path, monkeypatch):
    """A disconnect watchdog must not overwrite normal shell finalization."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    run = rl.init("vision", prune_first=False, emit_banner=False)
    run.finalize(130)

    second = rl.RunLog(run.dir, "vision", run.run_id)
    second.finalize(0)

    meta = json.loads((run.dir / "meta.json").read_text())
    events = [json.loads(line) for line in (run.dir / "run.jsonl").read_text().splitlines()]
    assert meta["status"] == "interrupted"
    assert meta["exit_code"] == 130
    assert sum(event["kind"] == "run.end" for event in events) == 1
