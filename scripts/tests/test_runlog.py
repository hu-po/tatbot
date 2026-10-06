#!/usr/bin/env python3
"""Pin the run-log layout and banner format to what the docs promise.

    uvx --with pytest pytest -q scripts/tests/test_runlog.py

AGENTS.md tells agents to `grep tatbot-run` and describes the run directory,
and the LAYOUT block in tatbot_runlog.py names every file a run writes. Those
are claims made TO agents, so they are the ones worth a test — if the writer drifts from them, an
agent follows an instruction that no longer works and falls back to asking a
human for terminal output, which is the whole failure this system exists to
end.

Runs from scripts/githooks/pre-commit when the log system is touched.
"""

import json
import os
import re
from pathlib import Path

import tatbot_runlog as rl  # noqa: E402

SRC = Path(rl.__file__).read_text()


def test_archive_release_uses_its_checked_in_log_policy(tmp_path, monkeypatch):
    release = tmp_path / "release" / "source"
    lib = release / "scripts" / "lib" / "tatbot_runlog.py"
    lib.parent.mkdir(parents=True)
    lib.write_text("# archived logger\n")
    (release / "scripts" / "tatbot").write_text("# archived CLI\n")
    (release / "AGENTS.md").write_text("# Repository Guidelines\n")
    (release / "config").mkdir()
    policy = {"log_root": str(tmp_path / "fleet-logs"),
              "workflows": {"fleet-service": {"keep_runs": 17}}}
    (release / "config" / "runlog.json").write_text(json.dumps(policy))
    monkeypatch.setattr(rl, "__file__", str(lib))
    monkeypatch.setenv("HOME", str(tmp_path / "account"))
    assert rl._repo_root() == release
    cfg = rl.load_config()
    assert cfg["log_root"] == policy["log_root"]
    assert cfg["workflows"]["fleet-service"]["keep_runs"] == 17
    # A directory with a coincidental AGENTS/config pair is not a source tree.
    (release / "scripts" / "tatbot").unlink()
    assert rl._repo_root() is None


def _layout_block() -> str:
    start = SRC.index("# BEGIN LAYOUT")
    end = SRC.index("# END LAYOUT")
    return SRC[start:end]


def test_layout_markers_exist():
    """test_run_dir_matches_documented_layout reads between these; losing them blinds it."""
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


def test_node_maps_a_hostname_alias_to_its_node(monkeypatch):
    """A node whose hostname is an alias: a run made there outside the CLI is still that node's."""
    import socket
    monkeypatch.delenv("TATBOT_NODE", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "Lab-Trainer.lan")
    monkeypatch.setattr(rl, "_fleet_map", lambda: {"trainer": {"hostname": "lab-trainer"}, "camera": {}})
    monkeypatch.setattr(rl, "_HOST_NODE", None)
    assert rl._node() == "trainer"
    monkeypatch.setenv("TATBOT_NODE", "arm")
    assert rl._node() == "arm"


def test_shell_begin_records_the_launcher_pid(tmp_path, monkeypatch, capsys):
    """runlog.sh's begin process exits at once; the run lives as long as the launcher shell."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    launcher = os.getppid()
    assert rl.main(["begin", "--workflow", "teleop", "--parent-pid", str(launcher)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    assert json.loads((run_dir / "meta.json").read_text())["node"]["pid"] == launcher
    row = [r for r in rl.index_runs() if r["run_id"] == run_dir.name][-1]
    assert row["pid"] == launcher and rl.resolve_status(row) == "running"


def test_logs_count_reconciles_launches_against_the_index(tmp_path, monkeypatch, capsys):
    """The 2026-08-24 lesson: count launches from the index, never from timing."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    assert rl.count_launches("rollout_async") == 0
    before = rl.count_launches("rollout_async")
    for _ in range(2):
        run = rl.init("rollout_async", prune_first=False, emit_banner=False)
        run.finalize(0)
    other = rl.init("rollout", prune_first=False, emit_banner=False)  # a different launcher
    other.finalize(0)
    assert rl.count_launches("rollout_async") == 2  # finalized rows do not double-count
    assert rl.count_launches("rollout") == 1
    assert rl.main(["count", "rollout_async"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert rl.main(["count", "rollout_async", "--expect", "2", "--before", str(before)]) == 0
    assert "OK" in capsys.readouterr().out
    assert rl.main(["count", "rollout_async", "--expect", "1", "--before", str(before)]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    assert rl.main(["compact"]) == 0
    assert rl.count_launches("rollout_async") == 2  # one merged row per run keeps its start


def test_logs_compact_keeps_a_row_appended_while_it_rewrites(tmp_path, monkeypatch):
    """A row appended between compact's read and its replace used to be lost."""
    import threading

    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    run = rl.init("rollout", prune_first=False, emit_banner=False)
    run.finalize(0)
    read = rl.index_runs
    late: list[threading.Thread] = []

    def read_then_race(*args, **kwargs):
        rows = read(*args, **kwargs)
        # Another workflow appends after the read: it opens the old file and
        # must wait for the lock, then reopen the file compact put in its place.
        late.append(threading.Thread(target=rl._append_line, args=(
            rl.index_path(), {"run_id": "late-run", "workflow": "teleop", "status": "running"})))
        late[0].start()
        late[0].join(0.5)
        return rows

    monkeypatch.setattr(rl, "index_runs", read_then_race)
    assert rl.main(["compact"]) == 0
    late[0].join(5)
    monkeypatch.setattr(rl, "index_runs", read)
    assert {r["run_id"] for r in rl.index_runs()} == {run.run_id, "late-run"}


def test_logs_list_reports_a_node_that_did_not_answer(tmp_path, monkeypatch, capsys):
    """An unreachable node must not read as 'that run does not exist'.

    `logs list --all-nodes` is what AGENTS.md tells an agent to run when it does
    not know where a run happened, so a node that fails to answer has to reach
    the exit code. It used to be discarded by `rc = rc or (0 if code == 0 else 0)`.
    """
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(rl, "_node", lambda: "here")
    monkeypatch.setattr(rl, "_remote", lambda node, args: (255, f"{node}: unreachable"))

    assert rl.main(["list", "--node", "elsewhere"]) == 255
    captured = capsys.readouterr()
    assert "elsewhere: listing failed (exit 255)" in captured.err
    assert "--- elsewhere ---" in captured.out

    # A node that answers normally still exits 0, listing or not.
    monkeypatch.setattr(rl, "_remote", lambda node, args: (0, ""))
    assert rl.main(["list", "--node", "elsewhere"]) == 0


def test_logs_list_json_across_nodes_is_one_json_array(tmp_path, monkeypatch, capsys):
    """--json with --node used to print text headers and the remote's text listing
    before the local array; --status never reached the remote node."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(rl, "_node", lambda: "here")
    calls = []

    def remote(node, args):
        calls.append(args)
        return 0, json.dumps([{"run_id": "20260821T164233Z-elsewhere-a3f1", "node": node}])

    monkeypatch.setattr(rl, "_remote", remote)
    assert rl.main(["list", "--node", "elsewhere", "--status", "ok", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["node"] for r in rows] == ["elsewhere"]
    assert calls == [["list", "-n", "20", "--status", "ok", "--json"]]
    # A node that answers with something other than a listing is a failed node.
    monkeypatch.setattr(rl, "_remote", lambda node, args: (0, "not json"))
    assert rl.main(["list", "--node", "elsewhere", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == [] and "elsewhere: listing failed" in captured.err


def test_logs_list_passes_the_positional_workflow_to_remote_nodes(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(rl, "_node", lambda: "here")
    calls = []
    monkeypatch.setattr(rl, "_remote", lambda node, args: (calls.append((node, args)) or 0, ""))

    assert rl.main(["list", "session", "--node", "elsewhere", "-n", "3"]) == 0
    assert calls == [("elsewhere", ["list", "session", "-n", "3"])]


# A fleet map in the shape of config/nodes.json, with documentation addresses
# (RFC 5737): this file is exported, so no real node name or tailnet address.
FLEET = {
    "camera": {"ssh": "camera@192.0.2.10", "checkout": "~/source", "roles": ["poe-cameras"]},
    "arm": {"ssh": "arm@192.0.2.11", "checkout": "~/source", "roles": ["arm"]},
    "retired": {"ssh": "retired@192.0.2.12", "checkout": "~/source", "roles": []},
}


def test_remote_command_resolves_the_checkout_like_the_on_hop():
    """`logs list --all-nodes` dials a node's checkout from config/nodes.json.

    It used to glob `$(cd ~/tatbot* && pwd)` on the remote, which `cd` refuses
    the moment the log root sits beside the checkout, so two fleet nodes ran
    `python3 /scripts/lib/tatbot_runlog.py` (2026-09-17). The target and the
    checkout now come from tatbot_cli.nodes, as `tatbot --on <node>` reads them.
    """
    from tatbot_cli import nodes

    cmd = rl._remote_argv("camera", ["list", "-n", "20"], FLEET)
    assert cmd[:len(rl.SSH)] == rl.SSH
    assert cmd[-2] == nodes.ssh_target(FLEET, "camera") == "camera@192.0.2.10"
    assert cmd[-1] == (f"python3 {nodes.remote_checkout(FLEET, 'camera')}/scripts/lib/tatbot_runlog.py "
                       "'list' '-n' '20'")
    assert "$HOME/source/scripts/lib/tatbot_runlog.py" in cmd[-1]
    assert "tatbot*" not in cmd[-1]
    # Arguments are quoted for the remote shell, whatever they hold.
    assert rl._remote_argv("camera", ["list", "--workflow", "it's"], FLEET)[-1].endswith("'it'\\''s'")


def test_fetch_dials_the_ssh_target_at_the_remote_log_root(tmp_path, monkeypatch):
    """rsync reaches the node as `list` and `show` do, not by a bare name ~/.ssh/config may lack."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(rl, "_node", lambda: "here")
    monkeypatch.setattr(rl, "_fleet_map", lambda: FLEET)
    monkeypatch.setattr(rl, "_remote", lambda node, args: (0, "/srv/camera-logs\n"))
    calls = []
    monkeypatch.setattr(rl.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert rl.main(["fetch", "20260821T164233Z-camera-a3f1"]) == 0
    assert calls[0][-2] == "camera@192.0.2.10:/srv/camera-logs/*/20260821T164233Z-camera-a3f1*/"


def test_remote_command_dials_an_unmapped_node_by_name():
    """A node the fleet map does not know (a bare --node alias, a public clone
    with no config/nodes.json) is dialed as given at the public repo's path."""
    cmd = rl._remote_argv("elsewhere", ["show", "x"], {})
    assert cmd[-2] == "elsewhere"
    assert cmd[-1].startswith("python3 $HOME/tatbot/scripts/lib/tatbot_runlog.py ")


def test_all_nodes_sweep_leaves_a_retired_node_out(tmp_path, monkeypatch, capsys):
    """A node with no roles is retired: nothing runs there and it is usually
    off, so dialing it cost every sweep a connect timeout and a non-zero exit.
    The sweep names it and moves on, the way `tatbot status --fleet` reports
    it retired."""
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("TATBOT_NODES", raising=False)
    monkeypatch.setattr(rl, "_node", lambda: "here")
    monkeypatch.setattr(rl, "_fleet_map", lambda: FLEET)
    sweep_set = ["arm", "camera", "here", "retired"]
    monkeypatch.setattr(rl, "load_config", lambda: {**rl.DEFAULT_CONFIG, "nodes": sweep_set})
    dialed = []
    monkeypatch.setattr(rl, "_remote", lambda node, args: dialed.append(node) or (0, ""))

    assert rl._sweep_nodes({"nodes": sweep_set}, FLEET) == (["arm", "camera", "here"], ["retired"])
    assert rl._sweep_nodes({}, FLEET) == (["camera", "arm"], ["retired"])
    assert rl.main(["list", "--all-nodes"]) == 0
    out = capsys.readouterr().out
    assert dialed == ["arm", "camera"]
    assert "--- retired ---\n(retired: no roles in config/nodes.json; not swept" in out

    # TATBOT_NODES is an explicit request: swept as given, retired or not.
    monkeypatch.setenv("TATBOT_NODES", "retired arm")
    assert rl._sweep_nodes({}, FLEET) == (["retired", "arm"], [])
