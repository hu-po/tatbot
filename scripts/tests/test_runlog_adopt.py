"""`tatbot logs adopt`: a sim lane's finished runs move into this node's log root, findable and prunable.

What would fail if it were wrong: a lane run that `logs show/last/list` cannot find, or that retention never
deletes; a page-carry file or an unfinished run moved where a real stack or a live writer would meet it; an
adopted run pushing a node's own runs out of its workflow's keep count; a root adopted into itself.
"""

import json
import os
from pathlib import Path

import pytest
import tatbot_runlog as rl  # noqa: E402

OLD = "2020-01-01T00:00:00.000Z"


def _run(root: Path, workflow: str, run_id: str, status: str = "ok", ended: str = OLD, size: int = 1000) -> Path:
    run = root / workflow / run_id
    run.mkdir(parents=True)
    meta = {"status": status, "started_at": ended, "node": {"hostname": run_id.split("-")[1], "pid": 7},
            "git": {"short": "abc1234", "dirty": False}}
    if status != "running":
        meta.update(ended_at=ended, exit_code=0, duration_s=1.0)
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "console.log").write_text("x" * size)
    with (root / "index.jsonl").open("a") as index:
        index.write(json.dumps({"run_id": run_id, "workflow": workflow, "node": run_id.split("-")[1],
                                "dir": str(run), "status": "running", "pid": 7, "started_at": ended}) + "\n")
        if status != "running":
            index.write(json.dumps({"run_id": run_id, "workflow": workflow, "status": status, "dir": str(run),
                                    "exit_code": 0, "ended_at": ended}) + "\n")
    return run


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """This node's log root (TATBOT_LOG_ROOT) and a sim lane's, with its page-carry file beside its runs."""
    node, lane = tmp_path / "tatbot-logs", tmp_path / "tatbot-ros-sim" / "21" / "logs"
    node.mkdir()
    lane.mkdir(parents=True)
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(node))
    _run(lane, "ros-stack", "20260101T000100Z-simnode-0a01")
    _run(lane, "ros-draw", "20260101T000200Z-simnode-0a02", size=5000)
    _run(lane, "ros-draw", "20260101T000300Z-simnode-0a03", status="running")
    (lane / "page-touched-right.json").write_text("{}")
    (lane / "ros-draw" / "latest").symlink_to("20260101T000200Z-simnode-0a02")
    return node, lane


def test_finished_lane_runs_move_into_this_nodes_root_and_index(roots, capsys):
    node, lane = roots
    found = rl.adopt(lane)
    assert sorted(e["run_id"] for e in found["adopted"]) == ["20260101T000100Z-simnode-0a01",
                                                              "20260101T000200Z-simnode-0a02"]
    assert found["left"] == [{"run_id": "20260101T000300Z-simnode-0a03", "workflow": "ros-draw",
                              "reason": "not finished"}]
    moved = sum(e["bytes"] for e in found["adopted"])
    assert moved > 6000 and found["bytes_after"] == rl._dir_size(lane)   # the lane's index grew a row a run
    assert found["bytes_after"] < found["bytes_before"] - moved / 2
    draw = node / "ros-draw" / "20260101T000200Z-simnode-0a02"
    assert draw.is_dir() and not (lane / "ros-draw" / draw.name).exists()
    assert json.loads((draw / rl.ADOPTED).read_text())["from"] == str(lane / "ros-draw" / draw.name)
    # the page-carry file, the unfinished run and the lane's own index stay in the lane
    assert (lane / "page-touched-right.json").is_file() and (lane / "ros-draw" / "20260101T000300Z-simnode-0a03").is_dir()
    assert not list(node.rglob("page-touched-*"))
    rows = {r["run_id"]: r for r in rl.index_runs()}
    assert rows[draw.name]["dir"] == str(draw) and rows[draw.name]["status"] == "ok" and "pid" not in rows[draw.name]
    assert rl.resolve_status(rows[draw.name]) == "ok"
    assert os.readlink(node / "ros-draw" / "latest") == draw.name
    lane_rows = rl._rows_by_run(lane / "index.jsonl")
    assert lane_rows[draw.name]["dir"] == str(draw) and lane_rows[draw.name]["adopted_to"] == str(draw)
    # logs show, last and list find it
    assert rl.main(["show", draw.name]) == 0 and str(draw) in capsys.readouterr().out
    assert rl.main(["last", "ros-draw"]) == 0 and draw.name in capsys.readouterr().out
    assert rl.main(["list"]) == 0
    listed = capsys.readouterr().out
    assert draw.name in listed and "20260101T000100Z-simnode-0a01" in listed
    # a second adopt moves nothing new
    assert rl.adopt(lane)["adopted"] == []


def test_a_dry_run_moves_nothing(roots, capsys):
    node, lane = roots
    assert rl.main(["adopt", str(lane), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would adopt 20260101T000200Z-simnode-0a02" in out and "before" in out and "after" in out
    assert not (node / "ros-draw").exists() and not (node / "index.jsonl").exists()


@pytest.mark.parametrize("which", ["same", "inside", "above"])
def test_a_root_is_never_adopted_into_itself(roots, which):
    node, _ = roots
    (node / "sim2").mkdir()
    source = {"same": node, "inside": node / "sim2", "above": node.parent}[which]
    with pytest.raises(ValueError):
        rl.adopt(source)


def test_adopted_runs_are_pruned_by_their_own_retention(roots):
    node, lane = roots
    for i in range(3):   # this node's own old runs of the same workflow
        _run(node, "ros-draw", f"2020010{i + 1}T000000Z-realnode-000{i}")
    rl.adopt(lane)
    cfg = rl.load_config()
    cfg["log_root"] = str(node)
    cfg["workflows"]["ros-draw"] = {"keep_days": 1, "keep_runs": 3}
    cfg["workflows"][rl.ADOPTED_POLICY] = {"keep_days": 1, "keep_runs": 0}
    named = {e["run_id"] for e in rl.prune("ros-draw", cfg=cfg, dry_run=True, verbose=False)}
    # the adopted run past its retention is named; the node's own three are its keep count, not displaced by it
    assert named == {"20260101T000200Z-simnode-0a02"}
    cfg["workflows"]["ros-draw"]["keep_runs"] = 2
    named = {e["run_id"] for e in rl.prune("ros-draw", cfg=cfg, dry_run=True, verbose=False)}
    assert named == {"20260101T000200Z-simnode-0a02", "20200101T000000Z-realnode-0000"}
    cfg["workflows"][rl.ADOPTED_POLICY] = {"keep_days": 1, "keep_runs": 5}
    named = {e["run_id"] for e in rl.prune("ros-draw", cfg=cfg, dry_run=True, verbose=False)}
    assert named == {"20200101T000000Z-realnode-0000"}


def test_the_checked_in_policy_holds_adopted_runs_to_sim_retention():
    cfg = rl.load_config()
    adopted = rl.retention_for(rl.ADOPTED_POLICY, cfg)
    assert adopted["keep_days"] <= rl.retention_for("sim2", cfg)["keep_days"]
    assert adopted.get("enabled", True)
