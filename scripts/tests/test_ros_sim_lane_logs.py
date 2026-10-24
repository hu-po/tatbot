"""A sim lane's run logs: `tatbot ros down` adopts the lane's finished runs into the node's log root.

What would fail if it were wrong: a lane that comes down leaving its runs where `tatbot logs` never looks and
retention never prunes; an adopt that runs before the lane's stack has finished writing its run (before
`run.sh down`); the command `down` prints not working against a real lane root's layout.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli_runner import REPO
from tatbot_cli import nodes

BACKEND = REPO / "scripts" / "lib" / "tatbot_cli" / "tatbot_ros.py"
spec = importlib.util.spec_from_file_location("tatbot_ros_backend_lane_logs", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

NMAP = nodes.load(REPO)
needs_sim_fleet = pytest.mark.skipif(
    not (len(nodes.nodes_with(NMAP, "ros-sim")) == 1 and len(nodes.nodes_with(NMAP, "ros")) == 1),
    reason=("config/nodes.json names no single ros-sim and ros node" if NMAP
            else "needs a described fleet (config/nodes.json)"))


@needs_sim_fleet
def test_a_lanes_down_adopts_its_runs_after_the_container_stops(capsys, monkeypatch):
    for name in ("TATBOT_ROS_NODE", "TATBOT_ROS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TATBOT_ROS_LANE", "21")
    assert backend.main(["--dry-run", "down"]) == 0
    out = capsys.readouterr().out
    stop = out.index("run.sh down $HOME/tatbot-ros-sim/21")
    adopt = out.index("$HOME/tatbot-ros-sim/21/repo/scripts/lib/tatbot_runlog.py adopt $HOME/tatbot-ros-sim/21/logs")
    assert stop < adopt   # the stack's own run is finished (finalized at its SIGINT) before it moves


def test_the_adopt_command_moves_a_lanes_runs_into_the_node_root(tmp_path):
    """``SIM_ADOPT`` run by a shell against a lane root laid out as deploy leaves it (``repo/`` the deployed
    checkout, ``logs/`` the container's TATBOT_LOG_ROOT)."""
    lane, node = tmp_path / "lane", tmp_path / "node-logs"
    (lane / "logs" / "ros-draw").mkdir(parents=True)
    (lane / "repo").symlink_to(REPO)
    run = lane / "logs" / "ros-draw" / "20260101T000200Z-simnode-0a02"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"status": "ok", "ended_at": "2026-10-19T08:13:29.703Z",
                                               "node": {"hostname": "simnode", "pid": 42}}))
    (lane / "logs" / "page-touched-right.json").write_text("{}")
    python = Path(sys.executable).parent   # its python3: a CI image keeps none in /usr/bin
    env = {"PATH": f"{python}:/usr/bin:/bin", "HOME": str(tmp_path), "TATBOT_LOG_ROOT": str(node)}
    proc = subprocess.run(["bash", "-c", backend.SIM_ADOPT.format(root=lane)], env=env, capture_output=True,
                          text=True, check=True)
    assert "adopted 20260101T000200Z-simnode-0a02" in proc.stdout, proc.stdout + proc.stderr
    assert (node / "ros-draw" / run.name / "ADOPTED").is_file() and not run.exists()
    assert (lane / "logs" / "page-touched-right.json").is_file() and not list(node.rglob("page-touched-*"))
    broken = subprocess.run(["bash", "-c", backend.SIM_ADOPT.format(root=tmp_path / "nowhere")], env=env,
                            capture_output=True, text=True)
    assert broken.returncode == 0 and "adopt failed" in broken.stdout   # the lane is down either way
