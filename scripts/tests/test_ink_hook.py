"""The ink hook's seams: the run stamp, the cross-node/session debit guards,
the arm-gate hand-down and the ballpoint/needle spelling of the tool.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_ink_hook.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import il_analyze_rollout as ana  # noqa: E402
import ink_session  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402


@pytest.fixture
def ink_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("TATBOT_INK_SESSION", str(tmp_path / "session.json"))
    monkeypatch.delenv("TATBOT_INK", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "nodea.local")
    return tmp_path


def _open_session():
    tool = tool_spec.load_tool("lutin-3rl-bugpin", REPO)
    pol = ink_spec.policy_for(tool)
    s = ink_session.start(tool, pol)
    ink_session.apply_dip(s, pol, "inkcap_medium_1", None, pol.uptake_ul, "session_start")
    return s


def _run(tmp_path, run_id="20260829T120000Z-nodea-ab12", **meta):
    d = tmp_path / run_id
    d.mkdir()
    if meta:
        (d / "meta.json").write_text(json.dumps(meta))
    a = {"run_id": run_id, "ink_contact": {"valid": True, "basis": "touchoff_tip", "contact_mm": 100.0, "contact_s": 5.0}}
    return d, a


# --- the run stamp outranks the environment -------------------------------------------


def test_run_stamp_outranks_the_environment(ink_env, monkeypatch):
    _open_session()
    d, a = _run(ink_env)
    (d / "ink.json").write_text(json.dumps({"tracking": False, "hook": "--no-ink"}))
    out = ana.debit_ink_session(d, a, log=lambda *_: None)
    assert out["skipped"].startswith("--no-ink (run stamp")
    # a stale TATBOT_INK=0 in this shell does not skip a run whose stamp says tracking
    (d / "ink.json").write_text(json.dumps({"tracking": True, "hook": "none"}))
    monkeypatch.setenv("TATBOT_INK", "0")
    out = ana.debit_ink_session(d, a, log=lambda *_: None)
    assert "session_id" in out and out["taken_ul"] > 0


def test_environment_is_the_fallback_without_a_stamp(ink_env, monkeypatch):
    _open_session()
    d, a = _run(ink_env)
    monkeypatch.setenv("TATBOT_INK", "0")
    assert ana.debit_ink_session(d, a, log=lambda *_: None)["skipped"] == "--no-ink (TATBOT_INK (no run stamp))"
    monkeypatch.delenv("TATBOT_INK")
    assert "session_id" in ana.debit_ink_session(d, a, log=lambda *_: None)


# --- the session is node-local ----------------------------------------------------------


def test_a_run_from_another_node_is_not_debited_here(ink_env):
    _open_session()
    d, a = _run(ink_env, run_id="20260829T120000Z-nodec-ab12")
    out = ana.debit_ink_session(d, a, log=lambda *_: None)
    assert "made on nodec" in out["skipped"]
    # meta.json's hostname is the authority when present
    d, a = _run(ink_env, run_id="20260829T120001Z-nodea-ab13", node={"hostname": "noded"})
    assert "made on noded" in ana.debit_ink_session(d, a, log=lambda *_: None)["skipped"]


def test_a_run_under_another_session_is_not_debited(ink_env):
    s = _open_session()
    d, a = _run(ink_env)
    (d / "ink.jsonl").write_text(json.dumps({"kind": "dip", "session_id": "20260801_000000-nodea-dead"}) + "\n")
    out = ana.debit_ink_session(d, a, log=lambda *_: None)
    assert "20260801_000000-nodea-dead" in out["skipped"] and s.session_id in out["skipped"]
    (d / "ink.jsonl").write_text(json.dumps({"kind": "dip", "session_id": s.session_id}) + "\n")
    assert ana.debit_ink_session(d, a, log=lambda *_: None)["session_id"] == s.session_id


# --- ink_hook writes the stamp ------------------------------------------------------------


def _hook(tmp_path, *args, env=None):
    script = f'''
set -e
REPO="{REPO}"
source "$REPO/scripts/lib/ink_hook.sh"
ink_hook::strip "$@"; set -- "${{INK_HOOK_ARGS[@]}}"
RUN_DIR="{tmp_path}"
ink_hook::stamp
echo "rest=$*"; echo "ink=$TATBOT_INK"
'''
    return subprocess.run(["bash", "-c", script, "x", *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def test_ink_hook_stamps_the_run_dir(tmp_path):
    r = _hook(tmp_path, "--no-ink", "pos")
    assert r.returncode == 0, r.stderr
    stamp = json.loads((tmp_path / "ink.json").read_text())
    assert stamp["tracking"] is False and stamp["hook"] == "--no-ink" and "ink=0" in r.stdout and "rest=pos" in r.stdout
    r = _hook(tmp_path, "pos")
    assert r.returncode == 0, r.stderr
    stamp = json.loads((tmp_path / "ink.json").read_text())
    assert stamp["tracking"] is True and stamp["hook"] == "none" and "ink=1" in r.stdout and "rest=pos" in r.stdout
    # A stale TATBOT_INK=0 left in the operator's shell does not outvote the stamp.
    r = _hook(tmp_path, "pos", env={"TATBOT_INK": "0"})
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "ink.json").read_text())["tracking"] is True and "ink=1" in r.stdout


# --- the arm gate hands its launch id to a child, and only to a child -------------------


def test_arm_gate_ancestry_check():
    script = f'''
source "{REPO}/scripts/lib/arm_gate.sh"
arm_gate::_is_ancestor "$PPID" && echo parent-ok
arm_gate::_is_ancestor 1 && echo init-ok || echo init-no
arm_gate::_is_ancestor 999999 && echo bogus-ok || echo bogus-no
'''
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "parent-ok" in r.stdout and "init-no" in r.stdout and "bogus-no" in r.stdout


def test_arm_gate_ledgers_and_audits_but_never_refuses():
    """Operator decision 2026-09-04: the launch id is automatic. A missing, stale or empty
    token mints one (`minted`), a repeated id is audited (`repeat`), a stale inherit mints
    fresh (`fresh-after-stale-inherit`); no path returns non-zero."""
    gate = (REPO / "scripts/lib/arm_gate.sh").read_text()
    require = gate[gate.index("arm_gate::require()"):]
    assert "REFUSING" not in gate and "return 2" not in require and "return 3" not in require
    for verdict in ("minted", "repeat", "fresh-after-stale-inherit", "pass-inherited", "pass"):
        assert f"arm_gate::audit {verdict} " in gate, verdict
    assert "arm_gate::mint()" in gate and "/var/tmp/tatbot-launch-ids" in gate


# --- ink.py takes the stated tool the way every tool does ---------------------------------


def test_ink_py_session_start_takes_ee_tool_and_the_environment(ink_env, monkeypatch):
    py = [sys.executable, str(REPO / "scripts/ink.py")]
    env = dict(os.environ)
    r = subprocess.run(py + ["session", "start", "--ee-tool", "lutin-3rl-bugpin"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert ink_session.current() is not None
    subprocess.run(py + ["session", "end"], capture_output=True, text=True, env=env, check=True)
    env["TATBOT_EE_TOOL"] = "lutin-3rl-bugpin"
    r = subprocess.run(py + ["session", "start"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    env.pop("TATBOT_EE_TOOL")
    subprocess.run(py + ["session", "end"], capture_output=True, text=True, env=env, check=True)
    r = subprocess.run(py + ["session", "start"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "--ee-tool" in (r.stderr + r.stdout)
    r = subprocess.run(py + ["plan", "--strokes", "100,5"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "--ee-tool" in (r.stderr + r.stdout)
