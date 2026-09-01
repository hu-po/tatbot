"""Hardware-profile gate: fail closed, structurally. Stdlib only.

Plan Phase 2 exit gates: a missing or malformed profile is a nonzero,
pre-connection failure; the example profile can never satisfy the hardware
gate; the scrubbed trossen-wxai profile satisfies it only once a deployment
states its own addresses; the private tatbot profile passes as-is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import tatbot_profile as tp  # noqa: E402
from tatbot_cli import nodes  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(tp.ENV, raising=False)


def test_repo_profiles_all_load_and_scan_shape():
    for name in tp.available(REPO):
        p = tp.load(REPO, name)
        assert p["name"] == name
        assert p["schema"] == tp.SCHEMA


def test_example_profile_is_structurally_gate_incapable():
    p = tp.load(REPO, "example")
    errs = tp.hardware_errors(p)
    assert any("hardware != true" in e for e in errs)
    assert any("no driver stanza" in e for e in errs)
    # flipping the single hardware field must NOT arm it: still no driver
    p["hardware"] = True
    assert tp.hardware_errors(p), "example armed by editing one field"


def test_trossen_profile_needs_deployment_addresses():
    p = tp.load(REPO, "trossen-wxai")
    errs = tp.hardware_errors(p)
    assert errs and all("not stated" in e for e in errs)
    p["driver"].update(leader_ip="203.0.113.3", follower_ip="203.0.113.2",
                       estop_device="/dev/ttyACM0")
    assert tp.hardware_errors(p) == []


def test_tatbot_profile_passes_the_gate():
    p = tp.load(REPO, "tatbot")
    assert tp.hardware_errors(p) == []
    ex = tp.env_exports(p)
    assert ex["TATBOT_FOLLOWER_IP"] and ex["TATBOT_LEADER_IP"]
    assert ex["TATBOT_ESTOP_DEVICE"]


def test_resolution_precedence(monkeypatch, tmp_path):
    # explicit > env > tatbot-if-present > none
    assert tp.resolve_name(REPO, "example") == "example"
    monkeypatch.setenv(tp.ENV, "trossen-wxai")
    assert tp.resolve_name(REPO) == "trossen-wxai"
    monkeypatch.delenv(tp.ENV)
    assert tp.resolve_name(REPO) == "tatbot"  # private file exists here
    (tmp_path / "config").mkdir()
    assert tp.resolve_name(tmp_path) is None  # public clone: no default


def test_missing_and_malformed_fail_closed(tmp_path):
    (tmp_path / "config" / "profiles").mkdir(parents=True)
    with pytest.raises(tp.ProfileError, match="no hardware profile stated"):
        tp.load(tmp_path)
    with pytest.raises(tp.ProfileError, match="not found"):
        tp.load(tmp_path, "ghost")
    bad = tmp_path / "config" / "profiles" / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(tp.ProfileError, match="unreadable"):
        tp.load(tmp_path, "bad")
    bad.write_text(json.dumps({"schema": 99, "name": "bad"}))
    with pytest.raises(tp.ProfileError, match="schema"):
        tp.load(tmp_path, "bad")


def test_cli_refuses_motion_without_valid_profile(tmp_path):
    """End to end: a motion verb on a profile-less tree exits nonzero at the
    profile gate, before any hardware path."""
    env = dict(os.environ, TATBOT_NODE=nodes.example_node("arm"), TATBOT_PROFILE="example",
               TATBOT_EE_TOOL="picosecond-laser-pen")
    r = subprocess.run(
        [str(REPO / "scripts" / "tatbot"), "--json", "teleop", "run"],
        capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    out = json.loads(r.stdout or r.stderr)
    assert out["gate"] == "profile"
    assert "cannot drive hardware" in out["reason"]
