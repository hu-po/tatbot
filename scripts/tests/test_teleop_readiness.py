"""Tool defaults and the teleop readiness model.

Two rules hold everything here together. A default is never mistaken for a
statement about what is physically in the mount, and a diagnosis is never
replaced by a generic instruction: an installed limits policy that did not take
effect must not be answered with "install it".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "scripts" / "lib"

import teleop_readiness  # noqa: E402
from cli_runner import tatbot  # noqa: E402
from tatbot_cli import gates, nodes  # noqa: E402

ARM = nodes.example_node("arm")
VIEWER = nodes.example_node("operator")
TOOL = "lutin-ballpoint-dot"
LASER = "picosecond-laser-pen"


# --- tool precedence -------------------------------------------------------------


def test_precedence_is_flag_then_environment_then_configured(monkeypatch):
    import tool_spec
    configured = tool_spec.active_tool_id(REPO)
    assert configured, "this checkout must name a fitted tool to exercise the default"

    monkeypatch.delenv("TATBOT_EE_TOOL", raising=False)
    assert gates.resolve_tool(REPO, None) == (configured, "configured", None)

    monkeypatch.setenv("TATBOT_EE_TOOL", LASER)
    assert gates.resolve_tool(REPO, None) == (LASER, "environment", None)
    assert gates.resolve_tool(REPO, TOOL) == (TOOL, "flag", None)


@pytest.mark.parametrize("stated,env,source", [("nope", None, "--ee-tool"),
                                               (None, "nope", "TATBOT_EE_TOOL")])
def test_an_explicit_value_never_falls_back_to_the_default(monkeypatch, stated, env, source):
    monkeypatch.delenv("TATBOT_EE_TOOL", raising=False)
    if env:
        monkeypatch.setenv("TATBOT_EE_TOOL", env)
    tool, _, error = gates.resolve_tool(REPO, stated)
    assert tool is None and "unknown tool 'nope'" in error and source in error


def test_a_missing_or_unknown_configured_pointer_says_exactly_what_to_do(monkeypatch, tmp_path):
    monkeypatch.delenv("TATBOT_EE_TOOL", raising=False)
    (tmp_path / "config" / "tools").mkdir(parents=True)
    (tmp_path / "config" / "tools" / f"{TOOL}.yaml").write_text("id: x\n")

    monkeypatch.setattr(gates, "configured_tool", lambda repo, arm: (None, None))
    tool, source, error = gates.resolve_tool(tmp_path, None)
    assert tool is None and source == "configured" and "--ee-tool <id>" in error

    monkeypatch.setattr(gates, "configured_tool", lambda repo, arm: ("gone-tool", None))
    tool, _, error = gates.resolve_tool(tmp_path, None)
    assert tool is None and "gone-tool" in error and "no datasheet" in error


def test_omitting_the_tool_uses_the_configured_one_and_names_its_source():
    plan = tatbot("--dry-run", "--json", "teleop", "start", node=ARM)
    assert plan.returncode == 0, plan.stderr
    selection = json.loads(plan.stdout)["tool"]
    assert selection["source"] == "configured" and selection["id"] and not selection["deferred"]


def test_a_remote_default_is_the_owners_and_planning_never_asks_that_node():
    """The client's configuration is not the operator's choice for another node."""
    plan = tatbot("--dry-run", "--json", "teleop", "start", node=VIEWER)
    assert plan.returncode == 0, plan.stderr
    report = json.loads(plan.stdout)
    assert report["hop"] == ARM
    assert report["tool"] == {"id": None, "source": None, "source_label": None,
                              "deferred": True, "resolved_on": ARM}
    assert "--ee-tool" not in report["argv"][-1]


def test_an_intentional_environment_tool_crosses_the_hop_as_an_explicit_flag():
    plan = tatbot("--dry-run", "--json", "teleop", "start", node=VIEWER,
                  env={"TATBOT_EE_TOOL": LASER})
    assert plan.returncode == 0, plan.stderr
    report = json.loads(plan.stdout)
    assert f"--ee-tool {LASER}" in report["argv"][-1]
    assert any("carried over" in note for note in report["notes"])


def test_a_mistyped_tool_is_refused_before_an_ssh_is_spent():
    refused = tatbot("--dry-run", "--json", "--ee-tool", "nope", "teleop", "start", node=VIEWER)
    assert refused.returncode == 3
    assert json.loads(refused.stderr)["gate"] == "ee_tool"


# --- real-time evidence ----------------------------------------------------------


def test_the_required_priority_is_the_one_the_control_loop_requests():
    """Parsed from the executable's source, so the two cannot drift apart."""
    assert teleop_readiness.priority_is_measured(REPO)
    source = (REPO / teleop_readiness.PRIORITY_SOURCE).read_text()
    assert f"int rt_priority = {teleop_readiness.required_priority(REPO)};" in source
    assert teleop_readiness.required_priority(REPO, 42) == 42


def test_limits_rules_are_matched_the_way_pam_matches_them(monkeypatch, tmp_path):
    path = tmp_path / "99-tatbot-realtime.conf"
    path.write_text("# comment\n@sudo   -   rtprio   90\nbogus line\nroot - rtprio 95\n")
    monkeypatch.setattr(teleop_readiness, "_account", lambda: ("someone", {"sudo", "users"}))
    policy = teleop_readiness.limits_policy(path)
    assert policy["installed"] and [r["value"] for r in policy["rules"]] == [90, 95]
    assert [r["domain"] for r in policy["applicable"]] == ["@sudo"]

    monkeypatch.setattr(teleop_readiness, "_account", lambda: ("someone", {"users"}))
    assert teleop_readiness.limits_policy(path)["applicable"] == []
    assert teleop_readiness.limits_policy(tmp_path / "absent.conf")["installed"] is False


def facts(**overrides):
    base = {"priority": 80, "priority_measured": True, "soft": 0, "hard": 0, "sufficient": False,
            "limits": {"path": "/etc/security/limits.d/99-tatbot-realtime.conf", "installed": True,
                       "rules": [{"domain": "@sudo", "type": "-", "value": 90}],
                       "applicable": [{"domain": "@sudo", "type": "-", "value": 90}],
                       "account": "someone", "groups": ["sudo"]},
            "login": {"over_ssh": True, "sshd_uses_pam": True, "sshd_pam_limits": True}}
    base.update(overrides)
    return base


def test_an_absent_policy_gets_the_exact_install_command():
    limits = dict(facts()["limits"], installed=False, rules=[], applicable=[])
    lines = teleop_readiness.explain_realtime(facts(limits=limits))
    assert "requires priority 80" in lines[1] and "limit is 0" in lines[1]
    assert any(line.startswith("Next: sudo cp config/limits/") for line in lines)
    assert any("NEW login session" in line for line in lines)


def test_an_installed_policy_is_never_answered_with_install_it_again():
    lines = teleop_readiness.explain_realtime(facts(), node="somewhere")
    text = " ".join(lines)
    assert "installed and grants this account rtprio 90" in text
    assert "not effective in this session" in text
    assert "sudo cp" not in text and "reinstall" not in text
    assert "Next: tatbot --on somewhere teleop check" in lines


def test_an_inapplicable_policy_names_the_account_and_its_groups():
    limits = dict(facts()["limits"], applicable=[])
    text = " ".join(teleop_readiness.explain_realtime(facts(limits=limits)))
    assert "grants rtprio to @sudo" in text and "'someone' is not matched" in text
    assert "sudo cp" not in text


@pytest.mark.parametrize("login,expected", [
    ({"over_ssh": True, "sshd_uses_pam": True, "sshd_pam_limits": False}, "does not call pam_limits.so"),
    ({"over_ssh": True, "sshd_uses_pam": False, "sshd_pam_limits": False}, "UsePAM no"),
    ({"over_ssh": True, "sshd_uses_pam": True, "sshd_pam_limits": True}, "has not been established"),
    ({"over_ssh": False, "sshd_uses_pam": True, "sshd_pam_limits": True}, "inherited from whatever"),
])
def test_the_ineffective_policy_explanation_follows_the_evidence(login, expected):
    assert expected in " ".join(teleop_readiness.explain_realtime(facts(login=login)))


def test_a_sufficient_limit_reports_availability_and_nothing_else():
    lines = teleop_readiness.explain_realtime(facts(soft=90, sufficient=True))
    assert len(lines) == 1 and "available" in lines[0]


def test_no_explanation_ever_offers_no_rt_as_the_production_fix():
    for login in ({"over_ssh": True, "sshd_uses_pam": True, "sshd_pam_limits": True},):
        text = " ".join(teleop_readiness.explain_realtime(facts(login=login)))
        assert "--no-rt" not in text or "not a way to drive the arms" in text


# --- the check command -----------------------------------------------------------


def test_teleop_check_is_read_only_and_routes_to_the_arm_owner():
    from tatbot_cli.registry import find
    v = find("teleop", "check")
    assert v is not None and v.role == "arm" and v.auto_hop and v.native
    assert not v.launch_id and not v.ink_hook and v.tier == "sensor"
    assert "autonomous_motion" not in v.effects and "human_motion" not in v.effects

    plan = tatbot("--dry-run", "--json", "teleop", "check", "--no-probe", node=VIEWER)
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["hop"] == ARM


def test_teleop_check_reports_observations_without_opening_any_device(tmp_path):
    result = tatbot("--json", "teleop", "check", "--no-probe", node=ARM)
    assert result.returncode in (0, 3), result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "tatbot.teleop-readiness/1"
    rows = {row["id"]: row for row in report["observations"]}
    assert {"executable", "profile", "tool", "realtime", "estop", "exclusivity"} <= set(rows)
    # A device path is not a heartbeat, and an unprobed arm is unknown, not ok.
    assert rows["arm-leader"]["state"] == "unknown"
    assert "launcher" in report["limits"]


def test_teleop_check_is_not_a_repository_check():
    from tatbot_cli.registry import find
    invariants = " ".join(find("teleop", "check").invariants)
    # The safety narration that used to ride here was removed as duplicated (4913c76f);
    # the invariant that stays is the one the name invites confusing.
    assert "tatbot check" in invariants and "Not a repository check" in invariants


def test_the_launcher_explains_a_scheduling_refusal_from_the_same_model():
    launcher = (REPO / "scripts/teleop_start.sh").read_text()
    assert 'if [ "$RC" = 3 ]' in launcher
    assert "teleop_readiness.py\" --explain-realtime" in launcher
    assert 'exit "$RC"' in launcher, "the launcher must still return the executable's code"


def test_the_executable_still_refuses_before_either_driver_is_constructed():
    source = (REPO / "cpp/teleop/wxai_teleop.cpp").read_text()
    refusal = source.index("Teleop could not start")
    assert refusal < source.index("std::make_unique<tatbot::DriverLease>")
    assert "opt.rt_priority" in source[refusal:refusal + 900]
    assert "setup.limits_installed" in source[refusal:refusal + 1600]


def test_the_check_does_not_tell_you_to_run_the_command_you_just_ran():
    """`teleop check` renders this account itself; a self-reference is not advice."""
    from_check = teleop_readiness.explain_realtime(facts(), node="somewhere", from_check=True)
    assert not any("teleop check" in line for line in from_check)
    assert any("console" in line for line in from_check)
    # The launcher, which is a different caller, still points at the check.
    assert any("teleop check" in line for line in teleop_readiness.explain_realtime(facts()))


def test_the_next_probe_names_a_session_that_measures_something_new():
    """Another SSH session ran the same PAM stack; the console is the discriminator."""
    lines = teleop_readiness.explain_realtime(facts(), node="somewhere", from_check=True)
    instruction = next(line for line in lines if line.startswith("Next:"))
    assert "ssh" not in instruction.lower(), "an SSH probe would measure what was just measured"
    assert "console of somewhere" in instruction and "ulimit -Sr -Hr" in instruction
    text = " ".join(lines)
    assert "0 at both" in text and "not being applied at all" in text


def sessions_facts(capable=True, **overrides):
    rows = [{"session": "c9", "type": "tty", "remote": True, "class": "user", "soft": 0, "hard": 0}]
    if capable:
        rows.append({"session": "2", "type": "x11", "remote": False, "class": "user",
                     "soft": 90, "hard": 90})
    base = facts()
    base.update(sessions=rows, capable_sessions=[r for r in rows if r["soft"] >= 80],
                host_capable=capable)
    base.update(overrides)
    return base


def test_a_rig_is_not_judged_by_the_session_the_check_happens_to_run_in():
    """A routed check runs in a one-shot ssh session; an operator never starts
    teleop from one, so its limit does not answer 'can this rig start teleop'."""
    lines = teleop_readiness.explain_realtime(sessions_facts(), node="somewhere", from_check=True)
    text = " ".join(lines)
    assert "The host itself can" in text and "holds 90" in text
    assert "one-shot `ssh host command` session does not get it" in text
    assert "Next: log in to somewhere and run `tatbot teleop start` there" in lines[-1]
    # It must NOT fall through to the "not effective" branch, which would be wrong.
    assert "not effective in this session" not in text
    assert "console" not in text


def test_that_case_reports_the_rig_as_able_rather_than_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(teleop_readiness, "realtime_facts",
                        lambda repo, requested=None: sessions_facts())
    rows = {row["id"]: row for row in teleop_readiness.observations(tmp_path, probe_arms=False)}
    assert rows["realtime"]["state"] == "ok"
    assert rows["realtime"]["value"]["host_capable"] is True
    assert rows["realtime"]["value"]["soft"] == 0, "the measured session limit is still reported"


def test_a_host_where_no_session_has_the_priority_still_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(teleop_readiness, "realtime_facts",
                        lambda repo, requested=None: sessions_facts(capable=False))
    rows = {row["id"]: row for row in teleop_readiness.observations(tmp_path, probe_arms=False)}
    assert rows["realtime"]["state"] == "failed"


def test_the_session_limit_is_read_from_the_kernel_not_from_configuration(tmp_path):
    """`/proc/<pid>/limits` is what a session actually holds; a limits.conf file
    is only what someone intended."""
    soft, hard = teleop_readiness._process_rtprio(os.getpid())
    assert soft is not None and hard is not None and soft >= 0
    assert teleop_readiness._process_rtprio(2 ** 22) == (None, None)


def test_peer_sessions_survives_a_host_without_loginctl(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("loginctl")
    monkeypatch.setattr(teleop_readiness.subprocess, "run", absent)
    assert teleop_readiness.peer_sessions() == []
