"""Physical identity survives role reversal; unsupported launches never reach hardware."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

from tatbot_cli import arms  # noqa: E402


@pytest.mark.parametrize("leader,follower,expected", [
    (None, None, ("left", "right")),
    ("left", None, ("left", "right")),
    (None, "right", ("left", "right")),
    ("right", None, ("right", "left")),
    (None, "left", ("right", "left")),
    ("right", "left", ("right", "left")),
])
def test_role_defaults_and_partial_selection(leader, follower, expected):
    assert arms.select_roles(leader, follower) == arms.TeleopRoles(*expected)


@pytest.mark.parametrize("leader,follower", [
    ("left", "left"), ("right", "right"), ("unknown", None), (None, ""),
])
def test_invalid_roles_refuse(leader, follower):
    with pytest.raises(ValueError):
        arms.select_roles(leader, follower)


def test_reversal_preserves_every_physical_binding():
    default = arms.teleop_plan(REPO, arms.select_roles())
    reverse = arms.teleop_plan(REPO, arms.select_roles("right"))
    assert default["assignments"]["leader"] == reverse["assignments"]["follower"]
    assert default["assignments"]["follower"] == reverse["assignments"]["leader"]
    assert reverse["tool_binding"]["section"] == "left"
    assert default["required_arm_claims"] == reverse["required_arm_claims"] == ["left", "right"]
    assert default["execution"]["implemented"]
    assert not reverse["execution"]["implemented"]
    assert not default["execution"]["motion_authorized"]
    assert not reverse["execution"]["motion_authorized"]


def write_registry(root, document):
    (root / "config").mkdir(exist_ok=True)
    (root / "config/arms.json").write_text(json.dumps(document))


@pytest.mark.parametrize("field,value", [
    ("workspace_section", "right"), ("urdf_prefix", "right"),
    ("controller_config", "../outside.yaml"),
    ("controller_config", "/absolute.yaml"),
    ("controller_config", "config/trossen/follower.yaml"),
    ("profile_ip_field", "follower_ip"), ("control_role", None),
])
def test_invalid_registry_cannot_reassign_identity(tmp_path, field, value):
    document = json.loads((REPO / "config/arms.json").read_text())
    document["arms"]["left"][field] = value
    write_registry(tmp_path, document)
    with pytest.raises(ValueError):
        arms.load(tmp_path)


def test_missing_registry_does_not_invent_a_default(tmp_path):
    with pytest.raises(ValueError, match="cannot read arm registry"):
        arms.require_current_executor(tmp_path, arms.select_roles())


def test_registry_edits_cannot_enable_a_new_executor(tmp_path):
    document = json.loads((REPO / "config/arms.json").read_text())
    document["arms"]["left"]["control_role"] = "second-arm"
    write_registry(tmp_path, document)
    result = arms.teleop_plan(tmp_path, arms.select_roles())
    assert not result["execution"]["implemented"]
    with pytest.raises(ValueError, match="physical binding"):
        arms.require_current_executor(tmp_path, arms.select_roles())


def cli(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TATBOT_")}
    return subprocess.run(
        [sys.executable, "-S", str(REPO / "scripts/lib/tatbot_cli"), *args],
        cwd="/tmp", env=env, text=True, capture_output=True, timeout=10,
    )


def test_plan_from_bare_python_has_no_hardware_or_tool_requirement():
    result = cli("teleop", "plan", "--leader", "right", "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "tatbot.teleop-assignment/1"
    assert report["assignments"]["follower"]["id"] == "left"
    assert not report["execution"]["implemented"]


@pytest.mark.parametrize("flags,code,reason", [
    (["--leader", "right"], 3, "Reverse teleop execution"),
    (["--follower=left"], 3, "Reverse teleop execution"),
    (["--leader", "left", "--follower", "left"], 2, "different physical arms"),
])
def test_start_refuses_before_routing_profile_and_tool_resolution(flags, code, reason):
    result = cli("teleop", "start", *flags, "--json")
    assert result.returncode == code, result.stderr
    assert reason in result.stderr
    assert json.loads(result.stderr)["schema"]
    assert not result.stdout


@pytest.fixture
def launcher(tmp_path):
    """Stop at the first post-selection source; no real launcher helpers run."""
    lib = tmp_path / "scripts/lib"
    lib.mkdir(parents=True)
    (lib / "tatbot_cli").symlink_to(REPO / "scripts/lib/tatbot_cli", target_is_directory=True)
    (lib / "cli_hint.sh").write_text('echo reached-existing-launcher-gates; exit 42\n')
    wrapper = tmp_path / "scripts/teleop_start.sh"
    wrapper.write_text((REPO / "scripts/teleop_start.sh").read_text())
    write_registry(tmp_path, json.loads((REPO / "config/arms.json").read_text()))
    return wrapper


@pytest.mark.parametrize("flags,code", [
    ([], 42), (["--leader", "left", "--follower=right"], 42),
    (["--leader=right"], 3), (["--follower", "left"], 3),
    (["--leader", "right", "--follower", "right"], 2),
    (["--leader"], 2), (["--leader=unknown"], 2),
    (["--leader="], 2), (["--follower", ""], 2),
])
def test_direct_launcher_guard_precedes_all_existing_gates(launcher, flags, code):
    result = subprocess.run(["bash", str(launcher), *flags], text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == code, result.stderr
    assert ("reached-existing-launcher-gates" in result.stdout) == (code == 42)


def test_default_cli_launch_argv_preserved():
    from tatbot_cli import cli as cli_module
    from tatbot_cli.registry import Ctx, find

    command = find("teleop", "start")
    parser = cli_module.build_noun_parser("teleop")
    ctx = Ctx(repo=REPO, node="test", ee_tool="fixture-pen")
    ns = parser.parse_args(["start"])
    plan = command.run(ctx, ns, ["--damping", "0.1"])
    assert plan.argv == [str(REPO / "scripts/teleop_start.sh"), "--ee-tool",
                         "fixture-pen", "--damping", "0.1"]
    explicit = command.run(ctx, parser.parse_args(["start", "--leader", "left"]), [])
    assert explicit.argv[-2:] == ["--leader", "left"]


def test_wrist_mode_assigns_right_leader_without_swapping_hardware(launcher):
    roles = arms.select_roles('right')
    arms.require_current_executor(REPO, roles, wrist_calibration=True)
    with pytest.raises(ValueError, match='right leader and left follower'):
        arms.require_current_executor(REPO, arms.select_roles(), wrist_calibration=True)
    result = subprocess.run(['bash', str(launcher), '--wrist-calibration'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 42, result.stderr
    result = subprocess.run(['bash', str(launcher), '--wrist-calibration', '--leader', 'left'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode in (2, 3) and 'reached-existing' not in result.stdout


def test_wrist_cli_carries_mode_and_correct_physical_roles():
    from tatbot_cli import cli as cli_module
    from tatbot_cli.registry import Ctx, find

    command = find('teleop', 'start')
    parser = cli_module.build_noun_parser('teleop')
    ctx = Ctx(repo=REPO, node='test', ee_tool='lutin-ballpoint-dot')
    ns = parser.parse_args(['start', '--wrist-calibration'])
    assert command.validate(ctx, command, ns) is None
    command.prepare(ctx, command, ns)
    plan = command.run(ctx, ns, [])
    assert plan.argv[-5:] == ['--wrist-calibration', '--leader', 'right', '--follower', 'left']
