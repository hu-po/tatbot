"""The `tatbot` CLI: grammar, gates, routing, schema, orphan scan. Stdlib only."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "scripts" / "lib"
sys.path.insert(0, str(LIB))

from tatbot_cli import gates, nodes, selfcheck  # noqa: E402
from tatbot_cli.registry import MOTION_AUTO, MOTION_HUMAN, TIERS, all_verbs  # noqa: E402

# Node names come from config/nodes.json, so the suite exercises THIS
# deployment's fleet without freezing its hostnames into the tests.
ARM = nodes.example_node("arm")
VIEWER = nodes.example_node("viewer")
TRAIN = nodes.example_node("train")
INKGEN = nodes.example_node("inkgen")
CAMS = nodes.example_node("poe-cameras")


def ssh_target(node: str) -> str:
    return nodes.ssh_target(nodes.load(REPO), node) or ""


INKGEN_API = f"http://{ssh_target(nodes.example_node('inkgen')).split(chr(64))[-1]}:8600"


def arm_ip(which: str) -> str:
    import json as _json
    d = _json.loads((REPO / "config/profiles/tatbot.json").read_text())["driver"]
    return str(d[f"{which}_ip"])


TOOL = "picosecond-laser-pen"


def tatbot(*args: str, node: str = VIEWER, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ, TATBOT_NODE=node, TATBOT_TRAIN_ROOT="/nonexistent")
    e.pop("TATBOT_EE_TOOL", None)
    e.update(env or {})
    return subprocess.run([sys.executable, str(LIB / "tatbot_cli"), *args], capture_output=True, text=True, env=e, timeout=60)


# --- grammar -----------------------------------------------------------------------


def test_root_help_lists_every_noun_in_order():
    r = tatbot("--help")
    assert r.returncode == 0
    nouns = [v.noun for v in all_verbs()]
    seen = list(dict.fromkeys(nouns))
    positions = [r.stdout.index(f"\n  {n:<9}") for n in seen]
    assert positions == sorted(positions)


def test_no_args_is_a_usage_error():
    assert tatbot().returncode == 2


def test_unknown_noun_is_exit_2():
    assert tatbot("bogus").returncode == 2


def test_every_verb_has_one_tier_and_an_example_that_dry_runs():
    for v in all_verbs():
        assert v.tier in TIERS, v.name
    assert selfcheck.check_help_and_dry_run(REPO) == []


def test_bare_global_flags_work_after_the_verb():
    a = tatbot("--dry-run", "logs", "last", "rollout")
    b = tatbot("logs", "last", "rollout", "--dry-run")
    assert a.returncode == b.returncode == 0
    assert a.stdout == b.stdout


def test_dashdash_passthrough_reaches_the_launcher_untouched():
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "record", "d", "t", "-n", "2", "--", "--robot.z_floor_m=0.04", "-x", node=ARM)
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout)["argv"]
    assert argv[-2:] == ["--robot.z_floor_m=0.04", "-x"]
    assert argv[1:4] == ["d", "t", "2"]


def test_rollout_bench_uses_lerobot_project_environment():
    r = tatbot(
        "--dry-run",
        "--json",
        "rollout",
        "bench",
        "wire",
        "--",
        "act_rgb",
        "--policy",
        "/models/act",
    )
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout)["argv"]
    script = str(REPO / "scripts/eval/wire_bench.py")
    assert argv[0] != "python3"
    script_index = argv.index(script)
    assert argv[script_index + 1 :] == ["act_rgb", "--policy", "/models/act"]


def test_train_manifest_render_uses_the_wrapped_tools_default_mode():
    r = tatbot(
        "--dry-run",
        "--json",
        "train",
        "manifest",
        "config/training/example.json",
        "candidate",
        "--render",
        node=TRAIN,
    )
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout)["argv"]
    env = json.loads(r.stdout)["env"]
    assert argv[-2:] == ["config/training/example.json", "candidate"]
    assert "--render" not in argv
    assert env["TATBOT_NODE"] == TRAIN


def test_train_offline_eval_uses_the_pinned_training_environment(tmp_path: Path):
    train_root = tmp_path / "il-train"
    r = tatbot(
        "--dry-run",
        "--json",
        "train",
        "offline-eval",
        "--",
        "--help",
        node=TRAIN,
        env={"TATBOT_TRAIN_ROOT": str(train_root)},
    )
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [
        str(train_root / ".venv" / "bin" / "python"),
        str(REPO / "scripts/train/offline_eval.py"),
        "--help",
    ]
    assert "pinned training environment" in plan["notes"][0]


# --- gates ---------------------------------------------------------------------------


def test_estop_override_list_matches_estop_guard_sh():
    text = (REPO / "scripts/lib/estop_guard.sh").read_text()
    m = re.search(r"^\s*(--no-estop\|.*?)\)\s*$", text, re.M)
    assert m, "could not find the case pattern in estop_guard.sh"
    patterns = m.group(1).split("|")
    want = set()
    for p in patterns:
        want.add(p.rstrip("*").removesuffix("="))
    have = {pat for pat, _ in gates.ESTOP_OVERRIDES}
    assert have == want


@pytest.mark.parametrize("arg", ["--no-estop", "--estop", "--estop=/tmp/x", "--robot.estop_required=false",
                                 "--robot.estop_device=", "--teleop.estop_required=false"])
def test_motion_verbs_refuse_estop_overrides_with_exit_3(arg):
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "record", "d", "t", "--", arg, node=ARM)
    assert r.returncode == 3
    assert json.loads(r.stderr)["gate"] == "estop_guard"


def test_safe_passthrough_is_not_refused():
    r = tatbot("--dry-run", "--ee-tool", TOOL, "record", "d", "t", "--", "--ff-gain", "0.1", node=ARM)
    assert r.returncode == 0, r.stderr


def test_tool_must_be_stated_for_motion_verbs():
    r = tatbot("--dry-run", "--json", "record", "d", "t", node=ARM)
    assert r.returncode == 3
    assert json.loads(r.stderr)["gate"] == "ee_tool"
    r = tatbot("--dry-run", "--json", "--ee-tool", "no-such-tool", "record", "d", "t", node=ARM)
    assert r.returncode == 3


def test_tool_from_environment_is_accepted():
    r = tatbot("--dry-run", "--json", "record", "d", "t", node=ARM, env={"TATBOT_EE_TOOL": TOOL})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["env"]["TATBOT_EE_TOOL"] == TOOL


def test_autonomous_verbs_need_a_literal_nonce():
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "rollout", "run", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "arm_gate"
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "rollout", "run", "--nonce", "bad nonce!", node=ARM)
    assert r.returncode == 3
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "rollout", "run", "--nonce", "ok-1", node=ARM)
    assert r.returncode == 0, r.stderr


def test_every_nonce_verb_is_motion_auto_and_every_motion_verb_needs_a_tool():
    for v in all_verbs():
        if v.nonce:
            assert v.tier == MOTION_AUTO, v.name
        # Every launcher that puts a tool on the paper states the tool; recover only lands the arm.
        if v.tier in (MOTION_AUTO, MOTION_HUMAN) and v.wraps and v.wraps[0].startswith("scripts/il_") \
                and "recover" not in v.wraps[0]:
            assert v.needs_tool, v.name


def test_dry_run_never_writes_the_arm_token(tmp_path, monkeypatch):
    token = tmp_path / "token"
    monkeypatch.setattr(gates, "ARM_TOKEN", token)
    r = tatbot("--dry-run", "--ee-tool", TOOL, "rollout", "run", "--nonce", "abc", node=ARM)
    assert r.returncode == 0
    assert not token.exists()


def test_write_nonce_is_what_the_launcher_reads(tmp_path, monkeypatch):
    token = tmp_path / "token"
    monkeypatch.setattr(gates, "ARM_TOKEN", token)
    gates.write_nonce("sleepy-1")
    assert token.read_text() == "sleepy-1\n"


def test_gates_come_from_what_a_verb_declares_not_its_tier():
    """A verb never advertises a gate its launcher does not run (dip used to
    print `single-use arm nonce` from its tier while il_dip.sh had no gate)."""
    from tatbot_cli.registry import GATE_DIP, GATE_NONCE, find
    dip = find("dip", "")
    assert dip.nonce and dip.nonce_exempt == ("plan", "connect_only")
    assert any(g.startswith(GATE_NONCE) for g in dip.gates)
    record = find("record", "")
    assert record.dip_hook and not record.nonce
    assert GATE_DIP in record.gates and not any(g.startswith(GATE_NONCE) for g in record.gates)
    for v in all_verbs():
        assert (GATE_NONCE in " ".join(v.gates)) == (v.nonce or v.dip_hook), v.name


# --- dip: plan / rehearse / yes ---------------------------------------------------------

BALLPOINT, NEEDLE = "lutin-ballpoint-dot", "lutin-3rl-bugpin"


def test_dip_plan_and_connect_only_command_nothing_and_need_no_nonce():
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "dip", "--plan", node=ARM)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][-1] == "--dry-run"
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "dip", "--connect-only", node=ARM)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][-1] == "--connect-only"
    sh = (REPO / "scripts/il_dip.sh").read_text()
    assert "--dry-run|--connect-only) moves=0" in sh


def test_dip_rehearse_is_a_moving_dip_with_a_rehearsal_tool():
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "dip", "--rehearse", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "arm_gate"
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "dip", "--rehearse", "--nonce", "r-1", node=ARM)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"][-1] == "--yes" and "--dry-run" not in plan["argv"]
    assert any("rehearsal" in n for n in plan["notes"])
    r = tatbot("--dry-run", "--json", "--ee-tool", NEEDLE, "dip", "--rehearse", "--nonce", "r-2", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "ink_mode"


def test_dip_yes_with_a_real_needle_needs_allow_real():
    r = tatbot("--dry-run", "--json", "--ee-tool", NEEDLE, "dip", "--yes", "--nonce", "y-1", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "ink_mode"
    r = tatbot("--dry-run", "--json", "--ee-tool", NEEDLE, "dip", "--yes", "--allow-real", "--nonce", "y-2", node=ARM)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][-2:] == ["--yes", "--allow-real"]


def test_dip_refuses_a_tool_that_never_dips_and_a_bare_call():
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "dip", "--yes", "--nonce", "n-1", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "ink_mode"
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "dip", "--nonce", "n-2", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "dip"


def test_record_with_dip_is_autonomous_motion():
    """--dip runs a scripted dip before the human takes the leader arm: nonce required, refused over --on."""
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "record", "d", "t", "--dip", node=ARM)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "arm_gate"
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "record", "d", "t", "--dip", "--nonce", "d-1", node=ARM)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][-1] == "--dip"
    r = tatbot("--dry-run", "--json", "--on", ARM, "--ee-tool", BALLPOINT, "record", "d", "t", "--dip", "--nonce", "d-2", node=VIEWER)
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "arm_gate"
    # without --dip, record stays a human-on-the-leader verb: hops, no nonce
    r = tatbot("--dry-run", "--json", "--on", ARM, "--ee-tool", BALLPOINT, "record", "d", "t", "--no-ink", node=VIEWER)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][0] == "ssh"


def test_record_dip_writes_the_nonce_the_dip_will_consume(tmp_path, monkeypatch):
    token = tmp_path / "token"
    monkeypatch.setattr(gates, "ARM_TOKEN", token)
    r = tatbot("--dry-run", "--ee-tool", BALLPOINT, "record", "d", "t", "--dip", "--nonce", "d-3", node=ARM)
    assert r.returncode == 0 and not token.exists()  # dry-run never arms


# --- ink: one verb per subcommand, honest tiers --------------------------------------------


def test_ink_verbs_have_honest_tiers():
    from tatbot_cli.registry import MUTATES_CONFIG, OFFLINE, REMOTE, verbs_of
    tiers = {v.verb: v.tier for v in verbs_of("ink")}
    assert tiers["sync"] == REMOTE
    for w in ("load", "dump", "bottle", "cartridge", "caps", "reconcile"):
        assert tiers[w] == MUTATES_CONFIG, w
    for r in ("status", "ledger", "fit", "plan", "mise-en-place", "session", "session start", "session end",
              "session rebuild", "weigh"):
        assert tiers[r] == OFFLINE, r
    assert all(v.wraps[0] == "scripts/ink.py" for v in verbs_of("ink"))


def test_ink_session_start_takes_the_stated_tool():
    r = tatbot("--dry-run", "--json", "--ee-tool", NEEDLE, "ink", "session", "start", "--need-ul", "5")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][2:] == ["session", "--ee-tool", NEEDLE, "start", "--need-ul", "5"]
    r = tatbot("--dry-run", "--json", "ink", "session", "start")
    assert r.returncode == 3 and json.loads(r.stderr)["gate"] == "ee_tool"
    r = tatbot("--dry-run", "--json", "ink", "session")
    assert r.returncode == 0 and json.loads(r.stdout)["argv"][-2:] == ["session", "status"]


def test_ink_options_pass_through_in_order():
    r = tatbot("--dry-run", "--json", "ink", "load", "inkcap_right_medium_0", "nighthawk_black", "--ul", "400", "--bottle", "b1")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["argv"][2:] == ["load", "inkcap_right_medium_0", "nighthawk_black", "--ul", "400", "--bottle", "b1"]
    r = tatbot("--dry-run", "--json", "ink", "bottle", "add", "nb_01", "--ink", "nighthawk_black", "--ml", "30")
    assert json.loads(r.stdout)["argv"][2:] == ["bottle", "add", "nb_01", "--ink", "nighthawk_black", "--ml", "30"]


# --- inkmap: dev, build, check, deploy --------------------------------------------


def test_inkmap_verbs_have_honest_tiers():
    from tatbot_cli.registry import OFFLINE, REMOTE, verbs_of
    tiers = {v.verb: v.tier for v in verbs_of("inkmap")}
    assert tiers["dev"] == OFFLINE
    assert tiers["build"] == OFFLINE
    assert tiers["check"] == OFFLINE
    assert tiers["deploy"] == REMOTE


def test_inkmap_dev_and_build_options_pass_through():
    r = tatbot("--dry-run", "--json", "inkmap", "dev", "--host", "0.0.0.0", "--port", "5000", "--api", INKGEN_API)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [str(REPO / "scripts/inkmap_dev.sh"), "--host", "0.0.0.0", "--port", "5000", "--api", INKGEN_API]
    assert "http://0.0.0.0:5000/" in plan["notes"][0]
    assert INKGEN_API in plan["notes"][1]

    r = tatbot("--dry-run", "--json", "inkmap", "build")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == ["npm", "run", "build"]
    assert plan["cwd"].endswith("web/inkmap")


def test_inkmap_check_delegates_to_scripts_check():
    r = tatbot("--dry-run", "--json", "inkmap", "check")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"][-2:] == [str(REPO / "scripts/check"), "web"]


def test_inkmap_deploy_constructs_deploy_script_argv():
    r = tatbot("--dry-run", "--json", "inkmap", "deploy", "--space", "custom/space", "--no-build")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["tier"] == "remote"
    assert plan["argv"][-4:] == ["--space", "custom/space", "--no-build", "--dry-run"]
    assert str(REPO / "scripts/inkmap_deploy.sh") in plan["argv"][0]
    assert "https://huggingface.co/spaces/custom/space" in plan["notes"][0]


# --- inkgen: start, stop, logs, serve, status, deploy -----------------------


def test_inkgen_verbs_have_honest_tiers():
    from tatbot_cli.registry import OFFLINE, REMOTE, SENSOR, verbs_of
    tiers = {v.verb: v.tier for v in verbs_of("inkgen")}
    assert tiers["start"] == OFFLINE
    assert tiers["stop"] == OFFLINE
    assert tiers["logs"] == OFFLINE
    assert tiers["serve"] == OFFLINE
    assert tiers["status"] == SENSOR
    assert tiers["deploy"] == REMOTE


def test_inkgen_serve_options_pass_through():
    r = tatbot("--dry-run", "--json", "inkgen", "serve", "--port", "8600", "--model", "custom/model", "--", "--cpu", node=INKGEN)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [str(REPO / "scripts/inkgen_serve.sh"), "--port", "8600", "--host", "0.0.0.0", "--model", "custom/model", "--cpu"]
    assert "foreground" in plan["notes"][0]


def test_inkgen_ctl_verbs_construct_argv():
    r = tatbot("--dry-run", "--json", "inkgen", "start", "--port", "8600", node=INKGEN)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [str(REPO / "scripts/inkgen_ctl.sh"), "start", "--port", "8600"]

    r = tatbot("--dry-run", "--json", "inkgen", "stop", node=INKGEN)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [str(REPO / "scripts/inkgen_ctl.sh"), "stop"]

    r = tatbot("--dry-run", "--json", "inkgen", "logs", "-n", "50", node=INKGEN)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [str(REPO / "scripts/inkgen_ctl.sh"), "logs", "-n", "50"]


def test_inkgen_deploy_constructs_deploy_script_argv():
    r = tatbot("--dry-run", "--json", "inkgen", "deploy", "--space", "custom/space")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["tier"] == "remote"
    assert plan["argv"] == [str(REPO / "scripts/inkgen_deploy.sh"), "--space", "custom/space", "--dry-run"]
    assert "custom/space" in plan["notes"][0]


def test_inkgen_status_constructs_curl_argv():
    r = tatbot("--dry-run", "--json", "inkgen", "status", "--url", "http://127.0.0.1:8600/")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["tier"] == "sensor"
    assert plan["argv"] == ["curl", "-fsS", "--max-time", "30", "http://127.0.0.1:8600/api/health"]
    assert "read-only GET" in plan["notes"][0]

    r = tatbot("--dry-run", "--json", "inkgen", "status", "--space")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == ["curl", "-fsS", "--max-time", "30", "https://hu-po-inkgen.hf.space/api/health"]


def test_inkgen_role_auto_hop_and_sync():
    r = tatbot("--dry-run", "--json", "inkgen", "start", node=VIEWER)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["hop"] == INKGEN
    assert plan["argv"][0] == "ssh"
    assert ssh_target(INKGEN) in plan["argv"]
    remote_cmd = plan["argv"][-1]
    assert "git pull -q --ff-only origin main" in remote_cmd
    assert "scripts/tatbot --no-hop" in remote_cmd
    assert "inkgen start" in remote_cmd

    r = tatbot("--dry-run", "--json", "inkgen", "serve", node=VIEWER)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["hop"] == INKGEN
    assert "-t" in plan["argv"]
    assert "git pull -q --ff-only origin main" in plan["argv"][-1]


# --- live: cockpit -------------------------------------------------------------------


def test_live_cockpit_requires_viewer_role_and_passes_flags():
    # Role gating: the arm node has no viewer role, the viewer node does
    r = tatbot("--dry-run", "--json", "live", "cockpit", node=ARM)
    assert r.returncode == 4
    assert json.loads(r.stderr)["gate"] == "node"

    # Default dry-run on viewer node
    r = tatbot("--dry-run", "--json", "live", "cockpit", node=VIEWER)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["tier"] == "sensor"
    assert plan["role"] == "viewer"
    assert plan["argv"] == [str(REPO / "scripts/live/cockpit.sh")]

    # Full flag options passthrough
    r = tatbot(
        "--dry-run",
        "--json",
        "live",
        "cockpit",
        "--fps",
        "4",
        "--rs-fps",
        "3",
        "--rs-scale",
        "0.5",
        "--stream",
        "main",
        "--duration",
        "1800",
        "--no-realsense",
        "--no-audio",
        "--no-teleop",
        node=VIEWER,
    )
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"] == [
        str(REPO / "scripts/live/cockpit.sh"),
        "--fps",
        "4",
        "--rs-fps",
        "3",
        "--rs-scale",
        "0.5",
        "--stream",
        "main",
        "--duration",
        "1800",
        "--no-realsense",
        "--no-audio",
        "--no-teleop",
    ]


def test_teleop_start_is_the_canonical_bare_teleop():
    r = tatbot("--dry-run", "--json", "--on", ARM, "--ee-tool", BALLPOINT, "teleop", "start", node=VIEWER)
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout)["argv"]
    assert argv[0] == "ssh" and "-t" in argv and "teleop start" in argv[-1]
    r = tatbot("--dry-run", "--json", "--ee-tool", BALLPOINT, "teleop", "start", "--touchoff", node=ARM)
    assert json.loads(r.stdout)["argv"][-1] == "--touchoff"
    sh = (REPO / "scripts/teleop_start.sh").read_text()
    assert "--touchoff) TOOL_ARGS+=(--tool-uncalibrated)" in sh
    assert 'if (arg == "--tool-uncalibrated")' in (REPO / "cpp/teleop/wxai_teleop.cpp").read_text()
    for needle in ("--telemetry-udp \"$TELEMETRY\"", 'TELEMETRY="${TATBOT_TELEMETRY_UDP:-}"',
                   '--estop "$TATBOT_ESTOP_DEVICE"', "--ee-tool \"$EE_TOOL\"",
                   "pgrep -f '[w]xai_teleop'", "runlog::run"):
        assert needle in sh, needle
    sweep = (REPO / "scripts/vision/calib_sweep.sh").read_text()
    assert "teleop start" in sweep and f"./cpp/teleop/build/wxai_teleop {arm_ip('leader')}" not in sweep


# --- routing --------------------------------------------------------------------------


def test_wrong_node_is_exit_4_with_the_on_form():
    r = tatbot("--json", "--ee-tool", TOOL, "record", "d", "t", node=VIEWER)
    assert r.returncode == 4
    err = json.loads(r.stderr)
    assert err["fix"].startswith(f"tatbot --on {ARM} ")


def test_hop_uses_the_canonical_ssh_target_and_no_hop():
    r = tatbot("--dry-run", "--json", "--on", ARM, "--ee-tool", TOOL, "teleop", "lerobot", node=VIEWER)
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout)["argv"]
    assert argv[0] == "ssh" and ssh_target(ARM) in argv and "-t" in argv
    assert "--no-hop" in argv[-1] and "--on" not in argv[-1]
    # a login shell, or ~/.local/bin/uv is not on PATH and the launcher dies with 127 after its gates
    assert argv[-1].startswith("bash -lc ") and argv[-1].count("'") >= 2


def test_hop_is_refused_for_autonomous_motion():
    r = tatbot("--dry-run", "--json", "--on", ARM, "--ee-tool", TOOL, "rollout", "run", "--nonce", "x", node=VIEWER)
    assert r.returncode == 3


def test_hop_refuses_a_node_without_a_checkout(tmp_path):
    # a synthetic node map where a node has no checkout
    fake = tmp_path / "repo"
    (fake / "config").mkdir(parents=True)
    (fake / "config" / "tools").symlink_to(REPO / "config" / "tools")
    nmap = json.loads((REPO / "config" / "nodes.json").read_text())
    nmap[CAMS]["checkout"] = None
    nmap[CAMS]["note"] = "no checkout here"
    (fake / "config" / "nodes.json").write_text(json.dumps(nmap))
    r = tatbot("--dry-run", "--json", "--on", CAMS, "vision", "track", node=VIEWER, env={"TATBOT_REPO": str(fake)})
    assert r.returncode == 2, r.stderr
    assert "checkout" in json.loads(r.stderr)["reason"] and json.loads(r.stderr)["fix"] == "no checkout here"


def test_unknown_on_node_is_exit_2():
    assert tatbot("--dry-run", "--on", "nowhere", "logs", "last").returncode == 2


def test_hostname_alias_resolves_to_the_node_name():
    from tatbot_cli import nodes
    nmap = nodes.load(REPO)
    assert nodes.this_node({TRAIN: {"hostname": "other-hostname"}}) in (TRAIN, nodes.this_node())  # env wins in CI
    import socket
    assert nodes.this_node({"x": {"hostname": socket.gethostname().split(".")[0]}}) in ("x", os.environ.get("TATBOT_NODE", "x"))
    assert TRAIN in nmap


def test_offline_verbs_run_anywhere():
    assert tatbot("--dry-run", "logs", "last", node="unknown-node").returncode == 0


# --- schema / docs / orphans ------------------------------------------------------------


def test_schema_json_round_trips_and_covers_every_verb():
    r = tatbot("schema", "--json")
    d = json.loads(r.stdout)
    assert {v["name"] for v in d["verbs"]} == {v.name for v in all_verbs()}
    assert set(d["tiers"]) == set(TIERS)
    assert "3" in d["exit_codes"]


def test_schema_markdown_has_no_trailing_spaces():
    from tatbot_cli import schema
    md = schema.as_markdown(REPO)
    for line in md.splitlines():
        assert line == line.rstrip(" "), f"Trailing spaces found in schema markdown line: {line!r}"

    r = tatbot("schema", "--md")
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        assert line == line.rstrip(" "), f"Trailing spaces found in CLI markdown line: {line!r}"


def test_explain_needs_no_positionals():
    r = tatbot("vision", "touchoff", "--explain", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["tier"] == "mutates-config"


def test_vision_touchoff_uses_vision_py_environment():
    venv_py = Path("~/.venvs/tatbot-vision/bin/python").expanduser()
    expected_interp = str(venv_py) if venv_py.exists() else "python3"
    r = tatbot("--dry-run", "--json", "--ee-tool", TOOL, "vision", "touchoff", "session_dir", "--write")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["tier"] == "mutates-config"
    assert plan["argv"] == [
        expected_interp,
        str(REPO / "scripts/il_touchoff.py"),
        "session_dir",
        "--ee-tool",
        TOOL,
        "--write",
    ]


def test_docs_cli_md_is_current():
    assert selfcheck.check_docs(REPO) == []


def test_cli_shim_is_not_orphan_or_entry_point():
    eps = selfcheck.entry_points(REPO)
    assert "scripts/tatbot" not in eps
    assert selfcheck.check_orphans(REPO) == []


def _wrapped_launchers():
    launchers = []
    for v in all_verbs():
        for w in v.wraps:
            if not w.endswith(".sh") or "/lib/" in w or (w.startswith("scripts/vision/") and w != "scripts/vision/calib_sweep.sh"):
                continue
            text = (REPO / w).read_text()
            if re.search(r"^#.*\bsource (it|this)\b", "\n".join(text.splitlines()[:20]), re.M | re.I):
                continue  # a sourced library (il_audio_record.sh), not a launcher
            launchers.append((w, v.name))
    return launchers


@pytest.mark.parametrize("script,verb_name", _wrapped_launchers())
def test_every_wrapped_shell_launcher_hints_its_verb(script, verb_name):
    verbs = {v.name for v in all_verbs()}
    missing_cli_hint_bugs = {
        "scripts/inkmap_dev.sh",
        "scripts/inkgen_serve.sh",
        "scripts/inkgen_deploy.sh",
        "scripts/inkgen_ctl.sh",
    }
    text = (REPO / script).read_text()
    m = re.search(r'cli_hint::note "tatbot ([a-z-]+(?: [a-z-]+)*)', text)
    if script in missing_cli_hint_bugs and not m:
        pytest.xfail(f"BUG: {script} wrapped by {verb_name} does not source cli_hint.sh")
    assert m, f"{script} does not source cli_hint"
    named = m.group(1)
    known = verbs | {v.noun for v in all_verbs()}
    assert any(named == k or named.startswith(k + " ") or k.startswith(named + " ") for k in known), (script, named)


def test_launcher_hint_is_silent_under_the_cli(tmp_path):
    hint = REPO / "scripts/lib/cli_hint.sh"
    loud = subprocess.run(["bash", "-c", f'source "{hint}"; cli_hint::note "tatbot x"'], capture_output=True, text=True)
    quiet = subprocess.run(["bash", "-c", f'source "{hint}"; cli_hint::note "tatbot x"'], capture_output=True, text=True,
                           env=dict(os.environ, TATBOT_VIA_CLI="1"))
    assert "tatbot x" in loud.stderr and quiet.stderr == "" and loud.returncode == quiet.returncode == 0
    r = tatbot("--dry-run", "--json", "logs", "last")
    assert json.loads(r.stdout)["env"]["TATBOT_VIA_CLI"] == "1"


def test_ee_tool_is_the_flag_every_python_tool_accepts():
    for path in ["scripts/il_tool_meta.py", "scripts/il_touchoff.py",
                 "scripts/check_tool_sync.py", "scripts/il_dip.py", "scripts/ink.py"]:
        text = (REPO / path).read_text()
        assert '"--ee-tool", "--tool-id", dest="tool_id"' in text, path
    assert "--ee-tool=*)" in (REPO / "scripts/vision/calib_sweep.sh").read_text()
    assert not (REPO / "scripts/il_compare_features.py").exists()


# --- phase 3: absorbed launchers ------------------------------------------------------


def _fake_python(dir_: Path, *, has_hub: bool) -> Path:
    py = dir_ / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\ncase \"$*\" in *huggingface_hub*) exit %d;; esac\nexit 0\n" % (0 if has_hub else 1))
    py.chmod(0o755)
    return py


def test_hub_python_prefers_the_plugin_venv_then_il_train_then_uv(tmp_path, monkeypatch):
    from tatbot_cli import interp
    repo = tmp_path / "repo"
    plugin = _fake_python(repo / "python" / "lerobot_robot_tatbot", has_hub=True)
    train = _fake_python(tmp_path / "il-train", has_hub=True)
    env = {"TATBOT_TRAIN_ROOT": str(tmp_path / "il-train"), "HOME": str(tmp_path)}
    assert interp.hub_python(repo, env=env)[0] == [str(plugin)]
    plugin.write_text("#!/bin/sh\nexit 1\n")
    assert interp.hub_python(repo, env=env)[0] == [str(train)]
    train.unlink()
    monkeypatch.setattr(interp, "uv_binary", lambda: "/opt/uv")
    argv, why = interp.hub_python(repo, env=env)
    assert argv[:2] == ["/opt/uv", "run"] and "--with" in argv and "uv" in why
    monkeypatch.setattr(interp, "uv_binary", lambda: None)
    assert interp.hub_python(repo, env=env)[0] is None


def test_data_push_dry_run_names_the_interpreter():
    r = tatbot("--dry-run", "--json", "data", "push", "--", "--root", "/x", "--repo-id", "a/b")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["argv"][-5:] == [str(REPO / "scripts/dataset_hub.py"), "push", "--root", "/x", "--repo-id", "a/b"][-5:]
    assert plan["notes"] and "interpreter" in plan["notes"][0]


def test_shims_delegate_to_the_cli():
    assert 'exec "$REPO/scripts/tatbot" data "$@"' in (REPO / "scripts/dataset_hub.sh").read_text()
    assert '/tatbot" logs "$@"' in (REPO / "scripts/tatbot-logs").read_text()
    r = subprocess.run([str(REPO / "scripts/tatbot-logs"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "last" in r.stdout


def test_logs_runs_in_process():
    r = tatbot("logs", "list", "-n", "1", "--json")
    assert r.returncode == 0, r.stderr


def test_nodes_json_agrees_with_the_network_document_when_present():
    try:
        from tatbot_cli import nodes_parity
    except ImportError:
        pytest.skip("no nodes_parity module (no fleet document for this checkout)")
    problems, skip = nodes_parity.check(REPO)
    if skip:
        pytest.skip(skip)
    assert problems == []
