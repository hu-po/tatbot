import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).parents[2]


def _text(path: str) -> str:
    return (REPO / path).read_text()


def test_guard_rejects_every_safety_override_surface():
    guard = REPO / "scripts" / "lib" / "estop_guard.sh"
    for argument in (
        "--no-estop",
        "--estop",
        "--estop=/tmp/fake",
        "--robot.estop_required=false",
        "--robot.estop_device=",
        "--teleop.estop_required=false",
        "--teleop.estop_device=",
    ):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; estop_guard::reject_overrides "$2"',
                "bash",
                str(guard),
                argument,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, argument
    safe = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; estop_guard::reject_overrides --ff-gain 0.1',
            "bash",
            str(guard),
        ]
    )
    assert safe.returncode == 0


def test_every_production_launcher_sources_the_guard():
    for path in (
        "scripts/il_rollout_async.sh",
    ):
        text = _text(path)
        assert "scripts/lib/estop_guard.sh" in text, path
        assert "estop_guard::reject_overrides" in text, path


def test_follower_and_recovery_are_all_required():
    assert "--robot.estop_required=true" in _text("scripts/il_rollout_async.sh")
    # The device comes from the hardware profile (profile_env exports it);
    # required=True is the safety property this test pins.
    # The recovery launcher hands the profile's device to the native landing
    # binary, which fails closed: no --estop, no landing.
    text = _text("scripts/il_recover_arm.sh")
    assert '--estop "$TATBOT_ESTOP_DEVICE"' in text
    assert "profile_env::require" in text
    assert "uv run" not in text and "lerobot" not in text.lower().replace("lerobot plugins", "")
    native = _text("cpp/teleop/arm_recover.cpp")
    assert "tatbot::estop::Monitor>(opt.estop_device, true, g_estop)" in native
    assert '"--estop DEV is required"' in native


def _fake_recover(path, body):
    """A stand-in for cpp/teleop/arm_recover: <ip> <role> --staged .. --golden .. --estop .."""
    path.write_text("#!/usr/bin/env bash\nip=$1; role=$2; shift 2\n" + body)
    path.chmod(0o755)


def test_both_arm_recovery_stops_after_failed_follower(tmp_path):
    fake = tmp_path / "arm_recover"
    _fake_recover(fake,
        'printf \'%s %s\\n\' "$role" "$ip"\n'
        'if [[ -n ${FAIL_ROLE:-} && $role == "$FAIL_ROLE" ]]; then exit 1; fi\n')
    profile = json.loads(_text("config/profiles/tatbot.json"))["driver"]
    env = dict(
        os.environ,
        TATBOT_ARM_RECOVER_BIN=str(fake),
        TATBOT_PROFILE="tatbot",
        TATBOT_VIA_CLI="1",
        TATBOT_RUNLOG="0",
    )

    result = subprocess.run(
        [str(REPO / "scripts/il_recover_arms.sh")], capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    expected = [
        f"follower {profile['follower_ip']}",
        f"leader {profile['leader_ip']}",
    ]
    assert result.stdout.splitlines() == expected

    result = subprocess.run(
        [str(REPO / "scripts/il_recover_arms.sh")],
        capture_output=True,
        text=True,
        env={**env, "FAIL_ROLE": "follower"},
    )
    assert result.returncode == 1
    assert result.stdout.splitlines() == expected[:1]
    assert "leader recovery was NOT started" in result.stderr


def test_single_arm_launcher_passes_staged_pose_golden_and_estop(tmp_path):
    """The native binary gets the golden staged pose (read from tatbot.yaml by
    the stdlib parser, never a literal copy), the role's golden, and the
    profile's e-stop device — in that spelling."""
    fake = tmp_path / "arm_recover"
    _fake_recover(fake, 'printf \'%s\\n\' "$ip" "$role" "$@"\n')
    env = dict(os.environ, TATBOT_ARM_RECOVER_BIN=str(fake), TATBOT_PROFILE="tatbot",
               TATBOT_VIA_CLI="1", TATBOT_RUNLOG="0")
    result = subprocess.run([str(REPO / "scripts/il_recover_arm.sh"), "192.0.2.7", "leader"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["192.0.2.7", "leader"]
    args = dict(zip(lines[2::2], lines[3::2], strict=True))
    pose = [float(v) for v in args["--staged"].split(",")]
    import tool_spec
    assert pose == tool_spec.staged_positions(REPO)
    assert args["--golden"] == str(REPO / "config/trossen/leader.yaml")
    profile = json.loads(_text("config/profiles/tatbot.json"))["driver"]
    assert args["--estop"] == profile["estop_device"]

    # A refusal before any command (e-stop engaged: 3, controller never
    # answered: 5, driver busy: 6, a joint past its limits: 7) passes through
    # without the "arm state is UNKNOWN" warning; a failure keeps it.
    for code, unknown in ((3, False), (5, False), (6, False), (7, False), (1, True)):
        _fake_recover(fake, f"exit {code}\n")
        result = subprocess.run([str(REPO / "scripts/il_recover_arm.sh"), "192.0.2.7", "leader"],
                                capture_output=True, text=True, env=env)
        assert result.returncode == code
        assert ("arm state is UNKNOWN" in result.stderr) is unknown


def test_native_deadline_ends_hung_child_and_never_starts_leader(tmp_path):
    # Exercise the real launcher and GNU timeout with a dummy process. The
    # fake timeout only shortens the fixed production budget for this test.
    native_timeout = shutil.which('timeout')
    assert native_timeout
    fake_timeout = tmp_path / 'timeout'
    fake_timeout.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$TRACE_ARGS"\n'
        'args=(); for arg in "$@"; do [[ $arg == 45s ]] && arg=0.2s; args+=("$arg"); done\n'
        'exec "$NATIVE_TIMEOUT" "${args[@]}"\n')
    fake_timeout.chmod(0o755)
    dummy = tmp_path / 'arm_recover'
    _fake_recover(dummy, 'echo "dummy blocked SDK"\necho $$ > "$TRACE_PID"\nexec sleep 60\n')
    env = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', TATBOT_PROFILE='tatbot',
               TATBOT_ARM_RECOVER_BIN=str(dummy),
               TATBOT_VIA_CLI='1', TATBOT_RUNLOG='1', TATBOT_LOG_ROOT=str(tmp_path / 'logs'),
               NATIVE_TIMEOUT=native_timeout,
               TRACE_ARGS=str(tmp_path / 'args'), TRACE_PID=str(tmp_path / 'pid'))
    started = time.monotonic()
    result = subprocess.run([str(REPO / 'scripts/il_recover_arms.sh')], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 124
    assert time.monotonic() - started < 4
    assert result.stdout.count('dummy blocked SDK') == 1
    assert 'arm state is UNKNOWN' in result.stdout and 'leader recovery was NOT started' in result.stderr
    args = (tmp_path / 'args').read_text().splitlines()
    assert args[:4] == ['--foreground', '--signal=TERM', '--kill-after=2s', '45s']
    assert args[4] == str(dummy)
    profile = json.loads(_text("config/profiles/tatbot.json"))["driver"]
    assert args[5:7] == [profile['follower_ip'], 'follower']
    assert args[-2:] == ['--estop', profile['estop_device']]
    assert not Path('/proc', (tmp_path / 'pid').read_text().strip()).exists()
    logs = sorted({path.resolve() for path in (tmp_path / 'logs' / 'arm-recover').glob('*/meta.json')})
    assert len(logs) == 1
    assert json.loads(logs[0].read_text())['exit_code'] == 124
    assert 'arm state is UNKNOWN' in (logs[0].parent / 'console.log').read_text()


def test_policy_launcher_fails_closed_on_floor_estop_and_surviving_client():
    launcher = _text("scripts/il_rollout_async.sh")
    shield = _text("scripts/il_client_shield.py")

    assert "--robot.require_z_floor=true" in launcher
    assert "--robot.abort_on_estop=true" in launcher
    assert '--robot.max_joint_velocity="$TARGET_VELOCITY"' in launcher
    assert '--robot.controller_velocity_limit="$CONTROLLER_VELOCITY"' in launcher
    assert "E-stop event makes the rollout a failure" in launcher
    assert "STATUS=137" in launcher
    assert "PR_SET_PDEATHSIG" in shield


def test_cpp_and_teleop_launcher_chain_uses_the_deployed_device():
    teleop = _text("cpp/teleop/wxai_teleop.cpp")
    assert "bool estop_required = true" in teleop
    friction = _text("cpp/teleop/friction_tune.cpp")
    # Device comes from the hardware profile env; required monitoring stays on.
    assert 'std::getenv("TATBOT_ESTOP_DEVICE")' in friction
    assert "estop_device, true, estop_state" in friction
    launcher = _text("scripts/teleop_start.sh")
    assert "scripts/lib/estop_guard.sh" in launcher
    assert "estop_guard::reject_overrides" in launcher
    assert '--estop "$TATBOT_ESTOP_DEVICE"' in launcher
