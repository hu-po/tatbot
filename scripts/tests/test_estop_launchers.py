import subprocess
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
        "scripts/record_session.sh",
        "scripts/il_teleop.sh",
        "scripts/il_record.sh",
        "scripts/il_rollout.sh",
        "scripts/il_rollout_async.sh",
        "scripts/il_tune.sh",
    ):
        text = _text(path)
        assert "scripts/lib/estop_guard.sh" in text, path
        assert "estop_guard::reject_overrides" in text, path


def test_leader_follower_and_recovery_are_all_required():
    for path in ("scripts/il_teleop.sh", "scripts/il_record.sh"):
        text = _text(path)
        assert "--robot.estop_required=true" in text
        assert "--teleop.estop_required=true" in text
    for path in ("scripts/il_rollout.sh", "scripts/il_rollout_async.sh"):
        assert "--robot.estop_required=true" in _text(path)
    # The device comes from the hardware profile (profile_env exports it);
    # required=True is the safety property this test pins.
    text = _text("scripts/il_recover_arm.sh")
    assert 'acquire_estop(os.environ["TATBOT_ESTOP_DEVICE"], required=True)' in text
    assert "profile_env::require" in text


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


def test_cpp_and_calibration_launcher_chain_uses_the_deployed_device():
    teleop = _text("cpp/teleop/wxai_teleop.cpp")
    assert "bool estop_required = true" in teleop
    friction = _text("cpp/teleop/friction_tune.cpp")
    # Device comes from the hardware profile env; required monitoring stays on.
    assert 'std::getenv("TATBOT_ESTOP_DEVICE")' in friction
    assert "estop_device, true, estop_state" in friction
    # The calibration sweep no longer prints a raw wxai_teleop command. It
    # delegates to the canonical launcher, so keep the safety assertion on the
    # whole chain instead of requiring the device flag in the caller's hint.
    calibration = _text("scripts/vision/calib_sweep.sh")
    assert "--ee-tool" in calibration and "teleop start" in calibration
    launcher = _text("scripts/teleop_start.sh")
    assert "scripts/lib/estop_guard.sh" in launcher
    assert "estop_guard::reject_overrides" in launcher
    assert '--estop "$TATBOT_ESTOP_DEVICE"' in launcher
