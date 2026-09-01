from __future__ import annotations

import pytest
from lerobot_robot_tatbot.motion_safety import MotionSafetyError, MotionSafetyWatchdog


def watchdog(**overrides) -> MotionSafetyWatchdog:
    values = {
        "velocity_limit": 2.5,
        "acceleration_limit": 80.0,
        "reversal_window_s": 1.0,
        "reversal_min_velocity": 0.2,
        "reversal_limit": 4,
        "clamp_grace_s": 0.0,
        "clamp_window_s": 1.0,
        "clamp_fraction": 0.8,
        "clamp_min_samples": 20,
        "overforce_limit": 9.0,
        "overforce_window_s": 0.5,
        "overforce_fraction": 0.5,
        "overforce_min_samples": 8,
    }
    values.update(overrides)
    return MotionSafetyWatchdog(**values)


def update(
    guard: MotionSafetyWatchdog,
    now: float,
    velocity: float = 0.0,
    effort: float = 0.0,
    clamped: bool = False,
) -> None:
    guard.update(
        now=now,
        velocities=[velocity] * 6,
        external_efforts=[effort] * 6,
        clamped=clamped,
    )


def test_successful_envelope_does_not_abort() -> None:
    guard = watchdog()
    for sample in range(61):
        update(
            guard,
            sample / 30,
            velocity=1.5 if sample % 20 else -0.1,
            effort=5.2,
            clamped=sample < 10,
        )


def test_measured_velocity_and_acceleration_abort() -> None:
    guard = watchdog(acceleration_limit=0)
    update(guard, 0.0)
    with pytest.raises(MotionSafetyError, match="velocity") as caught:
        update(guard, 1 / 30, velocity=5.31)
    assert caught.value.code == "measured_velocity"

    guard = watchdog(velocity_limit=0)
    update(guard, 0.0)
    with pytest.raises(MotionSafetyError, match="acceleration") as caught:
        update(guard, 1 / 30, velocity=3.0)
    assert caught.value.code == "measured_acceleration"


def test_repeated_fast_reversals_abort() -> None:
    guard = watchdog(velocity_limit=0, acceleration_limit=0)
    for sample, value in enumerate((0.3, -0.3, 0.3, -0.3)):
        update(guard, sample * 0.2, velocity=value)
    with pytest.raises(MotionSafetyError, match="reversed") as caught:
        update(guard, 0.8, velocity=0.3)
    assert caught.value.code == "measured_reversals"


def test_sustained_clamp_ignores_startup_then_aborts() -> None:
    guard = watchdog(clamp_grace_s=2.5)
    for sample in range(75):
        update(guard, sample / 30, clamped=True)
    # The grace window itself contributes no clamp evidence.
    with pytest.raises(MotionSafetyError, match="clamp") as caught:
        for sample in range(75, 106):
            update(guard, sample / 30, clamped=True)
    assert caught.value.code == "sustained_clamp"


def test_rolling_overforce_catches_alternating_load() -> None:
    guard = watchdog()
    # No ten consecutive samples are high; 9/16 in a half-second are.
    with pytest.raises(MotionSafetyError, match="effort") as caught:
        for sample in range(16):
            update(guard, sample / 30, effort=12.0 if sample % 2 == 0 else 4.0)
    assert caught.value.code == "rolling_overforce"


def test_carriage_effort_is_not_part_of_arm_watchdog() -> None:
    guard = watchdog()
    for sample in range(31):
        guard.update(
            now=sample / 30,
            velocities=[0.0] * 6,
            external_efforts=[5.15] * 6,
            clamped=False,
        )
