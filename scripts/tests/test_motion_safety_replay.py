"""Replay flight-CSV telemetry through the measured-motion watchdog, hardware-free.

This used to be `scripts/eval/replay_motion_safety.py` behind `tatbot rollout
replay-safety` — a verb nobody ever typed. An incident regression is a test,
not an operator verb: the fixture CSVs are synthesised here in the flight
recorder's column layout (`t_mono`, `pos_joint_i`, `raw_joint_i`,
`vel_joint_i`, `eff_joint_i`), so the assertion is that the watchdog the
follower runs (`lerobot_robot_tatbot.motion_safety`) aborts on the measured
dynamics of a recorded run and stays quiet on a calm one. Only the CSV reader
and the watchdog are exercised; no robot, camera or driver import.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MOTION_SAFETY_SOURCE = REPO / "python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/motion_safety.py"

# The same limits the retired replay script pinned: velocity 2.5 rad/s,
# acceleration 80 rad/s^2, reversal 4/s above 0.2 rad/s, a 2.5 s clamp grace.
LIMITS = {
    "velocity_limit": 2.5,
    "acceleration_limit": 80.0,
    "reversal_window_s": 1.0,
    "reversal_min_velocity": 0.2,
    "reversal_limit": 4,
    "clamp_grace_s": 2.5,
    "clamp_window_s": 1.0,
    "clamp_fraction": 0.8,
    "clamp_min_samples": 20,
    "overforce_limit": 9.0,
    "overforce_window_s": 0.5,
    "overforce_fraction": 0.5,
    "overforce_min_samples": 8,
}
DT = 0.01  # 100 Hz rows, inside the watchdog's 1-250 ms acceleration window
JOINTS = 6


def _load_motion_safety():
    spec = importlib.util.spec_from_file_location("tatbot_motion_safety", MOTION_SAFETY_SOURCE)
    assert spec is not None and spec.loader is not None, MOTION_SAFETY_SOURCE
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(path: Path) -> dict:
    """The retired script's core: feed every CSV row to a fresh watchdog."""
    ms = _load_motion_safety()
    guard = ms.MotionSafetyWatchdog(**LIMITS)
    samples = 0
    first_t = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            now = float(row["t_mono"])
            if first_t is None:
                first_t = now
                guard.reset(now)
            velocity = [float(row[f"vel_joint_{j}"]) for j in range(JOINTS)]
            effort = [float(row[f"eff_joint_{j}"]) for j in range(JOINTS)]
            clamped = any(abs(float(row[f"raw_joint_{j}"]) - float(row[f"pos_joint_{j}"])) > 0.5 for j in range(JOINTS))
            samples += 1
            try:
                guard.update(now=now, velocities=velocity, external_efforts=effort, clamped=clamped)
            except ms.MotionSafetyError as error:
                return {"verdict": "abort", "code": error.code, "samples_read": samples,
                        "elapsed_s": now - first_t, "metrics": error.metrics}
    return {"verdict": "no_abort", "samples_read": samples}


def _write_flight_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    fields = ["t_mono"]
    for prefix in ("pos", "raw", "vel", "eff"):
        fields += [f"{prefix}_joint_{j}" for j in range(JOINTS)]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(t: float, *, vel: float = 0.0, eff: float = 0.0, clamp: float = 0.0, joint: int = 2) -> dict[str, float]:
    row: dict[str, float] = {"t_mono": t}
    for j in range(JOINTS):
        row[f"pos_joint_{j}"] = 0.1 * j
        row[f"raw_joint_{j}"] = 0.1 * j + (clamp if j == joint else 0.0)
        row[f"vel_joint_{j}"] = vel if j == joint else 0.0
        row[f"eff_joint_{j}"] = eff if j == joint else 0.0
    return row


def _calm(seconds: float, t0: float = 100.0) -> list[dict[str, float]]:
    return [_row(t0 + i * DT, vel=0.05) for i in range(int(seconds / DT))]


def test_calm_flight_log_never_aborts(tmp_path: Path) -> None:
    result = replay(_write_flight_csv(tmp_path / "flight-calm.csv", _calm(6.0)))
    assert result["verdict"] == "no_abort"
    assert result["samples_read"] == 600


def test_measured_velocity_spike_after_the_clamp_grace_aborts(tmp_path: Path) -> None:
    rows = _calm(4.0)
    rows.append(_row(rows[-1]["t_mono"] + DT, vel=3.0))  # 3.0 rad/s > 2.5 limit
    rows += [_row(rows[-1]["t_mono"] + (i + 1) * DT, vel=3.0) for i in range(20)]
    result = replay(_write_flight_csv(tmp_path / "flight-spike.csv", rows))
    assert result["verdict"] == "abort"
    assert result["code"] == "measured_velocity"
    assert result["metrics"]["joint"] == 2
    assert result["samples_read"] == 401  # the first over-limit row, not a later one


def test_the_same_spike_inside_the_grace_window_is_staging_not_a_fault(tmp_path: Path) -> None:
    rows = [_row(100.0 + i * DT, vel=3.0 if i == 50 else 0.05) for i in range(200)]  # spike at 0.5 s
    result = replay(_write_flight_csv(tmp_path / "flight-staging.csv", rows))
    assert result["verdict"] == "no_abort"


def test_sustained_clamp_after_grace_aborts(tmp_path: Path) -> None:
    rows = _calm(3.0)
    # raw target held 0.5+ rad away from the measured joint for a full window
    rows += [_row(rows[-1]["t_mono"] + (i + 1) * DT, vel=0.05, clamp=0.8) for i in range(150)]
    result = replay(_write_flight_csv(tmp_path / "flight-clamp.csv", rows))
    assert result["verdict"] == "abort"
    assert result["code"] == "sustained_clamp"


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_non_finite_telemetry_is_refused(tmp_path: Path, bad: str) -> None:
    rows = _calm(3.0)
    poisoned = _row(rows[-1]["t_mono"] + DT, vel=0.05)
    poisoned["vel_joint_0"] = float(bad)
    rows.append(poisoned)
    result = replay(_write_flight_csv(tmp_path / "flight-nan.csv", rows))
    assert result["verdict"] == "abort"
    assert result["code"] == "non_finite_telemetry"
