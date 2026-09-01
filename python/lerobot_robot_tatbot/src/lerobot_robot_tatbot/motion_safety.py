"""Measured-motion safety gates independent of the robot driver.

The policy target slew limiter constrains commands, not the arm's response.
This watchdog consumes only already-measured telemetry so its incident replay
and unit tests remain hardware-free.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable


class MotionSafetyError(RuntimeError):
    """A measured safety envelope was exceeded and policy motion must stop."""

    def __init__(self, code: str, message: str, **metrics: Any):
        super().__init__(message)
        self.code = code
        self.metrics = metrics


class MotionSafetyWatchdog:
    """Fail closed on measured dynamics, repeated clamp, or rolling load."""

    def __init__(
        self,
        *,
        velocity_limit: float,
        acceleration_limit: float,
        reversal_window_s: float,
        reversal_min_velocity: float,
        reversal_limit: int,
        clamp_grace_s: float,
        clamp_window_s: float,
        clamp_fraction: float,
        clamp_min_samples: int,
        overforce_limit: float,
        overforce_window_s: float,
        overforce_fraction: float,
        overforce_min_samples: int,
    ) -> None:
        self.velocity_limit = float(velocity_limit)
        self.acceleration_limit = float(acceleration_limit)
        self.reversal_window_s = float(reversal_window_s)
        self.reversal_min_velocity = float(reversal_min_velocity)
        self.reversal_limit = int(reversal_limit)
        self.clamp_grace_s = float(clamp_grace_s)
        self.clamp_window_s = float(clamp_window_s)
        self.clamp_fraction = float(clamp_fraction)
        self.clamp_min_samples = int(clamp_min_samples)
        self.overforce_limit = float(overforce_limit)
        self.overforce_window_s = float(overforce_window_s)
        self.overforce_fraction = float(overforce_fraction)
        self.overforce_min_samples = int(overforce_min_samples)
        self.reset()

    @classmethod
    def from_config(cls, config: Any) -> MotionSafetyWatchdog:
        return cls(
            velocity_limit=config.measured_velocity_abort,
            acceleration_limit=config.measured_acceleration_abort,
            reversal_window_s=config.reversal_window_s,
            reversal_min_velocity=config.reversal_min_velocity,
            reversal_limit=config.reversal_abort_count,
            clamp_grace_s=config.clamp_abort_grace_s,
            clamp_window_s=config.clamp_abort_window_s,
            clamp_fraction=config.clamp_abort_fraction,
            clamp_min_samples=config.clamp_abort_min_samples,
            overforce_limit=config.overforce_limit,
            overforce_window_s=config.overforce_window_s,
            overforce_fraction=config.overforce_window_fraction,
            overforce_min_samples=config.overforce_window_min_samples,
        )

    def reset(self, now: float | None = None) -> None:
        self._started = now
        self._previous_t: float | None = None
        self._previous_velocity: list[float] | None = None
        self._previous_sign: list[int] | None = None
        self._reversals: list[deque[float]] = []
        self._clamps: deque[tuple[float, bool]] = deque()
        self._overforce: deque[tuple[float, bool]] = deque()

    @staticmethod
    def _finite(values: Iterable[float], label: str) -> list[float]:
        result = [float(value) for value in values]
        if not result or not all(math.isfinite(value) for value in result):
            raise MotionSafetyError(
                "non_finite_telemetry",
                f"{label} contains no finite joint vector",
                telemetry=label,
            )
        return result

    @staticmethod
    def _trim(window: deque[tuple[float, bool]], cutoff: float) -> None:
        while window and window[0][0] < cutoff:
            window.popleft()

    @staticmethod
    def _window_ready(
        window: deque[tuple[float, bool]], now: float, duration: float, minimum: int
    ) -> bool:
        return (
            len(window) >= minimum
            and bool(window)
            and now - window[0][0] >= duration * 0.8
        )

    def update(
        self,
        *,
        now: float,
        velocities: Iterable[float],
        external_efforts: Iterable[float],
        clamped: bool,
    ) -> None:
        velocity = self._finite(velocities, "measured velocity")
        effort = self._finite(external_efforts, "external effort")
        if len(velocity) != len(effort):
            raise MotionSafetyError(
                "telemetry_width",
                "measured velocity and effort widths differ",
                velocities=len(velocity),
                efforts=len(effort),
            )
        if self._started is None:
            self._started = now
        previous_signs = self._previous_sign
        if previous_signs is None or len(previous_signs) != len(velocity) or not self._reversals:
            previous_signs = [0 for _ in velocity]
            self._previous_sign = previous_signs
            self._reversals = [deque() for _ in velocity]

        previous_t = self._previous_t
        previous_velocity = self._previous_velocity
        self._previous_t = now
        self._previous_velocity = velocity

        # Staging-to-policy-pose really does clamp for roughly 1-2 seconds.
        # Exclude it instead of teaching a persistent-clamp gate to tolerate it.
        if now - self._started < self.clamp_grace_s:
            self._clamps.clear()
            self._overforce.clear()
            self._previous_sign = [0 for _ in velocity]
            for history in self._reversals:
                history.clear()
            return

        peak_velocity = max(abs(value) for value in velocity)
        if self.velocity_limit > 0 and peak_velocity > self.velocity_limit:
            joint = max(range(len(velocity)), key=lambda i: abs(velocity[i]))
            raise MotionSafetyError(
                "measured_velocity",
                f"measured joint {joint} velocity {velocity[joint]:.3f} rad/s exceeds "
                f"{self.velocity_limit:.3f} rad/s",
                joint=joint,
                value=velocity[joint],
                limit=self.velocity_limit,
            )

        if previous_t is not None and previous_velocity is not None:
            dt = now - previous_t
            if 0.001 <= dt <= 0.25 and self.acceleration_limit > 0:
                acceleration = [
                    (value - old) / dt
                    for value, old in zip(velocity, previous_velocity, strict=True)
                ]
                peak_acceleration = max(abs(value) for value in acceleration)
                if peak_acceleration > self.acceleration_limit:
                    joint = max(
                        range(len(acceleration)), key=lambda i: abs(acceleration[i])
                    )
                    raise MotionSafetyError(
                        "measured_acceleration",
                        f"measured joint {joint} acceleration {acceleration[joint]:.1f} "
                        f"rad/s^2 exceeds {self.acceleration_limit:.1f} rad/s^2",
                        joint=joint,
                        value=acceleration[joint],
                        limit=self.acceleration_limit,
                        dt=dt,
                    )

        for joint, value in enumerate(velocity):
            sign = 1 if value >= self.reversal_min_velocity else (
                -1 if value <= -self.reversal_min_velocity else 0
            )
            previous_sign = previous_signs[joint]
            if sign and previous_sign and sign != previous_sign:
                self._reversals[joint].append(now)
            if sign:
                previous_signs[joint] = sign
            cutoff = now - self.reversal_window_s
            while self._reversals[joint] and self._reversals[joint][0] < cutoff:
                self._reversals[joint].popleft()
            if self.reversal_limit > 0 and len(self._reversals[joint]) >= self.reversal_limit:
                raise MotionSafetyError(
                    "measured_reversals",
                    f"measured joint {joint} reversed {len(self._reversals[joint])} times "
                    f"within {self.reversal_window_s:.2f} s",
                    joint=joint,
                    count=len(self._reversals[joint]),
                    window_s=self.reversal_window_s,
                    min_velocity=self.reversal_min_velocity,
                )

        self._clamps.append((now, bool(clamped)))
        self._trim(self._clamps, now - self.clamp_window_s)
        if self._window_ready(
            self._clamps, now, self.clamp_window_s, self.clamp_min_samples
        ):
            fraction = sum(value for _, value in self._clamps) / len(self._clamps)
            if fraction >= self.clamp_fraction:
                raise MotionSafetyError(
                    "sustained_clamp",
                    f"safety clamp active on {fraction:.1%} of {len(self._clamps)} "
                    f"samples in {self.clamp_window_s:.2f} s",
                    fraction=fraction,
                    samples=len(self._clamps),
                    window_s=self.clamp_window_s,
                    limit=self.clamp_fraction,
                )

        peak_effort = max(abs(value) for value in effort)
        self._overforce.append((now, peak_effort > self.overforce_limit))
        self._trim(self._overforce, now - self.overforce_window_s)
        if self.overforce_limit > 0 and self._window_ready(
            self._overforce,
            now,
            self.overforce_window_s,
            self.overforce_min_samples,
        ):
            fraction = sum(value for _, value in self._overforce) / len(self._overforce)
            if fraction >= self.overforce_fraction:
                raise MotionSafetyError(
                    "rolling_overforce",
                    f"arm effort exceeded {self.overforce_limit:.1f} Nm on "
                    f"{fraction:.1%} of {len(self._overforce)} samples in "
                    f"{self.overforce_window_s:.2f} s",
                    peak_effort=peak_effort,
                    fraction=fraction,
                    samples=len(self._overforce),
                    window_s=self.overforce_window_s,
                    limit=self.overforce_fraction,
                )
