"""What the arm does with a position command: late, and a little behind.

The dataset's ``action`` is what the policy would send; its
``observation.state`` is what the controller reports back. Between them the
real arm adds transport and firmware delay and tracks with a lag, so the
simulated measurement is a delayed first-order follower of the executed
command, read through the encoders' 14-bit quantum. Fitted to the blue arm's
own run logs (gain 1, group delay 1.7-2.7 ticks at 30 Hz, readings constant
at standstill in 0.383 mrad steps): a time constant of 30-70 ms after 0-1
ticks of delay, drawn per episode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServoConfig:
    tau_s: tuple[float, float] = (0.03, 0.07)
    delay_ticks: tuple[int, int] = (0, 1)
    quantum_rad: float = 2 * np.pi / 16384  # the encoders' step
    noise_rad: float = 0.0


class Servo:
    def __init__(self, q0: np.ndarray, rng: np.random.Generator, cfg: ServoConfig, dt: float):
        self.rng, self.cfg = rng, cfg
        self.tau = float(rng.uniform(*cfg.tau_s))
        self.delay = int(rng.integers(cfg.delay_ticks[0], cfg.delay_ticks[1] + 1))
        self.alpha = 1.0 - np.exp(-dt / self.tau)
        self.q = np.asarray(q0, dtype=float).copy()
        self.pending: deque[np.ndarray] = deque([self.q.copy()] * (self.delay + 1), maxlen=self.delay + 1)

    def measured(self) -> np.ndarray:
        q = self.q + (self.rng.normal(0.0, self.cfg.noise_rad, self.q.shape) if self.cfg.noise_rad else 0.0)
        return np.round(q / self.cfg.quantum_rad) * self.cfg.quantum_rad

    def step(self, command: np.ndarray) -> None:
        """Advance one control tick with ``command`` executed."""
        self.pending.append(np.asarray(command, dtype=float).copy())
        self.q = self.q + self.alpha * (self.pending[0] - self.q)
