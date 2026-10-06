"""Plans in a process of their own, so the policy's Python never shares a GIL with the control loop.

In one process the planner thread, the camera thread and the 30 Hz loop take
turns at the interpreter: a one-step plan measured 1.0 s alone on the arm
node and 1.45 s beside the camera and the loop. Here the policy (and torch,
and CUDA) live in a spawned child; the loop sends it a history window when a
plan is due and picks finished plans up without waiting.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


def _serve(conn, checkpoint: str, dataset: str, steps: int | None, guidance: float | None, seed: int | None) -> None:
    from tatbot_travel.evaluate import LeRobotPolicy

    policy = LeRobotPolicy(Path(checkpoint), Path(dataset), steps=steps, guidance=guidance, seed=seed)
    conn.send(("ready", policy.n_obs_steps, policy.task, policy.cameras))
    while True:
        message = conn.recv()
        if message is None:
            return
        tick, window = message
        started = time.monotonic()
        commands = policy.plan(*window)  # (images, states, commands[, scenes])
        conn.send(("plan", tick, commands, time.monotonic() - started))


class ProcessPlanner:
    """One plan at a time in a spawned child; ``take`` returns the finished ones."""

    def __init__(self, checkpoint: Path, dataset: Path, steps: int | None = None, guidance: float | None = None,
                 seed: int | None = None):
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_serve, name="travel-planner", daemon=True,
                                       args=(child, str(checkpoint), str(dataset), steps, guidance, seed))
        self.process.start()
        _, self.n_obs_steps, self.task, self.cameras = self.conn.recv()  # blocks until the policy is loaded
        self.busy = False

    def request(self, tick: int, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
        if self.busy:
            return False
        self.conn.send((tick, window))
        self.busy = True
        return True

    def take(self) -> list[tuple[int, np.ndarray, float]]:
        done = []
        while self.conn.poll():
            _, tick, commands, seconds = self.conn.recv()
            done.append((tick, commands, seconds))
            self.busy = False
        return done

    def plan_now(self, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, float]:
        """A plan, waited for (warm-up)."""
        self.request(-1, window)
        while True:
            done = self.take()
            if done:
                return done[-1][1], done[-1][2]
            time.sleep(0.005)

    def join(self, timeout_s: float = 10.0) -> None:
        """Let a plan in flight finish before the process is asked to stop."""
        deadline = time.monotonic() + timeout_s
        while self.busy and time.monotonic() < deadline:
            self.take()
            time.sleep(0.01)

    def close(self) -> None:
        self.join()
        if self.process.is_alive():
            self.conn.send(None)
            self.process.join(timeout=10.0)
