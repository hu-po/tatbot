"""Asynchronous action chunking: the next chunk is computed while the current one plays.

A FLUX 3 Action chunk takes longer than a control tick -- most of a second on
a Thor at one sampler step, 2-3 s at the checkpoints' own four steps with
guidance. ``sync`` is the recipe's own loop, and the default: plan, execute
the first ``execute`` (32) commands from the plan's arrival, hold while the
next is computed. The overlapped timings keep the arm moving instead. Plans are
computed back to back from the history as it stands, and each arrives about
``latency`` ticks after the observation it answers. By default (``aligned``)
a plan takes over at its arrival from the tick it has reached: its first
commands are already past, so the handover catches up to where the plan says
the arm should be by now, blended in over ``blend`` ticks. ``Plan.rebased``
plays a plan from its arrival instead -- its motion relative to the command it
was planned from, started at the command in force -- which removes the
handover step but runs every plan ``latency`` ticks late; in sim that moved
less and approached the ink less (26 Sep). If a plan runs out before the next
one arrives, the arm carries on at its last velocity for a few ticks, then
holds.

The history is the recipe's: the last ``n`` ticks' images and measured
states, and the command in force when each was observed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


class History:
    """The last ``n`` ticks: image, measured state, and the command in force when they were observed."""

    def __init__(self, n: int):
        self.n = n
        self.images: deque[np.ndarray] = deque(maxlen=n)
        self.scenes: deque[np.ndarray] = deque(maxlen=n)  # a third-person view, for checkpoints trained on one
        self.states: deque[np.ndarray] = deque(maxlen=n)
        self.commands: deque[np.ndarray] = deque(maxlen=n)

    def push(self, image: np.ndarray, state: np.ndarray, command: np.ndarray, scene: np.ndarray | None = None) -> None:
        # Like the recipe's synchronous start, the first tick fills the whole window.
        for _ in range(self.n if not self.images else 1):
            self.images.append(image)
            self.states.append(np.asarray(state, dtype=float))
            self.commands.append(np.asarray(command, dtype=float))
            if scene is not None:
                self.scenes.append(scene)

    def scene_window(self) -> np.ndarray | None:
        """(n, H, W, 3) scene images, oldest first, or None when no scene view is kept."""
        return np.stack(self.scenes) if len(self.scenes) == self.n else None

    def window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(images (n, H, W, 3), states (n, d), commands (n, d)), oldest first."""
        return np.stack(self.images), np.stack(self.states), np.stack(self.commands)


def condition_plan(commands: np.ndarray, q_start: np.ndarray, held: list[int], smooth_ticks: int) -> np.ndarray:
    """A plan as it is executed: smoothed over ``smooth_ticks`` (the policy's per-tick deltas carry noise that
    reverses every few ticks; the expert's motions are smooth on this scale), ``held`` joints at the start pose."""
    out = np.asarray(commands, dtype=float).copy()
    if smooth_ticks > 1:
        pad = np.pad(out, ((smooth_ticks // 2, smooth_ticks - 1 - smooth_ticks // 2), (0, 0)), mode="edge")
        kernel = np.ones(smooth_ticks) / smooth_ticks
        out = np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(out.shape[1])], 1)
    if held:
        out[:, held] = q_start[held]
    return out


@dataclass(frozen=True)
class Plan:
    start: int  # the tick commands[0] is for: the observation's tick, or the arrival's once rebased
    commands: np.ndarray  # (T, d) absolute joint commands

    def rebased(self, tick: int, anchor: np.ndarray, current: np.ndarray) -> Plan:
        """The same motion started at ``tick`` from ``current``: the commands relative to ``anchor``, the command in
        force at the observation the plan answers (the delta representation's integration start)."""
        return Plan(tick, np.asarray(self.commands, dtype=float) - anchor + current)

    def at(self, tick: int) -> np.ndarray | None:
        k = tick - self.start
        return self.commands[k] if 0 <= k < len(self.commands) else None

    def left(self, tick: int) -> int:
        return self.start + len(self.commands) - tick


class ChunkScheduler:
    """Which command to send each tick, and when to ask for the next plan.

    With ``latency`` 0 it replans after ``horizon`` ticks of each plan, as the
    recipe's synchronous loop does; with latency it asks that many ticks
    earlier. ``back_to_back`` (rebased plans) asks again as soon as a plan is in.
    """

    def __init__(self, latency: int, horizon: int = 32, blend: int = 8, back_to_back: bool = False,
                 coast: int = 4, sync: bool = False):
        self.latency, self.horizon, self.blend, self.back_to_back = latency, horizon, blend, back_to_back
        self.sync = sync  # the recipe's loop: execute ``horizon`` commands of a (rebased) plan, then plan again
        self.coast = coast  # ticks to carry on at a finished plan's last velocity before holding
        self.plan: Plan | None = None
        self.previous: Plan | None = None
        self.adopted_at = 0
        self.waiting = False

    def due(self, tick: int) -> bool:
        """Ask for the next plan now, so it lands as this one reaches its horizon."""
        if self.waiting:
            return False
        if self.plan is None or self.back_to_back:
            return True
        if self.sync:
            return self.plan.left(tick) <= 0
        return self.plan.left(tick) <= len(self.plan.commands) - self.horizon + self.latency

    def requested(self) -> None:
        self.waiting = True

    def adopt(self, plan: Plan, tick: int) -> None:
        if self.sync:
            plan = Plan(plan.start, plan.commands[:self.horizon])
        self.previous, self.plan, self.adopted_at, self.waiting = self.plan, plan, tick, False

    def command(self, tick: int, holding: np.ndarray) -> np.ndarray:
        """This tick's command: the plan's, blended in after a switch; ``holding`` when there is none."""
        new = self.plan.at(tick) if self.plan is not None else None
        if new is None:
            return self._coasting(tick, holding)
        k = tick - self.adopted_at
        if k >= self.blend:
            return new
        old = self.previous.at(tick) if self.previous is not None else None
        w = (k + 1) / (self.blend + 1)
        return (1.0 - w) * (holding if old is None else old) + w * new

    def _coasting(self, tick: int, holding: np.ndarray) -> np.ndarray:
        """Past the plan's end: its last velocity for up to ``coast`` ticks (the next plan is a tick or two late)."""
        plan = self.plan
        if plan is None or len(plan.commands) < 2:
            return holding
        over = tick - (plan.start + len(plan.commands) - 1)
        if not 0 < over <= self.coast:
            return holding
        last, velocity = plan.commands[-1], plan.commands[-1] - plan.commands[-2]
        return last + velocity * over
