"""The carriage as a drawing DOF, on the executor's own terms.

The tool rides a prismatic carriage on the follower's left finger, and the real
executor already solves it as a seventh axis (``cpp/teleop/square_probe.cpp``,
``weighted_carriage_dls``). Its envelope, centering gain and rate caps live in
``config/motion_constants.json``, which also renders
``cpp/teleop/motion_constants.hpp``; this module reads that same file through
``scripts/lib/motion_constants.py`` rather than restating a number, so sim, the
Python planner and the executor cannot disagree about the envelope.

**Why the carriage follows surface-normal error rather than its share of the
twist.** The executor's weighted solve hands the carriage most of any motion
along the tool axis. On a flat sheet that is what you want. On a curved one it
is not: the wrist turns to follow the normal, and that rotation swings the tip
through a ~200 mm lever arm. The weighted solve reads the swing as tool-axis
motion and recruits the carriage to serve it, saturating the 1 mm/s cap while
the tip is genuinely moving at 0.5 mm/s. It walked the carriage out of its
envelope within 25 s on the first live bottle path, and every real draw session
since has run ``carriage_ik 0`` -- the carriage locked -- with only the flat
paper-validated spiral keeping it on (square_probe.cpp, the pen-up branch).

So the carriage here serves only the component of tip position error along the
LOCAL SURFACE NORMAL, and the arm takes the whole remainder, lever-arm swing
included. The normal comes from the surface itself, so a plane, a deformed
cylinder and a posed body patch all drive it identically; nothing here knows
which shape it is on.

This is offline simulation. Nothing in this module authorizes powered motion,
and it does not re-enable ``carriage_ik`` on the real executor.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

CARRIAGE_JOINT = "left_carriage_joint"
"""The derived URDF's prismatic carriage, and the 7th action channel."""


def _motion_constants():
    """The one reader of config/motion_constants.json (stdlib, scripts/lib)."""
    return importlib.import_module("motion_constants")


@dataclass(frozen=True)
class CarriagePolicy:
    """The guarded drawing envelope and rates, as the executor states them."""

    bias_m: float
    """Where the carriage sits when nothing asks it to move, and what the
    centering term pulls it back to. Off its hard stop, so it has authority in
    both directions."""

    min_m: float
    max_m: float
    """The guarded drawing envelope. Narrower than the joint's mechanical
    range on purpose: leaving it is the failure this module exists to avoid."""

    center_gain_s: float
    max_velocity_m_s: float
    max_acceleration_m_s2: float
    constants_sha: str
    """Digest of config/motion_constants.json, recorded per run so a dataset
    names the numbers it was generated under."""

    @classmethod
    def from_repo(cls) -> CarriagePolicy:
        module = _motion_constants()
        data = module.load()
        constants = module._ns(data)
        return cls(
            bias_m=float(constants.carriage_ik.bias_m),
            min_m=float(constants.carriage_ik.min_m),
            max_m=float(constants.carriage_ik.max_m),
            center_gain_s=float(constants.planner.carriage_center_gain_s),
            max_velocity_m_s=float(constants.planner.max_carriage_velocity_m_s),
            max_acceleration_m_s2=float(constants.planner.max_carriage_acceleration_m_s2),
            constants_sha=module.sha_of(data),
        )

    def __post_init__(self) -> None:
        if not self.min_m <= self.bias_m <= self.max_m:
            raise ValueError(
                f"carriage bias {self.bias_m} is outside its envelope "
                f"[{self.min_m}, {self.max_m}]"
            )

    def max_step_m(self, control_hz: float) -> float:
        """How far the carriage may travel between two control frames.

        At 30 Hz and the executor's 1 mm/s cap this is 33 um -- the whole
        envelope takes three seconds to cross. Solving each timestep
        independently would let it jump the envelope in one frame and call the
        result a demonstration, so the rate limit belongs in the sequential
        sweep, against the previous step's value.
        """
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        return self.max_velocity_m_s / float(control_hz)

    def centering_per_iteration(self, control_hz: float, iters: int) -> float:
        """The executor centers once per control tick at ``center_gain_s``.

        The batched solver refines a timestep ``iters`` times, so the per-tick
        pull is spread across them and sums to the same relaxation rather than
        being applied ``iters`` times over.
        """
        if iters <= 0:
            raise ValueError("iters must be positive")
        return self.center_gain_s / float(control_hz) / float(iters)

    def as_metadata(self) -> dict:
        """What a dataset records about the axis it was solved with."""
        return {
            "bias_m": self.bias_m,
            "min_m": self.min_m,
            "max_m": self.max_m,
            "center_gain_s": self.center_gain_s,
            "max_velocity_m_s": self.max_velocity_m_s,
            "max_acceleration_m_s2": self.max_acceleration_m_s2,
            "constants_sha": self.constants_sha,
            "source": "config/motion_constants.json",
        }
