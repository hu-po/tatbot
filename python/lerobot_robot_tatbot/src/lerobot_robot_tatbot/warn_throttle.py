"""Collapse upstream's per-tick clamp WARNING into one line plus a summary.

lerobot's ``ensure_safe_goal_position`` logs a multi-line pformat WARNING on
EVERY tick where ``max_relative_target`` bites, through the bare ``logging``
module — i.e. the root logger. The startup traverse from ``staged_positions``
to whatever pose the policy wants bites for ~1.5 s, which is ~30 identical
warnings scrolled past before the run has begun (measured on all five rollouts
of the 2026-08-21 DiT sweep).

THIS IS PRESENTATION ONLY. It is a ``logging.Filter``; it touches no clamp, no
limit and no command. It attaches to the ROOT logger, where ``Logger.handle()``
runs filters only for records that ORIGINATED there — records from named
loggers (ours, lerobot's own module loggers) reach the handlers by propagation
and never pass through a root-logger filter. So the blast radius is exactly the
bare ``logging.warning()`` calls upstream makes, and nothing else.

It is never fully silent. A clamp still biting after the startup transient
means something real — a wedged controller, a runaway policy, an operator
outrunning the slew limiter — so one warning is always let through every
``REPEAT_S``. The analyzer recomputes clamp ticks from the flight CSV
independently (``raw_*`` is the pre-clamp goal), so throttling the display
costs no data.
"""

import logging
import time

logger = logging.getLogger(__name__)


class ClampWarningThrottle(logging.Filter):
    PREFIX = "Relative goal position magnitude had to be clamped"
    REPEAT_S = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.first: float | None = None
        self.last: float | None = None
        self._last_emitted = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.msg
        if not isinstance(msg, str) or not msg.startswith(self.PREFIX):
            return True
        now = time.monotonic()
        self.count += 1
        if self.first is None:
            self.first = now
        self.last = now
        if self.count == 1:
            record.msg = (
                msg + "\n(expected during the staged -> policy-pose traverse; "
                "further clamp warnings are summarized at disconnect)"
            )
            self._last_emitted = now
            return True
        if now - self._last_emitted >= self.REPEAT_S:
            self._last_emitted = now
            record.msg = (
                f"{self.PREFIX} (still clamping — {self.count} ticks so far, "
                f"{now - self.first:.1f} s since the first). Past the startup "
                "transient this means the arm is not tracking: check the "
                "controller and the policy."
            )
            record.args = ()
            return True
        return False

    def summary(self) -> str | None:
        if not self.count or self.first is None or self.last is None:
            return None
        return (
            f"max_relative_target clamp engaged on {self.count} ticks over "
            f"{self.last - self.first:.1f} s ({max(self.count - 1, 0)} warnings "
            "suppressed); clamping behaviour unchanged"
        )


def install() -> ClampWarningThrottle:
    throttle = ClampWarningThrottle()
    logging.getLogger().addFilter(throttle)
    return throttle


def remove(throttle: ClampWarningThrottle | None) -> None:
    if throttle is None:
        return
    logging.getLogger().removeFilter(throttle)
    text = throttle.summary()
    if text:
        logger.info(text)
