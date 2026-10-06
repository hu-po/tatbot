"""Which Decide answers an arm accepts while it waits (ros/README.md section 8). Pure Python.

An arm waits for one of:
- `pause`: a pause op holds at the standoff; CONTINUE resumes, LAND lands.
- `latched`: the driver holds after an e-stop, a stall, a contact trip or a failed goal; CONTINUE
  unlatches (never a step) and re-plans the op, LAND lands. CONTINUE is refused, leaving LAND, while
  the e-stop is pressed, after a controller error, when the pose moved more than 0.05 rad or 5 mm
  from the latched pose, or when the fitted tool changed. Past resume_timeout_s the arm announces and
  lands by itself.
- `uncertain`: an op a crash left `sent`; REDRAW draws it from its start, SKIP marks it skipped.

A landed arm takes no Draw, Touch or trajectory goal at all until it is woken (landed_refusal).
"""
from __future__ import annotations

CONTINUE, LAND, SKIP, REDRAW = 1, 2, 3, 4  # tatbot_interfaces/srv/Decide
NAMES = {CONTINUE: "continue", LAND: "land", SKIP: "skip", REDRAW: "redraw"}
ALLOWED = {
    "pause": (CONTINUE, LAND),
    "latched": (CONTINUE, LAND),
    "uncertain": (REDRAW, SKIP, LAND),
}
MOVED_RAD = 0.05
MOVED_M = 0.005


def continue_refusal(*, estop_ok: bool, controller_error: bool, joint_moved_rad: float, tip_moved_m: float,
                     tool_changed: bool, moved_rad: float = MOVED_RAD, moved_m: float = MOVED_M) -> str | None:
    """Why CONTINUE after a latch is refused (LAND stays available), or None."""
    if not estop_ok:
        return "the e-stop is pressed or silent"
    if controller_error:
        return "the controller reported an error: land"
    if joint_moved_rad > moved_rad or tip_moved_m > moved_m:
        return (f"the arm moved {joint_moved_rad:.3f} rad / {tip_moved_m * 1000:.1f} mm from the latched pose "
                f"(> {moved_rad} rad / {moved_m * 1000:.0f} mm): land")
    if tool_changed:
        return "the fitted tool changed: land"
    return None


def landed_refusal(arm: str) -> str:
    """What a landed arm answers every motion goal. The driver keeps it idle until it is woken (the arm alone,
    `client wake`), while its trajectory controller would still report each goal a success with the arm at rest
    (2026-09-29)."""
    return (f"the {arm} arm has landed and stays idle until it is woken: `tatbot ros draw` and `tatbot ros calib "
            f"run` wake it, or `ros2 run tatbot_session client wake --arm {arm}` on the ros node")


def check(waiting: str | None, decision: int) -> str | None:
    """Why `decision` does not fit what the arm waits for, or None."""
    if decision not in NAMES:
        return f"unknown decision {decision}"
    if waiting is None:
        return None if decision in (CONTINUE, LAND) else "no op waits for redraw or skip"
    if decision not in ALLOWED[waiting]:
        wanted = "|".join(NAMES[d] for d in ALLOWED[waiting])
        return f"the arm waits on {waiting}: {wanted}"
    return None


def resume_expired(released_at: float | None, now: float, resume_timeout_s: float) -> bool:
    """True once a released latch has waited for a decision longer than resume_timeout_s."""
    return released_at is not None and now - released_at > resume_timeout_s
