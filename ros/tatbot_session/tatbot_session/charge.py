"""A dipping cartridge's charge while it draws (ros/README.md 6). Pure Python, no ROS.

A dip charges the cartridge for the datasheet's `mm_per_dip` (the program resource's `dip`) times the operator's dip
scale for that resource, which the numpad's + (a dip at once, and every later one a step sooner) and - (a step later)
move from 1; each change is a `dip_scale` ledger row, and a resume starts each resource at its last one. Until + or -
is pressed for a resource in a run its dips are the program's (a bench program may draw past mm_per_dip on purpose, to
find where a line fades). After, the executor counts the line drawn since the last dip against that budget:
- a stroke the charge left cannot finish dips first, unless the charge is full; one that starts a source path
  commits to the whole path, every section of it, as the compiler weighs a path;
- a stroke longer than a full charge stops where the charge runs out and dips there;
- a compiled replenishing dip the charge left still covers is skipped.
At scale 1 none of these moves a compiled program's dip: the compiler placed them by the same count (tatbot_ink
schedule). SLACK_M keeps rounding from moving one.
"""
from __future__ import annotations

from tatbot_session import geometry

STEP = 0.1                  # one press: a tenth of the datasheet's line per dip
LOWEST, HIGHEST = 0.2, 2.0  # the scale's range
SLACK_M = 0.001             # how far past (or short of) its budget the count must be to move a dip


def stepped(scale: float, key: str) -> float:
    """The scale after a press: `dip` (+) shortens the line per dip a step, `fewer_dips` (-) lengthens it."""
    steps = round(scale / STEP) + (-1 if key == "dip" else 1)
    return round(min(HIGHEST, max(LOWEST, steps * STEP)), 6)


def scales(rows, arm: str) -> dict[str, float]:
    """resource -> its last dip scale in a run's ledger."""
    return {row["resource"]: float(row["scale"]) for row in rows
            if row.get("event") == "dip_scale" and row.get("arm") == arm}


def budget_m(resource: dict | None, scale: float = 1.0) -> float | None:
    """The line one dip draws now (m), or None for a resource that takes no dips."""
    dip = (resource or {}).get("dip")
    return None if dip is None else float(dip["mm_per_dip"]) / 1000.0 * scale


def contact_m(op: dict) -> float:
    """A stroke's contact, as the compiler counts it."""
    return geometry.stroke_length(op["points_m"], closed=bool(op.get("closed")))


def _continues(last: dict, op: dict) -> bool:
    """`op` goes on with `last`'s source path: pen-down, or its next section after a dip (the arc goes on)."""
    if op.get("continues"):
        return True
    a, b = last.get("src") or {}, op.get("src") or {}
    return (a.get("path") is not None and (a.get("artwork_sha256"), a["path"]) == (b.get("artwork_sha256"), b.get("path"))
            and abs(a["arc_m"][1] - b["arc_m"][0]) < 1e-9)


def need_m(ops: list[dict], index: int, from_arc: float = 0.0) -> float:
    """The line the stroke at `index` commits to: its whole source path when it starts one (every section, as the
    compiler weighs a path before it dips), else its own rest from from_arc."""
    need, last = contact_m(ops[index]) - from_arc, ops[index]
    if from_arc > 0 or last.get("continues"):
        return need
    for op in ops[index + 1:]:
        if op.get("op") == "dip":
            continue
        if op.get("op") != "stroke" or not _continues(last, op):
            break
        need, last = need + contact_m(op), op
    return need


def dip_first(inked_m: float, need: float, budget: float) -> bool:
    """A stroke dips first: the charge left cannot finish the line it commits to, and is not full."""
    return inked_m > 0 and inked_m + need > budget + SLACK_M


def covered(inked_m: float, need: float, budget: float) -> bool:
    """A compiled replenishing dip is skipped: the charge left covers the line that follows it."""
    return inked_m + need <= budget - SLACK_M


def spent(inked_m: float, drawn_m: float, budget: float) -> bool:
    """A running stroke stops for a dip: the charge has drawn its line."""
    return inked_m + drawn_m > budget + SLACK_M
