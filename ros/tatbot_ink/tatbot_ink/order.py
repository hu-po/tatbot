"""The order inks go onto the skin: a lighter ink never follows a darker one its marks touch.

A needle that draws through wet darker ink smears it into its own mark and carries it back to its cap at the next dip:
2026-11-10's red cap turned dark after the red's strokes ran along the black drawn before it, the separation having
grown the red 0.4 mm under the black on the understanding that the light ink goes first.

The inks are taken in the order they draw, run by run: the programs drawn earlier on the same print (`earlier`,
oldest first), then this one's. A run is one ink's strokes between two changes of ink. A later run of a lighter ink is
refused when its mark overlaps an earlier run's darker one, unless
- the design was separated for that order, the lighter ink kept clear of the darker (`tatbot art separate
  --overlap-mm` negative): then the marks do not overlap and there is nothing to refuse; or
- the print is wiped between them: a `tool_change` that carries `wipe` (tatbot-inks/3 `wipe_before: true`, the arm
  landed for it), or an earlier program stated as wiped after.

Marks are the strokes at the width `passes` counts them at. Lightness is CIE L* of the catalogue colour, the lab's
rule for which ink goes first (tatbot_lab.plan.lightness).
"""
from __future__ import annotations

import numpy as np
from tatbot_motion.timelaw import arc_lengths

from tatbot_ink import passes
from tatbot_ink.errors import CompileError

LIGHTER_L = 10.0      # an ink this much lighter in L* than one drawn before it is the lighter of the two
TOUCH_MM2 = 0.5       # marks overlapping by more than this touch; under it is the raster's own rounding at an edge
CELL_M = 0.0001       # the raster two marks are compared on


def lightness(rgb) -> float | None:
    """CIE L* (0 black, 100 white) of an 8-bit sRGB colour; None for a resource that declares no pigment (the
    fitted tool drawing one pen), which is never judged lighter or darker than another."""
    if rgb is None:
        return None
    def linear(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = 0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2])
    return 116.0 * y ** (1 / 3) - 16.0 if y > 216 / 24389 else 24389 / 27 * y


def runs(program: dict, *, wiped_after: bool = False) -> list[dict]:
    """The program's inks in drawing order, one entry per run of one ink: {ink_id, lightness, lines, first, last (op
    ids), wiped (the print was wiped before it drew)}. `wiped_after` marks the print wiped once the program ended."""
    table = {row["id"]: row for row in program["resources"]}
    width = passes._widths(program)
    out, wipe = [], False
    for op in program["ops"]:
        if op["op"] == "tool_change":
            wipe = wipe or bool(op.get("wipe"))
        if op["op"] != "stroke":
            continue
        resource = table[op["resource_id"]]
        if not out or out[-1]["ink_id"] != resource["ink_id"] or wipe:
            out.append({"ink_id": resource["ink_id"], "lightness": lightness(resource.get("rgb")), "lines": [],
                        "first": op["id"], "wiped": wipe, "wiped_after": False})
            wipe = False
        points = op["points_m"] + ([op["points_m"][0]] if op.get("closed") else [])
        out[-1]["lines"].append((np.asarray(points, float), width[op["id"]]))
        out[-1]["last"] = op["id"]
    if out and wiped_after:
        out[-1]["wiped_after"] = True
    return out


def _mask(lines, origin, shape) -> np.ndarray:
    """The cells of a raster (origin, CELL_M) within half a mark of any line."""
    mask = np.zeros(shape, bool)
    for points, width in lines:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
        arc = arc_lengths(points)
        s = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / CELL_M)) + 2)
        xy = np.c_[np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])] - origin
        reach = int(np.ceil(width / 2 / CELL_M))
        offsets = np.arange(-reach, reach + 1)
        disc = (offsets[:, None] ** 2 + offsets[None, :] ** 2) * CELL_M ** 2 <= (width / 2) ** 2
        dr, dc = np.nonzero(disc)
        centre = np.unique(np.floor(xy / CELL_M).astype(int), axis=0)
        near = ((centre >= -reach) & (centre < np.array(shape[::-1]) + reach)).all(axis=1)
        centre = centre[near]   # only what can reach the raster: two inks' boxes often share a corner
        for part in np.array_split(centre, max(1, len(centre) // 20000)):   # bounded memory on metres of line
            rows = part[:, 1, None] + offsets[dr][None, :]
            cols = part[:, 0, None] + offsets[dc][None, :]
            inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
            mask[rows[inside], cols[inside]] = True
    return mask


def overlap_mm2(first: list, then: list) -> float:
    """The area two sets of (points m, width m) lines' marks share, mm2."""
    reach = max(width for _, width in [*first, *then]) / 2 + CELL_M
    lo = np.maximum(np.vstack([p for p, _ in first]).min(axis=0), np.vstack([p for p, _ in then]).min(axis=0)) - reach
    hi = np.minimum(np.vstack([p for p, _ in first]).max(axis=0), np.vstack([p for p, _ in then]).max(axis=0)) + reach
    if (hi <= lo).any():
        return 0.0   # their boxes do not meet
    shape = tuple(np.ceil((hi - lo) / CELL_M).astype(int)[::-1] + 1)
    shared = _mask(first, lo, shape) & _mask(then, lo, shape)
    return float(shared.sum()) * (CELL_M * 1000) ** 2


def check(program: dict, earlier=()) -> dict:
    """The program's ink order against itself and the runs drawn before it on the print. `earlier` is (program,
    wiped_after) pairs, oldest first. Returns {"order", "after", "pairs"}; CompileError names every lighter ink
    that would draw over a darker one's mark."""
    before = [run for prior, wiped in earlier for run in runs(prior, wiped_after=wiped)]
    own = runs(program)
    sequence = [*before, *own]
    pairs, refused = [], []
    for at in range(len(before), len(sequence)):
        later = sequence[at]
        for index in range(at):
            dark = sequence[index]
            if dark["ink_id"] == later["ink_id"] or None in (later["lightness"], dark["lightness"]) \
                    or later["lightness"] - dark["lightness"] < LIGHTER_L:
                continue
            between = sequence[index + 1:at + 1]
            wiped = dark["wiped_after"] or any(run["wiped"] for run in between) \
                or any(run["wiped_after"] for run in between[:-1])
            shared = round(overlap_mm2(dark["lines"], later["lines"]), 2)
            verdict = "clear" if shared <= TOUCH_MM2 else "wiped" if wiped else "refused"
            pairs.append({"darker": dark["ink_id"], "lighter": later["ink_id"], "from_op": later["first"],
                          "earlier_run": index < len(before), "overlap_mm2": shared, "verdict": verdict})
            if verdict == "refused":
                refused.append(f"{later['ink_id']} (from {later['first']}) draws after {dark['ink_id']}, a darker ink"
                               + (" of an earlier run" if index < len(before) else "")
                               + f", and their marks overlap by {shared:g} mm2")
    report = {"order": [run["ink_id"] for run in own], "after": [run["ink_id"] for run in before], "pairs": pairs}
    if refused:
        raise CompileError(
            "ink order: " + "; ".join(refused) + ". Its needles would run through the darker ink and carry it into "
            "its own cap. Draw the lighter ink first; or separate the design for this order, the lighter ink kept "
            "clear of the darker (tatbot art separate --overlap-mm negative); or wipe the print between them (a "
            "tatbot-inks/3 resource's wipe_before: true; after an earlier run, --after PROGRAM --wiped)")
    return report
