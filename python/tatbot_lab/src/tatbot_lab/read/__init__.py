"""Reading drawn cells with the rig's own cameras (docs/lab.md).

`frames` puts a view's before and after frames on a cell's raster by the print's markers and fuses a scan's views;
`diff` differences them, cancelling the print; `measure` turns the difference into a cell's numbers. `read_cell` is
the whole read of one cell from (before, after) frame pairs: one pair from a rest view, or one per hold of a scan. OpenCV and the fiducials package (scripts/vision) are needed here and nowhere else in the lab.
"""
from __future__ import annotations

import numpy as np

from tatbot_lab.read import diff, frames, measure

STRAIGHT = ("line", "carry", "solid", "whip")


def _box(cell: dict) -> tuple[float, float, float, float]:
    """A cell's strokes' box grown by the keep-out, page mm."""
    pts = np.vstack([np.asarray(s, float) for s in cell["strokes_mm"]])
    lo, hi = pts.min(axis=0) - diff.KEEP_OUT_MM, pts.max(axis=0) + diff.KEEP_OUT_MM
    return (*lo, *hi)


def read_cell(cell: dict, technique: str, settings: dict, views: list[tuple[np.ndarray, np.ndarray]], t,
              others: list[dict] = (), *, instrument: str = "wrist", roi=None) -> dict:
    """One cell's numbers from (before, after) frame pairs, each pair from one pose. `cell` is a tatbot.lab-cells/1
    entry; `t` the print's fiducials tile (frames.tile); `others` the cells drawn between the same frames, left out
    of the alignment and the brightness match with this one."""
    strokes = [np.asarray(s, float) for s in cell["strokes_mm"]]
    frame = frames.frame_for(strokes)
    pts = np.vstack(strokes)
    centre = pts.mean(axis=0)
    lo, hi = pts.min(axis=0) - diff.KEEP_OUT_MM, pts.max(axis=0) + diff.KEEP_OUT_MM
    changed = [(*lo, *hi)] + [_box(o) for o in others]
    pairs, unplaced = [], []
    for before, after in views:   # a run's holds aim one at each cell: one aimed at a cell across the sheet may not
        try:                      # see this one's markers, and is left out of its read (2026-11-09)
            pairs.append(frames.pair(before, after, t, frame, centre, changed, instrument, roi))
        except frames.UnplacedError as error:
            unplaced.append(str(error))
    if not pairs:
        raise ValueError(f"cell {cell['id']}: no view places it: {unplaced[0]}")
    fused = frames.fuse(pairs)
    x, y = frame.page()
    keep_out = np.zeros(x.shape, bool)
    for x0, y0, x1, y1 in changed:
        keep_out |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    d = diff.difference(fused["before"], fused["after"], fused["valid"], keep_out)
    out = {"cell": cell["id"], "trial": cell["trial"], "views": len(pairs),
           "markers": sorted({m for p in pairs for m in p["markers"]}),
           "ecc": round(min(p["ecc"] for p in pairs), 4), "noise_dL": round(d.noise, 2)}
    if technique in STRAIGHT:
        out.update(measure.straight(d, frame, strokes, technique, settings, cell.get("dips_mm", ())))
    else:
        out.update(measure.corners(d, frame, strokes))
    return out
