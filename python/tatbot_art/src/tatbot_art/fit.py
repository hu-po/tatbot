"""`--fit` on a fiducial-tile print: the widest a design is drawn and still leaves the cameras the print's markers.

The page's room is the first bound (`image.fit_width`). A design that fills a tile print covers the markers the print
is tracked by, so the width is then stepped down until the marker budget holds (tatbot_ink.markers, stack.yaml
page.markers): first on the separation's own regions, each taken as inked all over, before DBV3 has drawn a path;
then, once the program is compiled, on its counts of the paths themselves (`stats.markers`). The record names the
width the page holds, the width the print can carry and each size tried.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STEP = 0.9             # each smaller size tried is this share of the last
MIN_WIDTH_MM = 6.0     # under this no design is left to draw
REFUSED = "covers the markers its print is tracked by"   # the compile's refusal (tatbot_ink.program._marker_budget)


def areas(sep, looks: dict, canvas_mm, at: str | None) -> list:
    """The separation's layers as tatbot_ink.markers.predict_areas takes them: drawn in order, on the page."""
    cx, cy = (float(v) for v in at.split(",")) if at else (0.0, 0.0)
    w, h = canvas_mm
    extent = tuple(v / 1000 for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return [(lay.ink_id, [round(255 * c) for c in looks[lay.ink_id].srgb], lay.source < 255, extent)
            for lay in sep.layers]


def counts(sep, looks: dict, canvas_mm, stencil: Path, at: str | None) -> dict | None:
    """The separation's marker counts on the print, or None: no markers on it, or no OpenCV to read them with."""
    from tatbot_ink import markers

    from .acquire import REPO

    try:
        return markers.predict_areas(areas(sep, looks, canvas_mm, at), stencil, REPO)
    except ImportError:
        return None


def step(width_mm: float, report: dict, by: str) -> dict:
    """One size tried: what the cameras would keep at it, by the separation's regions or the program's paths."""
    from tatbot_ink import markers

    last = report["inks"][-1] if report["inks"] else {"overhead": report["clean"]["overhead"], "anchors": report["anchors"]}
    return {"width_mm": round(width_mm, 2), "by": by, "overhead": last["overhead"], "anchors": last["anchors"],
            "wrist": report["wrist"]["min"], "within": not markers.refusals(report)}


def smaller(width_mm: float, px_per_mm: float) -> float:
    """The next width down, in whole output pixels; ValueError when nothing is left to draw."""
    width = np.floor(width_mm * STEP * px_per_mm) / px_per_mm
    if width < MIN_WIDTH_MM:
        raise ValueError(f"no size of this design down to {MIN_WIDTH_MM:g} mm leaves the print the markers it is "
                         "tracked by (stack.yaml page.markers): compose it off the markers, or draw it with "
                         "--allow-marker-loss on a sheet fixed on its mount")
    return float(width)


def carried(plan_at, width_mm: float, px_per_mm: float, stencil: Path, at: str | None) -> tuple[float, dict | None]:
    """(the widest width at or under `width_mm` whose separation keeps the marker budget, the record) for
    `plan_at(width_mm)` -> (sep, looks, canvas_mm); the record is None on a print without markers."""
    record = {"page_mm": round(width_mm, 2), "steps": []}
    while True:
        sep, looks, canvas_mm = plan_at(width_mm)
        report = counts(sep, looks, canvas_mm, stencil, at)
        if report is None:
            return width_mm, None
        record["steps"].append(step(width_mm, report, "separation"))
        if record["steps"][-1]["within"]:
            return width_mm, {**record, "carried_mm": round(width_mm, 2)}
        width_mm = smaller(width_mm, px_per_mm)


def line(record: dict) -> str:
    tried = len(record["steps"])
    return (f"the page holds {record['page_mm']:g} mm; the print's markers carry {record['carried_mm']:g} mm "
            f"({tried} size{'s' if tried != 1 else ''} tried)")
