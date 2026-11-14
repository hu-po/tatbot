"""Cells on a dense tag print: where trials go, which markers each covers, and the order to spend them in.

The lab draws on the prints the rig tracks (2026-11-08: the dense tag tiles stay), so every cell sits over
markers and inking it costs the readings those markers gave. A cell is centred on one grid marker and no wider than
the grid's pitch less a millimetre, so it costs that marker alone (2026-11-05: a 20 mm patch centred on a marker
costs only that marker). Anchors, the large corner markers the overhead reads the print by, carry no cell. Cells are
named by their marker, so a cell is the same place on every sheet of one print.

The fill order goes from the edge inward, each next cell as far as it can be from those already drawn, so the
markers still readable stay spread over the sheet: the wrist reads the edge's markers least (2026-11-05: 2-4 a
reading at the corners, 12-15 near the middle).

Page frame: millimetres about the tile's centre, x right, y toward the print's top (tatbot_ink's page frame).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Print:
    pattern_id: str
    layout: str
    directory: Path
    tile_mm: tuple[float, float]
    pitch_mm: float
    markers: dict[int, np.ndarray]   # id -> (4, 2) page mm, TL/TR/BR/BL of the upright marker
    roles: dict[int, str]            # id -> "marker" (the grid's) or "anchor"


@dataclass(frozen=True)
class Cell:
    id: str
    box_mm: tuple[float, float, float, float]   # x0, y0, x1, y1

    @property
    def centre(self) -> np.ndarray:
        x0, y0, x1, y1 = self.box_mm
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def load(directory) -> Print:
    """A generated print's directory (tile.json and tracking.json beside its stencil.png)."""
    directory = Path(directory).expanduser()
    tracking = json.loads((directory / "tracking.json").read_text())
    layout = json.loads((directory / "tile.json").read_text())
    w, h = (float(v) for v in layout["tile_mm"])
    markers = {int(m["id"]): np.array([[x - w / 2, h / 2 - y] for x, y in m["corners_mm"]], float)
               for m in layout["markers"]}
    roles = {int(m["id"]): m["role"] for m in layout["markers"]}
    return Print(tracking["pattern_id"], layout["layout"], directory, (w, h), float(layout["pitch_mm"]), markers,
                 roles)


def cells(p: Print, cell_mm: float) -> list[Cell]:
    """One cell_mm square centred on each grid marker, top row first; ids m<marker id>."""
    if cell_mm > p.pitch_mm - 1.0:
        raise ValueError(f"a {cell_mm:g} mm cell overlaps its neighbours at this print's {p.pitch_mm:g} mm pitch")
    out = []
    for marker_id, corners in p.markers.items():
        if p.roles[marker_id] != "marker":
            continue
        c = corners.mean(axis=0)
        out.append(Cell(f"m{marker_id}", tuple(round(float(v), 3) for v in (c[0] - cell_mm / 2, c[1] - cell_mm / 2,
                                                                            c[0] + cell_mm / 2, c[1] + cell_mm / 2))))
    if not out:
        raise ValueError(f"the {p.layout} print has no grid markers to centre cells on")
    return sorted(out, key=lambda cell: (-cell.centre[1], cell.centre[0]))


def covered(p: Print, box_mm) -> list[int]:
    """Markers whose square overlaps the box."""
    x0, y0, x1, y1 = box_mm
    out = []
    for marker_id, corners in p.markers.items():
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        if lo[0] < x1 and hi[0] > x0 and lo[1] < y1 and hi[1] > y0:
            out.append(marker_id)
    return sorted(out)


def fill_order(cells: list[Cell], used: set[str] = frozenset()) -> list[Cell]:
    """The free cells in the order to draw them: the first farthest from the tile's centre, each next farthest from
    every cell already drawn (ties by id)."""
    by_id = {c.id: c for c in cells}
    drawn = [by_id[cell_id].centre for cell_id in used if cell_id in by_id]
    free, order = sorted((c for c in cells if c.id not in used), key=lambda c: c.id), []
    while free:
        def far(c: Cell) -> float:
            return min(float(np.linalg.norm(c.centre - d)) for d in drawn) if drawn else float(np.linalg.norm(c.centre))
        best = max(free, key=lambda c: round(far(c), 6))   # the first of equals: the lowest id
        order.append(best)
        free.remove(best)
        drawn.append(best.centre)
    return order
