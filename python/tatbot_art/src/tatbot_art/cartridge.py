"""A cartridge's mark: how wide a stroke it lays in each direction on the page.

The needle count and wire diameter are the tool datasheet's (config/tools/<id>.yaml `tip_detail`). Round liner
or magnum is the simulator's sim-tool fact (config/sim2/tools/<id>.yaml `grouping`), and the tips' layout is the
simulator's own geometry (tatbot_site.needle.groups) at nominal values of its unknowns. A round liner's tips
converge to 0.89 of a full ring (mu0.group_convergence spans 0.85-0.93). A magnum is weaved (M1), about one
needle diameter per needle across, as the datasheets say; a stacked M2 would be about half that.

A round liner lays about the same width whichever way it moves. A magnum lays its row's length only across the
row; dragged along it, it cuts a thin line, and it starts and stops square (the render sweeps the row). The planner holds one tool heading
per page, so the row's angle on the page is set by the cartridge's keyed seat and the page's placement:
`row_deg`, counter-clockwise from the page's +x. Its default of 0 is the rig's: on 2026-10-08 the 9M's and 5M's
strokes along the print's y inked wide, to within about 22 deg. A datasheet's `line.width_mm`, measured or
assumed, overrides a round mark's width.
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
CONVERGENCE = 0.89
MAGNUM_LAYOUT, ROW_OFFSET_FRACTION = "M1", 0.5   # mu0.magnum_row_offset spans 0-1 of the stacked pair's offset


def _groups():
    """tatbot_site.needle.groups, loaded by its file: the package's __init__ loads the whole needle model."""
    name = "tatbot_art._needle_groups"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "python/tatbot_site/src/tatbot_site/needle/groups.py")
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class Cartridge:
    tool_id: str
    kind: str                 # round, magnum or ball
    offsets_mm: np.ndarray    # (n, 2) tip centres on the page, a magnum's row turned to row_deg
    wire_mm: float            # a tip's diameter, or a round mark's width where it is one point
    source: str               # where the width comes from: the needle layout, or the datasheet's line block
    row_deg: float | None = None

    def swath_mm(self, heading_deg: float) -> float:
        """The width laid across a stroke along `heading_deg`: the tips' extent across it plus one wire."""
        a = math.radians(heading_deg)
        across = self.offsets_mm @ np.array([-math.sin(a), math.cos(a)])
        return float(across.max() - across.min()) + self.wire_mm

    def across_deg(self) -> float | None:
        """A magnum's widest heading, across its row (or that +180); None for a round mark."""
        return None if self.row_deg is None else self.row_deg + 90.0

    def magnum(self) -> dict | None:
        """A magnum's mark for the render: its row on the page, end to end less one thickness, and that thickness
        (the swath dragged along the row), so the row swept across a stroke lays the widest swath."""
        if self.row_deg is None:
            return None
        thickness = self.swath_mm(self.row_deg)
        return {"row_deg": self.row_deg, "row_mm": round(self.swath_mm(self.row_deg + 90.0) - thickness, 4),
                "thickness_mm": round(thickness, 4)}

    def describe(self, heading_deg: float) -> dict:
        widths = [self.swath_mm(h) for h in range(0, 180, 5)]
        return {"tool_id": self.tool_id, "kind": self.kind, "source": self.source,
                "width_mm": round(self.swath_mm(heading_deg), 3), "widest_mm": round(max(widths), 3),
                "narrowest_mm": round(min(widths), 3), "magnum": self.magnum()}


def load(tool_id: str, row_deg: float = 0.0, repo: Path = REPO) -> Cartridge:
    path = repo / "config" / "tools" / f"{tool_id}.yaml"
    if not path.is_file():
        raise ValueError(f"no tool datasheet {path.relative_to(repo)}")
    sheet = yaml.safe_load(path.read_text())
    tip, line = sheet.get("tip_detail") or {}, sheet.get("line") or {}
    if tip.get("kind") == "ball" or (tip.get("kind") == "needles" and line.get("width_mm")):
        width = float(line.get("width_mm") or 2000.0 * float(tip["radius_m"]))
        return Cartridge(tool_id, "round" if tip.get("kind") == "needles" else "ball", np.zeros((1, 2)), width,
                         f"line.width_mm ({line.get('status', 'assumed')})" if line.get("width_mm") else "ball diameter")
    if tip.get("kind") != "needles":
        raise ValueError(f"{tool_id} lays no mark to shade with (tip {tip.get('kind')!r})")
    sim_path = repo / "config" / "sim2" / "tools" / f"{tool_id}.yaml"
    if not sim_path.is_file():
        raise ValueError(f"no sim-tool file says whether {tool_id} is a round liner or a magnum ({sim_path.name})")
    # sim-tool files are read by the simulator's simple parser, not as YAML: take the one line needed
    found = re.search(r"^grouping:\s*(round|magnum)\s*$", sim_path.read_text(), re.MULTILINE)
    if found is None:
        raise ValueError(f"{sim_path.name} names no grouping: round or magnum")
    kind = found.group(1)
    count, radius_mm = int(tip["count"]), 1000.0 * float(tip["radius_m"])
    groups = _groups()
    if kind == "round":
        layout = "ring" if count < 7 else "ring+centre"
        offsets = groups.layout_offsets("round", layout, count, radius_mm, convergence=CONVERGENCE)
        return Cartridge(tool_id, "round", offsets, 2.0 * radius_mm, f"{count}-needle round liner, {layout}")
    offsets = groups.layout_offsets("magnum", MAGNUM_LAYOUT, count, radius_mm, row_offset_fraction=ROW_OFFSET_FRACTION)
    return Cartridge(tool_id, "magnum", groups.turned(offsets, math.radians(row_deg)), 2.0 * radius_mm,
                     f"{count}-needle magnum, {MAGNUM_LAYOUT}", float(row_deg))
