"""Measured tool profile and exact acquired-pen to physical-ink bindings."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml

from tatbot_ink.errors import CompileError


def load_tool(repo, arm, tool_id, *, allow_dip=False):
    if tool_id is None:
        tool_id = (yaml.safe_load((repo / "config/workspace.yaml").read_text()).get(arm) or {}).get("tool_id")
    if not isinstance(tool_id, str) or not tool_id or Path(tool_id).name != tool_id:
        raise CompileError("select a tool datasheet with --ee-tool or config/workspace.yaml")
    path = repo / "config/tools" / f"{tool_id}.yaml"
    if not path.is_file():
        raise CompileError(f"no tool datasheet {path}")
    raw = path.read_bytes()
    sheet = yaml.safe_load(raw)
    line, ink = sheet.get("line") or {}, sheet.get("ink") or {}
    width = line.get("width_mm")
    if width is not None and (type(width) not in (int, float) or not math.isfinite(width) or not 0 < width <= 20):
        raise CompileError("tool line.width_mm must be a positive finite width or null")
    status = "unknown" if width is None else "measured" if line.get("status") == "measured" else "assumed"
    mode = ink.get("mode", "none")
    if allow_dip and mode == "real":
        mode = "dip"
    if mode not in (("none", "cartridge", "dip") if allow_dip else ("none", "cartridge")):
        raise CompileError("a dipping tool requires an explicit resource naming its cap")
    result = {"id": tool_id, "line_width_m": None if width is None else width / 1000,
            "line_width_status": status, "ink": mode, "datasheet_sha256": hashlib.sha256(raw).hexdigest(),
            "substrates": sheet.get("substrates") or [sheet.get("substrate")],
            "stroke_mm": sheet.get("stroke_mm"), "tip_out_at_top_mm": sheet.get("tip_out_at_top_mm"),
            "contact_reference": (sheet.get("calibration") or {}).get("contact_reference"),
            "cartridge_activation": sheet.get("cartridge_activation") or {"action": "exchange"}}
    if allow_dip:
        result['dip_block'] = sheet.get('dip')
    return result
