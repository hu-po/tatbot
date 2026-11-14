"""What a trial can vary, and where each setting takes effect.

A setting lives at one of four levels, which is what a plan has to respect:
- run: `trim_mm`, the depth trim `tatbot ros draw --trim-mm` starts a whole run at;
- program: `speed_mm_s`, the draw speed `tatbot ros compile --speed` gives a whole program;
- resource: `mm_per_dip`, `dip_dwell_s`, `dip_above_ink_mm`, `dip_drain_s`, `dip_drain_above_rim_mm`, the trial's own
  resource's dip over the cartridge datasheet's (tatbot-inks/3 `dip:`); unset, the datasheet's;
- geometry: everything else, the strokes a technique lays in its cell.

Trials that differ in a run or program setting cannot share a run; every trial has its own resource, so resource
settings never split one.

The dose, needle pokes per millimetre of line in a stroke's middle, is the speed seen through the machine's rate the
study holds (`constants.machine_hz`): a study may bound `dose_per_mm` in place of `speed_mm_s` (speed_bounds), and
every trial's is `dose_per_mm`. How the ends and corners of a stroke differ from that is the planner's time law
(`tatbot_ink.dose`, the program's `stats.dose`).
"""
from __future__ import annotations

from tatbot_contracts.lab import SETTINGS

REQUIRED = ("trim_mm", "speed_mm_s")                   # never defaulted: a depth or a speed is always a choice
DOSE = "dose_per_mm"                                   # a study's bound that stands for its speed's (speed_bounds)
DIP = {"mm_per_dip": ("mm_per_dip", 1.0), "dip_dwell_s": ("dwell_s", 1.0),   # setting -> (inks-file dip key, scale)
       "dip_above_ink_mm": ("above_ink_m", 0.001), "dip_drain_s": ("drain_s", 1.0),
       "dip_drain_above_rim_mm": ("drain_above_rim_m", 0.001)}
LEVEL = {"trim_mm": "run", "speed_mm_s": "program", **dict.fromkeys(DIP, "resource")}   # the rest are geometry

# technique -> its geometry settings and their defaults. Each lays its strokes about the cell's centre.
TECHNIQUES: dict[str, dict[str, float]] = {
    # parallel strokes: line weight, darkness and edges at a depth and speed
    "line": {"lines": 5, "length_mm": 10.0, "pitch_mm": 2.0, "passes": 1, "angle_deg": 0.0},
    # a serpentine of strokes 1 mm apart, read stroke by stroke from each dip: how far a dip carries (2026-11-05's
    # blocks). The compiler dips again every mm_per_dip: set it above the cell's 168 mm to read one dip whole
    "carry": {"lines": 14, "length_mm": 12.0, "pitch_mm": 1.0, "passes": 1, "angle_deg": 0.0},
    # a square of strokes at the pitch: a fill's darkness and its bands
    "solid": {"length_mm": 10.0, "pitch_mm": 0.4, "passes": 1, "angle_deg": 0.0},
    # parallel strokes fading from full deposition to `deposition` at their far end: a whip's fade
    "whip": {"lines": 12, "length_mm": 12.0, "pitch_mm": 0.8, "passes": 1, "angle_deg": 90.0, "deposition": 0.0},
    # an L, a zigzag and three short strokes: what stopping, turning and lifting leave
    "corners": {"length_mm": 10.0, "passes": 1, "angle_deg": 0.0},
}


def level(name: str) -> str:
    return LEVEL.get(name, "geometry")


def resolve(technique: str, settings: dict) -> dict:
    """A trial's full settings: its own over its technique's defaults, every key one the technique uses. Resource
    settings are never defaulted here: unset, the cartridge's datasheet gives them at compile."""
    if technique not in TECHNIQUES:
        raise ValueError(f"technique {technique!r} is not one of {', '.join(TECHNIQUES)}")
    if missing := [name for name in REQUIRED if name not in settings]:
        raise ValueError(f"a trial states {', '.join(missing)}: a depth and a speed are never defaulted")
    allowed = {*TECHNIQUES[technique], *REQUIRED, *DIP}
    if stray := sorted(set(settings) - allowed):
        raise ValueError(f"{technique} does not use {', '.join(stray)}; it uses {', '.join(sorted(allowed))}")
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    return {**TECHNIQUES[technique], **settings}


def speed_bounds(dose_bounds, machine_hz: float) -> list[float]:
    """The speed bounds [low, high] mm/s that a dose's [low, high] pokes per mm ask of a machine at machine_hz: the
    most pokes is the slowest."""
    low, high = (float(v) for v in dose_bounds)
    if not 0.0 < low <= high:
        raise ValueError(f"{DOSE} bounds are positive, low to high, got {low:g}:{high:g}")
    return [round(machine_hz / high, 3), round(machine_hz / low, 3)]


def dose_per_mm(study: dict, settings: dict) -> float:
    """A trial's steady dose: its study's machine rate over its speed, pokes per millimetre of line."""
    return float(study["constants"]["machine_hz"]) / float(settings["speed_mm_s"])


def dip_override(settings: dict) -> dict:
    """The trial's resource dip values, as tatbot-inks/3's `dip:` takes them; empty for the datasheet's."""
    return {key: round(settings[name] * scale, 6) for name, (key, scale) in DIP.items() if name in settings}


def run_key(trial: dict) -> tuple:
    """What trials sharing a run must agree on: the cartridge, the depth and the speed."""
    s = trial["settings"]
    return trial["cartridge"]["type"], float(s["trim_mm"]), float(s["speed_mm_s"])
